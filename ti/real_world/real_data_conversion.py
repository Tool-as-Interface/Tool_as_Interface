from typing import Sequence, Tuple, Dict, Optional, Union
import os
import pathlib
import numpy as np
import matplotlib.pyplot as plt
import av
import zarr
import numcodecs
import multiprocessing
import concurrent.futures
from tqdm import tqdm
from ti.common.replay_buffer import ReplayBuffer, get_optimal_chunks
from ti.real_world.video_recorder import read_video, read_cam_video
from ti.common.trans_utils import are_joints_close
from ti.utils.segmentation import get_segmentation_mask, initialize_segmentation_models
import cv2
import glob
import base64
from omegaconf import OmegaConf
import h5py
from pathlib import Path
from ti.codecs.imagecodecs_numcodecs import (
    register_codecs,
    Jpeg2k,
    JpegXl,
)
import cProfile
import pstats
import io
import dask.array as da
from dask import delayed
import dask.bag as db
import gc
from ti.common.cv2_util import get_image_transform, load_images_mast3r
register_codecs()
from ti.common.data_utils import  _convert_actions, point_cloud_proc, feature_proc
from ti.common.pcd_utils import visualize_pcd
from ti.common.kin_utils import RobotKinHelper
import starster
import time
from contextlib import contextmanager

import sys
import re
import torch
import dill
from dask.distributed import Client

from mast3r.model import AsymmetricMASt3R
from mast3r.fast_nn import fast_reciprocal_NNs
from dust3r.inference import inference
from dust3r.utils.image import load_images




def real_zarr_to_replay_buffer(dataset_path, out_store, shape_meta, rotation_transformer, 
        n_workers=None, max_inflight_tasks=None, robot_name='ur5e', expected_labels=None,
        exclude_colors=[],
        lowdim_compressor: Optional[numcodecs.abc.Codec]=None,
        skip_close_joint: bool=True, profile: bool=False, debug=False):

    n_workers = n_workers or multiprocessing.cpu_count()
    max_inflight_tasks = max_inflight_tasks or n_workers * 5
    image_compressor = Jpeg2k(level=50)
    image_compressor_lossless = Jpeg2k(level=0)  # Use level=0 for lossless compression

    root = zarr.group(out_store)
    data_group = root.require_group('data', overwrite=True)
    meta_group = root.require_group('meta', overwrite=True)

    # verify input
    input_path = pathlib.Path(dataset_path).expanduser()
    in_zarr_path = input_path.joinpath('replay_buffer.zarr')
    assert in_zarr_path.is_dir(), f"{in_zarr_path} does not exist"


    in_replay_buffer = ReplayBuffer.create_from_path(str(in_zarr_path.absolute()), mode='r')
    episode_ends = in_replay_buffer.episode_ends[:] 
    n_steps = in_replay_buffer.n_steps
    episode_starts = in_replay_buffer.episode_ends[:] - in_replay_buffer.episode_lengths[:]
    episode_lengths = in_replay_buffer.episode_lengths
    timestamps = in_replay_buffer['timestamp'][:]
    
    if debug:
        episode_ends = episode_ends[:1]
        episode_starts = episode_starts[:1]
        episode_lengths = episode_lengths[:1]
        n_steps = episode_ends[-1]
        
    device = 'cuda'
    aug_mult = 1  # Set the augmentation multiplier to the desired number of repetitions
    
    # parse shape_meta
    rgb_keys = list()
    depth_keys = list()
    lowdim_keys = list()
    spatial_keys = list()
    feature_keys = list()
    # construct compressors and chunks
    obs_shape_meta = shape_meta['obs']
    for key, attr in obs_shape_meta.items():
        shape = attr['shape']
        type = attr.get('type', 'low_dim')
        if type == 'rgb':
            rgb_keys.append(key)
        if type == 'depth':
            depth_keys.append(key)
        elif type == 'low_dim':
            lowdim_keys.append(key)
        elif type == 'spatial':
            spatial_keys.append(key)
            max_pts_num = obs_shape_meta[key]['shape'][1]
        elif type == 'feature':
            feature_keys.append(key)

    lowdim_data_dict = dict()
    rgb_data_dict = dict()
    depth_data_dict = dict()
    spatial_data_dict = dict()
    feature_data_dict = dict()
    
    if skip_close_joint:
        not_moving_mask = are_joints_close(in_replay_buffer['observations']['robot_joint'], in_replay_buffer['joint_action'])
        moving_mask = ~not_moving_mask
    else:
        moving_mask = np.ones(n_steps, dtype=bool)
    
    updated_episode_end = 0
    updated_episode_ends_ls = []
    if 'point_cloud' in shape_meta['obs']:
        tool_names = [None, None]
        if 'left_tool' in shape_meta['obs']['point_cloud']['info']:
            tool_names[0] = shape_meta['obs']['point_cloud']['info']['left_tool']

        if 'right_tool' in shape_meta['obs']['point_cloud']['info']:
            tool_names[1] = shape_meta['obs']['point_cloud']['info']['right_tool']
        N_per_link = shape_meta['obs']['point_cloud']['info']['N_per_link']
        N_eef = shape_meta['obs']['point_cloud']['info']['N_eef']

        kin_helper = RobotKinHelper(robot_name, tool_name=tool_names[1], N_per_link=N_per_link, N_tool=N_eef)

    if 'camera_features' in shape_meta['obs']:
        mast3r_size = max(shape_meta['obs']['camera_features']['original_shape'])
        mast3r_model_name = "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"
        starster_mast3r_model = starster.Mast3rModel.from_pretrained(mast3r_model_name).to(device)
        
        grounding_dino_model, sam_predictor = initialize_segmentation_models()
        remove_labels = ['human arms.sleeve.shirt.pants.hoody.jacket.robot.kinova.ur5e']
        
        feature_view_keys = shape_meta['obs']['camera_features']['info']['view_keys']
        aug_img_num = shape_meta['obs']['camera_features']['info']['aug_img_num']
        
        aug_mult = aug_mult + aug_img_num
                
        init_features = []
        novel_view = shape_meta['obs']['camera_features']['info']['novel_view']
        segment = shape_meta['obs']['camera_features']['info'].get('segment', False)
    
    aug_mult_epi_ends = np.arange(1, aug_mult + 1)
    
    for epi_idx, episode_length in tqdm(enumerate(episode_lengths), desc=f"Loading episodes"):
        episode_start = episode_starts[epi_idx]
        episode_slice = slice(episode_start, episode_start + episode_length, 1)
        episode_mask = moving_mask[episode_slice]
        
        updated_episode_length = np.sum(moving_mask[episode_slice])-1
        
        augmented_ends = updated_episode_end + aug_mult_epi_ends * updated_episode_length
        updated_episode_ends_ls.extend(augmented_ends)
        updated_episode_end = updated_episode_ends_ls[-1]

        # save lowdim data to lowedim_data_dict
        for key in lowdim_keys + ['action']:
            data_key = f"observations/images/{key}" if 'camera' in key else f"observations/{key}"
            if key == 'action':
                data_key = 'cartesian_action' if 'type' not in shape_meta['action'] else shape_meta['action']['type']
            if key not in lowdim_data_dict:
                lowdim_data_dict[key] = list()
            if key != 'action':
                this_data = in_replay_buffer[data_key][episode_slice][episode_mask][:-1]
            if key == 'action':
                # add this because action reading sometimes jump to some value that is far away from the previous value
                if data_key == 'cartesian_action':
                    this_data = in_replay_buffer['observations/robot_eef_pose'][episode_slice][episode_mask][1:]
                elif data_key == 'joint_action':
                    this_data = in_replay_buffer['observations/robot_joint'][episode_slice][episode_mask][1:]
                else:
                    raise ValueError(f"Unexpected action type: {data_key}")
                this_data = _convert_actions(
                    raw_actions=this_data,
                    rotation_transformer=rotation_transformer,
                    action_key=data_key,
                )
                # this_data = this_data[:,:shape_meta['action']['shape'][0]]
                assert this_data.shape == (updated_episode_length,) + tuple(shape_meta['action']['shape'])
            elif key == 'robot_eef_pose':
                this_data = _convert_actions(
                    raw_actions=this_data,
                    rotation_transformer=rotation_transformer,
                    action_key=key,
                )
                this_data = this_data[:,:shape_meta['obs'][key]['shape'][0]]
                assert this_data.shape == (updated_episode_length,) + tuple(shape_meta['obs'][key]['shape'])
            else:
                assert this_data.shape == (updated_episode_length,) + tuple(shape_meta['obs'][key]['shape'])
            # lowdim_data_dict[key].append(this_data.astype(np.float32))
            lowdim_data_dict[key].extend([np.copy(this_data).astype(np.float32)] * aug_mult)

        for key in rgb_keys:
            if key not in rgb_data_dict:
                rgb_data_dict[key] = list()
            imgs = in_replay_buffer['observations']['images'][key][episode_slice][episode_mask][:-1]
            shape = tuple(shape_meta['obs'][key]['shape'])
            c,h,w = shape
            resize_imgs = [cv2.resize(img, (w,h), interpolation=cv2.INTER_AREA) for img in imgs]
            imgs = np.stack(resize_imgs, axis=0)
            assert imgs[0].shape == (h,w,c)
            rgb_data_dict[key].append(imgs)
        
        for key in depth_keys:
            if key not in depth_data_dict:
                depth_data_dict[key] = list()
            imgs = in_replay_buffer['observations']['images'][key][episode_slice][episode_mask][:-1]
            shape = tuple(shape_meta['obs'][key]['shape'])
            h,w = shape
            resize_imgs = [cv2.resize(img, (w,h), interpolation=cv2.INTER_AREA) for img in imgs]
            imgs = np.stack(resize_imgs, axis=0)[..., None]
            imgs = np.clip(imgs, 0, 1000).astype(np.uint16)
            assert imgs[0].shape == (h,w,1)
            depth_data_dict[key].append(imgs)
        
        for key in spatial_keys:
            assert key == 'point_cloud' 
            
            # construct inputs for d3fields processing
            view_keys = shape_meta['obs'][key]['info']['view_keys']
            color_seq = np.stack([in_replay_buffer['observations']['images'][f'{k}_color'][episode_slice][episode_mask][:-1] for k in view_keys], axis=1) # (T, V, H ,W, C)
            depth_seq = np.stack([in_replay_buffer['observations']['images'][f'{k}_depth'][episode_slice][episode_mask][:-1] for k in view_keys], axis=1) / 1000. # (T, V, H ,W)
            extri_seq = np.stack([in_replay_buffer['observations']['images'][f'{k}_extrinsics'][episode_slice][episode_mask][:-1] for k in view_keys], axis=1) # (T, V, 4, 4)
            intri_seq = np.stack([in_replay_buffer['observations']['images'][f'{k}_intrinsics'][episode_slice][episode_mask][:-1] for k in view_keys], axis=1) # (T, V, 3, 3)
            qpos_seq = in_replay_buffer['observations']['robot_joint'][episode_slice][episode_mask][:-1] # (T, -1)
            robot_base_pose_in_world_seq = in_replay_buffer['observations']['robot_base_pose_in_world'][episode_slice][episode_mask][:-1]
            exclude_colors = shape_meta['obs'][key]['info'].get('exclude_colors', [])

            aggr_src_pts_ls = point_cloud_proc(
                shape_meta=shape_meta['obs'][key],
                color_seq=color_seq,
                depth_seq=depth_seq,
                extri_seq=extri_seq,
                intri_seq=intri_seq,
                robot_base_pose_in_world_seq=robot_base_pose_in_world_seq,
                qpos_seq=qpos_seq,
                expected_labels=expected_labels,
                exclude_colors=exclude_colors,
                teleop_robot=kin_helper,
                tool_names=tool_names,
            )

            
            if key not in spatial_data_dict:
                spatial_data_dict[key] = list()
            
            spatial_data_dict[key] = spatial_data_dict[key] + aggr_src_pts_ls


        for key in feature_keys:
            assert key == 'camera_features'
            if key not in feature_data_dict:
                feature_data_dict[key] = list()

            color_seq = np.stack([in_replay_buffer['observations']['images'][f'{k}_color'][episode_slice][episode_mask][:-1] for k in feature_view_keys], axis=1) # (T, V, H ,W, C)                        
            if segment:
                processed_images = np.array([
                    get_segmentation_mask(
                        view_image,  # Each image is already in BGR format
                        segmentation_labels=remove_labels,
                        saved_path=None,
                        grounding_dino_model=grounding_dino_model,
                        sam_predictor=sam_predictor,
                        debug=False,
                        save_img=False,
                        remove_segmented=True
                    )[1][..., ::-1]
                    for frame in color_seq[..., ::-1] for view_image in frame  # Iterate over each frame and each image within it
                ])
                processed_images = processed_images.reshape(color_seq.shape)  # Now both are (T, V, H, W, C)
                color_seq = processed_images
            view_keys = shape_meta['obs'][key]['info']['view_keys']
            episode_views_features = feature_proc(color_seq,
                                                aug_img_num, shape_meta, 
                                                aug_mult, updated_episode_length, 
                                                mast3r_size, starster_mast3r_model, 
                                                device, novel_view)

            feature_data_dict[key].append(episode_views_features)


            init_features.append(feature_data_dict[key][-1][0]) 
            
            gc.collect()
            torch.cuda.empty_cache()




    def img_copy(zarr_arr, zarr_idx, hdf5_arr, hdf5_idx):
        try:
            zarr_arr[zarr_idx] = hdf5_arr[hdf5_idx]
            # make sure we can successfully decode
            _ = zarr_arr[zarr_idx]
            return True
        except Exception as e:
            print(f"Error copying data from hdf5_idx {hdf5_idx} to zarr_idx {zarr_idx} : {e}")
            return False
        
    # dump data_dict
    print('Dumping meta data')
    n_steps = updated_episode_ends_ls[-1]
    _ = meta_group.array('episode_ends', updated_episode_ends_ls, 
        dtype=np.int64, compressor=None, overwrite=True)

    print('Dumping lowdim data')
    for key, data in lowdim_data_dict.items():
        data = np.concatenate(data, axis=0)
        data = data.astype(np.float32)  
        
        _ = data_group.array(
            name=key,
            data=data,
            shape=data.shape,
            chunks=data.shape,
            compressor=None,
            dtype=data.dtype
        )
    
    print('Dumping rgb data')
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = set()
        for key, data in rgb_data_dict.items():
            hdf5_arr = np.concatenate(data, axis=0)
            # hdf5_arr = np.tile(hdf5_arr, (aug_mult, 1, 1, 1))

            shape = tuple(shape_meta['obs'][key]['shape'])
            c,h,w = shape
            # total_steps = n_steps * aug_mult  
            
            if key not in data_group:
                img_arr = data_group.require_dataset(
                    name=key,
                    shape=(n_steps,h,w,c),
                    chunks=(1,h,w,c),
                    compressor=image_compressor,
                    dtype=np.uint8
                )
            for hdf5_idx in tqdm(range(n_steps)):
                if len(futures) >= max_inflight_tasks:
                    # limit number of inflight tasks
                    completed, futures = concurrent.futures.wait(futures, 
                        return_when=concurrent.futures.FIRST_COMPLETED)
                    for f in completed:
                        if not f.result():
                            raise RuntimeError('Failed to encode image!')
                zarr_idx = hdf5_idx
                futures.add(
                    executor.submit(img_copy, img_arr, zarr_idx, hdf5_arr, hdf5_idx)
                )
                

        completed, futures = concurrent.futures.wait(futures)
        for f in completed:
            if not f.result():
                raise RuntimeError('Failed to encode image!')

    print('Dumping depth data')
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = set()
        for key, data in depth_data_dict.items():
            hdf5_arr = np.concatenate(data, axis=0)
            shape = tuple(shape_meta['obs'][key]['shape'])
            c = 1
            h,w = shape
            img_arr = data_group.require_dataset(
                name=key,
                shape=(n_steps,h,w,c),
                chunks=(1,h,w,c),
                compressor=image_compressor_lossless,
                dtype=np.uint16
            )
            for hdf5_idx in tqdm(range(hdf5_arr.shape[0])):
                if len(futures) >= max_inflight_tasks:
                    # limit number of inflight tasks
                    completed, futures = concurrent.futures.wait(futures, 
                        return_when=concurrent.futures.FIRST_COMPLETED)
                    for f in completed:
                        if not f.result():
                            raise RuntimeError('Failed to encode image!')
                zarr_idx = hdf5_idx
                futures.add(
                    executor.submit(img_copy, 
                        img_arr, zarr_idx, hdf5_arr, hdf5_idx))
        completed, futures = concurrent.futures.wait(futures)
        for f in completed:
            if not f.result():
                raise RuntimeError('Failed to encode image!')
            
    # dump spatial data
    print('Dumping spatial data')
    for key, data in spatial_data_dict.items():
        data = np.stack(data, axis=0) # (T, N, 1000)
        _ = data_group.array(
            name=key,
            data=data,
            shape=data.shape,
            chunks=(1,) + data.shape[1:],
            compressor=None,
            dtype=data.dtype
        )
        
    print('Dumping mast3r feature data')
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = set()
        for key, data in feature_data_dict.items():
            c, h, w = shape_meta['obs'][key]['shape']
            hdf5_arr = np.concatenate(data, axis=0)


            hdf5_arr = (hdf5_arr * 255).astype(np.uint8)
            img_arr = data_group.require_dataset(
                name=key,
                shape=(n_steps,h,w,c),
                chunks=(1,h,w,c),
                compressor=image_compressor,
                dtype=np.uint8
            )
            for hdf5_idx in tqdm(range(hdf5_arr.shape[0])):
                if len(futures) >= max_inflight_tasks:
                    # limit number of inflight tasks
                    completed, futures = concurrent.futures.wait(futures, 
                        return_when=concurrent.futures.FIRST_COMPLETED)
                    for f in completed:
                        if not f.result():
                            raise RuntimeError('Failed to encode image!')
                zarr_idx = hdf5_idx
                futures.add(
                    executor.submit(img_copy, 
                        img_arr, zarr_idx, hdf5_arr, hdf5_idx))
        completed, futures = concurrent.futures.wait(futures)
        for f in completed:
            if not f.result():
                raise RuntimeError('Failed to encode image!')

    replay_buffer = ReplayBuffer(root)
    return replay_buffer