from typing import Dict, Tuple
import numpy as np
from einops import rearrange, reduce
from ti.common.cv2_util import get_image_transform, load_images_mast3r, ImgNorm_inv

import os
from scipy.spatial.transform import Rotation as R
from ti.common.data_utils import  _convert_actions, point_cloud_proc
from ti.model.common.rotation_transformer import RotationTransformer
from ti.common.data_utils import _convert_actions
from ti.utils.segmentation import get_segmentation_mask
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

import cv2
import re
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../../third_party'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../third_party/FoundationPose'))
from FoundationPose.estimater import *
from FoundationPose.datareader import *

def get_real_obs_dict(
        env_obs: Dict[str, np.ndarray], 
        shape_meta: dict,
        task_conf=None,
        kin_helper=None,
        real_obs_info={},
        ) -> Dict[str, np.ndarray]:
    rotation_transformer = RotationTransformer(
        from_rep='axis_angle', to_rep='rotation_6d')
    segmentation_labels = task_conf.get('segmentation_labels', None)
    obs_dict_np = dict()
    obs_shape_meta = shape_meta['obs']
    device = 'cuda'
    if 'robot_eef_pose' in env_obs.keys() and 'keypoint' in env_obs.keys():
        obs_dict_np = {'obs': env_obs['keypoint_in_world']}
        return obs_dict_np
    for key, attr in obs_shape_meta.items():
        type = attr.get('type', 'low_dim')
        shape = attr.get('shape')
        if type == 'rgb':
            this_imgs_in = env_obs[key]
            t,hi,wi,ci = this_imgs_in.shape
            co,ho,wo = shape
            assert ci == co
            out_imgs = this_imgs_in
            if (ho != hi) or (wo != wi) or (this_imgs_in.dtype == np.uint8):
                tf = get_image_transform(
                    input_res=(wi,hi), 
                    output_res=(wo,ho), 
                    bgr_to_rgb=False)
                out_imgs = np.stack([tf(x) for x in this_imgs_in])
                if this_imgs_in.dtype == np.uint8:
                    out_imgs = out_imgs.astype(np.float32) / 255
            # THWC to TCHW
            obs_dict_np[key] = np.moveaxis(out_imgs,-1,1)
        if type == 'depth':
            this_imgs_in = env_obs[key]
            t,hi,wi = this_imgs_in.shape
            ho,wo = shape
            out_imgs = this_imgs_in
            if (ho != hi) or (wo != wi) or (this_imgs_in.dtype == np.uint8):
                out_imgs = np.stack([cv2.resize(x, (wo,ho), cv2.INTER_LANCZOS4) for x in this_imgs_in])
                if this_imgs_in.dtype == np.uint16:
                    out_imgs = out_imgs.astype(np.float32) / 1000
            obs_dict_np[key] = out_imgs
        elif type == 'low_dim':
            if 'tracks' in key or 'vis' in key:
                if 'vis' in key:
                    continue
                camera = '_'.join(key.split('_')[:2])
                camera_img = env_obs[f'{camera}_color']
                obs_len, H, W, C = camera_img.shape
                num_points = 32
                hand_masks, object_masks = zip(*[get_segmentation_mask(img[..., ::-1], segmentation_labels, 'tmp', save_img=False) for img in camera_img])
                hand_points = np.concatenate(
                    [
                        np.expand_dims(
                            sample_from_mask(
                                np.expand_dims(hand_mask, axis=-1) * 255,
                                num_samples=int(obs_shape_meta[key].shape[0] / 2)
                            ), axis=0
                        ).astype(np.float32)
                        for hand_mask in hand_masks
                    ], axis=0
                )
                object_points = np.concatenate(
                    [
                        np.expand_dims(
                            sample_from_mask(
                                np.expand_dims(object_mask, axis=-1) * 255,
                                num_samples=int(obs_shape_meta[key].shape[0] / 2)
                            ), axis=0
                        ).astype(np.float32)
                        for object_mask in object_masks
                    ], axis=0
                )
                all_points =(np.concatenate([hand_points, object_points], axis=1))

                all_points[..., 0] /= H
                all_points[..., 1] /= W
                

                all_vis = np.ones(all_points.shape[:-1])
                
                # tracks, vis = sample_tracks_fps(all_points, all_vis, num_samples=num_points)
                obs_dict_np[f'{camera}_tracks']= all_points
                obs_dict_np[f'{camera}_vis'] = all_vis
                continue

            this_data_in = env_obs[key]
            if 'pose' in key and shape == (2,):
                # take X,Y coordinates
                this_data_in = this_data_in[...,[0,1]]
            if key =='robot_eef_pose':
                this_data_in = _convert_actions(this_data_in, rotation_transformer, 'cartesian_action')
                this_data_in = this_data_in[:, :shape_meta['obs']['robot_eef_pose'].shape[0]]
            obs_dict_np[key] = this_data_in
        elif type == 'spatial':
            try:
                assert key == 'point_cloud'
            except AssertionError:
                raise RuntimeError('Only support point_cloud as spatial type.')
            # construct inputs for d3fields processing
            view_keys = attr['info']['view_keys']
            tool_names = [None, None]
            if 'right_tool' in attr['info']:
                tool_names[0] = attr['info']['right_tool']
            if 'left_tool' in attr['info']:
                tool_names[1] = attr['info']['left_tool']
            color_seq = np.stack([env_obs[f'{k}_color'] for k in view_keys], axis=1) # (T, V, H ,W, C)
            depth_seq = np.stack([env_obs[f'{k}_depth'] for k in view_keys], axis=1) / 1000. # (T, V, H ,W)
            extri_seq = np.stack([env_obs[f'{k}_extrinsics'] for k in view_keys], axis=1) # (T, V, 4, 4)
            intri_seq = np.stack([env_obs[f'{k}_intrinsics'] for k in view_keys], axis=1) # (T, V, 3, 3)
            robot_base_pose_in_world_seq = env_obs['robot_base_pose_in_world'] # (T, 4, 4)
            qpos_seq = env_obs['robot_joint']# (T, -1)
            exclude_colors = attr['info'].get('exclude_colors', [])
            aggr_src_pts_ls = point_cloud_proc(
                shape_meta=shape_meta['obs'][key],
                color_seq=color_seq,
                depth_seq=depth_seq,
                extri_seq=extri_seq,
                intri_seq=intri_seq,
                robot_base_pose_in_world_seq=robot_base_pose_in_world_seq,
                qpos_seq=qpos_seq,
                expected_labels=None,
                exclude_colors=exclude_colors,
                teleop_robot=kin_helper,
                tool_names=tool_names,

            )
            aggr_src_pts = np.stack(aggr_src_pts_ls)
            obs_dict_np[key] = aggr_src_pts.transpose(0,2,1)
        elif type == 'feature':
            mast3r_size = max(shape_meta['obs']['camera_features']['original_shape'])
            test_feature_view_key = real_obs_info['feature_view_key'] 
            segment = shape_meta['obs']['camera_features']['info'].get("segment", False)

            
            view_image = env_obs[f'{test_feature_view_key}_color'] # (T, H ,W, C)
            remove_labels = ['human arms.sleeve.shirt.pants.hoody.jacket.robot.kinova.ur5e']
            if segment:
                color_seq = np.array([
                    get_segmentation_mask(
                        frame,  # Each image is already in BGR format
                        segmentation_labels=remove_labels,
                        saved_path=None,
                        grounding_dino_model=real_obs_info['grounding_dino_model'],
                        sam_predictor=real_obs_info['sam_predictor'],
                        debug=False,
                        save_img=False,
                        remove_segmented=True
                    )[1][..., ::-1]
                    for frame in view_image[..., ::-1]
                ])
            else:
                color_seq = view_image


            mast3r_imgs = load_images_mast3r(color_seq, size=mast3r_size, verbose=False)
            c,h,w = tuple(shape_meta['obs']['camera_features']['shape'])

            obs_dict_np[key] = F.interpolate(torch.concat([ImgNorm_inv(mast3r_imgs[i]['img']) for i in range(len(mast3r_imgs))]),size=(h,w), mode='area').cpu().numpy()

    return obs_dict_np



def get_tool_in_cam(pose_model, segmentation_labels, obs, shape_meta, to_origin, bbox, camera_key, vis_img=True, init=False, grounding_dino_model=None, sam_predictor=None):
    # pattern = re.compile(r'camera_\d')
    # if 'camera_features' in shape_meta['obs']:
    #     camera_key = [key for key in shape_meta['obs']['camera_features']['info']['view_keys'] if pattern.match(key)][0]
    # else:
    #     camera_color_key = [key for key in shape_meta['obs'].keys() if pattern.match(key)][0]
    #     camera_key = camera_color_key.split('_')[0] + '_' + camera_color_key.split('_')[1]
    rgb = obs[camera_key + '_color'][-1].copy()
    depth = obs[camera_key + '_depth'][-1].astype(np.float64)/1000
    intrinsics = obs[camera_key + '_intrinsics'][-1].astype(np.float64)
    if init==True:
        tool_mask, _ = get_segmentation_mask(rgb[..., ::-1], segmentation_labels, 'tmp.png', grounding_dino_model, sam_predictor)
        pose = pose_model.register(K=intrinsics, rgb=rgb, depth=depth, ob_mask=tool_mask, iteration=5)
    else:
        pose = pose_model.track_one(rgb=rgb, depth=depth, K=intrinsics, iteration=2)
    if vis_img:
        center_pose = pose@np.linalg.inv(to_origin)
        vis_rgb = draw_posed_3d_box(intrinsics, img=rgb, ob_in_cam=center_pose, bbox=bbox)
        vis_rgb = draw_xyz_axis(rgb, ob_in_cam=center_pose, scale=0.1, K=intrinsics, thickness=3, transparency=0, is_input_rgb=True)

    return pose, vis_rgb

def get_real_obs_resolution(
        shape_meta: dict
        ) -> Tuple[int, int]:
    out_res = None
    obs_shape_meta = shape_meta['obs']
    for key, attr in obs_shape_meta.items():
        type = attr.get('type', 'low_dim')
        shape = attr.get('shape')
        if type == 'rgb':
            co,ho,wo = shape
            if out_res is None:
                out_res = (wo, ho)
            assert out_res == (wo, ho)
    return out_res