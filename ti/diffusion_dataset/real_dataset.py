from typing import Dict, List
import torch
import numpy as np
import zarr
import os
import shutil
from filelock import FileLock
from threadpoolctl import threadpool_limits
from omegaconf import OmegaConf
import cv2
import json
import hashlib
import copy
import h5py
import sys, pathlib
ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
sys.path.append(ROOT_DIR)
from ti.common.pytorch_util import dict_apply
from ti.diffusion_dataset.base_image_dataset import BaseImageDataset
from ti.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from ti.model.common.rotation_transformer import RotationTransformer
from ti.common.replay_buffer import ReplayBuffer
from ti.common.sampler import (
    SequenceSampler, get_val_mask, downsample_mask)
from ti.real_world.real_data_conversion import (real_zarr_to_replay_buffer)
from ti.common.normalize_util import (
    get_range_normalizer_from_stat,
    get_image_range_normalizer,
    get_identity_normalizer_from_stat,
    array_to_stats,
    calculate_max_abs_scaling_parameters
)
from ti.common.pcd_utils import visualize_pcd
import matplotlib.pyplot as plt
import cProfile
import pstats
import pathlib
import psutil


def check_available_ram():
    # Get the virtual memory statistics
    memory_info = psutil.virtual_memory()

    # Calculate available RAM in GB
    available_ram_gb = memory_info.available / (1024 ** 3)
    return available_ram_gb

def normalizer_from_stat(stat):
    """
    Creates a normalizer based on the Maximum Absolute Scaling technique.
    
    This function calculates the normalization parameters (scale and offset)
    based on the maximum absolute value in the provided statistics. The resulting
    normalizer scales data to the range [-1, 1], ensuring that the largest
    absolute value in the dataset, whether positive or negative, is normalized
    to 1 or -1, respectively. The offset is set to zero, indicating no shift
    after scaling.

    Parameters:
    - stat (dict): A dictionary containing the statistics of the dataset. Expected
                   to have 'min' and 'max' keys with corresponding values.

    Returns:
    - SingleFieldLinearNormalizer: An instance of SingleFieldLinearNormalizer
                                   configured with calculated scale and offset
                                   for Maximum Absolute Scaling normalization.
    """

    max_abs = np.maximum(stat['max'].max(), np.abs(stat['min']).max())
    scale = np.full_like(stat['max'], fill_value=1/max_abs)
    offset = np.zeros_like(stat['max'])
    return SingleFieldLinearNormalizer.create_manual(
        scale=scale,
        offset=offset,
        input_stats_dict=stat
    )

class RealDataset(BaseImageDataset):
    def __init__(self,
            shape_meta: dict,
            dataset_path: str,
            horizon=1,
            pad_before=0,
            pad_after=0,
            rotation_rep='rotation_6d',
            n_obs_steps=None,
            n_latency_steps=0,
            use_cache=False,
            seed=42,
            val_ratio=0.0,
            max_train_episodes=None,
            delta_action=False,
            debug_overfit=False,
            debug=False,
            use_robomimic_legacy_normalizer=True
        ):
        """Initializes the dataset.

        Args:
            shape_meta (Dict): The shape meta-data.
            dataset_path (str): The path to the dataset.
            horizon (int, optional): The horizon value. Defaults to 1.
            pad_before (int, optional): The pad before value. Defaults to 0.
            pad_after (int, optional): The pad after value. Defaults to 0.
            n_obs_steps (None, optional): The number of observation steps. Defaults to None.
            n_latency_steps (int, optional): The number of latency steps. Defaults to 0.
            use_cache (bool, optional): Flag to use cache. Defaults to False.
            seed (int, optional): The seed value. Defaults to 42.
            val_ratio (float, optional): The validation ratio. Defaults to 0.0.
            max_train_episodes (None, optional): The maximum number of training episodes. Defaults to None.
            delta_action (bool, optional): The delta action flag. Defaults to False.
        """

        assert os.path.isdir(dataset_path)
        rotation_transformer = RotationTransformer(
            from_rep='axis_angle', to_rep=rotation_rep)
        self.debug_overfit = debug_overfit
        save_format = 'zarr'
        replay_buffer = True
        # use_cache = False
        if use_cache:
            # fingerprint shape_meta
            shape_meta_json = json.dumps(OmegaConf.to_container(shape_meta), sort_keys=True)
            shape_meta_hash = hashlib.md5(shape_meta_json.encode('utf-8')).hexdigest()
            cache_zarr_path = os.path.join(dataset_path, shape_meta_hash + '.zarr.zip')
            cache_lock_path = cache_zarr_path + '.lock'
            print('Acquiring lock on cache.')
            with FileLock(cache_lock_path):
                if not os.path.exists(cache_zarr_path):
                    # cache does not exists
                    try:
                        print('Cache does not exist. Creating!')
                        replay_buffer = _get_replay_buffer(
                            dataset_path=dataset_path,
                            shape_meta=shape_meta,
                            store=zarr.MemoryStore(),
                            rotation_transformer=rotation_transformer,
                            save_format=save_format,
                            debug=debug
                        )
                        print('Saving cache to disk.')
                        with zarr.ZipStore(cache_zarr_path) as zip_store:
                            replay_buffer.save_to_store(
                                store=zip_store
                            )
                    except Exception as e:
                        shutil.rmtree(cache_zarr_path)
                        raise e
                else:
                    print('Loading cached ReplayBuffer from Disk.')
                    with zarr.ZipStore(cache_zarr_path, mode='r') as zip_store:
                        if check_available_ram() > 32:
                            replay_buffer = ReplayBuffer.copy_from_store(
                                src_store=zip_store, store=zarr.MemoryStore())
                        else:
                            replay_buffer = ReplayBuffer(root=zarr.group(zarr.ZipStore(cache_zarr_path, mode='r')))
                    print('Loaded!')
        else:
            replay_buffer = _get_replay_buffer(
                dataset_path=dataset_path,
                shape_meta=shape_meta,
                store=zarr.MemoryStore(),
                rotation_transformer=rotation_transformer,
                save_format=save_format
            )
        
        if delta_action:
            # replace action as relative to previous frame
            actions = replay_buffer['action'][:]
            # support positions only at this time
            assert actions.shape[1] <= 3
            actions_diff = np.zeros_like(actions)
            episode_ends = replay_buffer.episode_ends[:]
            for i in range(len(episode_ends)):
                start = 0
                if i > 0:
                    start = episode_ends[i-1]
                end = episode_ends[i]
                # delta action is the difference between previous desired position and the current
                # it should be scheduled at the previous timestep for the current timestep
                # to ensure consistency with positional mode
                actions_diff[start+1:end] = np.diff(actions[start:end], axis=0)
            replay_buffer['action'][:] = actions_diff

        rgb_keys = list()
        lowdim_keys = list()
        depth_keys = list()
        spatial_keys = list()
        feature_keys = list()
        obs_shape_meta = shape_meta['obs']
        for key, attr in obs_shape_meta.items():
            type = attr.get('type', 'low_dim')
            if type == 'rgb':
                rgb_keys.append(key)
            elif type == 'low_dim':
                lowdim_keys.append(key)
            elif type == 'depth':
                depth_keys.append(key)
            elif type == 'spatial':
                spatial_keys.append(key)
            elif type == 'feature':
                feature_keys.append(key)

        # key_first_k = dict()
        # if n_obs_steps is not None:
        #     # only take first k obs from images
        #     for key in rgb_keys + lowdim_keys + depth_keys:
        #         key_first_k[key] = n_obs_steps

        val_mask = get_val_mask(
            n_episodes=replay_buffer.n_episodes, 
            val_ratio=val_ratio,
            seed=seed)
        if self.debug_overfit:
            val_mask[:] = False
            val_mask[78] = True
            train_mask = val_mask
        else:
            train_mask = ~val_mask
        train_mask = downsample_mask(
            mask=train_mask, 
            max_n=max_train_episodes, 
            seed=seed)

        sampler = SequenceSampler(
            replay_buffer=replay_buffer, 
            sequence_length=horizon+n_latency_steps,
            pad_before=pad_before, 
            pad_after=pad_after,
            episode_mask=train_mask)
            # key_first_k=key_first_k
        
        self.replay_buffer = replay_buffer
        self.sampler = sampler
        self.shape_meta = shape_meta
        self.rgb_keys = rgb_keys
        self.depth_keys = depth_keys
        self.lowdim_keys = lowdim_keys
        self.spatial_keys = spatial_keys
        self.feature_keys = feature_keys
        self.obs_shape_meta = obs_shape_meta
        
        self.n_obs_steps = n_obs_steps
        self.val_mask = val_mask
        self.horizon = horizon
        self.n_latency_steps = n_latency_steps
        self.pad_before = pad_before
        self.pad_after = pad_after
        
        self.use_robomimic_legacy_normalizer = use_robomimic_legacy_normalizer

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=self.horizon+self.n_latency_steps,
            pad_before=self.pad_before, 
            pad_after=self.pad_after,
            episode_mask=self.val_mask
            )
        val_set.val_mask = ~self.val_mask
        return val_set

    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        normalizer = LinearNormalizer()

        # action
        stat = array_to_stats(self.replay_buffer['action'])
        if self.use_robomimic_legacy_normalizer:
            this_normalizer = normalizer_from_stat(stat)
            normalizer['action'] = this_normalizer
        else:
            normalizer['action'] = get_identity_normalizer_from_stat(stat)

        # obs
        for key in self.lowdim_keys:
            stat = array_to_stats(self.replay_buffer[key])
            if key.endswith('pose'): # eef pose: x, y, z, 6D rotation representation
                if self.use_robomimic_legacy_normalizer:
                    this_normalizer = normalizer_from_stat(stat)
                else:
                    this_normalizer = get_identity_normalizer_from_stat(stat)
            elif key.endswith('quat'):
                # quaternion is in [-1,1] already
                this_normalizer = get_identity_normalizer_from_stat(stat)
            elif key.endswith('joint'):
                this_normalizer = get_range_normalizer_from_stat(stat)
            elif key.endswith('vel'):
                this_normalizer = get_identity_normalizer_from_stat(stat)
            elif key.endswith('wrench'):
                this_normalizer = get_range_normalizer_from_stat(stat)
            else:
                this_normalizer = get_identity_normalizer_from_stat(stat)

            normalizer[key] = this_normalizer

            # normalizer[key] = SingleFieldLinearNormalizer.create_fit(
            #     self.replay_buffer[key])
        # image
        for key in self.rgb_keys:
            normalizer[key] = get_image_range_normalizer()
        for key in self.depth_keys:
            normalizer[key] = get_image_range_normalizer()
            
        for key in self.spatial_keys:
            B, N, C = self.replay_buffer[key].shape
            dtype = self.replay_buffer[key].dtype
            stat = array_to_stats(self.replay_buffer[key][()].reshape(B * N, C))
            xyz_stat = {key: value[:3] for key, value in stat.items()}
            xyz_scale, xyz_offset = calculate_max_abs_scaling_parameters(xyz_stat)
            scale = np.ones(C, dtype=dtype)  # Set scale to 1 for all elements
            offset = np.zeros(C, dtype=dtype)  # Set offset to 0 for all elements
            scale[:3] = xyz_scale
            offset[:3] = xyz_offset
            normalizer[key] = SingleFieldLinearNormalizer.create_manual(
                            scale=scale,
                            offset=offset,
                            input_stats_dict=stat
                        )
        for key in self.feature_keys:
            normalizer[key] = get_image_range_normalizer() #get_identity_normalizer_from_stat(array_to_stats(self.replay_buffer[key]))
            
        return normalizer

    def get_all_actions(self) -> torch.Tensor:
        return torch.from_numpy(self.replay_buffer['action'])

    def __len__(self):
        return len(self.sampler)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        threadpool_limits(1)
        data = self.sampler.sample_sequence(idx)

        # to save RAM, only return first n_obs_steps of OBS
        # since the rest will be discarded anyway.
        # when self.n_obs_steps is None
        # this slice does nothing (takes all)
        # T_slice = slice(self.n_obs_steps)
        T_slice = slice(None)

        obs_dict = dict()
        for key in self.rgb_keys:
            # move channel last to channel first
            # T,H,W,C
            # convert uint8 image to float32
            obs_dict[key] = np.moveaxis(data[key][T_slice],-1,1
                ).astype(np.float32) / 255.
            # T,C,H,W
            # save ram
            del data[key]
        for key in self.depth_keys:
            # move channel last to channel first
            # T,H,W,C
            # convert uint16 image to float32
            obs_dict[key] = np.moveaxis(data[key][T_slice],-1,1
                ).astype(np.float32) / 1000.
            # T,C,H,W
            del data[key]
        for key in self.lowdim_keys:
            obs_dict[key] = data[key][T_slice].astype(np.float32)
            # save ram
            del data[key]
        for key in self.spatial_keys:
            obs_dict[key] = np.moveaxis(data[key][T_slice],1,2).astype(np.float32)
            del data[key]
        for key in self.feature_keys:
            obs_dict[key] = np.moveaxis(data[key][T_slice],-1,1
                ).astype(np.float32) / 255.
            del data[key]

        action = data['action'].astype(np.float32)
        # handle latency by dropping first n_latency_steps action
        # observations are already taken care of by T_slice
        if self.n_latency_steps > 0:
            action = action[self.n_latency_steps:]

        torch_data = {
            'obs': dict_apply(obs_dict, torch.from_numpy),
            'action': torch.from_numpy(action)
        }
        return torch_data

def zarr_resize_index_last_dim(zarr_arr, idxs):
    actions = zarr_arr[:]
    actions = actions[...,idxs]
    zarr_arr.resize(zarr_arr.shape[:-1] + (len(idxs),))
    zarr_arr[:] = actions
    return zarr_arr

def _get_replay_buffer(dataset_path, shape_meta, store, rotation_transformer, save_format='hdf5', debug=False):

    # load data
    cv2.setNumThreads(1)
    with threadpool_limits(1):
        replay_buffer = real_zarr_to_replay_buffer(
            dataset_path=dataset_path,
            out_store=store,
            shape_meta=shape_meta,
            rotation_transformer=rotation_transformer,
            skip_close_joint=False,
            debug=debug
        )


    return replay_buffer


def test():
    import hydra
    from omegaconf import OmegaConf
    OmegaConf.register_new_resolver("eval", eval, replace=True)

    with hydra.initialize('../config'):
        cfg = hydra.compose('train_robomimic_real_image_workspace')
        OmegaConf.resolve(cfg)
        dataset = hydra.utils.instantiate(cfg.task.dataset)

    from matplotlib import pyplot as plt
    normalizer = dataset.get_normalizer()
    nactions = normalizer['action'].normalize(dataset.replay_buffer['action'][:])
    diff = np.diff(nactions, axis=0)
    dists = np.linalg.norm(np.diff(nactions, axis=0), axis=-1)
    plt.hist(dists, bins=100)
    _ = plt.title('real action velocity')
    plt.show(block = True)

if __name__ == '__main__':
    test()