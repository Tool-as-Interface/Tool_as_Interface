import numpy as np
import copy

import h5py
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.env_utils as EnvUtils
from scipy.spatial.transform import Rotation

from robomimic.config import config_factory


class RobomimicAbsoluteActionConverter:
    def __init__(self, dataset_path, algo_name='bc'):
        # default BC config
        config = config_factory(algo_name=algo_name)

        # read config to set up metadata for observation modalities (e.g. detecting rgb observations)
        # must ran before create dataset
        ObsUtils.initialize_obs_utils_with_config(config)

        env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path)
        abs_env_meta = copy.deepcopy(env_meta)
        abs_env_meta['env_kwargs']['controller_configs']['control_delta'] = False

        env = EnvUtils.create_env_from_metadata(
            env_meta=env_meta,
            render=False, 
            render_offscreen=False,
            use_image_obs=False, 
        )
        assert len(env.env.robots) in (1, 2)

        abs_env = EnvUtils.create_env_from_metadata(
            env_meta=abs_env_meta,
            render=False, 
            render_offscreen=False,
            use_image_obs=False, 
        )
        assert not abs_env.env.robots[0].controller.use_delta

        self.env = env
        self.abs_env = abs_env
        self.file = h5py.File(dataset_path, 'r')
    
    def __len__(self):
        return len(self.file['data'])

    def convert_actions(self, 
            states: np.ndarray, 
            actions: np.ndarray) -> np.ndarray:
        """
        Given state and delta action sequence
        generate equivalent goal position and orientation for each step
        keep the original gripper action intact.
        """
        # in case of multi robot
        # reshape (N,14) to (N,2,7)
        # or (N,7) to (N,1,7)
        stacked_actions = actions.reshape(*actions.shape[:-1],-1,7)

        env = self.env
        # generate abs actions
        action_goal_pos = np.zeros(
            stacked_actions.shape[:-1]+(3,), 
            dtype=stacked_actions.dtype)
        action_goal_ori = np.zeros(
            stacked_actions.shape[:-1]+(3,), 
            dtype=stacked_actions.dtype)
        action_gripper = stacked_actions[...,[-1]]
        for i in range(len(states)):
            _ = env.reset_to({'states': states[i]})

            # taken from robot_env.py L#454
            for idx, robot in enumerate(env.env.robots):
                # run controller goal generator
                robot.control(stacked_actions[i,idx], policy_step=True)
            
                # read pos and ori from robots
                controller = robot.controller
                action_goal_pos[i,idx] = controller.goal_pos
                action_goal_ori[i,idx] = Rotation.from_matrix(
                    controller.goal_ori).as_rotvec()

        stacked_abs_actions = np.concatenate([
            action_goal_pos,
            action_goal_ori,
            action_gripper
        ], axis=-1)
        abs_actions = stacked_abs_actions.reshape(actions.shape)
        return abs_actions

    def convert_idx(self, idx):
        file = self.file
        demo = file[f'data/demo_{idx}']
        # input
        states = demo['states'][:]
        actions = demo['actions'][:]

        # generate abs actions
        abs_actions = self.convert_actions(states, actions)
        return abs_actions

    def convert_and_eval_idx(self, idx):
        env = self.env
        abs_env = self.abs_env
        file = self.file
        # first step have high error for some reason, not representative
        eval_skip_steps = 1

        demo = file[f'data/demo_{idx}']
        # input
        states = demo['states'][:]
        actions = demo['actions'][:]

        # generate abs actions
        abs_actions = self.convert_actions(states, actions)

        # verify
        robot0_eef_pos = demo['obs']['robot0_eef_pos'][:]
        robot0_eef_quat = demo['obs']['robot0_eef_quat'][:]

        delta_error_info = self.evaluate_rollout_error(
            env, states, actions, robot0_eef_pos, robot0_eef_quat, 
            metric_skip_steps=eval_skip_steps)
        abs_error_info = self.evaluate_rollout_error(
            abs_env, states, abs_actions, robot0_eef_pos, robot0_eef_quat,
            metric_skip_steps=eval_skip_steps)

        info = {
            'delta_max_error': delta_error_info,
            'abs_max_error': abs_error_info
        }
        return abs_actions, info

    @staticmethod
    def evaluate_rollout_error(env, 
            states, actions, 
            robot0_eef_pos, 
            robot0_eef_quat, 
            metric_skip_steps=1):
        # first step have high error for some reason, not representative

        # evaluate abs actions
        rollout_next_states = list()
        rollout_next_eef_pos = list()
        rollout_next_eef_quat = list()
        obs = env.reset_to({'states': states[0]})
        for i in range(len(states)):
            obs = env.reset_to({'states': states[i]})
            obs, reward, done, info = env.step(actions[i])
            obs = env.get_observation()
            rollout_next_states.append(env.get_state()['states'])
            rollout_next_eef_pos.append(obs['robot0_eef_pos'])
            rollout_next_eef_quat.append(obs['robot0_eef_quat'])
        rollout_next_states = np.array(rollout_next_states)
        rollout_next_eef_pos = np.array(rollout_next_eef_pos)
        rollout_next_eef_quat = np.array(rollout_next_eef_quat)

        next_state_diff = states[1:] - rollout_next_states[:-1]
        max_next_state_diff = np.max(np.abs(next_state_diff[metric_skip_steps:]))

        next_eef_pos_diff = robot0_eef_pos[1:] - rollout_next_eef_pos[:-1]
        next_eef_pos_dist = np.linalg.norm(next_eef_pos_diff, axis=-1)
        max_next_eef_pos_dist = next_eef_pos_dist[metric_skip_steps:].max()

        next_eef_rot_diff = Rotation.from_quat(robot0_eef_quat[1:]) \
            * Rotation.from_quat(rollout_next_eef_quat[:-1]).inv()
        next_eef_rot_dist = next_eef_rot_diff.magnitude()
        max_next_eef_rot_dist = next_eef_rot_dist[metric_skip_steps:].max()

        info = {
            'state': max_next_state_diff,
            'pos': max_next_eef_pos_dist,
            'rot': max_next_eef_rot_dist
        }
        return info



def robomimic_process_observations(obs_keys, raw_obs, quat_transformer, obj_pos_idx, obj_quat_idx, single_obs, find_obs_length=False):
    """
    Processes observation data by converting quaternion data to a 6D continuous rotation 
    representation and reordering observation elements based on position and quaternion indices. 
    Optionally, calculates the total lengths of position and quaternion elements.

    The function handles two types of data in 'raw_obs': quaternion data (to be transformed) 
    and non-quaternion data. The order of elements in the output is first all position elements, 
    then quaternion elements (in 6D representation), and finally any remaining elements.

    Parameters:
    obs_keys (list): List of keys to process in raw_obs.
    raw_obs (dict): Dictionary containing raw observation data.
    quat_transformer (object): Object with a 'forward' method for quaternion transformation.
    obj_pos_idx (list of tuples): Index ranges for position data in 'object' keys.
    obj_quat_idx (list of tuples): Index ranges for quaternion data in 'object' keys.
    single_obs (bool): Flag to determine slicing behavior (single observation or batch).
    find_obs_length (bool): Flag to determine if total lengths should be calculated.

    Returns:
    np.ndarray: The processed observation data.
    int (optional): Total length of position elements, if find_obs_length is True.
    int (optional): Total length of quaternion elements in 6D representation, if find_obs_length is True.
    """

    def slice_func(data, start, end):
        """
        Slices the given data from 'start' to 'end'. Slicing behavior depends on 'single_obs'.
        For single observation, slices along the first dimension; for batch, slices along the second dimension.
        """
        return data[start:end] if single_obs else data[:, start:end]

    def calculate_segment_length(index_ranges):
        """
        Calculates the total length of segments defined in 'index_ranges'.
        Each range in 'index_ranges' is a tuple (start, end).
        """
        return sum(end - start for start, end in index_ranges)

    # Initialize total lengths for position and quaternion segments
    obj_pos_length = calculate_segment_length(obj_pos_idx) 

    # Calculate the length of a single transformed quaternion to estimate total length
    quat_sample = np.zeros((4,))
    rotation_6d_sample_length = quat_transformer.forward(quat_sample).shape[-1]
    obj_quat_length = sum((end - start) for start, end in obj_quat_idx) 
    obj_rot_length = sum((end - start) * rotation_6d_sample_length // 4 for start, end in obj_quat_idx) 

    obs = [None] * len(obs_keys)  # Preallocate list for observation data
    for i, key in enumerate(obs_keys):
        if 'object' in key:
            # Adjust the total length calculation to account for the increased size of 6D quaternions
            object_length = len(raw_obs[key]) + obj_rot_length - obj_quat_length  if single_obs else raw_obs[key].shape[1]  + obj_rot_length - obj_quat_length

            # Calculate the total length of the object observation array
            # last_range_end = max(obj_pos_idx[-1][-1], obj_quat_idx[-1][-1])
            # object_length = len(raw_obs[key])  if single_obs else raw_obs[key].shape[1] 

            # Preallocate array for object observations
            obj_obs_array = np.empty((object_length, ), dtype=np.float64) if single_obs else np.empty((raw_obs[key].shape[0], object_length), dtype=np.float64)

            current_idx = 0
            # Process and insert position elements
            for start, end in obj_pos_idx:
                obj_obs_array[..., current_idx:current_idx + (end - start)] = slice_func(raw_obs[key], start, end)
                current_idx += end - start

            # Process, transform, and insert quaternion elements
            for start, end in obj_quat_idx:
                quat_slice = slice_func(raw_obs[key], start, end)
                rotation_6d_slice = quat_transformer.forward(np.array(quat_slice, dtype=np.float64))
                obj_obs_array[..., current_idx:current_idx + rotation_6d_slice.shape[-1]] = rotation_6d_slice
                current_idx += rotation_6d_slice.shape[-1]


            # Fill in any remaining elements
            if current_idx < object_length:
                obj_obs_array[..., current_idx:] = slice_func(raw_obs[key], obj_pos_length+obj_quat_length, None)

            obs[i] = obj_obs_array
                        
        elif 'quat' in key:
            # Transform quaternion data using the provided transformer
            obs[i] = quat_transformer.forward(np.array(raw_obs[key], dtype=np.float64))

        else:
            obs[i] = raw_obs[key]

    # Concatenate all observation data along the last dimension and convert to float32
    processed_obs = np.concatenate(obs, axis=-1).astype(np.float32)
    return (processed_obs, obj_pos_length, obj_rot_length) if find_obs_length else processed_obs

