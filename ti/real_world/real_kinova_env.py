from typing import Optional
import pathlib
import numpy as np
import time
import shutil
import math
import cv2
from multiprocessing.managers import SharedMemoryManager
import sys
sys.path.insert(1, '.')
from os.path import dirname, abspath
import zarr
from ti.real_world.multi_realsense import MultiRealsense, SingleRealsense
from ti.real_world.video_recorder import VideoRecorder
from ti.common.timestamp_accumulator import (
    TimestampObsAccumulator,
    TimestampActionAccumulator,
    align_timestamps
)
from ti.real_world.multi_camera_visualizer import MultiCameraVisualizer
from ti.common.replay_buffer import ReplayBuffer
from ti.common.cv2_util import (
    get_image_transform, optimal_row_cols, draw_keypoints_on_image, visualize_keypoints_in_workspace)
import open3d as o3d
import os
import matplotlib.pyplot as plt
import cProfile
import pstats
import io
import json
from ti.model.common.tensor_util import index_at_time
import yaml
from ti.real_world.kinova_interpolation_controller import KortexInterpolationController
import glob
import re
import cv2  
import warnings
from filelock import FileLock

from ti.common.data_utils import save_dict_to_hdf5
from ti.common.precise_sleep import precise_wait
from ti.model.common.rotation_transformer import RotationTransformer
from ti.common.cv2_util import get_image_transform
from ti.common.data_utils import  policy_action_to_env_action, _homo_to_9d_action, _9d_to_homo_action, _6d_axis_angle_to_homo, _homo_to_6d_axis_angle
from ti.real_world.real_inference_util import get_tool_in_cam
sys.path.append(os.path.join(os.path.dirname(__file__), '../third_party'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../third_party/FoundationPose'))
from FoundationPose.estimater import *
from FoundationPose.datareader import *
import omegaconf
from ti.utils.segmentation import initialize_segmentation_models


DEFAULT_OBS_KEY_MAP = {
    # robot
    'ActualTCPPose': 'robot_eef_pose',
    'ActualTCPSpeed': 'robot_eef_pose_vel',
    'ActualQ': 'robot_joint',
    # timestamps
    'step_idx': 'step_idx',
    'timestamp': 'timestamp',
    'FtRawWrench': 'robot_ft_wrench',
    'robot_base_pose_in_world': 'robot_base_pose_in_world',
}


class RealKinovaEnv:
    def __init__(self,
                 # required params
                 output_dir,
                 robot_ip='192.168.1.10',
                 # env params
                 frequency=10,
                 n_obs_steps=2,
                 # obs
                 obs_image_resolution=(640, 480),
                 max_obs_buffer_size=30,
                 camera_serial_numbers=None,
                 obs_key_map=DEFAULT_OBS_KEY_MAP,
                 obs_float32=False,
                 # action
                 max_pos_speed=0.05,
                 max_rot_speed=0.1,
                 speed_slider_value=0.1,
                 # robot
                 tcp_offset=0.13,
                 init_joints=False,
                 j_init=None,
                 # video capture params
                 video_capture_fps=15, # 6, 15, 30
                 video_capture_resolution=(640, 480),
                 # saving params
                 record_raw_video=True,
                 thread_per_video=2,
                 video_crf=21,
                 # vis params
                 enable_multi_cam_vis=False,
                 multi_cam_vis_resolution=(640, 480),
                 # shared memory
                 shm_manager=None,
                 enable_depth=True,
                 debug=False,
                 dummy_robot=False,
                 enable_pose=True,
                 bimanual=False,
                 single_arm_type='right',
                 ctrl_mode='eef',
                 use_gripper=True,
                 ):
        
        video_capture_resolution = obs_image_resolution
        multi_cam_vis_resolution = obs_image_resolution
        
        assert frequency <= video_capture_fps
        output_dir = pathlib.Path(output_dir)
        if not output_dir.parent.is_dir():
            print(f"Output directory {output_dir.parent} does not exist! Creating...")
            output_dir.parent.mkdir(parents=True, exist_ok=True)

        # assert output_dir.parent.is_dir()
        video_dir = output_dir.joinpath('videos')
        video_dir.mkdir(parents=True, exist_ok=True)

        self.episode_id = 0

        zarr_path = str(output_dir.joinpath('replay_buffer.zarr').absolute())
        replay_buffer = ReplayBuffer.create_from_path(
            zarr_path=zarr_path, mode='a')
        self.debug = debug
        self.dummy_robot = dummy_robot
        self.enable_depth = enable_depth
        self.enable_pose = enable_pose

        if shm_manager is None:
            shm_manager = SharedMemoryManager()
            shm_manager.start()
        if camera_serial_numbers is None:
            camera_serial_numbers = SingleRealsense.get_connected_devices_serial()

        color_tf = get_image_transform(
            input_res=video_capture_resolution,
            output_res=obs_image_resolution,
            # obs output rgb
            bgr_to_rgb=True)
        color_transform = color_tf
        if obs_float32:
            color_transform = lambda x: color_tf(x).astype(np.float32) / 255

        def transform(data):
            data['color'] = color_transform(data['color'])
            # data['depth'] = refine_depth_image(data['depth'])
            return data

        rw, rh, col, row = optimal_row_cols(
            n_cameras=len(camera_serial_numbers),
            in_wh_ratio=obs_image_resolution[0]/obs_image_resolution[1],
            max_resolution=multi_cam_vis_resolution
        )
        vis_color_transform = get_image_transform(
            input_res=video_capture_resolution,
            output_res=(rw,rh),
            bgr_to_rgb=False
        )

        def vis_transform(data):
            data['color'] = vis_color_transform(data['color'])
            return data

        recording_transfrom = None
        recording_fps = video_capture_fps
        recording_pix_fmt = 'bgr24'
        if not record_raw_video:
            recording_transfrom = transform
            recording_fps = frequency
            recording_pix_fmt = 'rgb24'
            
        # video_zarr_path = [str(output_dir.joinpath(f'camera_{i}.zarr').absolute()) for i in range(len(camera_serial_numbers))]
        video_zarr_path = str(output_dir.joinpath(f'videos.zarr').absolute())
        video_recorder = VideoRecorder.create_h264(
            fps=recording_fps,
            codec='h264',
            input_pix_fmt=recording_pix_fmt,
            crf=video_crf,
            thread_type='FRAME',
            thread_count=thread_per_video,
            video_capture_resolution=video_capture_resolution,
            video_zarr_path=video_zarr_path,
            num_cams=len(camera_serial_numbers),
        )
        # self.video_recorder = video_recorder

        

        if self.debug:
            print("Initializing RealSense cameras...")
        realsense = MultiRealsense(
            serial_numbers=camera_serial_numbers,
            shm_manager=shm_manager,
            resolution=video_capture_resolution,
            capture_fps=video_capture_fps,
            put_fps=video_capture_fps,
            # send every frame immediately after arrival
            # ignores put_fps
            put_downsample=False,
            record_fps=recording_fps,
            enable_color=True,
            enable_depth=enable_depth,
            enable_infrared=False,
            enable_pointcloud=False,
            process_pcd=False,
            draw_keypoints=False,
            keypoint_kwargs=None,
            get_max_k=max_obs_buffer_size,
            # TODO: check why is transform and vis_transform blocking the program
            transform=transform,
            # vis_transform=vis_transform if enable_multi_cam_vis else None, # TODO: it is blocking the program
            # recording_transform=recording_transfrom,
            video_recorder=video_recorder,
            verbose=False,
        )
        if self.debug:
            print("RealSense cameras initialized!")


        multi_cam_vis = None
        if enable_multi_cam_vis:
            multi_cam_vis = MultiCameraVisualizer(
                realsense=realsense,
                row=row,
                col=col,
                rgb_to_bgr=False
            )

        cube_diag = np.linalg.norm([1, 1, 1])
        # j_init = np.array([-140, -120, -95, -150, -55, -180,
        #                    30, -120, -95, -150, -55, 180]) / 180 * np.pi
        if not init_joints:
            j_init = None
            
        self.bimanual = bimanual
        
        RobotController = KortexInterpolationController
        robot_ip = robot_ip
        
        robot = RobotController(
            shm_manager=shm_manager,
            robot_ip=robot_ip,
            frequency=125,  # UR5 CB3 RTDE
            lookahead_time=0.2,
            gain=100, # TODO: check this, 100
            max_pos_speed=max_pos_speed*cube_diag,
            max_rot_speed=max_rot_speed*cube_diag,
            speed_slider_value=speed_slider_value,
            launch_timeout=3,
            tcp_offset_pose=None, #[0, 0, tcp_offset, 0, 0, 0],
            payload_mass=None,
            payload_cog=None,
            joints_init=j_init,
            joints_init_speed=0.5, # 1.05,
            soft_real_time=False,
            # verbose=False,
            receive_keys=None,
            get_max_k=max_obs_buffer_size,
            dummy_robot=self.dummy_robot,
            ctrl_mode=ctrl_mode,
        )
        self.realsense = realsense
        self.robot = robot
        self.multi_cam_vis = multi_cam_vis
        self.video_capture_fps = video_capture_fps
        self.frequency = frequency
        self.n_obs_steps = n_obs_steps
        self.max_obs_buffer_size = max_obs_buffer_size
        self.max_pos_speed = max_pos_speed
        self.max_rot_speed = max_rot_speed
        self.obs_key_map = obs_key_map
        # recording
        self.output_dir = output_dir
        self.video_dir = video_dir
        self.replay_buffer = replay_buffer
        self.episode_id = replay_buffer.n_episodes
        # temp memory buffers
        self.last_realsense_data = None
        # recording buffers
        self.obs_accumulator = None
        # self.action_accumulator = None
        self.joint_action_accumulator = None
        self.eef_action_accumulator = None
        self.stage_accumulator = None

        self.start_time = None
        
        
    # ======== start-stop API =============
    @property
    def is_ready(self):
        return self.realsense.is_ready and self.robot.is_ready

    def start(self, wait=True):
        if self.debug:
            print("Starting RealKinovaEnv...")
        self.realsense.start(wait=False)
        self.robot.start(wait=False)
        if self.multi_cam_vis is not None:
            self.multi_cam_vis.start(wait=False)
        if wait:
            self.start_wait()
        if self.debug:
            print("RealKinovaEnv started!")

    def stop(self, wait=True):
        self.end_episode()
        if self.multi_cam_vis is not None:
            self.multi_cam_vis.stop(wait=False)
        self.robot.stop(wait=False)
        self.realsense.stop(wait=False)
        if wait:
            self.stop_wait()

    def start_wait(self):
        self.realsense.start_wait()
        self.robot.start_wait()
        if self.multi_cam_vis is not None:
            self.multi_cam_vis.start_wait()

    def stop_wait(self):
        self.robot.stop_wait()
        self.realsense.stop_wait()
        if self.multi_cam_vis is not None:
            self.multi_cam_vis.stop_wait()

    # ========= context manager ===========
    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    # ========= async env API ===========
    def get_obs(self, skip_keypoint=False, profile=False) -> dict:
        "observation dict"
        assert self.is_ready
        if profile:
            # Initialize profiler
            pr = cProfile.Profile()
            pr.enable()
        
        # get data
        # 30 Hz, camera_receive_timestamp
        k = math.ceil(self.n_obs_steps * (self.video_capture_fps / self.frequency)) # how many of the most recent observations from the camera we want to fetch
        self.last_realsense_data = self.realsense.get(
            k=k, 
            out=self.last_realsense_data)
        
        # 125 hz, robot_receive_timestamp
        last_robot_data = self.robot.get_all_state()
        # both have more than n_obs_steps data

        # align camera obs timestamps
        dt = 1 / self.frequency
        last_timestamp = np.max([x['timestamp'][-1] for x in self.last_realsense_data.values()]) # the most recent timestamp from the collected camera data
        obs_align_timestamps = last_timestamp - (np.arange(self.n_obs_steps)[::-1] * dt)


        camera_obs = dict()
        for camera_idx, value in self.last_realsense_data.items():
            this_timestamps = value['timestamp']
            this_idxs = list()
            for t in obs_align_timestamps:
                is_before_idxs = np.nonzero(this_timestamps < t)[0]
                this_idx = 0
                if len(is_before_idxs) > 0:
                    this_idx = is_before_idxs[-1]
                this_idxs.append(this_idx)
            # remap key
            camera_obs[f'camera_{camera_idx}_color'] = value['color'][this_idxs]
            camera_obs[f'camera_{camera_idx}_depth'] = value['depth'][this_idxs]
            camera_obs[f'camera_{camera_idx}_intrinsics'] = value['intrinsics'][this_idxs]
            camera_obs[f'camera_{camera_idx}_extrinsics'] = value['extrinsics'][this_idxs]



        robot_timestamps = last_robot_data['robot_receive_timestamp']
        this_timestamps = robot_timestamps
        this_idxs = list()
        for t in obs_align_timestamps:
            is_before_idxs = np.nonzero(this_timestamps < t)[0]
            this_idx = 0
            if len(is_before_idxs) > 0:
                this_idx = is_before_idxs[-1]
            this_idxs.append(this_idx)

        robot_obs_raw = dict()
        for k, v in last_robot_data.items():
            if k in self.obs_key_map:
                robot_obs_raw[self.obs_key_map[k]] = v
        
        robot_obs = dict()
        for k, v in robot_obs_raw.items():
            robot_obs[k] = v[this_idxs]

        # return obs
        obs_data = dict(camera_obs)
        obs_data.update(robot_obs)
        
        # accumulate obs
        if self.obs_accumulator is not None:
            self.obs_accumulator.put(
                obs_data,
                obs_align_timestamps,
            )
            
        obs_data['timestamp'] = obs_align_timestamps

        if profile:
            pr.disable()
            s = io.StringIO()
            sortby = 'cumulative'
            ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
            ps.print_stats()
            print(s.getvalue())
        return obs_data


    def exec_actions(self,
                     joint_actions: np.ndarray,
                     eef_actions: np.ndarray,
                     timestamps: np.ndarray,
                     mode = 'eef', # 'joint' or 'eef'
                     dt: float = 0.1,
                     stages: Optional[np.ndarray] = None):
        """
        Execute a series of robot actions at specified times.

        Args:
            actions (np.ndarray): An array of actions to be executed by the robot. Each action
                                should correspond to a robot pose or similar set of instructions.
            timestamps (np.ndarray): An array of timestamps (in seconds since the epoch) when each 
                                    corresponding action should be executed. These should be future 
                                    times relative to the current time when exec_actions is called.
            stages (Optional[np.ndarray], optional): An optional array of stage identifiers for each 
                                                    action. Defaults to None. If provided, it helps 
                                                    in categorizing or identifying the stage of each action.

        Raises:
            AssertionError: If the method is called when the object is not ready to execute actions.

        Note:
            - The method filters out any actions and their corresponding timestamps and stages 
            if the timestamp is earlier than the current time.
            - Actions are scheduled as waypoints for the robot to follow.
            - If accumulators for actions or stages are set, the actions and stages are recorded.
        """
        # Ensure that the object is ready to execute actions
        assert self.is_ready

        # Convert input actions and timestamps to numpy arrays if they aren't already
        if not isinstance(joint_actions, np.ndarray):
            joint_actions = np.array(joint_actions)
        if not isinstance(eef_actions, np.ndarray):
            eef_actions = np.array(eef_actions) 
            
        if not isinstance(timestamps, np.ndarray):
            timestamps = np.array(timestamps)
            
        # Initialize stages array if not provided, or convert to numpy array if needed
        if stages is None:
            stages = np.zeros_like(timestamps, dtype=np.int64) # timestamps is dummy
        elif not isinstance(stages, np.ndarray):
            stages = np.array(stages, dtype=np.int64)        
            
        # real timestamp
        receive_time = time.time()

        
        is_new = timestamps > receive_time
        new_joint_actions = joint_actions[is_new]
        new_eef_actions = eef_actions[is_new]
        new_timestamps = timestamps[is_new]
        new_stages = stages[is_new]

        # schedule waypoints
        for i in range(len(new_eef_actions)):
            self.robot.schedule_waypoint(new_eef_actions[i], new_timestamps[i])

        # record actions
        # if self.action_accumulator is not None:
        #     self.action_accumulator.put(
        #         actions,
        #         timestamps
        #     )
        # record joint_actions
        if self.joint_action_accumulator is not None:
            self.joint_action_accumulator.put(
                new_joint_actions,
                new_timestamps
            )
        if self.eef_action_accumulator is not None:
            self.eef_action_accumulator.put(
                new_eef_actions,
                new_timestamps
            )

        if self.stage_accumulator is not None:
            self.stage_accumulator.put(
                stages,
                timestamps
            )

    def get_robot_state(self):
        if self.dummy_robot:
            return {'robot_eef_pose': np.zeros((6,), dtype=np.float32),
                    'TargetTCPPose': np.zeros((6,), dtype=np.float32),}
        else:
            return self.robot.get_state() 

    # recording API
    def start_episode(self, start_time=None):
        "Start recording and return first obs"
        if start_time is None:
            start_time = time.time()
        self.start_time = start_time

        assert self.is_ready

        # prepare recording stuff
        episode_id = self.replay_buffer.n_episodes
        
        this_video_dir = self.video_dir.joinpath(str(episode_id))
        this_video_dir.mkdir(parents=True, exist_ok=True)
        n_cameras = self.realsense.n_cameras
        video_paths = list()
        hdf5_paths = list()
        zarr_paths = list()
        cam_ids = list()
        for i in range(n_cameras):
            video_paths.append(
                str(this_video_dir.joinpath(f'{i}.mp4').absolute()))
            hdf5_paths.append(
                str(this_video_dir.joinpath(f'{i}.hdf5').absolute()))
            zarr_paths.append(
                str(this_video_dir.joinpath(f'{i}.zarr').absolute()))
            cam_ids.append(i)

        # start recording on realsense
        self.realsense.restart_put(start_time=start_time)
        self.realsense.start_recording(
            video_path=video_paths, 
            hdf5_path=hdf5_paths, 
            zarr_path=zarr_paths, 
            episode_id=episode_id, 
            cam_ids=cam_ids, 
            start_time=start_time)

        # create accumulators
        self.obs_accumulator = TimestampObsAccumulator(
            start_time=start_time,
            dt=1/self.frequency
        )
        self.joint_action_accumulator = TimestampActionAccumulator(
            start_time=start_time,
            dt=1/self.frequency
        )
        self.eef_action_accumulator = TimestampActionAccumulator(
            start_time=start_time,
            dt=1/self.frequency
        )

        self.stage_accumulator = TimestampActionAccumulator(
            start_time=start_time,
            dt=1/self.frequency
        )
        self.episode_id = self.replay_buffer.n_episodes
        print(f'Episode {self.episode_id} started!')

    def end_episode(self, incr_epi=True):
        "Stop recording"
        assert self.is_ready

        # stop video recorder
        self.realsense.stop_recording()

        if self.obs_accumulator is not None:
            # recording
            # assert self.action_accumulator is not None
            assert self.joint_action_accumulator is not None
            assert self.eef_action_accumulator is not None
            assert self.stage_accumulator is not None

            # Since the only way to accumulate obs and action is by calling
            # get_obs and exec_actions, which will be in the same thread.
            # We don't need to worry new data come in here.
            obs_data = self.obs_accumulator.data
            obs_timestamps = self.obs_accumulator.timestamps            

            num_cam = 0
            cam_width = -1
            cam_height = -1
            for key in obs_data.keys():
                if 'camera' in key and 'color' in key:
                    num_cam += 1
                    cam_height, cam_width = obs_data[key].shape[1:3]

            # actions = self.action_accumulator.actions
            # action_timestamps = self.action_accumulator.timestamps
            joint_actions = self.joint_action_accumulator.actions
            eef_actions = self.eef_action_accumulator.actions
            action_timestamps = self.joint_action_accumulator.timestamps
            
            stages = self.stage_accumulator.actions
            n_steps = min(len(obs_timestamps), len(action_timestamps))
            num_cam = self.realsense.n_cameras
            if n_steps > 0:
                ### init episode data
                episode = {
                    'timestamp': None,
                    'stage': None,
                    'observations': 
                        {'robot_joint': [],
                        #  'robot_joint_vel': [], 
                         'robot_base_pose_in_world': [],
                         'robot_eef_pose': [],
                         'robot_eef_pose_vel': [],
                        #  'robot_gripper_pos': {},
                         'images': {},},
                    'joint_action': [],
                    'cartesian_action': [],
                }
                for cam in range(num_cam):
                    episode['observations']['images'][f'camera_{cam}_color'] = []
                    episode['observations']['images'][f'camera_{cam}_depth'] = []
                    episode['observations']['images'][f'camera_{cam}_intrinsics'] = []
                    episode['observations']['images'][f'camera_{cam}_extrinsics'] = []

                attr_dict = {
                }

                ### create config dict
                config_dict = {
                    'observations': {
                        'images': {}
                    },
                    'timestamp': {
                        'dtype': 'float64'
                    },
                }
                
                for cam in range(num_cam):
                    color_save_kwargs = {
                        'chunks': (1, cam_height, cam_width, 3), # (1, 480, 640, 3)
                        'compression': 'gzip',
                        'compression_opts': 9,
                        'dtype': 'uint8',
                    }
                    depth_save_kwargs = {
                        'chunks': (1, cam_height, cam_width), # (1, 480, 640)
                        'compression': 'gzip',
                        'compression_opts': 9,
                        'dtype': 'uint16',
                    }
                    config_dict['observations']['images'][f'camera_{cam}_color'] = color_save_kwargs
                    config_dict['observations']['images'][f'camera_{cam}_depth'] = depth_save_kwargs
               
                ### load episode data
                episode['timestamp'] = obs_timestamps[:n_steps]
                episode['joint_action'] = joint_actions[:n_steps]
                episode['cartesian_action'] = eef_actions[:n_steps]
                episode['stage'] = stages[:n_steps]
                # for key, value in obs_data.items():
                #     episode[key] = value[:n_steps]
                for key, value in obs_data.items():
                    if 'camera' in key:
                        episode['observations']['images'][key] = value[:n_steps]
                    else:
                        episode['observations'][key] = value[:n_steps]
                        
                self.replay_buffer.add_episode(episode, compressors='disk')
                episode_id = self.replay_buffer.n_episodes - 1
                print(f'Episode {episode_id} saved!')

            self.obs_accumulator = None
            self.joint_action_accumulator = None
            self.eef_action_accumulator = None
            self.stage_accumulator = None

    def drop_episode(self):
        self.end_episode()
        if self.replay_buffer.n_episodes == 0:
            print("No episode to drop!")
            return
        self.replay_buffer.drop_episode()
        episode_id = self.replay_buffer.n_episodes 
        # self.episode_id = episode_id
        this_video_dir = self.video_dir.joinpath(str(episode_id))
        if this_video_dir.exists():
            shutil.rmtree(str(this_video_dir))
        print(f'Episode {episode_id} dropped!')



def test_episode_start():
    # create env
    os.system('mkdir -p tmp')
    with RealKinovaEnv(
            output_dir='tmp',
        ) as env:
        print('Created env!')
        
        env.start_episode()
        print('Started episode!')

def test_env_obs_latency():
    os.system('mkdir -p tmp')
    with RealKinovaEnv(
            output_dir='tmp',
        ) as env:
        print('Created env!')

        for i in range(100):
            start_time = time.time()
            obs = env.get_obs()
            end_time = time.time()
            print(f'obs latency: {end_time - start_time}')
            time.sleep(0.1)




def test_env_human_replay():
    d = dirname(dirname(dirname(abspath(__file__))))
    
    replay_from_cache = False
    episode_num = 3
    zarr_path = os.path.join(d, 'data/hammer_human/replay_buffer.zarr')
    task_conf_path = os.path.join(d, 'conf/train_bc/task/real_hammer_human.yaml')
    cfg = omegaconf.OmegaConf.load(task_conf_path)
    action_key = 'action' # robot_eef_pose or action
    if not replay_from_cache:
        action_key = 'observations/robot_eef_pose'
    
    assert os.path.exists(zarr_path)
    group = zarr.open(zarr_path, 'r')
    src_root = zarr.group(group.store)


    if replay_from_cache:
        dataset_path = os.path.join(d, 'data/hammer_human')
        
        
        shape_meta_hash = '21e799f9ac859c561bd980afe0d3869a'
        cache_zarr_path = os.path.join(dataset_path, shape_meta_hash + '.zarr.zip')
        cache_lock_path = cache_zarr_path + '.lock'
        print('Acquiring lock on cache.')
        with FileLock(cache_lock_path):
            print('Loading cached ReplayBuffer from Disk.')
            with zarr.ZipStore(cache_zarr_path, mode='r') as zip_store:
                replay_buffer = ReplayBuffer.copy_from_store(
                    src_store=zip_store, store=zarr.MemoryStore())
            print('Loaded!')
        demo_dict = replay_buffer.data
        episode_ends = replay_buffer.episode_ends
    else:
        demo_dict = src_root['data']
        episode_ends = src_root['meta']['episode_ends']

    
    if episode_num == 0:
        episode_start = 0
        episode_end = episode_ends[0]
    else:
        episode_start = episode_ends[episode_num-1]
        episode_end = episode_ends[episode_num]
        
    episode_slice = slice(episode_start, episode_end)

    tool_actions = demo_dict[action_key][episode_slice].copy()
    if 'scoop' in zarr_path:
        repeated_z_rot = np.tile([ 0.10653476,  1.63334494,  2.42727399, -2.22424441], (tool_actions.shape[0], 1))
        tool_actions = np.concatenate((tool_actions, repeated_z_rot), axis=1)
    camera_key = 'camera_1'
    
    mat_6d_transformer = RotationTransformer(from_rep='matrix', to_rep='rotation_6d')
    rotation_transformer1 = RotationTransformer('axis_angle', 'rotation_6d')
    rotation_transformer2 = RotationTransformer('rotation_6d', 'axis_angle')


    robot_side = cfg.robot_side
    cam_serials = cfg.cam_serials 
    views = [camera_key]
    robot_extrinsics_path=os.path.join(d, f'ti/real_world/robot_extrinsics/{robot_side}_base_pose_in_world.npy')
    camera_extrinsics_path=os.path.join(d, f'ti/real_world/cam_extrinsics/{cam_serials[views[0]]}.npy')
    
    if not os.path.exists(robot_extrinsics_path):
        base_pose_in_world = np.ones((4,4))
        warnings.warn(f'extrinsics_path {robot_extrinsics_path} does not exist, using identity matrix.')
    else:
        base_pose_in_world = np.load(robot_extrinsics_path)

    if not os.path.exists(camera_extrinsics_path):
        cam_extrinsics = np.ones((4,4))
        warnings.warn(f'extrinsics_path {camera_extrinsics_path} does not exist, using identity matrix.')
    else:
        cam_extrinsics = np.load(camera_extrinsics_path)
        
    cam_in_base = np.linalg.inv(base_pose_in_world)@cam_extrinsics
    if replay_from_cache:
        homo_tool_in_base = _9d_to_homo_action(tool_actions)
        tool_actions = _homo_to_6d_axis_angle(homo_tool_in_base)
    else:
        homo_tool_in_base = _6d_axis_angle_to_homo(tool_actions)
        
    homo_tool_in_cam = np.linalg.inv(cam_in_base)@homo_tool_in_base
    
    tool_actions_in_cam = _homo_to_9d_action(homo_tool_in_cam)
        
    rotation_6d = rotation_transformer1.forward(tool_actions[:,3:])
    tool_actions_9d = np.concatenate([tool_actions[:,:3], rotation_6d], axis=-1)
    tool_actions[:,3:] = rotation_transformer2.forward(rotation_6d)

    
    mesh = trimesh.load(cfg.mesh_file)
    to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
    bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2,3)
    scorer = ScorePredictor()
    refiner = PoseRefinePredictor()
    glctx = dr.RasterizeCudaContext()
    pose_model = FoundationPose(model_pts=mesh.vertices, model_normals=mesh.vertex_normals, mesh=mesh, scorer=scorer, refiner=refiner, debug_dir=os.path.join(code_dir,'debug'), debug=1, glctx=glctx)
    
    grounding_dino_model, sam_predictor = initialize_segmentation_models()

    with RealKinovaEnv(
            output_dir='tmp',
            ctrl_mode='eef',
            speed_slider_value=0.15,
            bimanual=False,
            use_gripper=False
        ) as env:
        
        print('Created env!')
        time.sleep(0.5)
        timestamps = time.time() + np.arange(len(tool_actions)) / 10 + 1.0
        start_step = 0
        
        obs = env.get_obs()
        tool_pose, pose_rgb = get_tool_in_cam(pose_model, cfg.segmentation_labels, obs=obs, shape_meta=cfg.shape_meta,to_origin=to_origin, bbox=bbox, camera_key=camera_key, init=True, grounding_dino_model=grounding_dino_model, sam_predictor=sam_predictor)
        init_tool_pose_9d = np.repeat(tool_pose[np.newaxis, :, :], obs['robot_eef_pose'].shape[0], axis=0)
        init_tool_pose_9d = _homo_to_9d_action(init_tool_pose_9d, mat_6d_transformer)
        
        tool_in_eef = np.linalg.inv(_6d_axis_angle_to_homo(obs['robot_eef_pose'])[-1])@np.linalg.inv(base_pose_in_world)@cam_extrinsics@tool_pose
        
        robot_action_6d = policy_action_to_env_action(tool_actions_9d, 
                                                      obs['robot_eef_pose'][-1], 
                                                      init_tool_pose_9d[-1], 
                                                      tool_in_eef,
                                                      'eef' 
                                                      )

        while True:
            curr_time = time.monotonic()
            loop_end_time = curr_time + 1.0
            end_step = min(start_step+10, len(robot_action_6d))
            action_batch = robot_action_6d[start_step:end_step]
            timestamp_batch = timestamps[start_step:end_step]
            
            homo_actions = _9d_to_homo_action(tool_actions_in_cam[start_step:end_step])
            for pose in homo_actions:
                # pattern = re.compile(r'camera_\d+_color')
                rgb_key = camera_key # [key for key in cfg.shape_meta['obs'] if pattern.match(key)][0]
                intrinsics = obs[f'{rgb_key}_intrinsics'][-1].astype(np.float64)
                center_pose = pose@np.linalg.inv(to_origin)
                # tool_pose, pose_rgb = get_tool_in_cam(pose_model, cfg.segmentation_labels, obs=obs, shape_meta=cfg.shape_meta,to_origin=to_origin, bbox=bbox, init=False, grounding_dino_model=grounding_dino_model, sam_predictor=sam_predictor)
                pose_rgb = draw_xyz_axis(pose_rgb, ob_in_cam=center_pose, scale=0.1, K=intrinsics, thickness=3, transparency=0, is_input_rgb=True)

            cv2.imshow('vis', pose_rgb[...,::-1])
            cv2.waitKey(0)

            for eef_action in action_batch:
                if 'scoop' in zarr_path:
                   eef_action[2] = np.clip(eef_action[2], 0.185, 0.195)
                env.robot.servo_ee_pose(eef_action, duration=0.1)
                time.sleep(0.1)

                # env.exec_actions(
                #     joint_actions=np.zeros((action_batch.shape[0], 7)),
                #     eef_actions=action_batch,
                #     timestamps=timestamp_batch,
                #     mode=ctrl_mode,
                # )
            print(f'executed {end_step - start_step} actions')
            start_step = end_step
            precise_wait(loop_end_time)
            if start_step >= len(tool_actions):
                break
            
if __name__ == '__main__':
    test_env_human_replay()