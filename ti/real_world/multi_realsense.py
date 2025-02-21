import gym.wrappers
from typing import List, Optional, Union, Dict, Callable
from collections import OrderedDict
import numbers
import time
import os
import pathlib
from multiprocessing.managers import SharedMemoryManager
import numpy as np
import open3d as o3d
import pyrealsense2 as rs
import cv2
import matplotlib.pyplot as plt
import sys
sys.path.insert(1, '.')

from ti.real_world.single_realsense import SingleRealsense
from ti.real_world.video_recorder import VideoRecorder
from ti.common.pcd_utils import aggr_point_cloud_from_data
from ti.common.cv2_util import refine_depth_image, draw_keypoints_on_image
from ti.common.kin_utils import RobotKinHelper
from ti.real_world.kinova_interpolation_controller import get_feedback
import ti.real_world.kinova_utilities as kinova_utilities
from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
from kortex_api.autogen.client_stubs.BaseCyclicClientRpc import BaseCyclicClient

import pdb
import rtde_receive

class MultiRealsense:
    def __init__(self,
        serial_numbers: Optional[List[str]]=None,
        shm_manager: Optional[SharedMemoryManager]=None,
        resolution=(1280,720),
        capture_fps=30,
        put_fps=None,
        put_downsample=True,
        record_fps=None,
        enable_color=True,
        enable_depth=True,
        enable_infrared=False,
        enable_pointcloud=False,
        process_pcd=False,
        draw_keypoints=False,
        keypoint_kwargs=None,
        get_max_k=30,
        advanced_mode_config: Optional[Union[dict, List[dict]]]=None,
        transform: Optional[Union[Callable[[Dict], Dict], List[Callable]]]=None,
        vis_transform: Optional[Union[Callable[[Dict], Dict], List[Callable]]]=None,
        recording_transform: Optional[Union[Callable[[Dict], Dict], List[Callable]]]=None,
        video_recorder: Optional[Union[VideoRecorder, List[VideoRecorder]]]=None,
        verbose=False
        ):
        if shm_manager is None:
            shm_manager = SharedMemoryManager()
            shm_manager.start()
        if serial_numbers is None:
            serial_numbers = SingleRealsense.get_connected_devices_serial()
        n_cameras = len(serial_numbers)

        advanced_mode_config = repeat_to_list(
            advanced_mode_config, n_cameras, dict)
        transform = repeat_to_list(
            transform, n_cameras, Callable)
        vis_transform = repeat_to_list(
            vis_transform, n_cameras, Callable)
        recording_transform = repeat_to_list(
            recording_transform, n_cameras, Callable)

        video_recorders = repeat_to_list(
            video_recorder, n_cameras, VideoRecorder)
        
        # self.extrinsics = load_extrinsic_matrices(os.path.join(os.path.dirname(__file__), "extrinsics.json"))

        cameras = OrderedDict()
        for i, serial in enumerate(serial_numbers):
            cameras[serial] = SingleRealsense(
                shm_manager=shm_manager,
                serial_number=serial,
                resolution=resolution,
                capture_fps=capture_fps,
                put_fps=put_fps,
                put_downsample=put_downsample,
                record_fps=record_fps,
                enable_color=enable_color,
                enable_depth=enable_depth,
                enable_infrared=enable_infrared,
                get_max_k=get_max_k,
                advanced_mode_config=advanced_mode_config[i],
                transform=transform[i],
                vis_transform=vis_transform[i],
                recording_transform=recording_transform[i],
                video_recorder=video_recorders[i],
                verbose=verbose,
                # extrinsics=self.extrinsics[serial] if serial in self.extrinsics else None
            )
        self.cameras = cameras
        self.shm_manager = shm_manager
        
        self.enable_pointcloud = enable_pointcloud
        self.process_pcd = process_pcd
        self.draw_keypoints = draw_keypoints
        self.keypoint_kwargs = keypoint_kwargs
        
        self.transform = transform
        
        self.prev_trans_local_world = [None]*2
        self.prev_trans_local_world_vis = [None]*2
        

    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
    
    @property
    def n_cameras(self):
        return len(self.cameras)
    
    @property
    def is_ready(self):
        is_ready = True
        for camera in self.cameras.values():
            if not camera.is_ready:
                is_ready = False
        return is_ready
    
    def start(self, wait=True, put_start_time=None):
        if put_start_time is None:
            put_start_time = time.time()
        for camera in self.cameras.values():
            camera.start(wait=False, put_start_time=put_start_time)
        
        if wait:
            self.start_wait()
    
    def stop(self, wait=True):
        for camera in self.cameras.values():
            camera.stop(wait=False)
        
        if wait:
            self.stop_wait()

    def start_wait(self):
        for camera in self.cameras.values():
            camera.start_wait()

    def stop_wait(self):
        for camera in self.cameras.values():
            camera.join()
    
    def get(self, k=None, out=None) -> Dict[int, Dict[str, np.ndarray]]:
        """
        Return order T,H,W,C
        {
            0: {
                'rgb': (T,H,W,C),
                'timestamp': (T,)
            },
            1: ...
        }
        """
        if out is None:
            out = dict()
        for i, camera in enumerate(self.cameras.values()):
            this_out = None
            if i in out:
                this_out = out[i]
            this_out = camera.get(k=k, out=this_out)
            out[i] = this_out


        return out

    def get_vis(self, out=None):
        results = list()
        for i, camera in enumerate(self.cameras.values()):
            this_out = None
            if out is not None:
                this_out = dict()
                for key, v in out.items():
                    # use the slicing trick to maintain the array
                    # when v is 1D
                    this_out[key] = v[i:i+1].reshape(v.shape[1:])
            this_out = camera.get_vis(out=this_out)
            if out is None:
                results.append(this_out)
        if out is None:
            out = dict()
            for key in results[0].keys():
                out[key] = np.stack([x[key] for x in results])
                
        if self.enable_pointcloud:
            boundaries = {'x_lower': -0.15, 'x_upper': 0.38, 'y_lower': -0.55, 'y_upper': 0.125, 'z_lower': -0.02, 'z_upper': 0.07}

            # shape: (n_cameras, H, W, C)
            colors = out['color'][..., ::-1] # please note that the color should be in RGB
            depths = out['depth'] 
            intrinsics = out['intrinsics']
            extrinsics = out['extrinsics']
            dist_coeffs = out['dist_coeffs']
            
            keypoints_world, all_trans_local_world = extract_keypoint_from_img(colors, depths, intrinsics, extrinsics, boundaries, self.prev_trans_local_world_vis, **self.keypoint_kwargs)
            self.prev_trans_local_world_vis = all_trans_local_world
            keypoints_world = np.concatenate(keypoints_world, axis=0)
            if self.draw_keypoints:
                for camera_idx in range(self.n_cameras):
                    draw_keypoints_on_image(camera_idx, keypoints_world, colors, intrinsics, extrinsics, dist_coeffs)


        return out
    
    def set_color_option(self, option, value):
        n_camera = len(self.cameras)
        value = repeat_to_list(value, n_camera, numbers.Number)
        for i, camera in enumerate(self.cameras.values()):
            camera.set_color_option(option, value[i])


    def set_depth_option(self, option, value):
        n_camera = len(self.cameras)
        value = repeat_to_list(value, n_camera, numbers.Number)
        for i, camera in enumerate(self.cameras.values()):
            camera.set_depth_option(option, value[i])

    def set_exposure(self, exposure=None, gain=None):
        """
        exposure: (1, 10000) 100us unit. (0.1 ms, 1/10000s)
        gain: (0, 128)
        """
        n_camera = len(self.cameras)
        exposure= repeat_to_list(exposure, n_camera, numbers.Number)
        gain= repeat_to_list(gain, n_camera, numbers.Number)
        for i, camera in enumerate(self.cameras.values()):
            camera.set_exposure(exposure[i], gain[i])


    def set_depth_exposure(self, exposure=None, gain=None):
        """
        exposure: (1, 10000) 100us unit. (0.1 ms, 1/10000s)
        gain: (0, 128)
        """
        n_camera = len(self.cameras)
        exposure= repeat_to_list(exposure, n_camera, numbers.Number)
        gain= repeat_to_list(gain, n_camera, numbers.Number)
        for i, camera in enumerate(self.cameras.values()):
            camera.set_depth_exposure(exposure[i], gain[i])

    def set_depth_preset(self, preset):
        n_camera = len(self.cameras)
        preset = repeat_to_list(preset, n_camera, str)
        for i, camera in enumerate(self.cameras.values()):
            camera.set_depth_preset(preset[i])

    def set_white_balance(self, white_balance=None):
        n_camera = len(self.cameras)
        white_balance= repeat_to_list(white_balance, n_camera, numbers.Number)
        for i, camera in enumerate(self.cameras.values()):
            camera.set_white_balance(white_balance[i])

    def set_contrast(self, contrast=None):
        n_camera = len(self.cameras)
        contrast= repeat_to_list(contrast, n_camera, numbers.Number)
        for i, camera in enumerate(self.cameras.values()):
            camera.set_contrast(contrast[i])

    def get_intr_mat(self):
        return np.array([c.get_intr_mat() for c in self.cameras.values()])
    
    def get_depth_scale(self):
        return np.array([c.get_depth_scale() for c in self.cameras.values()])
    
    def start_recording(self, video_path: Union[str, List[str]], 
                        hdf5_path: Union[str, List[str]], 
                        zarr_path: Union[str, List[str]],
                        episode_id: int, 
                        cam_ids: Union[str, List[int]],
                        start_time: float):
        if isinstance(video_path, str):
            # directory
            video_dir = pathlib.Path(video_path)
            assert video_dir.parent.is_dir()
            video_dir.mkdir(parents=True, exist_ok=True)
            video_path = list()
            for i in range(self.n_cameras):
                video_path.append(
                    str(video_dir.joinpath(f'{i}.mp4').absolute()))
        assert len(video_path) == self.n_cameras

        if isinstance(hdf5_path, str):
            # directory
            hdf5_dir = pathlib.Path(hdf5_path)
            assert hdf5_dir.parent.is_dir()
            hdf5_dir.mkdir(parents=True, exist_ok=True)
            hdf5_path = list()
            for i in range(self.n_cameras):
                hdf5_path.append(
                    str(hdf5_dir.joinpath(f'{i}.hdf5').absolute()))
                
        if isinstance(zarr_path, str):
            # directory
            zarr_dir = pathlib.Path(zarr_path)
            assert zarr_dir.parent.is_dir()
            zarr_dir.mkdir(parents=True, exist_ok=True)
            zarr_path = list()
            for i in range(self.n_cameras):
                zarr_path.append(
                    str(zarr_dir.joinpath(f'{i}.zarr').absolute()))

        for i, camera in enumerate(self.cameras.values()):
            camera.start_recording(video_path[i], 
                                   hdf5_path[i] if hdf5_path is not None else None, 
                                   zarr_path[i] if zarr_path is not None else None,
                                   episode_id, 
                                   cam_ids[i],
                                   start_time)
    
    def stop_recording(self):
        for i, camera in enumerate(self.cameras.values()):
            camera.stop_recording()
    
    def restart_put(self, start_time):
        for camera in self.cameras.values():
            camera.restart_put(start_time)

    def calibrate_extrinsics(self, visualize=True, board_size=(6, 9), squareLength=0.03, markerLength=0.022):
        for camera in self.cameras.values():
            camera.calibrate_extrinsics(visualize=visualize, board_size=board_size, squareLength=squareLength, markerLength=markerLength)


def visualize_calibration_result():
    
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    pcd = o3d.geometry.PointCloud()
    vis.add_geometry(pcd)
    
    origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    vis.add_geometry(origin)
    ctr = vis.get_view_control()
    ctr.change_field_of_view(step=90)

    cam_params = ctr.convert_to_pinhole_camera_parameters()
    # Assuming the point cloud is centered around (0, 0, 0), adjust as necessary
    cam_params.extrinsic = np.array([
        [1, 0, 0, 0], 
        [0, 1, 0, -0.5],  # Move camera slightly up
        [0, 0, 1, 10000],    # Move camera back
        [0, 0, 0, 1]
    ])
    ctr.convert_from_pinhole_camera_parameters(cam_params)

    boundaries = {'x_lower': 0.02, 'x_upper': 0.29, 'y_lower': -0.29, 'y_upper': 0.14, 'z_lower': 0.19, 'z_upper': 0.35}
    boundaries = None

    with MultiRealsense(resolution=(640, 480), put_downsample=False, enable_color=True, enable_depth=True, enable_infrared=False, verbose=False) as realsense:
        cv2.setNumThreads(1)

        for _ in range(200):
            out = realsense.get()

            colors = np.stack(value['color'] for value in out.values())[..., ::-1]
            depths = np.stack(value['depth'] for value in out.values()) / 1000.
            intrinsics = np.stack(value['intrinsics'] for value in out.values())
            extrinsics = np.stack(value['extrinsics'] for value in out.values())

            aggregated_pcd = aggr_point_cloud_from_data(colors=colors, depths=depths, Ks=intrinsics, tf_world2cams=extrinsics, downsample=True,
                                                        boundaries=boundaries)

            # Update only the points    
            pcd.points = o3d.utility.Vector3dVector(np.asarray(aggregated_pcd.points))
            pcd.colors = o3d.utility.Vector3dVector(np.asarray(aggregated_pcd.colors))

            vis.update_geometry(pcd)
            vis.poll_events()
            vis.update_renderer()

    vis.destroy_window()

def test_calibration(vis_each_cam = False, gen_robot_pcd = True):
    task = 'bimanual'
    use_gripper = True
    if task == 'bimanual':
        sides=['left', 'right']

        # bimanual case
        robot_base_in_world = np.array([[[1.0, 0, 0.0, 0.80],
                                        [0, 1.0, 0.0, -0.22],
                                        [0.0, 0.0, 1.0, 0.02], 
                                        [0.0, 0.0, 0.0, 1.0]],
                                        [[1.0, 0, 0.0, -0.515],
                                        [0, 1.0, 0.0, -0.22],
                                        [0.0, 0.0, 1.0, 0.02],
                                        [0.0, 0.0, 0.0, 1.0]],])

    elif task == 'kinova':
        sides=['kinova']
        robot_base_in_world = np.array([[[1.0, 0, 0.0, -0.509],
                                        [0, 1.0, 0.0, -0.18],
                                        [0.0, 0.0, 1.0, 0.05], 
                                        [0.0, 0.0, 0.0, 1.0]],])


    # save current robot base pose in world
    real_world_path = os.path.dirname(os.path.abspath(__file__))
    save_dir = os.path.join(real_world_path, 'robot_extrinsics')
    os.system(f'mkdir -p {save_dir}')

    # visualize robot base point cloud
    shm_manager = SharedMemoryManager()
    shm_manager.start()
    
    # get joint angles of current robot state
    if task == 'pushL':    
        robot_ip = ['192.168.0.3']
    elif task == 'right':
        robot_ip = ['192.168.0.2']
    elif task == 'bimanual':
        robot_ip = ['192.168.0.3', '192.168.0.2']
    elif task == 'kinova':
        robot_ip = ['192.168.1.10']
        
    all_robot_pcd = []
    for rob_i, side in enumerate(sides):
        if task == 'bimanual':
            np.save(os.path.join(save_dir, f'{side}_base_pose_in_world.npy'), robot_base_in_world[rob_i])
        elif task == 'kinova':
            np.save(os.path.join(save_dir, f'kinova_base_pose_in_world.npy'), robot_base_in_world[0])

    if gen_robot_pcd:
        for rob_i, side in enumerate(sides):
            if task != 'kinova':
                rtde_r = rtde_receive.RTDEReceiveInterface(hostname = robot_ip[rob_i])
                robot_q =  rtde_r.getActualQ()   # padding, getActualQ returns 6 joint angles while setConfig requires 8
                rtde_r.disconnect()
            
                gripper_q = None
                if use_gripper:
                    from gello.robots.robotiq_gripper import RobotiqGripper
                    gripper = RobotiqGripper()
                    gripper.connect(hostname=robot_ip[rob_i], port=63352)
                    print("gripper connected")
                    gripper_pos = gripper.get_current_position()
                    assert 0 <= gripper_pos <= 255, "Gripper position must be between 0 and 255"
                    gripper_q = gripper_pos / 255

                tool_name = 'soft_hande'
                kin_helper = RobotKinHelper('ur5e', tool_name=tool_name)
            else:
                kinova_args = kinova_utilities.parseConnectionArguments()

                with kinova_utilities.DeviceConnection.createTcpConnection(kinova_args) as router:

                    # Create required services
                    base = BaseClient(router)
                    base_cyclic = BaseCyclicClient(router)
            
                    obs = get_feedback(base, base_cyclic)
                    tool_name=None
                    kin_helper = RobotKinHelper('kinova', tool_name=tool_name)
                    robot_q = obs['ActualQ']
                    gripper_q = obs['gripper']

            base_pcd = kin_helper.get_robot_pcd(robot_q, out_o3d=True)
            

            base_in_world =  robot_base_in_world[rob_i] 
            robot_pcd_points = (base_in_world @ np.concatenate([np.asarray(base_pcd.points).T, np.ones((1, np.asarray(base_pcd.points).shape[0]))], axis=0))[:3, :].T
            robot_pcd = o3d.geometry.PointCloud()
            robot_pcd.points = o3d.utility.Vector3dVector(robot_pcd_points)
            if tool_name != None:
                tool_pcd = kin_helper.get_tool_pcd(robot_q + [gripper_q] if gripper_q is not None else robot_q)
                tool_pcd_points = (base_in_world @ np.concatenate([(tool_pcd).T, np.ones((1, (tool_pcd).shape[0]))], axis=0))[:3, :].T
                tool_pcd = o3d.geometry.PointCloud()
                tool_pcd.points = o3d.utility.Vector3dVector(tool_pcd_points)
                all_robot_pcd.extend([robot_pcd, tool_pcd])
            else:
                all_robot_pcd.extend([robot_pcd])

    # visualize realsense point cloud
    with MultiRealsense(
        resolution=(1280, 720), # (1280, 720), (640, 480)
        put_downsample=False,
        enable_color=True,
        enable_depth=True,
        verbose=False
        ) as realsense:
        cv2.setNumThreads(1)
        
        realsense.set_depth_preset('Default')
        realsense.set_depth_exposure(33000, 16)

        realsense.set_exposure(exposure=115, gain=64)
        realsense.set_white_balance(white_balance=3100)
        
        time.sleep(1)
        for _ in range(30):
            out = realsense.get()
            time.sleep(0.1)

        colors = np.stack([value['color'] for value in out.values()])[..., ::-1]
        depths = np.stack([(value['depth']) for value in out.values()]) / 1000.
        intrinsics = np.stack([value['intrinsics'] for value in out.values()])
        extrinsics = np.stack([value['extrinsics'] for value in out.values()])
        
        if vis_each_cam:
            for i in range(colors.shape[0]):
                pcd_i = aggr_point_cloud_from_data(colors=colors[i:i+1],
                                                depths=depths[i:i+1],
                                                Ks=intrinsics[i:i+1],
                                                tf_world2cams=extrinsics[i:i+1],
                                                downsample=False,
                                                voxel_size=0.001)
                origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        
                # visualize pcd_i
                o3d.visualization.draw_geometries([pcd_i, origin, *all_robot_pcd] )
        
        # visualize all
        boundaries = {'x_lower': -0.34, 'x_upper': 0.55, 'y_lower': -0.63, 'y_upper': 0.3, 'z_lower': 0.0, 'z_upper': 0.6}
        boundaries = None

        pcd = aggr_point_cloud_from_data(colors=colors,
                                         depths=depths,
                                         Ks=intrinsics,
                                         tf_world2cams=extrinsics,
                                         boundaries=boundaries,
                                         downsample=False,
                                        #  N_total=2000,
                                        #  voxel_size=0.001
                                         )
        origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        o3d.visualization.draw_geometries([pcd, origin, *all_robot_pcd] )

def calibrate_all():
    with MultiRealsense(
        put_downsample=False,
        enable_color=True,
        enable_depth=False,
        enable_infrared=False,
        verbose=False
        ) as realsense:
        cv2.setNumThreads(1)

        # Wait until the cameras are ready (optional, based on your setup)
        while not realsense.is_ready:
            time.sleep(0.1)
            
        realsense.set_exposure(exposure=115, gain=64)
        realsense.set_white_balance(white_balance=3100)
        # realsense.set_depth_preset('High Density')
        # realsense.set_depth_exposure(7000, 16)

        time.sleep(3)
        while 1:
            realsense.calibrate_extrinsics(visualize=True, board_size=(7,10), squareLength=0.04, markerLength=0.03)
            # realsense.calibrate_extrinsics(visualize=True, board_size=(5,6), squareLength=0.0415, markerLength=0.031)
            time.sleep(0.1)

def repeat_to_list(x, n: int, cls):  
    if x is None:
        x = [None] * n
    if isinstance(x, cls):
        x = [x] * n
    assert len(x) == n
    return x

if __name__ == '__main__':
    calibrate_all()
    # visualize_calibration_result()
    # test_calibration(vis_each_cam = False, gen_robot_pcd = False)