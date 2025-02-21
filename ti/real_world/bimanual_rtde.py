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

import multiprocessing as mp
import scipy.interpolate as si
import scipy.spatial.transform as st
from rtde_control import RTDEControlInterface
from rtde_receive import RTDEReceiveInterface
from ti.shared_memory.shared_memory_queue import (
    SharedMemoryQueue, Empty)
from ti.shared_memory.shared_memory_ring_buffer import SharedMemoryRingBuffer
from ti.common.pose_trajectory_interpolator import PoseTrajectoryInterpolator
from ti.real_world.rtde_interpolation_controller import RTDEInterpolationController


class BimanualRTDEInterpolationController:
    def __init__(self,
        shm_manager: Optional[SharedMemoryManager],
        robot_ip,
        frequency=125, 
        lookahead_time=0.1, 
        gain=300,
        max_pos_speed=0.25, # 5% of max speed
        max_rot_speed=0.16, # 5% of max speed
        speed_slider_value=0.1,
        launch_timeout=3,
        tcp_offset_pose=None,
        payload_mass=None,
        payload_cog=None,
        joints_init=None,
        joints_init_speed=1.05,
        soft_real_time=False,
        verbose=False,
        receive_keys=None,
        get_max_k=128,
        dummy_robot=False,
        use_gripper=False,
        ctrl_mode='eef',
        ):
        if shm_manager is None:
            shm_manager = SharedMemoryManager()
            shm_manager.start()

        self.dummy_robot = dummy_robot
        if robot_ip is None:
            raise ValueError("robot_ip is None")
        
        # check parameters
        assert ctrl_mode in ['eef', 'joint'], "ctrl_mode must be either 'eef' or 'joint'"

        # robot_ip is of the form [left_ip, right_ip]
        robots = OrderedDict()
        for i, ip in enumerate(robot_ip):
            if i == 0:
                side = 'left'
            else:
                side = 'right'
            robots[ip] = RTDEInterpolationController(
                shm_manager=shm_manager,
                robot_ip=ip,
                frequency=125,  # UR5 CB3 RTDE
                lookahead_time=0.2,
                gain=300,
                max_pos_speed=max_pos_speed,
                max_rot_speed=max_rot_speed,
                speed_slider_value=speed_slider_value,
                launch_timeout=3,
                tcp_offset_pose=tcp_offset_pose,
                payload_mass=None,
                payload_cog=None,
                joints_init=joints_init,
                joints_init_speed=0.5, # 1.05,
                soft_real_time=False,
                # verbose=False,
                receive_keys=None,
                get_max_k=get_max_k,
                
                dummy_robot=self.dummy_robot,
                use_gripper=use_gripper,
                ctrl_mode=ctrl_mode,
                extrinsics_name=f'{side}_base_pose_in_world.npy'
            )

        self.shm_manager = shm_manager
        self.robots = robots
        self.num_joints = 6 + int(use_gripper)
        self.eef_dof = 6 + int(use_gripper)
        
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
    
    
    @property
    def is_ready(self):
        is_ready = True
        for robot in self.robots.values():
            if not robot.is_ready:
                is_ready = False
        return is_ready
    
    def start(self, wait=True):
        for robot in self.robots.values():
            robot.start(wait=False)
        if wait:
            self.stop_wait()

    def stop(self, wait=True):
        for robot in self.robots.values():
            robot.stop(wait=False)

        if wait:
            self.stop_wait()

    def start_wait(self):
        for robot in self.robots.values():
            robot.start_wait()

    def stop_wait(self):
        for robot in self.robots.values():
            robot.join()

    # ========= command methods ============
    def servo_ee_pose(self, pose, duration=0.1):
        """
        duration: desired time to reach pose
        """
        if self.dummy_robot:
            return True
        for i, robot in enumerate(self.robots.values()):
            robot.servo_ee_pose(pose[self.eef_dof*i:self.eef_dof*i+self.eef_dof], duration)
        
    def servoJ(self, pose, duration=0.1):
        """
        duration: desired time to reach pose
        """
        if self.dummy_robot:
            return True
        num_joints = 6 if len(pose) == 12 else self.num_joints
        for i, robot in enumerate(self.robots.values()):
            robot.servoJ(pose[num_joints*i:num_joints*i+num_joints], duration)

    def schedule_waypoint(self, pose, target_time):
        if self.dummy_robot:
            return True
        for i, robot in enumerate(self.robots.values()):
            robot.schedule_waypoint(pose[self.eef_dof*i:self.eef_dof*i+self.eef_dof], target_time)

    def schedule_joints(self, pose, target_time):
        if self.dummy_robot:
            return True
        for i, robot in enumerate(self.robots.values()):
            robot.schedule_joints(pose[self.num_joints*i:self.num_joints*i+self.num_joints], target_time)

    def get_forward_kinematics(self, joints):
        if self.dummy_robot:
            return True
        eef_poses = []
        for i, robot in enumerate(self.robots.values()):
            eef_pose = robot.get_forward_kinematics(joints[self.num_joints*i:self.num_joints*i+self.num_joints])
            eef_poses.extend(eef_pose)
        return eef_poses

    def get_inverse_kinematics(self, eef_pose):
        if self.dummy_robot:
            return True
        joints = []
        for i, robot in enumerate(self.robots.values()):
            single_joints = robot.get_inverse_kinematics(eef_pose[self.eef_dof*i:self.eef_dof*i+self.num_joints])
            joints.extend(single_joints)
        return joints

    # ========= receive APIs =============
    def get_state(self, k=None, out=None):
            
        data_lists = {}

        for i, robot in enumerate(self.robots.values()):
            this_out = robot.get_state(k=k)

            for key, value in this_out.items():
                if key in data_lists:
                    data_lists[key].append(value)
                else:
                    data_lists[key] = [value]

                
        for key in data_lists:
            if key == 'robot_receive_timestamp':
                data_lists[key] = np.max(data_lists[key])
            else:
                data_lists[key] = np.hstack(data_lists[key])
        return data_lists

    def get_all_state(self):
        data_lists = {}

        for i, robot in enumerate(self.robots.values()):
            this_out = robot.get_all_state()

            for key, value in this_out.items():
                if key in data_lists:
                    data_lists[key].append(value)
                else:
                    data_lists[key] = [value]

        for key in data_lists:
            if key == 'robot_receive_timestamp':
                data_lists[key] = np.max(data_lists[key], axis=0)
            else:
                data_lists[key] = np.concatenate(data_lists[key], axis=-1)

        return data_lists
    
    # ========= helper function ===========
    def set_robot_joints(self, set_joints):
        """
        Resets the robot joints to a specified configuration.

        Parameters:
        - env: The environment object containing the robot.
        - set_joints: The target joint positions in radians.
        """
        if len(set_joints) == 12:
            num_joints = 6
        else:
            num_joints = self.num_joints
        for i, robot in enumerate(self.robots.values()):
            robot.set_robot_joints(set_joints[num_joints*i:num_joints*i+num_joints])



if __name__ == '__main__':
    pass