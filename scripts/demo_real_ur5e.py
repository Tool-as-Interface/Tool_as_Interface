"""
Usage:
(robodiff)$ python demo_real_robot.py -o <demo_save_dir> --robot_ip <ip_of_ur5>

Robot movement:
Move your SpaceMouse to move the robot EEF (locked in xy plane).
Press SpaceMouse right button to unlock z axis.
Press SpaceMouse left button to enable rotation axes.

Recording control:
Click the opencv window (make sure it's in focus).
Press "C" to start recording.
Press "S" to stop recording.
Press "Q" to exit program.
Press "Backspace" to delete the previously recorded episode.
"""

# %%
import time
from multiprocessing.managers import SharedMemoryManager
import click
import cv2
import numpy as np
import scipy.spatial.transform as st
from ti.real_world.real_ur5e_env import RealUR5eEnv
from ti.real_world.real_kinova_env import RealKinovaEnv
from ti.real_world.spacemouse_shared_memory import Spacemouse
from ti.devices.gello_shared_memory import Gello
from ti.common.precise_sleep import precise_sleep, precise_wait
from ti.real_world.keystroke_counter import (
    KeystrokeCounter, Key, KeyCode
)
from ti.common.trans_utils import transform_to_world, transform_from_world, interpolate_poses
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
import os
import yaml


@click.command()
@click.option('--output', '-o', default='', help="Directory to save demonstration dataset.")
@click.option('--vis_camera_idx', default=1, type=int, help="Which RealSense camera to visualize.")
@click.option('--init_joints', '-j', is_flag=True, default=True, help="Whether to initialize robot joint configuration in the beginning.")
@click.option('--frequency', '-f', default=10, type=float, help="Control frequency in Hz.")
@click.option('--command_latency', '-cl', default=0.01, type=float, help="Latency between receiving SapceMouse command to executing on Robot in Sec.")
@click.option('--debug', '-d', is_flag=True, default=False, help="Debug mode.")
@click.option('--dummy_robot', '-dr', is_flag=True, default=False, help="Dummy robot.")
@click.option('--count_time', '-ct', is_flag=True, default=False, help="Count the time to execute the program.", type=bool)
@click.option('--input_device', '-id', default='spacemouse', type=click.Choice(['spacemouse', 'gello'], case_sensitive=False), help="spacemouse")
@click.option('--bimanual', '-sr', default=False)
@click.option('--use_gripper', '-ug', default=True, help="Use gripper or not.")
@click.option('--robot_type', '-rt', default='ur5e')

def main(output, vis_camera_idx, init_joints, frequency, command_latency, 
         debug, dummy_robot, count_time, input_device, bimanual, use_gripper, robot_type):
    if input_device == 'spacemouse':
        ctrl_mode = 'eef'

    else:
        ctrl_mode = 'joint'
    max_pos_speed=0.2
    max_rot_speed=0.4 

    if robot_type != 'kinova':
        # dynamixel control box port map (to distinguish left and right gello)
        gello_config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ti/devices/gello_software/gello.yaml")
        
        with open(gello_config_path) as stream:
            try:
                gello_config = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
    
        single_robot_type = gello_config['single_arm_type']
        reset_joints = np.deg2rad(gello_config[f"{single_robot_type}_reset_joints"])
    else:
        single_robot_type = 'kinova'
        reset_joints = None
    dt = 1/frequency
    num_joints_per_bot = 6 + int(use_gripper)

    device_drivers = {
        'gello': Gello,
        'spacemouse': Spacemouse,
    }

    # Attempt to get the device driver class from the dictionary using the input_device key
    device_driver = device_drivers.get(input_device)

    # Raise an error if the input_device is not found in the dictionary (device_driver is None)
    if device_driver is None:
        raise ValueError(f'input_device {input_device} not supported')
    robot_env = RealKinovaEnv if robot_type=='kinova' else RealUR5eEnv 

    with SharedMemoryManager() as shm_manager:
        with KeystrokeCounter() as key_counter, \
            device_driver(shm_manager=shm_manager, 
                          bimanual=bimanual,
                          use_gripper=use_gripper) as device, \
            robot_env(
                output_dir=output, 
                # recording resolution
                obs_image_resolution=(640, 480), # (1280, 720), (640, 480), (480, 270)
                frequency=frequency,
                init_joints=init_joints,
                j_init=reset_joints,
                max_pos_speed=max_pos_speed,
                max_rot_speed=max_rot_speed,
                speed_slider_value=1,
                enable_multi_cam_vis=False,
                record_raw_video=True,
                # number of threads per camera view for video recording (H.264)
                thread_per_video=3,
                # video recording quality, lower is better (but slower).
                video_crf=21,
                shm_manager=shm_manager,
                enable_depth=True, # TODO: when set to True, it makes the robot show jittery behavior and color image fronzen
                debug=debug,
                dummy_robot=dummy_robot,
                bimanual=bimanual,
                single_arm_type=single_robot_type,
                ctrl_mode=ctrl_mode,
                use_gripper=use_gripper
            ) as env:
            cv2.setNumThreads(1)

            # realsense exposure
            env.realsense.set_depth_preset('Default')
            env.realsense.set_depth_exposure(33000, 16)

            env.realsense.set_exposure(exposure=115, gain=64)
            env.realsense.set_contrast(contrast=60)
            env.realsense.set_white_balance(white_balance=3100)


            obs_duration = 0.0
            

            time.sleep(5.0)
            print("Number of cameras: ", env.realsense.n_cameras)
            print('Ready!')
            
            print("Resetting robot to initial position from yaml config.")
            if input_device == 'gello':
                if bimanual:
                    # For bimanual robots, concatenate left and right reset joints
                    reset_joints_left = np.deg2rad(gello_config['left_reset_joints'])[:num_joints_per_bot]
                    reset_joints_right = np.deg2rad(gello_config['right_reset_joints'])[:num_joints_per_bot]
                    reset_joints = np.concatenate([reset_joints_left, reset_joints_right])
                    env.robot.set_robot_joints(reset_joints)
                else:
                    # For single-arm robots, use the reset joints based on the robot type
                    reset_joints = np.deg2rad(gello_config[f"{single_robot_type}_reset_joints"])[:num_joints_per_bot]
                    env.robot.set_robot_joints(reset_joints)
                    

                print("Comparing the initial gello joint positions with the robot's current joint positions.")
                gello_start_pos = device.get_joint_state()
                curr_joints = env.get_robot_state()['ActualQ']

                abs_deltas = np.abs(gello_start_pos - curr_joints)
                id_max_joint_delta = np.argmax(abs_deltas)

                max_joint_delta = 0.8
                if abs_deltas[id_max_joint_delta] > max_joint_delta:
                    id_mask = abs_deltas > max_joint_delta
                    print()
                    ids = np.arange(len(id_mask))[id_mask]
                    for i, delta, joint, current_j in zip(
                        ids,
                        abs_deltas[id_mask],
                        gello_start_pos[id_mask],
                        curr_joints[id_mask],
                    ):
                        print(
                            f"joint[{i}]: \t delta: {delta:4.3f} , leader: \t{joint:4.3f} , follower: \t{current_j:4.3f}"
                        )
                    return
                print(f"Start pos: {len(gello_start_pos)}", f"Joints: {len(curr_joints)}")
                assert len(gello_start_pos) == len(
                    curr_joints
                ), f"agent output dim = {len(gello_start_pos)}, but env dim = {len(curr_joints)}"

                env.robot.set_robot_joints(device.get_joint_state())
            
                joints = env.get_robot_state()['ActualQ']
                action = device.get_joint_state() 
                if (action - joints > 0.5).any():
                    print("Action is too big")

                    # print which joints are too big
                    joint_index = np.where(action - joints > 0.8)
                    for j in joint_index:
                        print(
                            f"Joint [{j}], leader: {action[j]}, follower: {joints[j]}, diff: {action[j] - joints[j]}"
                        )
                    exit()

            state = env.get_robot_state()
            if not dummy_robot:
                target_pose = state['TargetTCPPose']

            t_start = time.monotonic()
            iter_idx = 0
            stop = False
            is_recording = False
            intermediate_pose = []

            cur_dir_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
            robot_extrinsics_path = os.path.join(cur_dir_path, 'ti/real_world/robot_extrinsics')
            if robot_type == 'kinova':
                robots = ['kinova']
            else:
                robots = ['left', 'right']

            robot_base_in_world = {}
            for robot in robots:
                robot_base_in_world[robot] = np.load(os.path.join(robot_extrinsics_path, f'{robot}_base_pose_in_world.npy'))
            

            while not stop:
                if count_time:
                    # Start of loop cycle
                    loop_start_time = time.monotonic()
                # calculate timing
                t_cycle_end = t_start + (iter_idx + 1) * (dt+obs_duration) # the end time of the current control cycle
                t_sample = t_cycle_end - command_latency # the time when the system should sample spacemouse input
                t_command_target = t_cycle_end + dt # the future time when the the target pose should be reached

                if count_time:
                    # Fetch observations
                    obs_start_time = time.monotonic()  # Start timer for get_obs()
                    
                obs = env.get_obs()
                
                if count_time:
                    obs_end_time = time.monotonic()  # End timer for get_obs()
                    # Calculate actual time taken by get_obs()
                    obs_duration = obs_end_time - obs_start_time
                    print("Time taken by get_obs(): ", obs_duration)

                else:
                    # pump obs
                    obs = env.get_obs()
                                    
                # handle key presses
                press_events = key_counter.get_press_events()
                for key_stroke in press_events:
                    if key_stroke == KeyCode(char='q'):
                        # Exit program
                        stop = True
                    elif key_stroke == KeyCode(char='c'):
                        # Start recording
                        env.start_episode(t_start + (iter_idx + 2) * dt - time.monotonic() + time.time())
                        key_counter.clear()
                        is_recording = True
                        print('Recording!')
                    elif key_stroke == KeyCode(char='s'):
                        # Stop recording
                        env.end_episode()
                        key_counter.clear()
                        is_recording = False
                        print('Stopped.')

                    elif key_stroke == Key.backspace:
                        # Delete the most recent recorded episode
                        if click.confirm('Are you sure to drop an episode?'):
                            env.drop_episode()
                            key_counter.clear()
                            is_recording = False


                stage = key_counter[Key.space]

                # visualize
                vis_color_img = obs[f'camera_{vis_camera_idx}_color'][-1, :, :, ::-1].copy()
                vis_depth_img = obs[f'camera_{vis_camera_idx}_depth'][-1].copy()
                if len(vis_depth_img.shape) == 2:
                    vis_depth_img = cv2.normalize(vis_depth_img, None, 0, 255, cv2.NORM_MINMAX)
                    vis_depth_img = cv2.cvtColor(vis_depth_img.astype(np.uint8), cv2.COLOR_GRAY2BGR)
                vis_img = np.concatenate((vis_color_img, vis_depth_img), axis=1)
                
                # episode_id = env.episode_id
                episode_id = env.replay_buffer.n_episodes

                text = f'Episode: {episode_id}, Stage: {stage}'
                if is_recording:
                    text += ', Recording!'
                cv2.putText(
                    vis_img,
                    text,
                    (10,30),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1,
                    thickness=2,
                    color=(255,255,255)
                )

                cv2.imshow('default', vis_img)
                cv2.pollKey()

                precise_wait(t_sample)
                # get teleop command
                if input_device == 'spacemouse':
                    device_state = device.get_motion_state_transformed()
                    
                    dpos = device_state[:3] * (env.max_pos_speed / frequency)
                    drot_xyz = device_state[3:] * (env.max_rot_speed / frequency)
                    # if left button and right button is not pressed, use 3D translation mode
                    # if left button is pressed, use rotation mode
                    # if right button is pressed, use 3D translation mode
                    if not device.is_button_pressed(0) and not device.is_button_pressed(1):
                        drot_xyz[:] = 0

                    elif device.is_button_pressed(0) and not device.is_button_pressed(1):
                        dpos[:] = 0
                        
                    elif not device.is_button_pressed(0) and device.is_button_pressed(1):
                        gripper_openning = target_pose[-1]
                        if gripper_openning < 0.5:
                            target_pose[-1] = 1
                        else:
                            target_pose[-1] = 0
                            
                        # dpos[:] = 0
                        # drot_xyz[:] = 0
                        

                    # if not device.is_button_pressed(0): # 0: left button
                    #     # translation mode
                    #     drot_xyz[:] = 0
                    # else:
                    #     dpos[:] = 0
                        
                    # if not device.is_button_pressed(1): # 1: right button
                    #     # 2D translation mode
                    #     drot_xyz[:] = 0
                    # else:
                    #     dpos[2] = 0
                    #     drot_xyz[:2] = 0

                    drot = st.Rotation.from_euler('xyz', drot_xyz)
                    
                    if not dummy_robot:
                        if not bimanual:
                            if intermediate_pose != []:
                                target_pose = intermediate_pose.pop(0)
                            else:
                                target_pose[:3] += dpos
                                target_pose[3:6] = (drot * st.Rotation.from_rotvec(
                                    target_pose[3:6])).as_rotvec()

                        else:
                            target_pose[:3] += dpos
                            target_pose[6:9] += dpos

                            left_position = target_pose[:3]
                            right_position = target_pose[6:9]
                            
                            left_position_world = transform_to_world(robot_base_in_world['left'], left_position)
                            right_position_world = transform_to_world(robot_base_in_world['right'], right_position)
                            
                                
                            # Transform adjusted positions back to robot base coordinates
                            target_pose[:3] = transform_from_world(robot_base_in_world['left'], left_position_world)
                            target_pose[6:9] = transform_from_world(robot_base_in_world['right'], right_position_world)

                            # Calculate midpoint in world coordinates
                            midpoint_world = (left_position_world + right_position_world) / 2

                            # Translate positions to be relative to midpoint for rotation
                            left_pos_relative_world = left_position_world - midpoint_world
                            right_pos_relative_world = right_position_world - midpoint_world
                            
                            # Apply rotation to the relative positions in world frame
                            left_pos_rotated_world = drot.apply(left_pos_relative_world) + midpoint_world
                            right_pos_rotated_world = drot.apply(right_pos_relative_world) + midpoint_world
                            
                            # Transform adjusted positions back to robot base coordinates if necessary
                            target_pose[:3] = transform_from_world(robot_base_in_world['left'], left_pos_rotated_world)
                            target_pose[6:9] = transform_from_world(robot_base_in_world['right'], right_pos_rotated_world)
                        # execute teleop command
                        # target_joints = env.robot.get_inverse_kinematics(target_pose)
                        target_joints = np.zeros((6,))
                        # target_joints = np.zeros(num_joints_per_bot*(bimanual+1))
                        env.exec_actions(
                            joint_actions=[target_joints], 
                            eef_actions=[target_pose],
                            mode='eef',
                            timestamps=[t_command_target-time.monotonic()+time.time()],
                            stages=[stage])

                elif input_device == 'gello':
                    device_state = device.get_joint_state()
                    
                    if not dummy_robot:
                        target_joints = device_state
                        # target_ee_pose = env.robot.get_forward_kinematics(target_joints)
                        dof = 6 + int(use_gripper)
                        dof = dof * 2 if bimanual else dof
                        target_ee_pose = np.zeros((dof,))
                        if debug:
                            send_command_time = time.time()
                            print(f"Sending command at time: {send_command_time}")
                            print(f"Expected to reach target pose at time: {t_command_target-time.monotonic()+time.time()}")

                        # execute teleop command
                        env.exec_actions(
                            joint_actions=[target_joints], 
                            eef_actions=[target_ee_pose],
                            timestamps=[t_command_target-time.monotonic()+time.time()],
                            stages=[stage])

                precise_wait(t_cycle_end)
                iter_idx += 1

# %%
if __name__ == '__main__':
    main()
