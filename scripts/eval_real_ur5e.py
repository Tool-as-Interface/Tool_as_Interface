"""
Usage:
(robodiff)$ python eval_real_robot.py -i <ckpt_path> -o <save_dir> --robot_ip <ip_of_ur5>

================ Human in control ==============
Robot movement:
Move your SpaceMouse to move the robot EEF (locked in xy plane).
Press SpaceMouse right button to unlock z axis.
Press SpaceMouse left button to enable rotation axes.

Recording control:
Click the opencv window (make sure it's in focus).
Press "C" to start evaluation (hand control over to policy).
Press "Q" to exit program.

================ Policy in control ==============
Make sure you can hit the robot hardware emergency-stop button quickly! 

Recording control:
Press "S" to stop evaluation and gain control back.
"""

# %%
import time
from multiprocessing.managers import SharedMemoryManager
import click
import cv2
import numpy as np
import torch
import dill
import hydra
import pathlib
import skvideo.io
from omegaconf import OmegaConf
from omegaconf import open_dict
import scipy.spatial.transform as st
from ti.real_world.real_ur5e_env import RealUR5eEnv
from ti.real_world.real_kinova_env import RealKinovaEnv
from ti.real_world.rtde_interpolation_controller import RTDEInterpolationController, Command
from ti.real_world.spacemouse_shared_memory import Spacemouse
from ti.common.precise_sleep import precise_wait
import diffusers
from ti.real_world.real_inference_util import (
    get_real_obs_resolution, 
    get_real_obs_dict)
from ti.common.pytorch_util import dict_apply
from scripts.train_diffusion_policy import TrainDiffusionUnetHybridWorkspace
from ti.policy.base_image_policy import BaseImagePolicy
from ti.common.cv2_util import get_image_transform, visualize_6d_trajectory
from ti.common.trans_utils import interpolate_poses
from ti.common.data_utils import  policy_action_to_env_action
from ti.utils.segmentation import initialize_segmentation_models
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from ti.common.kin_utils import RobotKinHelper

OmegaConf.register_new_resolver("eval", eval, replace=True)

@click.command()
@click.option('--input', '-i', default='', help='Path to checkpoint')
@click.option('--output', '-o', default='', help='Directory to save recording')
@click.option('--match_dataset', '-m', default=None, help='Dataset used to overlay and adjust initial condition')
@click.option('--match_episode', '-me', default=None, type=int, help='Match specific episode from the match dataset')
@click.option('--vis_camera_idx', default=2, type=int, help="Which RealSense camera to visualize.")
@click.option('--init_joints', '-j', is_flag=True, default=False, help="Whether to initialize robot joint configuration in the beginning.")
@click.option('--steps_per_inference', '-si', default=16, type=int, help="Action horizon for inference.")
@click.option('--max_duration', '-md', default=120, help='Max duration for each epoch in seconds.')
@click.option('--frequency', '-f', default=10, type=float, help="Control frequency in Hz.")
@click.option('--command_latency', '-cl', default=0.01, type=float, help="Latency between receiving SapceMouse command to executing on Robot in Sec.")
@click.option('--bimanual', '-b', default=False, type=bool)
@click.option('--use_gripper', '-ug', default=True, type=bool)
@click.option('--ctrl_mode', '-cm', default='eef', type=str)
@click.option('--debug', '-d', default=False, type=bool)
def main(input, output, match_dataset, match_episode,
    vis_camera_idx, init_joints, 
    steps_per_inference, max_duration,
    frequency, command_latency,
    bimanual,use_gripper, ctrl_mode, debug):

    # load match_dataset
    match_camera_idx = 0
    episode_first_frame_map = dict()
    if match_dataset is not None:
        match_dir = pathlib.Path(match_dataset)
        match_video_dir = match_dir.joinpath('videos')
        for vid_dir in match_video_dir.glob("*/"):
            episode_idx = int(vid_dir.stem)
            match_video_path = vid_dir.joinpath(f'{match_camera_idx}.mp4')
            if match_video_path.exists():
                frames = skvideo.io.vread(
                    str(match_video_path), num_frames=1)
                episode_first_frame_map[episode_idx] = frames[0]
    print(f"Loaded initial frame for {len(episode_first_frame_map)} episodes")
    
    # load checkpoint
    ckpt_path = input
    payload = torch.load(open(ckpt_path, 'rb'), pickle_module=dill)
    cfg = payload['cfg']

    workspace = TrainDiffusionUnetHybridWorkspace(cfg)
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    num_joints = 6 + int(use_gripper)
    num_joints = num_joints * 2 if bimanual else num_joints
    

    # hacks for method-specific setup.
    action_offset = 1
    delta_action = False
    if 'diffusion' in cfg.name:
        # diffusion model
        policy: BaseImagePolicy
        policy = workspace.model
        if cfg.training.use_ema:
            policy = workspace.ema_model

        device = torch.device('cuda')
        policy.eval().to(device)

        # set inference params
        policy.num_inference_steps = 16 # DDIM inference iterations
        policy.n_action_steps = policy.horizon - policy.n_obs_steps + 1
        noise_scheduler = diffusers.schedulers.scheduling_ddim.DDIMScheduler(
            num_train_timesteps=100,
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule='squaredcos_cap_v2',
            clip_sample=True,
            set_alpha_to_one=True,
            steps_offset=0,
            prediction_type='epsilon'
        )
        policy.noise_scheduler = noise_scheduler
        if 'type' not in cfg.task.shape_meta['action'] or cfg.task.shape_meta['action']['type'] == 'cartesian_action':
            ctrl_mode = 'eef'
        elif cfg.task.shape_meta['action']['type'] == 'joint_action':
            ctrl_mode = 'joint'
        else:
            raise RuntimeError("Unsupported action mode: ", cfg.task.shape_meta['action']['type'])
        


    elif 'robomimic' in cfg.name:
        # BCRNN model
        policy: BaseImagePolicy
        policy = workspace.model

        device = torch.device('cuda')
        policy.eval().to(device)

        # BCRNN always has action horizon of 1
        steps_per_inference = 1
        action_offset = cfg.n_latency_steps
        delta_action = cfg.task.dataset.get('delta_action', False)

    elif 'ibc' in cfg.name:
        policy: BaseImagePolicy
        policy = workspace.model
        policy.pred_n_iter = 5
        policy.pred_n_samples = 4096

        device = torch.device('cuda')
        policy.eval().to(device)
        steps_per_inference = 1
        action_offset = 1
        delta_action = cfg.task.dataset.get('delta_action', False)
    else:
        raise RuntimeError("Unsupported policy type: ", cfg.name)
    
    max_pos_speed=0.2
    max_rot_speed=0.4 

    # setup experiment
    dt = 1/frequency
    
        
    tool_names = [None, None]
    shape_meta = cfg.task.shape_meta
    if 'point_cloud' in shape_meta:
        if 'left_tool' in shape_meta['obs']['point_cloud']['info']:
            tool_names[0] = shape_meta['obs']['point_cloud']['info']['left_tool']

        if 'right_tool' in shape_meta['obs']['point_cloud']['info']:
            tool_names[1] = shape_meta['obs']['point_cloud']['info']['right_tool']
        N_per_link = shape_meta['obs']['point_cloud']['info']['N_per_link']
        N_eef = shape_meta['obs']['point_cloud']['info']['N_eef']
        kin_helper = RobotKinHelper('ur5e', tool_name=tool_names[1], N_per_link=N_per_link, N_tool=N_eef)
    else:
        kin_helper = None
    if 'shape_meta' in cfg.task:
        out_obs_res = get_real_obs_resolution(cfg.task.shape_meta)
    obs_res = None
    if obs_res is None:
        obs_res = (640, 480)  # current task does not have shape_meta


    real_obs_info = {}
    if 'camera_features' in cfg.task.shape_meta.obs:
            
        real_obs_info['feature_view_key'] = 'camera_2' 
        grounding_dino_model, sam_predictor = initialize_segmentation_models()

        real_obs_info['grounding_dino_model'] = grounding_dino_model
        real_obs_info['sam_predictor'] = sam_predictor

    # obs_res = get_real_obs_resolution(cfg.task.shape_meta)
    n_obs_steps = cfg.n_obs_steps
    print("n_obs_steps: ", n_obs_steps)
    print("steps_per_inference:", steps_per_inference)
    print("action_offset:", action_offset)
    
    robot_env = RealKinovaEnv if cfg.task.robot_side=='kinova' else RealUR5eEnv 

    with SharedMemoryManager() as shm_manager:
        with Spacemouse(shm_manager=shm_manager,
                        bimanual=False,
                        use_gripper=use_gripper) as sm, robot_env(
            output_dir=output, 
            frequency=frequency,
            n_obs_steps=n_obs_steps,
            obs_image_resolution=obs_res,
            obs_float32=False,
            init_joints=init_joints,
            video_capture_fps=15, # 6, 15, 30
            max_pos_speed=max_pos_speed,
            max_rot_speed=max_rot_speed,
            speed_slider_value=1,
            lookahead_time=0.1,
            gain=100, 
            enable_multi_cam_vis=False,
            record_raw_video=True,
            # number of threads per camera view for video recording (H.264)
            thread_per_video=3,
            # video recording quality, lower is better (but slower).
            video_crf=21,
            shm_manager=shm_manager,
            bimanual=bimanual,
            single_arm_type='right',
            ctrl_mode=ctrl_mode,
            use_gripper=use_gripper) as env:
            cv2.setNumThreads(1)

            env.realsense.set_depth_preset('Default')
            env.realsense.set_depth_exposure(33000, 16)

            env.realsense.set_exposure(exposure=115, gain=64)
            env.realsense.set_contrast(contrast=60)
            env.realsense.set_white_balance(white_balance=3100)

            print("Waiting for realsense")
            time.sleep(3)

            print("Warming up policy inference")
            obs = env.get_obs()
            with torch.no_grad():
                policy.reset()
                obs_dict_np = get_real_obs_dict(
                    env_obs=obs, shape_meta=cfg.task.shape_meta,
                    task_conf=cfg.task,
                    kin_helper=kin_helper,
                    real_obs_info=real_obs_info)

                obs_dict = dict_apply(obs_dict_np, 
                    lambda x: torch.from_numpy(x).unsqueeze(0).to(device))
                result = policy.predict_action(obs_dict)
                action = result['action'][0].detach().to('cpu').numpy()
                action = policy_action_to_env_action(raw_actions=action, 
                                                     cur_eef_pose_6d=obs['robot_eef_pose'][-1], 
                                                     action_mode=ctrl_mode)
                del result
                
                
            intermediate_pose = []

            print('Ready!')
            while True:
                # ========= human control loop ==========
                print("Human in control!")
                state = env.get_robot_state()
                target_pose = state['TargetTCPPose']
                # target_joints = state['TargetQ']
                t_start = time.monotonic()
                iter_idx = 0
                while True:
                    # calculate timing
                    t_cycle_end = t_start + (iter_idx + 1) * dt
                    t_sample = t_cycle_end - command_latency
                    t_command_target = t_cycle_end + dt 

                    # pump obs
                    obs = env.get_obs()

                    # visualize
                    episode_id = env.replay_buffer.n_episodes
                    vis_img = obs[f'camera_{vis_camera_idx}_color'][-1]
                    match_episode_id = episode_id
                    if match_episode is not None:
                        match_episode_id = match_episode
                    if match_episode_id in episode_first_frame_map:
                        match_img = episode_first_frame_map[match_episode_id]
                        ih, iw, _ = match_img.shape
                        oh, ow, _ = vis_img.shape
                        tf = get_image_transform(
                            input_res=(iw, ih), 
                            output_res=(ow, oh), 
                            bgr_to_rgb=False)
                        match_img = tf(match_img).astype(np.float32) / 255
                        vis_img = np.minimum(vis_img, match_img)

                    text = f'Episode: {episode_id}'
                    cv2.putText(
                        vis_img,
                        text,
                        (10,20),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5,
                        thickness=1,
                        color=(255,255,255)
                    )
                    cv2.imshow('default', vis_img[...,::-1])
                    key_stroke = cv2.pollKey()
                    if key_stroke == ord('q'):
                        # Exit program
                        env.end_episode()
                        exit(0)
                    elif key_stroke == ord('c'):
                        # Exit human control loop
                        # hand control over to the policy
                        break
                    elif key_stroke == ord('l') and ctrl_mode=='eef' and bimanual==False : # hammer 
                        state = env.get_robot_state()
                        start_pose = state['ActualTCPPose']
                        final_pose = np.array([0.428, 0.042, -0.003, 2.2, -2.2, 0])   
                        
                        for goal_pose in [final_pose]:
                            step_intermediate_pose, _, _ = interpolate_poses(start_pose, goal_pose, 0.005)
                            intermediate_pose.extend(step_intermediate_pose)
                            intermediate_pose.append(goal_pose)
                            start_pose = goal_pose


                    precise_wait(t_sample)
                    if bimanual == False:
                        # get teleop command
                        sm_state = sm.get_motion_state_transformed()
                        # print(sm_state)
                        dpos = sm_state[:3] * (env.max_pos_speed / frequency)
                        drot_xyz = sm_state[3:] * (env.max_rot_speed / frequency)
    
                        if not sm.is_button_pressed(0) and not sm.is_button_pressed(1):
                            # 2D translation mode
                            drot_xyz[:2] = 0
                        elif sm.is_button_pressed(0) and not sm.is_button_pressed(1):
                            dpos[:] = 0
                        elif not sm.is_button_pressed(0) and sm.is_button_pressed(1):
                            dpos[:2] = 0
                            drot_xyz[:2] = 0
                            

                        drot = st.Rotation.from_euler('xyz', drot_xyz)
                        if intermediate_pose != []:
                            target_pose = intermediate_pose.pop(0)
                        else:
                            target_pose[:3] += dpos
                            target_pose[3:] = (drot * st.Rotation.from_rotvec(
                                target_pose[3:])).as_rotvec()
                            
                    if intermediate_pose != [] and bimanual==True:
                        target_pose = intermediate_pose.pop(0)

                    target_joints = np.zeros((num_joints,))

                    # execute teleop command
                    # env.robot.servo_ee_pose(target_pose, duration=0.1)
                    env.exec_actions(
                        joint_actions=[target_joints], 
                        eef_actions=[target_pose],
                        mode=ctrl_mode,
                        timestamps=[t_command_target-time.monotonic()+time.time()])
                        
                    precise_wait(t_cycle_end)
                    iter_idx += 1
                
                # ========== policy control loop ==============
                try:
                    # start episode
                    policy.reset()
                    start_delay = 1.0
                    eval_t_start = time.time() + start_delay
                    t_start = time.monotonic() + start_delay
                    env.start_episode(eval_t_start)
                    # wait for 1/30 sec to get the closest frame actually
                    # reduces overall latency
                    frame_latency = 1/30
                    precise_wait(eval_t_start - frame_latency, time_func=time.time)
                    print("Started!")
                    iter_idx = 0
                    term_area_start_timestamp = float('inf')
                    perv_target_pose = None
                    

                    while True:
                        # calculate timing
                        t_cycle_end = t_start + (iter_idx + steps_per_inference) * dt

                        # get obs
                        print('get_obs')
                        obs = env.get_obs()
                        obs_timestamps = obs['timestamp']
                        print(f'Obs latency {time.time() - obs_timestamps[-1]}')

                        # run inference
                        with torch.no_grad():
                            s = time.time()
                            obs_dict_np = get_real_obs_dict(
                                env_obs=obs, shape_meta=cfg.task.shape_meta,
                                task_conf=cfg.task,
                                kin_helper=kin_helper,
                                real_obs_info=real_obs_info)

                            obs_dict = dict_apply(obs_dict_np, 
                                lambda x: torch.from_numpy(x).unsqueeze(0).to(device))
                            result = policy.predict_action(obs_dict)
                            # this action starts from the first obs step
                            action = result['action'][0].detach().to('cpu').numpy()
                            action = policy_action_to_env_action(raw_actions=action, 
                                                                cur_eef_pose_6d=obs['robot_eef_pose'][-1], 
                                                                action_mode=ctrl_mode)
                            if debug:
                                visualize_6d_trajectory(action)

                            print('Inference latency:', time.time() - s)
                        
                        # convert policy action to env actions
                        if delta_action:
                            assert len(action) == 1
                            if perv_target_pose is None:
                                perv_target_pose = obs['robot_eef_pose'][-1]
                            this_target_pose = perv_target_pose.copy()
                            this_target_pose[[0,1]] += action[-1]
                            perv_target_pose = this_target_pose
                            this_target_poses = np.expand_dims(this_target_pose, axis=0)
                        else:
                            # this_target_poses = np.zeros((len(action), len(target_pose)), dtype=np.float64)
                            # this_target_poses[:] = target_pose
                            # this_target_poses[:,[0,1]] = action
                            this_target_poses = action
                            
                        # deal with timing
                        # the same step actions are always the target for
                        action_timestamps = (np.arange(len(action), dtype=np.float64) + action_offset
                            ) * dt + obs_timestamps[-1]
                        action_exec_latency = 0.01
                        curr_time = time.time()
                        is_new = action_timestamps > (curr_time + action_exec_latency)
                        if np.sum(is_new) == 0:
                            # exceeded time budget, still do something
                            this_target_poses = this_target_poses[[-1]]
                            # schedule on next available step
                            next_step_idx = int(np.ceil((curr_time - eval_t_start) / dt))
                            action_timestamp = eval_t_start + (next_step_idx) * dt
                            print('Over budget', action_timestamp - curr_time)
                            action_timestamps = np.array([action_timestamp])
                        else:
                            this_target_poses = this_target_poses[is_new]
                            action_timestamps = action_timestamps[is_new]


                        # execute actions
                        this_target_joints = [np.zeros(num_joints) for _ in this_target_poses]
                        this_target_joints = np.stack(this_target_joints, axis=0)

                        env.exec_actions(
                            joint_actions=this_target_joints, 
                            eef_actions=this_target_poses,
                            mode='eef',
                            timestamps=action_timestamps)

                        
                        print(f"Submitted {len(this_target_poses)} steps of actions.")

                        # visualize
                        episode_id = env.replay_buffer.n_episodes
                        vis_img = obs[f'camera_{vis_camera_idx}_color'][-1]
                        text = 'Episode: {}, Time: {:.1f}'.format(
                            episode_id, time.monotonic() - t_start
                        )
                        cv2.putText(
                            vis_img,
                            text,
                            (10,20),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=0.5,
                            thickness=1,
                            color=(255,255,255)
                        )
                        cv2.imshow('default', vis_img[...,::-1])


                        key_stroke = cv2.pollKey()
                        if key_stroke == ord('s'):
                            # Stop episode
                            # Hand control back to human
                            env.end_episode()
                            print('Stopped.')
                            break

                        # auto termination
                        terminate = False
                        if time.monotonic() - t_start > max_duration:
                            terminate = True
                            print('Terminated by the timeout!')

                        if terminate:
                            env.end_episode()
                            break

                        # wait for execution
                        precise_wait(t_cycle_end - frame_latency)
                        iter_idx += steps_per_inference
                        
                except KeyboardInterrupt:
                    print("Interrupted!")
                    # stop robot.
                    env.end_episode()
                
                print("Stopped.")



# %%
if __name__ == '__main__':
    main()
