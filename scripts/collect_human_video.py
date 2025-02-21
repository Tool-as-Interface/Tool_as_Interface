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
from ti.real_world.realsense_env import RealsenseEnv
from ti.common.precise_sleep import precise_sleep, precise_wait
from ti.real_world.keystroke_counter import (
    KeystrokeCounter, Key, KeyCode
)

@click.command()
@click.option('--output', '-o', default='data/hammer_human', help="Directory to save demonstration dataset.")
@click.option('--vis_camera_idx', default=2, type=int, help="Which RealSense camera to visualize.")
@click.option('--frequency', '-f', default=10, type=float, help="Control frequency in Hz.")
@click.option('--command_latency', '-cl', default=0.0, type=float, help="Latency between receiving SapceMouse command to executing on Robot in Sec.")
@click.option('--debug', '-d', is_flag=True, default=False, help="Debug mode.")
@click.option('--count_time', '-ct', is_flag=True, default=False, help="Count the time to execute the program.", type=bool)

def main(output, vis_camera_idx, frequency, command_latency, debug, count_time):

    dt = 1/frequency
    # Raise an error if the input_device is not found in the dictionary (device_driver is None)

    with SharedMemoryManager() as shm_manager:
        with KeystrokeCounter() as key_counter, \
            RealsenseEnv(
            output_dir=output, 
            # recording resolution
            obs_image_resolution=(640, 480), # (1280, 720), (640, 480), (480, 270)
            obs_float32=False,
            video_capture_fps=15, # 6, 15, 30
            frequency=frequency,
            n_obs_steps=1,
            enable_multi_cam_vis=False,
            record_raw_video=True,
            # number of threads per camera view for video recording (H.264)
            thread_per_video=3,
            # video recording quality, lower is better (but slower).
            video_crf=21,
            shm_manager=shm_manager,
            enable_depth=True, # TODO: when set to True, it makes the robot show jittery behavior and color image fronzen
            debug=debug,
            ) as env:
                
            cv2.setNumThreads(1)
            
            env.realsense.set_depth_preset('Default')
            env.realsense.set_depth_exposure(33000, 16)

            # wine: 200
            # others :115
            env.realsense.set_exposure(exposure=115, gain=64) 
            
            env.realsense.set_contrast(contrast=60)
            env.realsense.set_white_balance(white_balance=3100)

            obs_duration = 0.0
                
            for _ in range(30):
                out = env.realsense.get()
                time.sleep(0.1)

            time.sleep(2.0)
            print("Number of cameras: ", env.realsense.n_cameras)
            print('Ready!')
            
                    
            t_start = time.monotonic()
            iter_idx = 0
            stop = False
            is_recording = False


            while not stop:
                # calculate timing
                t_cycle_end = t_start + (iter_idx + 1) * (dt+obs_duration) # the end time of the current control cycle

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

                text = f'Saved Episode: {episode_id}'
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

                precise_wait(t_cycle_end)
                iter_idx += 1

# %%
if __name__ == '__main__':
    main()
