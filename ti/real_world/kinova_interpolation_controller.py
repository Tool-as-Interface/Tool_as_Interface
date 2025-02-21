import os
import sys
sys.path.insert(1, '.')
import time
import enum
import warnings
import multiprocessing as mp
from multiprocessing.managers import SharedMemoryManager
import scipy.interpolate as si
import scipy.spatial.transform as st
import numpy as np
from ti.shared_memory.shared_memory_queue import (
    SharedMemoryQueue, Empty)
from ti.shared_memory.shared_memory_ring_buffer import SharedMemoryRingBuffer
from ti.shared_memory.shared_ndarray import SharedNDArray
from ti.common.pose_trajectory_interpolator import PoseTrajectoryInterpolator
from ti.common.linear_interpolator import LinearInterpolator
from ti.devices.gello_shared_memory import Gello
import yaml
from ti.model.common.tensor_util import to_list
from ti.common.kin_utils import RobotKinHelper
from scipy.spatial.transform import Rotation as R
from kortex_api.autogen.messages import Base_pb2, BaseCyclic_pb2, Common_pb2, Session_pb2
import threading
from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
from kortex_api.autogen.client_stubs.BaseCyclicClientRpc import BaseCyclicClient
import ti.real_world.kinova_utilities as kinova_utilities
from scipy.spatial.transform import Rotation

import pdb

TIMEOUT_DURATION = 200

# Create closure to set an event after an END or an ABORT
def check_for_end_or_abort(e):
    def check(notification, e = e):
        if notification.action_event == Base_pb2.ACTION_END \
        or notification.action_event == Base_pb2.ACTION_ABORT:
            e.set()
    return check

def populateCartesianCoordinate(waypointInformation):
    
    waypoint = Base_pb2.CartesianWaypoint()  
    waypoint.pose.x = waypointInformation[0]
    waypoint.pose.y = waypointInformation[1]
    waypoint.pose.z = waypointInformation[2]
    waypoint.blending_radius = waypointInformation[3]
    waypoint.pose.theta_x = waypointInformation[4]
    waypoint.pose.theta_y = waypointInformation[5]
    waypoint.pose.theta_z = waypointInformation[6] 
    waypoint.reference_frame = Base_pb2.CARTESIAN_REFERENCE_FRAME_BASE
    
    return waypoint


def example_move_to_home_position(base):
    # Make sure the arm is in Single Level Servoing mode
    base_servo_mode = Base_pb2.ServoingModeInformation()
    base_servo_mode.servoing_mode = Base_pb2.SINGLE_LEVEL_SERVOING
    base.SetServoingMode(base_servo_mode)
    
    # Move arm to ready position
    print("Moving the arm to a safe position")
    action_type = Base_pb2.RequestedActionType()
    action_type.action_type = Base_pb2.REACH_JOINT_ANGLES
    action_list = base.ReadAllActions(action_type)
    action_handle = None
    for action in action_list.action_list:
        if action.name == "Home":
            action_handle = action.handle

    if action_handle == None:
        print("Can't reach safe position. Exiting")
        return False

    e = threading.Event()
    notification_handle = base.OnNotificationActionTopic(
        check_for_end_or_abort(e),
        Base_pb2.NotificationOptions()
    )

    base.ExecuteActionFromReference(action_handle)
    finished = e.wait(TIMEOUT_DURATION)
    base.Unsubscribe(notification_handle)

    if finished:
        print("Safe position reached")
    else:
        print("Timeout on action notification wait")
    return finished

def move_to_desired_pose(base, base_cyclic, pose):
    
    x, y, z, rotvec_x, rotvec_y, rotvec_z = pose
    theta_x, theta_y, theta_z = Rotation.from_rotvec([rotvec_x, rotvec_y, rotvec_z], degrees=False).as_euler('xyz', degrees=True).tolist()
    # Construct action
    action = Base_pb2.Action()
    action.name = "Move to desired position"
    action.application_data = ""
    
    # Set target pose
    cartesian_pose = action.reach_pose.target_pose
    cartesian_pose.x = x
    cartesian_pose.y = y
    cartesian_pose.z = z
    cartesian_pose.theta_x = theta_x
    cartesian_pose.theta_y = theta_y
    cartesian_pose.theta_z = theta_z

    # Start action
    # print("Executing action")
    # base.ExecuteAction(action)

    # Wait for action to finish
    e = threading.Event()
    notification_handle = base.OnNotificationActionTopic(check_for_end_or_abort(e), Base_pb2.NotificationOptions())
    base.ExecuteAction(action)
    finished = e.wait(TIMEOUT_DURATION)
    base.Unsubscribe(notification_handle)

    # if finished:
    #     print("Movement completed")
    # else:
    #     print("Timeout on action notification wait")
    return finished

def get_feedback(base, base_cyclic):
    # Refresh feedback
    feedback = base_cyclic.RefreshFeedback()

    # Get the end effector pose
    pose = feedback.base

    # start from base to eef: 7 joints in total
    joint = np.array([actuator.position for actuator in feedback.actuators])

    # Get the gripper state
    gripper_feedback = feedback.interconnect.gripper_feedback
    motor = gripper_feedback.motor[0]

    obs = {
        'ActualTCPPose': np.array([pose.tool_pose_x, pose.tool_pose_y, pose.tool_pose_z] + \
            Rotation.from_euler('xyz', [pose.tool_pose_theta_x, pose.tool_pose_theta_y, pose.tool_pose_theta_z], degrees=True).as_rotvec().tolist()),
        'ActualTCPSpeed': np.array([pose.tool_twist_linear_x, pose.tool_twist_linear_y, pose.tool_twist_linear_z] + \
            Rotation.from_euler('xyz', [pose.tool_twist_angular_x, pose.tool_twist_angular_y, pose.tool_twist_angular_z], degrees=True).as_rotvec().tolist()),
        'TargetTCPPose': np.array([pose.commanded_tool_pose_x, pose.commanded_tool_pose_y, pose.commanded_tool_pose_z] + \
            Rotation.from_euler('xyz', [pose.commanded_tool_pose_theta_x, pose.commanded_tool_pose_theta_y, pose.commanded_tool_pose_theta_z], degrees=True).as_rotvec().tolist()),
        'FtRawWrench': np.array([pose.tool_external_wrench_force_x, pose.tool_external_wrench_force_y, pose.tool_external_wrench_force_z] + \
            [pose.tool_external_wrench_torque_x, pose.tool_external_wrench_torque_y, pose.tool_external_wrench_torque_z]),
        'ActualQ': joint,
        'IMUAcceleration': np.array([pose.imu_acceleration_x, pose.imu_acceleration_y, pose.imu_acceleration_z]),
        'IMUAngularVelocity': np.array([pose.imu_angular_velocity_x, pose.imu_angular_velocity_y, pose.imu_angular_velocity_z]),
        'gripper': motor.position
    }

    return obs



class Command(enum.Enum):
    STOP = 0
    SERVO_EE_POSE = 1
    SCHEDULE_WAYPOINT = 2
    SERVOJ = 3
    SCHEDULE_JOINTS = 4
    GET_FK = 5
    GET_IK = 6
    SET_FREEDRIVE = 7


class KortexInterpolationController(mp.Process):
    """
    To ensure sending command to the robot with predictable latency
    this controller need its separate process (due to python GIL)
    """


    def __init__(self,
            shm_manager: SharedMemoryManager, 
            robot_ip, 
            frequency=125, 
            lookahead_time=0.1, 
            gain=100,
            max_pos_speed=0.25, # 5% of max speed
            max_rot_speed=0.16, # 5% of max speed
            speed_slider_value=0.1,
            launch_timeout=3,
            tcp_offset_pose=None,
            payload_mass=None,
            payload_cog=None,
            joints_init=None,
            joints_init_speed=0.5,
            soft_real_time=False,
            verbose=False,
            receive_keys=None,
            get_max_k=128,
            dummy_robot=False,
            ctrl_mode='eef',
            extrinsics_dir=os.path.join(os.path.dirname(__file__), 'robot_extrinsics'),
            extrinsics_name=f'right_base_pose_in_world.npy',
            ):
        """
        frequency: CB2=125, UR3e=500
        lookahead_time: [0.03, 0.2]s smoothens the trajectory with this lookahead time
        gain: [100, 2000] proportional gain for following target position
        max_pos_speed: m/s
        max_rot_speed: rad/s
        tcp_offset_pose: 6d pose
        payload_mass: float
        payload_cog: 3d position, center of gravity
        soft_real_time: enables round-robin scheduling and real-time priority
            requires running scripts/rtprio_setup.sh before hand.

        """
        # verify
        assert 0 < frequency <= 500
        assert 0.03 <= lookahead_time <= 0.2
        assert 100 <= gain <= 2000
        assert 0 < max_pos_speed
        assert 0 < max_rot_speed
        if tcp_offset_pose is not None:
            tcp_offset_pose = np.array(tcp_offset_pose)
            assert tcp_offset_pose.shape == (6,)
        if payload_mass is not None:
            assert 0 <= payload_mass <= 5
        if payload_cog is not None:
            payload_cog = np.array(payload_cog)
            assert payload_cog.shape == (3,)
            assert payload_mass is not None
        if joints_init is not None:
            joints_init = np.array(joints_init)
            assert joints_init.shape == (7,)

        super().__init__(name="Kinova Controller")
        self.dummy_robot = dummy_robot
        self.robot_ip = robot_ip
        self.frequency = frequency
        self.lookahead_time = lookahead_time
        self.gain = gain
        self.max_pos_speed = max_pos_speed
        self.max_rot_speed = max_rot_speed
        self.speed_slider_value = speed_slider_value
        self.launch_timeout = launch_timeout
        self.tcp_offset_pose = tcp_offset_pose
        self.payload_mass = payload_mass
        self.payload_cog = payload_cog
        self.joints_init = joints_init
        self.joints_init_speed = joints_init_speed
        self.soft_real_time = soft_real_time
        self.verbose = verbose
        
        assert ctrl_mode in ['eef', 'joint'], "ctrl_mode must be either 'eef' or 'joint'"
        self.ctrl_mode = ctrl_mode
        
        self.num_joints = 6 
        
        os.system(f'mkdir -p {extrinsics_dir}')
        if extrinsics_dir is None:
            self.base_pose_in_world = np.eye(4)
            warnings.warn("extrinsics_dir is None, using identity matrix as base pose in world")
        else:
            extrinsics_path = os.path.join(extrinsics_dir, extrinsics_name) 
            self.base_pose_in_world = np.eye(4)
            if not os.path.exists(extrinsics_path):
                self.base_pose_in_world = np.eye(4)
                warnings.warn(f"extrinsics_path {extrinsics_path} does not exist, using identity matrix as base pose in world")
            else:
                self.base_pose_in_world = np.load(extrinsics_path)

        # build input queue
        example = {
            'cmd': Command.SERVO_EE_POSE.value,
            'target_ee_pose': np.zeros((self.num_joints,), dtype=np.float64),
            'target_joints': np.zeros((self.num_joints,), dtype=np.float64), 
            'joints': np.zeros((7,), dtype=np.float64),
            'duration': 0.0,
            'target_time': 0.0,
            'speed': 0.0,
            'acceleration': 0.0,
            'asynchronous': False,
            'free_drive': 0
        }
        input_queue = SharedMemoryQueue.create_from_examples(
            shm_manager=shm_manager,
            examples=example,
            buffer_size=256
        )

        # build ring buffer
        if receive_keys is None:
            receive_keys = [
                'ActualTCPPose',
                'ActualTCPSpeed',
                'ActualQ',
                'ActualQd',

                'TargetTCPPose',
                'TargetTCPSpeed',
                'TargetQ',
                'TargetQd',
                
                'FtRawWrench'
            ]
        example = dict()
        if not self.dummy_robot:

            self.kinova_args = kinova_utilities.parseConnectionArguments()

            # Create connection to the device and get the router
            with kinova_utilities.DeviceConnection.createTcpConnection(self.kinova_args) as router:

                # Create required services
                base = BaseClient(router)
                base_cyclic = BaseCyclicClient(router)
        
                obs = get_feedback(base, base_cyclic)
                example.update(obs)

        example['robot_receive_timestamp'] = time.time()
        example['robot_base_pose_in_world'] = self.base_pose_in_world
        ring_buffer = SharedMemoryRingBuffer.create_from_examples(
            shm_manager=shm_manager,
            examples=example,
            get_max_k=get_max_k,
            get_time_budget=0.2,
            put_desired_frequency=frequency
        )

        self.ready_event = mp.Event()
        self.input_queue = input_queue
        self.ring_buffer = ring_buffer
        self.receive_keys = receive_keys
            
        

    # ========= launch method ===========
    def start(self, wait=True):
        super().start()
        if self.dummy_robot:
            return True
        if wait:
            self.start_wait()
        if self.verbose:
            print(f"[Kinova Kortex2] Controller process spawned at {self.pid}")

    def stop(self, wait=True):
        if self.dummy_robot:
            return True
        message = {
            'cmd': Command.STOP.value
        }
        self.input_queue.put(message)
        if wait:
            self.stop_wait()

    def start_wait(self):
        if self.dummy_robot:
            return True

        self.ready_event.wait(self.launch_timeout)
        assert self.is_alive()
    
    def stop_wait(self):
        self.join()
        if self.dummy_robot:
            return True

    @property
    def is_ready(self):
        if self.dummy_robot:
            return True

        return self.ready_event.is_set()

    # ========= context manager ===========
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        
    # ========= command methods ============
    def servo_ee_pose(self, pose, duration=0.1):
        """
        duration: desired time to reach pose
        """
        assert self.ctrl_mode == 'eef', "This function is only supported in eef mode"
        if self.dummy_robot:
            return True
        assert self.is_alive()
        assert(duration >= (1/self.frequency))
        pose = np.array(pose)
        assert pose.shape == (6, )
        target_joint_pos = np.zeros((self.num_joints,), dtype=np.float64)
            
        message = {
            'cmd': Command.SERVO_EE_POSE.value,
            'target_joints': target_joint_pos, # dummy value
            'target_ee_pose': pose,
            'duration': duration
        }
        self.input_queue.put(message)
        

    def schedule_waypoint(self, pose, target_time):
        assert self.ctrl_mode == 'eef', "This function is only supported in eef mode"
        assert target_time > time.time()
        pose = np.array(pose)
        assert pose.shape == (6, )

        message = {
            'cmd': Command.SCHEDULE_WAYPOINT.value,
            'target_ee_pose': pose,
            'target_time': target_time
        }
        self.input_queue.put(message)





    # ========= receive APIs =============
    def get_state(self, k=None, out=None):
        if k is None:
            return self.ring_buffer.get(out=out)
        else:
            return self.ring_buffer.get_last_k(k=k,out=out)
    
    def get_all_state(self):
        return self.ring_buffer.get_all()


    
    # ========= main loop in process ============
    def run(self):
        if self.dummy_robot:
            return
        # enable soft real-time
        if self.soft_real_time:
            os.sched_setscheduler(
                0, os.SCHED_RR, os.sched_param(20))

        # start rtde
        robot_ip = self.robot_ip
        
        # try:
        #     # Parse arguments
        #     # kinova_args = kinova_utilities.parseConnectionArguments()

        #     # Create connection to the device and get the router
        #     device_connection = kinova_utilities.DeviceConnection.createTcpConnection(self.kinova_args)

        #     # Manually do what __enter__ would have done
        #     router = device_connection.__enter__()
            
        #     # Create required services
        #     base = BaseClient(router)
        #     base_cyclic = BaseCyclicClient(router)
        # except:
        #     print("Failed to connect to Kinova Gen3 Robot")
        
        try:
            with kinova_utilities.DeviceConnection.createTcpConnection(self.kinova_args) as router:
                base = BaseClient(router)
                base_cyclic = BaseCyclicClient(router)

                # main loop
                dt = 1. / self.frequency
                curr_ee_pose = get_feedback(base, base_cyclic)['ActualTCPPose']
                
                # use monotonic time to make sure the control loop never go backward
                curr_t = time.monotonic()
                last_waypoint_time = curr_t

                pose_interp = PoseTrajectoryInterpolator(
                    times=[curr_t],
                    poses=[curr_ee_pose]
                )

                iter_idx = 0
                keep_running = True
                while keep_running:

                    # start control iteration
                    t_start = time.perf_counter()

                    # send command to robot
                    t_now = time.monotonic()
                    

                    pose_command = pose_interp(t_now)
                    assert move_to_desired_pose(base, base_cyclic, pose_command)


                    # update robot state
                    state = dict()
                    obs = get_feedback(base, base_cyclic)
                    state.update(obs)
                    state['robot_receive_timestamp'] = time.time()
                    state['robot_base_pose_in_world'] = self.base_pose_in_world

                    self.ring_buffer.put(state)

                    # fetch command from queue
                    try:
                        commands = self.input_queue.get_all()
                        n_cmd = len(commands['cmd'])
                    except Empty:
                        n_cmd = 0

                    # execute commands
                    for i in range(n_cmd):
                        command = dict()
                        for key, value in commands.items():
                            command[key] = value[i]
                        cmd = command['cmd']

                        if cmd == Command.STOP.value:
                            keep_running = False
                            # stop immediately, ignore later commands
                            break
                        elif cmd == Command.SERVO_EE_POSE.value:
                            # since curr_pose always lag behind curr_target_pose
                            # if we start the next interpolation with curr_pose
                            # the command robot receive will have discontinouity 
                            # and cause jittery robot behavior.
                            target_pose = command['target_ee_pose']
                            duration = float(command['duration'])
                            curr_time = t_now + dt
                            t_insert = curr_time + duration
                            pose_interp = pose_interp.drive_to_waypoint(
                                pose=target_pose,
                                time=t_insert,
                                curr_time=curr_time,
                                max_pos_speed=self.max_pos_speed,
                                max_rot_speed=self.max_rot_speed
                            )
                            last_waypoint_time = t_insert
                            if self.verbose:
                                print("[Kortex Kinova] New pose target:{} duration:{}s".format(
                                    target_pose, duration))
                                
                        elif cmd == Command.SCHEDULE_WAYPOINT.value:
                            target_pose = command['target_ee_pose']
                            target_time = float(command['target_time'])
                            # translate global time to monotonic time
                            target_time = time.monotonic() - time.time() + target_time
                            curr_time = t_now + dt
                            pose_interp = pose_interp.schedule_waypoint(
                                pose=target_pose[:6],
                                time=target_time,
                                max_pos_speed=self.max_pos_speed,
                                max_rot_speed=self.max_rot_speed,
                                curr_time=curr_time,
                                last_waypoint_time=last_waypoint_time
                            )
                            last_waypoint_time = target_time
                        else:
                            keep_running = False
                            break
                    


                    # first loop successful, ready to receive command
                    if iter_idx == 0:
                        self.ready_event.set()
                    iter_idx += 1

                    if self.verbose:
                        print(f"[Kinova] Actual frequency {1/(time.perf_counter() - t_start)}")

        finally:
            # manditory cleanup
            # decelerate
            self.ready_event.set()





def test_ee_rot():
    frequency = 10
    dt = 1 / frequency

    with SharedMemoryManager() as shm_manager:
        with KortexInterpolationController(
                shm_manager=shm_manager, 
                robot_ip='192.168.1.10', 
                frequency=frequency,
                max_pos_speed=0.5,
                max_rot_speed=0.5,
                dummy_robot=False,
                ctrl_mode='eef'
            ) as controller:
            while not controller.is_ready:
                time.sleep(0.1)

            i = 0
            while i < 10:  # Limit the number of iterations to 10 for testing
                start_time = time.monotonic()
                cur_robot_state = controller.get_state()
                eef_pose = cur_robot_state['ActualTCPPose']

                # Extract rotation vector (assumed to be the last three elements)
                rot_vec = np.array(eef_pose[3:6])
                print(f"Original Rotation Vector: {rot_vec}")

                # Convert rotation vector to quaternion
                quaternion = R.from_rotvec(rot_vec).as_quat()
                print(f"Quaternion: {quaternion}")

                # Modify the quaternion slightly
                quaternion[3] += 0.1  # increment quaternion's real part slightly
                new_rotvec = R.from_quat(quaternion).as_rotvec()
                print(f"Modified Rotation Vector: {new_rotvec}")

                # Apply the new rotation vector back to eef_pose
                eef_pose[3:6] = new_rotvec

                target_time = time.time() + 0.2
                controller.schedule_waypoint(eef_pose, target_time=target_time)

                time.sleep(max(0, dt - (time.monotonic() - start_time)))
                i += 1



def test_ee_translation():
    frequency = 10
    dt = 1 / frequency

    with SharedMemoryManager() as shm_manager:
        with KortexInterpolationController(
                shm_manager=shm_manager, 
                robot_ip='192.168.1.10', 
                frequency=frequency,
                max_pos_speed=0.5,
                max_rot_speed=0.5,
                dummy_robot=False,
                ctrl_mode='eef'
            ) as controller:
            while not controller.is_ready:
                time.sleep(0.1)

            i = 0
            while i < 10:  # Limit the number of iterations to 10 for testing
                start_time = time.monotonic()
                cur_robot_state = controller.get_state()
                eef_pose = cur_robot_state['ActualTCPPose']

                # Extract translation vector (first three elements) and rotation vector (last three elements)
                translation_vec = np.array(eef_pose[0:3])
                rot_vec = np.array(eef_pose[3:6])
                print(f"Original Translation Vector: {translation_vec}")
                print(f"Original Rotation Vector: {rot_vec}")

                # Apply a small translation adjustment
                translation_vec += np.array([0.0, 0.0, 0.05])  # Adjust translation as desired
                print(f"Modified Translation Vector: {translation_vec}")

                # Modify the quaternion slightly
                eef_pose[0:3] = translation_vec  # Updated translation

                target_time = time.time() + 0.2
                controller.schedule_waypoint(eef_pose, target_time=target_time)

                time.sleep(max(0, dt - (time.monotonic() - start_time)))
                i += 1

if __name__ == '__main__':
    # test_joint_teleop()
    # test_ee_teleop()
    # test_free_drive()
    # test_ee_rot()
    test_ee_translation()