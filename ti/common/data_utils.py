import einops
from typing import Optional
import time

import numpy as np
import h5py
import cv2
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F

from PIL import Image
import matplotlib.pyplot as plt

from tqdm import tqdm
from ti.common.pcd_utils import (aggr_point_cloud_from_data, remove_plane_from_mesh,
    visualize_pcd)
from ti.common.trans_utils import transform_to_world, transform_from_world, interpolate_poses
from ti.model.common.rotation_transformer import RotationTransformer
import cProfile
import pstats
import io
import os
import sys
from contextlib import contextmanager
import dask.array as da
from dask import delayed
import dask.bag as db
from scipy.spatial.transform import Rotation as R, Slerp

from ti.common.cv2_util import ImgNorm_inv, load_images_mast3r

from mast3r.model import AsymmetricMASt3R
from mast3r.fast_nn import fast_reciprocal_NNs
from dust3r.inference import inference
from dust3r.utils.image import load_images
import starster

from sklearn.neighbors import NearestNeighbors
from scipy.ndimage import gaussian_filter


@contextmanager
def suppress_output():
    """
    Suppresses stdout and stderr, including tqdm outputs, within the context.
    """
    dummy = open(os.devnull, 'w')
    original_stdout, original_stderr = sys.stdout, sys.stderr
    try:
        sys.stdout, sys.stderr = dummy, dummy
        yield  # Run the code within this block with stdout/stderr suppressed
    finally:
        sys.stdout, sys.stderr = original_stdout, original_stderr
        dummy.close()

def save_dict_to_hdf5(dic, config_dict, filename, attr_dict=None):
    """
    ....
    """
    with h5py.File(filename, 'w') as h5file:
        if attr_dict is not None:
            for key, item in attr_dict.items():
                h5file.attrs[key] = item
        recursively_save_dict_contents_to_group(h5file, '/', dic, config_dict)

def recursively_save_dict_contents_to_group(h5file, path, dic, config_dict):
    """
    ....
    """
    for key, item in dic.items():
        if isinstance(item, np.ndarray):
            # h5file[path + key] = item
            if key not in config_dict:
                config_dict[key] = {}
            dset = h5file.create_dataset(path + key, shape=item.shape, **config_dict[key])
            dset[...] = item
        elif isinstance(item, dict):
            if key not in config_dict:
                config_dict[key] = {}
            recursively_save_dict_contents_to_group(h5file, path + key + '/', item, config_dict[key])
        else:
            raise ValueError('Cannot save %s type'%type(item))

def load_dict_from_hdf5(filename):
    """
    ....
    """
    h5file = h5py.File(filename, 'r')
    return recursively_load_dict_contents_from_group(h5file, '/'), h5file

def recursively_load_dict_contents_from_group(h5file, path):
    """
    ....
    """
    ans = {}
    for key, item in h5file[path].items():
        if isinstance(item, h5py._hl.dataset.Dataset):
            # ans[key] = np.array(item)
            ans[key] = item
        elif isinstance(item, h5py._hl.group.Group):
            ans[key] = recursively_load_dict_contents_from_group(h5file, path + key + '/')
    return ans

def modify_hdf5_from_dict(filename, dic):
    """
    Modify hdf5 file from a dictionary
    """
    with h5py.File(filename, 'r+') as h5file:
        recursively_modify_hdf5_from_dict(h5file, '/', dic)

def recursively_modify_hdf5_from_dict(h5file, path, dic):
    """
    Modify hdf5 file from a dictionary recursively
    """
    for key, item in dic.items():
        if isinstance(item, np.ndarray) and key in h5file[path]:
            h5file[path + key][...] = item
        elif isinstance(item, dict):
            recursively_modify_hdf5_from_dict(h5file, path + key + '/', item)
        else:
            raise ValueError('Cannot modify %s type'%type(item))

def _convert_actions(raw_actions, rotation_transformer, action_key):        
    act_num, act_dim = raw_actions.shape
    if act_dim == 2:
        return raw_actions
    is_bimanual = (act_dim == 14) or (act_dim == 12) 
    if is_bimanual:
        raw_actions = raw_actions.reshape(act_num * 2, act_dim // 2)
    
    if action_key == 'cartesian_action' or action_key == 'robot_eef_pose':
        pos = raw_actions[...,:3]
        rot = raw_actions[...,3:6]
        gripper = raw_actions[...,6:]
        rot = rotation_transformer.forward(rot)
        raw_actions = np.concatenate([
            pos, rot, gripper
        ], axis=-1).astype(np.float32)
    elif action_key == 'joint_action':
        pass
    else:
        raise RuntimeError('unsupported action_key')
    if is_bimanual:
        proc_act_dim = raw_actions.shape[-1]
        raw_actions = raw_actions.reshape(act_num, proc_act_dim * 2)
    actions = raw_actions
    return actions

def _homo_to_9d_action(raw_actions, rotation_transformer=None):
    if rotation_transformer is None:
        rotation_transformer = RotationTransformer(
            from_rep='matrix', to_rep='rotation_6d')

    trans = raw_actions[:, :3, 3]  # Extract translation vectors
    rot_mat = raw_actions[:, :3, :3]    # Extract rotation matrices
    rot_vec = rotation_transformer.forward(rot_mat)  # Convert rotation matrices to rotation vectors
    pose_vec = np.concatenate([
        trans, rot_vec
    ], axis=-1).astype(np.float32)
    return pose_vec

def _9d_to_homo_action(raw_actions):
    
    trans = raw_actions[:, :3]  
    rot_vec = raw_actions[:, 3:]  
    rot_mat = convert_rotation_6d_to_matrix(rot_vec)
    
    batch_size = raw_actions.shape[0]
    homo_action = np.zeros((batch_size, 4, 4))  # Initialize homogeneous matrix

    homo_action[:, :3, :3] = rot_mat  # Set rotation part
    homo_action[:, :3, 3] = trans  # Set translation part
    homo_action[:, 3, 3] = 1.0  # Set bottom-right part to 1 (for homogeneous coordinates)

    return homo_action

def _6d_axis_angle_to_homo(raw_actions):
    trans = raw_actions[:, :3]  
    axis_angle = raw_actions[:, 3:]  
    rot_mat = convert_axis_angle_to_matrix(axis_angle)
    
    batch_size = raw_actions.shape[0]
    homo_action = np.zeros((batch_size, 4, 4))  # Initialize homogeneous matrix

    homo_action[:, :3, :3] = rot_mat  # Set rotation part
    homo_action[:, :3, 3] = trans  # Set translation part
    homo_action[:, 3, 3] = 1.0  # Set bottom-right part to 1 (for homogeneous coordinates)

    return homo_action

def _homo_to_6d_axis_angle(homo_action):
    trans = homo_action[:, :3, 3]  
    rot_mat = homo_action[:, :3, :3]  
    axis_angle = convert_matrix_to_axis_angle(rot_mat)
    
    batch_size = homo_action.shape[0]
    raw_actions = np.zeros((batch_size, 6))  # Initialize raw action

    raw_actions[:, :3] = trans  # Set translation part
    raw_actions[:, 3:] = axis_angle  # Set rotation part

    return raw_actions

def eef_action_to_6d(raw_actions):
    rotation_transformer = RotationTransformer(
        from_rep='rotation_6d', to_rep='axis_angle')
    act_num, act_dim = raw_actions.shape
    is_bimanual = (act_dim == 18) or (act_dim == 20)
    if is_bimanual:
        raw_actions = raw_actions.reshape(act_num * 2, act_dim // 2)
    
    pos = raw_actions[...,:3]
    rot = raw_actions[...,3:9]
    gripper = raw_actions[...,9:]
    rot = rotation_transformer.forward(rot)
    raw_actions = np.concatenate([
        pos, rot, gripper
    ], axis=-1).astype(np.float32)
    if is_bimanual:
        proc_act_dim = raw_actions.shape[-1]
        raw_actions = raw_actions.reshape(act_num, proc_act_dim * 2)
    actions = raw_actions
    # vis_post_actions(actions[:,10:])
    return actions

def convert_rotation_6d_to_matrix(rot_6d):
    rotation_transformer = RotationTransformer(
        from_rep='rotation_6d', to_rep='matrix')
    rot_mat = rotation_transformer.forward(rot_6d)
    return rot_mat

def convert_matrix_to_rotation_6d(rot_mat):
    rotation_transformer = RotationTransformer(
        from_rep='matrix', to_rep='rotation_6d')
    rot_6d = rotation_transformer.forward(rot_mat)
    return rot_6d

def convert_axis_angle_to_rotation_6d(axis_angle):
    rotation_transformer = RotationTransformer(
        from_rep='axis_angle', to_rep='rotation_6d')
    rot_6d = rotation_transformer.forward(axis_angle)
    return rot_6d

def convert_rotation_6d_to_axis_angle(rot_6d):
    rotation_transformer = RotationTransformer(
        from_rep='rotation_6d', to_rep='axis_angle')
    axis_angle = rotation_transformer.forward(rot_6d)
    return axis_angle

def convert_axis_angle_to_matrix(axis_angle):
    rotation_transformer = RotationTransformer(
        from_rep='axis_angle', to_rep='matrix')
    rot_mat = rotation_transformer.forward(axis_angle)
    return rot_mat

def convert_matrix_to_axis_angle(matrix):
    rotation_transformer = RotationTransformer(
        from_rep='matrix', to_rep='axis_angle')
    axis_angle = rotation_transformer.forward(matrix)
    return axis_angle

def policy_action_to_env_action(raw_actions, cur_eef_pose_6d=None, init_tool_pose_9d=None, 
                                tool_in_eef=None,action_mode='eef', debug=False):
    '''
    Convert policy actions to environment actions based on the given action mode.
    - cur_eef_pose_6d: Current end-effector pose in 6D (position + axis-angle).
    - init_tool_pose_9d: Initial tool pose in 9D (position + 6D rotation).
    - action_mode: 'eef' for end-effector control, 'joint' for joint control.
    - tool_in_eef: Transformation of the tool in the end-effector frame.
    - robot_base_pose_in_world: Transformation of the robot base in the world frame.
    - cam_extrinsic: Camera extrinsics in the world frame.
    - num_bots: Number of robots (defaults to 1).
    - debug: Whether to enable debug logging and assertions.
    '''
    if action_mode == 'eef' and raw_actions.shape[1] == 3:
        assert cur_eef_pose_6d is not None
        # raw_actions is of shape (T, 3). cur_eef_pose is of shape (6)
        # convert cur_eef_pose to (T, 6) and then concatenate with raw_actions
        raw_actions = np.concatenate([raw_actions, cur_eef_pose_6d[None,3:].repeat(raw_actions.shape[0], axis=0)], axis=1)
        return raw_actions
    elif action_mode == 'eef':
        if init_tool_pose_9d is not None:            
            pred_eef_pose = []
            
            tool_in_base = _9d_to_homo_action(raw_actions)
            pred_eef_pose = tool_in_base@np.linalg.inv(tool_in_eef) 
                
            pred_eef_pose_6d = _homo_to_6d_axis_angle(pred_eef_pose)

            if debug:
                tool_in_base_ls = np.array(tool_in_base_ls)
            return pred_eef_pose_6d 
        else:
            return eef_action_to_6d(raw_actions)
        
    elif action_mode == 'joint':
        return raw_actions
    
def point_cloud_proc(shape_meta, color_seq, depth_seq, extri_seq, intri_seq,
                  robot_base_pose_in_world_seq = None, qpos_seq=None, expected_labels=None, 
                  exclude_threshold=0.01, exclude_colors=[], teleop_robot=None, tool_names=[None], profile=False):
    # shape_meta: (dict) shape meta data for d3fields
    # color_seq: (np.ndarray) (T, V, H, W, C)
    # depth_seq: (np.ndarray) (T, V, H, W)
    # extri_seq: (np.ndarray) (T, V, 4, 4)
    # intri_seq: (np.ndarray) (T, V, 3, 3)
    # robot_name: (str) name of robot
    # meshes: (list) list of meshes
    # offsets: (list) list of offsets
    # finger_poses: (dict) dict of finger poses, mapping from finger name to (T, 6)
    # expected_labels: (list) list of expected labels
    if profile:
        # Initialize profiler
        pr = cProfile.Profile()
        pr.enable()


    boundaries = shape_meta['info']['boundaries']
    if boundaries == 'none':
        boundaries = None
    N_total = shape_meta['shape'][1]
    max_pts_num = shape_meta['shape'][1]
    
    resize_ratio = shape_meta['info']['resize_ratio']
    reference_frame = shape_meta['info']['reference_frame'] if 'reference_frame' in shape_meta['info'] else 'world'
    
    num_bots = 2 if robot_base_pose_in_world_seq.shape[2] == 8 else 1
    robot_base_pose_in_world_seq = robot_base_pose_in_world_seq.reshape(robot_base_pose_in_world_seq.shape[0], 4, num_bots, 4)
    robot_base_pose_in_world_seq = robot_base_pose_in_world_seq.transpose(0, 2, 1, 3)
    
    H, W = color_seq.shape[2:4]
    resize_H = int(H * resize_ratio)
    resize_W = int(W * resize_ratio)
    
    new_color_seq = np.zeros((color_seq.shape[0], color_seq.shape[1], resize_H, resize_W, color_seq.shape[-1]), dtype=np.uint8)
    new_depth_seq = np.zeros((depth_seq.shape[0], depth_seq.shape[1], resize_H, resize_W), dtype=np.float32)
    new_intri_seq = np.zeros((intri_seq.shape[0], intri_seq.shape[1], 3, 3), dtype=np.float32)
    for t in range(color_seq.shape[0]):
        for v in range(color_seq.shape[1]):
            new_color_seq[t,v] = cv2.resize(color_seq[t,v], (resize_W, resize_H), interpolation=cv2.INTER_NEAREST)
            new_depth_seq[t,v] = cv2.resize(depth_seq[t,v], (resize_W, resize_H), interpolation=cv2.INTER_NEAREST)
            new_intri_seq[t,v] = intri_seq[t,v] * resize_ratio
            new_intri_seq[t,v,2,2] = 1.
    color_seq = new_color_seq
    depth_seq = new_depth_seq
    intri_seq = new_intri_seq
    T, V, H, W, C = color_seq.shape
    # assert H == 240 and W == 320 and C == 3
    aggr_src_pts_ls = []
    
    use_robot_pcd = shape_meta['info']['rob_pcd']
    use_tool_pcd = shape_meta['info']['eef_pcd']
    N_per_link = shape_meta['info']['N_per_link']
    N_eef = shape_meta['info']['N_eef']
    N_joints = shape_meta['info']['N_joints']
    voxel_size = shape_meta['info'].get('voxel_size', 0.02)
    remove_plane = shape_meta['info'].get('remove_plane', False)
    plane_dist_thresh = shape_meta['info'].get('plane_dist_thresh', 0.01)

    # Assigning a bright magenta to the robot point cloud (robot_pcd)
    # RGB for a bright magenta: [1.0, 0, 1.0]
    robot_color = np.array([1.0, 0, 1.0])  # Bright Magenta

    # Assigning a fluorescent green to the end-effector point cloud (ee_pcd)
    # RGB for a fluorescent green: [0.5, 1.0, 0]
    ee_color = shape_meta['info'].get('ee_color', [0.5, 1.0, 0])  # Fluorescent Green
    # ee_color = np.array([0.5, 1.0, 0])  # Fluorescent Green

    for t in range(T):
        curr_qpos = qpos_seq[t]
        qpos_dim = curr_qpos.shape[0] // num_bots
        
        robot_pcd_ls = []
        ee_pcd_ls = []
        

        for rob_i in range(num_bots):
            # transform robot pcd to world frame    
            robot_base_pose_in_world = robot_base_pose_in_world_seq[t, rob_i] if robot_base_pose_in_world_seq is not None else None

            tool_name = tool_names[rob_i]
            # compute robot pcd
            if use_robot_pcd:
                robot_pcd_pts = teleop_robot.get_robot_pcd(curr_qpos[qpos_dim*rob_i:qpos_dim*(rob_i+1)], N_joints=N_joints, N_per_link=N_per_link, out_o3d=False)
                robot_pcd_pts = (robot_base_pose_in_world @ np.concatenate([robot_pcd_pts, np.ones((robot_pcd_pts.shape[0], 1))], axis=-1).T).T[:, :3]

                robot_pcd_color = np.tile(robot_color, (robot_pcd_pts.shape[0], 1))
                robot_pcd_pts = np.concatenate([robot_pcd_pts, robot_pcd_color], axis=-1)
                robot_pcd_ls.append(robot_pcd_pts)

            if use_tool_pcd:
                ee_pcd_pts = teleop_robot.get_tool_pcd(curr_qpos[qpos_dim*rob_i:qpos_dim*(rob_i+1)], tool_name, N_per_inst=N_eef)    
                ee_pcd_pts = (robot_base_pose_in_world @ np.concatenate([ee_pcd_pts, np.ones((ee_pcd_pts.shape[0], 1))], axis=-1).T).T[:, :3]
            
                ee_pcd_color = np.tile(ee_color, (ee_pcd_pts.shape[0], 1))
                ee_pcd_pts = np.concatenate([ee_pcd_pts, ee_pcd_color], axis=-1)
                ee_pcd_ls.append(ee_pcd_pts)
                
        robot_pcd = np.concatenate(robot_pcd_ls , axis=0) if use_robot_pcd else None
        ee_pcd = np.concatenate(ee_pcd_ls, axis=0) if use_tool_pcd  else None

        aggr_src_pts = aggr_point_cloud_from_data(color_seq[t], depth_seq[t], intri_seq[t], extri_seq[t], 
                                                  downsample=True, N_total=N_total,
                                                  boundaries=boundaries, 
                                                  out_o3d=False, 
                                                  exclude_colors=exclude_colors,
                                                  voxel_size=voxel_size,
                                                  remove_plane=remove_plane,
                                                  plane_dist_thresh=plane_dist_thresh,)
        aggr_src_pts = np.concatenate(aggr_src_pts, axis=-1) # (N_total, 6), 6: (x, y, z, r, g, b)
        aggr_src_pts = np.concatenate([aggr_src_pts, robot_pcd], axis=0) if use_robot_pcd else aggr_src_pts
        aggr_src_pts = np.concatenate([aggr_src_pts, ee_pcd], axis=0) if use_tool_pcd else aggr_src_pts
        
        # transform to reference frame
        if reference_frame == 'world':
            pass
        elif reference_frame == 'robot':
            transformed_xyz = (np.linalg.inv(robot_base_pose_in_world_seq[t, 0]) @ np.concatenate([aggr_src_pts[:, :3], np.ones((aggr_src_pts.shape[0], 1))], axis=-1).T).T[:, :3]
            aggr_src_pts = np.concatenate([transformed_xyz, aggr_src_pts[:, 3:]], axis=1)
            
        aggr_src_pts_ls.append(aggr_src_pts.astype(np.float32))
            
        if profile:
            pr.disable()
            s = io.StringIO()
            sortby = 'cumulative' # or 'time'
            ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
            ps.print_stats(10)
            print(s.getvalue())
 
    
    return aggr_src_pts_ls


def generate_novel_view(gs, aug_img_num, new_w2c, device, mast3r_size):
    
    random_indices = torch.randint(low=0, high=2, size=(aug_img_num,), device=device)
    selected_intrinsics = (gs.scene.intrinsics())[random_indices]

    render_view_arg = {'w2c': new_w2c, 
                       'intrinsics': selected_intrinsics,
                       'width': mast3r_size, 
                       'height': mast3r_size}
    
    aug_img, alpha, info = gs.render_views(**render_view_arg)

    return aug_img


def interpolate_homogeneous_matrix(w2c1, w2c2, alpha):
    """
    Interpolates two homogeneous transformation matrices w2c1 and w2c2.

    Args:
        w2c1: First homogeneous matrix (shape [4, 4]).
        w2c2: Second homogeneous matrix (shape [4, 4]).
        alpha: Interpolation factor (0.0 to 1.0).

    Returns:
        Interpolated homogeneous matrix (shape [4, 4]).
    """
    # Extract rotation and translation
    R1, t1 = w2c1[:3, :3], w2c1[:3, 3]
    R2, t2 = w2c2[:3, :3], w2c2[:3, 3]

    # Convert rotation matrices to quaternions
    q1 = R.from_matrix(R1.cpu().numpy()).as_quat()
    q2 = R.from_matrix(R2.cpu().numpy()).as_quat()

    # Interpolate rotations using Slerp
    key_times = [0, 1]  # Start and end times for Slerp
    key_rotations = R.from_quat([q1, q2])
    slerp = Slerp(key_times, key_rotations)
    interpolated_rotation = slerp([alpha])[0]  # Interpolate for given alpha
    R_interpolated = torch.tensor(interpolated_rotation.as_matrix(), device=w2c1.device)

    # Interpolate translation
    t_interpolated = (1 - alpha) * t1 + alpha * t2

    # Construct the interpolated homogeneous matrix
    interpolated = torch.eye(4, device=w2c1.device)
    interpolated[:3, :3] = R_interpolated
    interpolated[:3, 3] = t_interpolated

    return interpolated

def generate_intermediate_matrices(
    w2c1, w2c2, num_matrices=6, random_matrices=False
):
    """
    Generates w2c matrices with a mix of those close to w2c1, w2c2, and optionally random matrices.

    Args:
        w2c1: First w2c matrix (shape [4, 4]).
        w2c2: Second w2c matrix (shape [4, 4]).
        num_matrices: Total number of matrices to generate (should be even if no random_matrices).
        random_matrices: If True, generates some matrices with fully random interpolation.

    Returns:
        Tensor of generated w2c matrices (shape [num_matrices, 4, 4]).
    """
    if not random_matrices and num_matrices % 2 != 0:
        raise ValueError("num_matrices must be even when random_matrices is False to split evenly between w2c1 and w2c2")

    generated_matrices = []

    # Determine split for w2c1 and w2c2 biased matrices
    if random_matrices:
        num_close = num_matrices // 2  # Half for biased matrices
        num_random = num_matrices - num_close  # Remaining for random matrices
    else:
        num_close = num_matrices // 2
        num_random = 0

    # Generate alphas biased towards w2c1
    alphas_w2c1 = torch.distributions.Beta(5, 2).sample([num_close]).tolist()

    # Generate alphas biased towards w2c2
    alphas_w2c2 = torch.distributions.Beta(2, 5).sample([num_close]).tolist()

    # Generate matrices close to w2c1
    for alpha in alphas_w2c1:
        interpolated = interpolate_homogeneous_matrix(w2c1, w2c2, alpha)
        generated_matrices.append(interpolated)

    # Generate matrices close to w2c2
    for alpha in alphas_w2c2:
        interpolated = interpolate_homogeneous_matrix(w2c1, w2c2, alpha)
        generated_matrices.append(interpolated)

    # Generate fully random matrices (random interpolation)
    for _ in range(num_random):
        alpha = torch.rand(1).item()  # Uniform random interpolation
        interpolated = interpolate_homogeneous_matrix(w2c1, w2c2, alpha)
        generated_matrices.append(interpolated)

    # Stack all generated matrices into a tensor
    return torch.stack(generated_matrices)

def setup_scene_and_gs(color_seq, mast3r_size, starster_mast3r_model, dummy_file_list, aug_img_num, device):

    src_img_list = (load_images_mast3r)(color_seq[0], size=mast3r_size, verbose=False, device='cpu')
    imgs = ([img_info['img'][0] for img_info in src_img_list])
    
    # Suppress output during scene reconstruction and make it a delayed operation
    with suppress_output():
        scene = (starster.reconstruct_scene)(starster_mast3r_model, imgs, dummy_file_list, device=device)

    gs = (starster.GSTrainer)(scene, device=device)
    
    # roll_range, pitch_range, yaw_range, distance_range = extract_rpy_distance_range(torch.inverse(gs.scene.w2c()))
    new_matrices = generate_intermediate_matrices(gs.scene.w2c()[0], gs.scene.w2c()[1], num_matrices=aug_img_num, random_matrices=False)

    # Delay optimization and rendering as well
    gs.run_optimization(500, enable_pruning=True, verbose=False) # 1000
    gs.run_optimization(100, enable_pruning=False, verbose=False) # 5000
    
    src_img_list[0]['img'] = (src_img_list[0]['img'].to)(device)


    return gs, src_img_list, new_matrices


def feature_proc(color_seq, 
                 aug_img_num, shape_meta, 
                 aug_mult, updated_episode_length, 
                 mast3r_size, starster_mast3r_model, 
                 device, 
                 novel_view):
    dummy_file_list = [f"dummy_{i}.png" for i in range(2)]


    c,h,w = tuple(shape_meta['obs']['camera_features']['shape'])
    episode_views_features = np.empty((aug_mult, updated_episode_length, h, w, c), dtype=np.float32)
    

    for batch_start in range(0, len(color_seq)):
        batch_end = min(batch_start + 1, len(color_seq))
        
        color_seq_batch = color_seq[batch_start:batch_end]
        if novel_view:
            gs, src_img_list, new_w2c = setup_scene_and_gs(color_seq_batch, mast3r_size, starster_mast3r_model, dummy_file_list, aug_img_num-1, device) 
        
        novel_view_generated = False
        use_other_view = False
        while not novel_view_generated:
            try:
                if novel_view: 
                    if use_other_view == False:
                        aug_img = generate_novel_view(gs, aug_img_num-1, new_w2c, device, mast3r_size)
                        aug_img_info = (load_images_mast3r)(aug_img.permute(0, 3, 1, 2), size=mast3r_size, verbose=False)
                        pair_img_list = [src_img_list[0]] + aug_img_info + [src_img_list[1]]
                    else:
                        pair_img_list = [src_img_list[0]] + [src_img_list[1]] * aug_img_num
                else:
                    src_img_list = load_images_mast3r(color_seq_batch[0], size=mast3r_size, verbose=False)
                    if len(src_img_list) == 1:
                        pair_img_list = [src_img_list[0]] * (aug_img_num+1)
                    else:
                        pair_img_list = [src_img_list[0]] + [src_img_list[1]] * aug_img_num
                    
                episode_views_features[:, batch_start:batch_end] = np.concatenate([
                    F.interpolate(ImgNorm_inv(img_info['img']).detach(), size=(h,w), mode='area').permute(0, 2, 3, 1).cpu().numpy() 
                    for img_info in pair_img_list
                ])[:,None,...]

                novel_view_generated = True
            except Exception as e:
                use_other_view = True
                print(f"Error in generate_correspondence: {e}. Retrying with a new novel view...")
                continue   
    return np.concatenate(episode_views_features)





