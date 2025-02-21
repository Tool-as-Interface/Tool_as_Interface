import numpy as np
import open3d as o3d
import cv2
from math import atan2, cos, sin, sqrt, pi
from concurrent.futures import ThreadPoolExecutor
from scipy.spatial.distance import cdist
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors
import concurrent
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from ti.common.trans_utils import inverse_transform, rotation_matrix_zaxis, remove_scaling_from_transformation_matrix, apply_constraints_to_transformation
from ti.common.cv2_util import visualize_hsv_channels
import torch
# import teaserpp_python

def update_intrinsics(intrinsics, original_resolution, new_resolution):
    """
    Adjust camera intrinsics for image resizing with support for batch size.

    Parameters:
    - intrinsics (torch.Tensor): Original intrinsics matrix of shape (n, 3, 3) where n is the batch size.
    - original_resolution (Tuple[int, int]): Original image resolution as (width, height).
    - new_resolution (Tuple[int, int]): New image resolution as (width, height).

    Returns:
    - updated_intrinsics (torch.Tensor): Updated intrinsics matrix with the same shape as the input.
    """
    
    # Unpack original and new resolutions
    wi, hi = original_resolution
    wo, ho = new_resolution

    # Calculate scaling factors
    s_w = wo / wi  # Width scaling factor
    s_h = ho / hi  # Height scaling factor

    # Create scaling matrix (for each intrinsics in the batch)
    scale_matrix = torch.tensor([
        [s_w, 0, s_w],
        [0, s_h, s_h],
        [0, 0, 1]
    ], dtype=intrinsics.dtype, device=intrinsics.device)

    # Apply scaling to each intrinsics matrix in the batch
    updated_intrinsics = intrinsics * scale_matrix

    return updated_intrinsics



def intrinsics_dict2mat(intrinsics_dict):
    # intrinsics_dict: {'width': 640, 'height': 480, 'ppx': 320.0, 'ppy': 240.0, 'fx': 615.0, 'fy': 615.0, 'model': 'brown_conrady', 'coeffs': [0.0, 0.0, 0.0, 0.0, 0.0]}
    # intrinsics: (3, 3)
    intrinsics = np.zeros((3, 3))
    intrinsics[0, 0] = intrinsics_dict['fx']
    intrinsics[1, 1] = intrinsics_dict['fy']
    intrinsics[0, 2] = intrinsics_dict['ppx']
    intrinsics[1, 2] = intrinsics_dict['ppy']
    intrinsics[2, 2] = 1
    return intrinsics

def depth2fgpcd(depth, cam_params, mask):
    # depth: (h, w)
    # fgpcd: (n, 3)
    # mask: (h, w)
    
    # Get camera parameters
    fx, fy, cx, cy = cam_params

    # Ensure valid depth values are considered
    valid_mask = mask & (depth > 0)

    # Calculate point cloud coordinates using only valid masked values
    z_vals = depth[valid_mask]
    y_indices, x_indices = np.nonzero(valid_mask)  # Positions where mask is True
    x_vals = (x_indices - cx) * z_vals / fx
    y_vals = (y_indices - cy) * z_vals / fy

    # Combine x, y, z coordinates into a single array
    fgpcd = np.stack((x_vals, y_vals, z_vals), axis=-1)

    return fgpcd


def convert_3d_to_2d(points_3d, intrinsics):
    """
    Convert 3D points back to 2D directly using camera intrinsics.

    Parameters:
    - points_3d: np.array of shape (N, 3) containing 3D points (X, Y, Z).
    - intrinsics: np.array of shape (3, 3) containing the camera intrinsic matrix.

    Returns:
    - points_2d: np.array of shape (N, 2) containing 2D points (u, v).
    """
    # Extract camera intrinsics
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]

    # Extract X, Y, Z coordinates
    x_vals = points_3d[:, 0]
    y_vals = points_3d[:, 1]
    z_vals = points_3d[:, 2]
    
    if np.any(z_vals == 0):
        raise ValueError("One or more z-values are zero, which will cause a division by zero error.")

    # Compute the 2D coordinates
    u_vals = (x_vals * fx / z_vals) + cx
    v_vals = (y_vals * fy / z_vals) + cy

    # Stack the 2D coordinates into a single array
    points_2d = np.stack((u_vals, v_vals), axis=-1)

    return points_2d

def convert_2d_to_3d(points_2d, depth_map, intrinsics):
    """
    Converts batched 2D points directly to 3D points using the depth map and camera intrinsics.

    Parameters:
    - points_2d: torch.Tensor of shape (B, N, 2) containing the (u, v) coordinates of points (B = batch size).
    - depth_map: torch.Tensor of shape (B, H, W) containing the depth values for each batch.
    - intrinsics: torch.Tensor of shape (B, 3, 3) containing the camera intrinsic matrices for each batch.

    Returns:
    - points_3d: torch.Tensor of shape (B, N, 3) containing 3D points (X, Y, Z).
    """
    B, N, _ = points_2d.shape  # Batch size, number of points
    H, W = depth_map.shape[1:]  # Height and width of depth map

    # Extract intrinsics components
    fx = intrinsics[:, 0, 0]  # Focal length x
    fy = intrinsics[:, 1, 1]  # Focal length y
    cx = intrinsics[:, 0, 2]  # Principal point x
    cy = intrinsics[:, 1, 2]  # Principal point y

    # points_2d = points_2d.long()
    # batch_x_vals, batch_y_vals, batch_z_vals = [], [], []
    # for batch_idx in range(B):
    #     z_vals = depth_map[batch_idx][points_2d[batch_idx,...,1], points_2d[batch_idx,...,0]]
    #     x_vals = (points_2d[batch_idx,...,0].float() - cx[batch_idx]) * z_vals / fx[batch_idx]
    #     y_vals = (points_2d[batch_idx,...,1].float() - cy[batch_idx]) * z_vals / fy[batch_idx]
        
    #     batch_z_vals.append(z_vals)
    #     batch_x_vals.append(x_vals)
    #     batch_y_vals.append(y_vals)
    # batch_z_vals = torch.stack(batch_z_vals, dim=0)
    # batch_x_vals = torch.stack(batch_x_vals, dim=0)
    # batch_y_vals = torch.stack(batch_y_vals, dim=0)

    # points_3d = torch.stack((batch_x_vals, batch_y_vals, batch_z_vals), dim=-1)  # Shape (B, N, 3)
    # xx = points_3d.clone()
    
    # Ensure points_2d are integers to index into depth_map
    points_2d = points_2d.long()

    # Extract depth values using batched indexing
    batch_indices = torch.arange(B, device=points_2d.device).unsqueeze(-1)  # Shape (B, 1)
    z_vals = depth_map[batch_indices, points_2d[..., 1], points_2d[..., 0]]  # Shape (B, N)

    # Compute x and y values for all batches simultaneously
    x_vals = (points_2d[..., 0].float() - cx[:, None]) * z_vals / fx[:, None]  # Shape (B, N)
    y_vals = (points_2d[..., 1].float() - cy[:, None]) * z_vals / fy[:, None]  # Shape (B, N)

    # Stack the x, y, z values to get the 3D points
    points_3d = torch.stack((x_vals, y_vals, z_vals), dim=-1)  # Shape (B, N, 3)
    
    # assert torch.allclose(xx, points_3d, atol=1e-6), "The two tensors are not close enough!"
    
    return points_3d

def np2o3d(pcd, color=None):
    """
    Convert a NumPy array to an Open3D point cloud object. Optionally, assign colors to the points.

    :param pcd: NumPy array with shape (N, 6), where N is the number of points. The first three columns are XYZ coordinates, and the last three columns are normal vectors. If color is provided, the array should have shape (N, 3) for XYZ coordinates.
    :param color: Optional NumPy array with shape (N, 3) representing RGB colors for each point. Values should be in the range [0, 1].
    :return: Open3D point cloud object.
    """
    pcd_o3d = o3d.geometry.PointCloud()
    pcd_o3d.points = o3d.utility.Vector3dVector(pcd[:, :3])

    if pcd.shape[1] == 6:
        pcd_o3d.normals = o3d.utility.Vector3dVector(pcd[:, 3:6])

    if color is not None:
        assert pcd.shape[0] == color.shape[0], "Point and color array must have the same number of points."
        assert color.max() <= 1, "Color values must be in the range [0, 1]."
        assert color.min() >= 0, "Color values must be in the range [0, 1]."
        pcd_o3d.colors = o3d.utility.Vector3dVector(color)

    return pcd_o3d

def voxel_downsample(pcd, voxel_size, pcd_color=None):
    # :param pcd: [N,3] numpy array
    # :param voxel_size: float
    # :return: [M,3] numpy array
    pcd = np2o3d(pcd, pcd_color)
    pcd_down = pcd.voxel_down_sample(voxel_size)
    if pcd_color is not None:
        return np.asarray(pcd_down.points), np.asarray(pcd_down.colors)
    else:
        return np.asarray(pcd_down.points)


def fps_downsample(pcd, N_total, pcd_color=None):
    pcd = np2o3d(pcd, pcd_color)
    pcd = pcd.farthest_point_down_sample(N_total)
    if pcd_color is not None:
        return np.asarray(pcd.points), np.asarray(pcd.colors)
    else:
        return np.asarray(pcd.points)

def color_seg(colors, color_name):
    """color segmentation

    Args:
        colors (np.ndaaray): (N, H, W, 3) RGB images in float32 ranging from 0 to 1
        color_name (string): color name of supported colors
    """
    color_name_to_seg_dict = {
        'yellow': {
            'h_min': 0.0,
            'h_max': 0.1,
            's_min': 0.2,
            's_max': 0.55,
            's_cond': 'and',
            'v_min': 0.8,
            'v_max': 1.01,
            'v_cond': 'or',
        },
        'black': {
            'h_min': 0.0,
            'h_max': 1.0,
            's_min': 0.0,
            's_max': 0.5,
            's_cond': 'and',
            'v_min': 0.0,
            'v_max': 0.45,
            'v_cond': 'and',
        }

    }
    
    colors_hsv = [cv2.cvtColor((color * 255).astype(np.uint8), cv2.COLOR_RGB2HSV) for color in colors]
    colors_hsv = np.stack(colors_hsv, axis=0)
    colors_hsv = colors_hsv / 255.
    seg_dict = color_name_to_seg_dict[color_name]
    h_min = seg_dict['h_min']
    h_max = seg_dict['h_max']
    s_min = seg_dict['s_min']
    s_max = seg_dict['s_max']
    s_cond = seg_dict['s_cond']
    v_min = seg_dict['v_min']
    v_max = seg_dict['v_max']
    v_cond = seg_dict['v_cond']
    mask = (colors_hsv[:, :, :, 0] > h_min) & (colors_hsv[:, :, :, 0] < h_max)
    if s_cond == 'and':
        mask = mask & (colors_hsv[:, :, :, 1] > s_min) & (colors_hsv[:, :, :, 1] < s_max)
    else:
        mask = mask | ((colors_hsv[:, :, :, 1] > s_min) & (colors_hsv[:, :, :, 1] < s_max))
    if v_cond == 'and':
        mask = mask & (colors_hsv[:, :, :, 2] > v_min) & (colors_hsv[:, :, :, 2] < v_max)
    else:
        mask = mask | ((colors_hsv[:, :, :, 2] > v_min) & (colors_hsv[:, :, :, 2] < v_max))
    return mask


def aggr_point_cloud_from_data(colors, depths, 
                               Ks, tf_world2cams, 
                               downsample=True, N_total=None,
                               masks=None, boundaries=None, 
                               out_o3d=True, exclude_colors=[],
                               voxel_size=0.02, remove_plane=False,
                               plane_dist_thresh=0.01):
    """
    aggregate point cloud data from multiple camera inputs.
    """
    N, H, W, _ = colors.shape
    
    pcds = []
    pcd_colors = []

    # Precompute the inverse transformations and color normalization factors
    inverse_transforms = [inverse_transform(tf) for tf in tf_world2cams]
    colors_normalized = colors / 255.0  # Normalize colors just once

    # Loop over each camera
    for i in range(N):
        depth = depths[i]
        mask = (depth > 0) & (depth < 2) if masks is None else masks[i] & (depth > 0)
        K = Ks[i]
        cam_param = [K[0, 0], K[1, 1], K[0, 2], K[1, 2]]  # fx, fy, cx, cy

        color = colors_normalized[i]

        if exclude_colors:
            for exclude_color in exclude_colors:
                mask &= (~color_seg(color[None], exclude_color))[0]
        
        color = color[mask]  # Apply mask to color early to reduce operations

        pcd = depth2fgpcd(depth, cam_param, mask)
        trans_pcd = (inverse_transforms[i] @ np.concatenate([pcd.T, np.ones((1, pcd.shape[0]))], axis=0))[:3, :].T

        if boundaries:
            x_lower, x_upper = boundaries['x_lower'], boundaries['x_upper']
            y_lower, y_upper = boundaries['y_lower'], boundaries['y_upper']
            z_lower, z_upper = boundaries['z_lower'], boundaries['z_upper']

            boundary_mask = ((trans_pcd[:, 0] > x_lower) & (trans_pcd[:, 0] < x_upper) &
                             (trans_pcd[:, 1] > y_lower) & (trans_pcd[:, 1] < y_upper) &
                             (trans_pcd[:, 2] > z_lower) & (trans_pcd[:, 2] < z_upper))
            trans_pcd = trans_pcd[boundary_mask]
            color = color[boundary_mask]

        pcds.append(trans_pcd)
        pcd_colors.append(color)

    # Concatenate all point clouds and colors
    pcds_pos = np.concatenate(pcds, axis=0)
    pcd_colors = np.concatenate(pcd_colors, axis=0)

    if out_o3d:
        pcd_o3d = np2o3d(pcds_pos, pcd_colors)
        if remove_plane:
            plane_model, inliers = pcd_o3d.segment_plane(distance_threshold=plane_dist_thresh,
                                                         ransac_n=3,
                                                         num_iterations=1000)
            pcd_o3d = pcd_o3d.select_by_index(inliers, invert=True)

        if downsample:
            pcd_o3d = pcd_o3d.voxel_down_sample(voxel_size)  # Adjust voxel size as needed
        if N_total is not None:
            pcd_o3d = pcd_o3d.farthest_point_down_sample(N_total)
        return pcd_o3d
    else:
        if remove_plane:
            pcd_o3d = np2o3d(pcds_pos, pcd_colors)
            plane_model, inliers = pcd_o3d.segment_plane(distance_threshold=plane_dist_thresh,
                                                         ransac_n=3,
                                                         num_iterations=1000)
            pcd_o3d = pcd_o3d.select_by_index(inliers, invert=True)
            pcds_pos = np.asarray(pcd_o3d.points)
            pcd_colors = np.asarray(pcd_o3d.colors)

        if downsample:
            pcds_pos, pcd_colors = voxel_downsample(pcds_pos, voxel_size, pcd_colors)  # Adjust voxel size as needed
        if N_total is not None:
            pcds_pos, pcd_colors = fps_downsample(pcds_pos, N_total, pcd_colors)

        return pcds_pos, pcd_colors

    
    

def get_color_mask(hsv_img, color):
    """
    Generates a mask for filtering a specified color in an HSV image.

    Args:
        hsv_img (numpy.ndarray): The HSV image to filter.
        color (str): The color to filter ('red', 'blue', or 'orange').

    Returns:
        numpy.ndarray: The mask for the specified color.
    """
    # Define color bounds in HSV space
    if color == 'red':
        lower_bounds = [np.array([0, 100, 100]), np.array([160, 100, 100])]
        upper_bounds = [np.array([5, 255, 255]), np.array([180, 255, 255])]
    elif color == 'blue':
        lower_bounds = [np.array([80, 100, 60])]
        upper_bounds = [np.array([110, 255, 255])]
    elif color == 'orange':
        lower_bounds = [np.array([0, 100, 100]), np.array([170, 100, 100])]
        upper_bounds = [np.array([15, 255, 255]), np.array([190, 255, 255])]
    else:
        raise ValueError("Unsupported color for masking.")

    # Create masks and merge them
    # mask = np.zeros_like(hsv_img[:, :, 0], dtype=bool)
    # for lower, upper in zip(lower_bounds, upper_bounds):
    #     color_mask = cv2.inRange(hsv_img, lower, upper)
    #     mask = np.logical_or(mask, color_mask.astype(bool))
        
    masks = [cv2.inRange(hsv_img, lower_bound, upper_bound) for lower_bound, upper_bound in zip(lower_bounds, upper_bounds)]
    return cv2.bitwise_or(*masks) if len(masks) > 1 else masks[0]
    # return mask.astype(np.uint8) * 255

def filter_point_cloud_by_color(pcd, color, color_space='rgb'):
    """
    Filters a point cloud to keep only points of a specific color.

    Args:
        pcd (open3d.geometry.PointCloud): The input point cloud.
        color (str): The color to keep ('red', 'blue', or 'orange').

    Returns:
        open3d.geometry.PointCloud: The filtered point cloud.
    """
    # Convert Open3D point cloud to NumPy arrays
    pcd_np = np.asarray(pcd.points)
    colors_np = (np.asarray(pcd.colors) * 255).astype(np.uint8)

    # Convert RGB to HSV and apply color mask
    hsv_colors = cv2.cvtColor(colors_np.reshape((1, -1, 3)), cv2.COLOR_RGB2HSV)
    mask = get_color_mask(hsv_colors, color)[0]

    # Filter the point cloud
    filtered_points = pcd_np[mask.astype(bool)]
    filtered_colors = colors_np[mask.astype(bool)] / 255.0

    # Create a new filtered point cloud
    filtered_pcd = o3d.geometry.PointCloud()
    filtered_pcd.points = o3d.utility.Vector3dVector(filtered_points)
    filtered_pcd.colors = o3d.utility.Vector3dVector(filtered_colors)

    return filtered_pcd



    
def remove_plane_from_mesh(mesh, plane_z_value, tolerance=0.01):
    # Get triangle vertices
    triangles = np.asarray(mesh.triangles)
    vertices = np.asarray(mesh.vertices)

    # Determine which triangles to keep (those not in the specified plane)
    triangles_to_keep = []
    for triangle in triangles:
        vertex_coords = vertices[triangle]
        if not np.all(np.abs(vertex_coords[:, 2] - plane_z_value) < tolerance):
            triangles_to_keep.append(triangle)

    # Create a new mesh from the remaining triangles
    new_mesh = o3d.geometry.TriangleMesh()
    new_mesh.vertices = o3d.utility.Vector3dVector(vertices)
    new_mesh.triangles = o3d.utility.Vector3iVector(np.array(triangles_to_keep))
    new_mesh.compute_vertex_normals()
    return new_mesh





def flip_all_inward_normals(pcd, center):
    # Get vertices and normals from the mesh
    points = np.asarray(pcd.points)
    normals = np.asarray(pcd.normals)

    # Compute vectors from vertices to the center and normalize them
    norm_ref = points - center

    # Dot product between normals and vectors to center
    dot_products = np.einsum('ij,ij->i', norm_ref, normals)

    # Identify normals that need to be flipped
    flip_mask = dot_products < 0

    # Flip the normals
    normals[flip_mask] = -normals[flip_mask]

    pcd.normals = o3d.utility.Vector3dVector(normals)
    return pcd

def segment_and_filp(pcd, visualize=False):
    labels = np.array(pcd.cluster_dbscan(eps=0.01, min_points=100))
    part_one = pcd.select_by_index(np.where(labels == 0)[0])
    part_two = pcd.select_by_index(np.where(labels > 0)[0])

    for part in [part_one, part_two]:
        if np.asarray(part.points).shape[0] > 0:
            # invalidate existing normals
            part.normals = o3d.utility.Vector3dVector(np.zeros((1, 3)))  
            part.estimate_normals()
            part.orient_normals_consistent_tangent_plane(100)
            
            # get an accurate center
            # center = part.get_center()
            hull, _ = part.compute_convex_hull()
            center = hull.get_center()
            
            part = flip_all_inward_normals(part, center)
            
            if visualize:
                visualize_o3d([part, hull], title='part_normals', show_normal=True)
    
    return part_one + part_two




def visualize_o3d(geometry_list, title='O3D', view_point=None, point_size=5, pcd_color=[0, 0, 0],
    mesh_color=[0.5, 0.5, 0.5], show_normal=False, show_frame=True, path=''):
    vis = o3d.visualization.Visualizer()
    vis.create_window(title)
    types = []

    for geometry in geometry_list:
        type = geometry.get_geometry_type()
        # Point Cloud
        # if type == o3d.geometry.Geometry.Type.PointCloud:
        #     geometry.paint_uniform_color(pcd_color)
        # Triangle Mesh
        if type == o3d.geometry.Geometry.Type.TriangleMesh:
            geometry.paint_uniform_color(mesh_color)
        types.append(type)

        vis.add_geometry(geometry)
        vis.update_geometry(geometry)
    
    vis.get_render_option().background_color = np.array([0., 0., 0.])
    if show_frame:
        mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.15)
        vis.add_geometry(mesh)
        vis.update_geometry(mesh)

    if o3d.geometry.Geometry.Type.PointCloud in types:
        vis.get_render_option().point_size = point_size
        vis.get_render_option().point_show_normal = show_normal
    if o3d.geometry.Geometry.Type.TriangleMesh in types:
        vis.get_render_option().mesh_show_back_face = True
        vis.get_render_option().mesh_show_wireframe = True

    vis.poll_events()
    vis.update_renderer()

    if view_point is None:
        vis.get_view_control().set_front(np.array([0.305, -0.463, 0.832]))
        vis.get_view_control().set_lookat(np.array([0.4, -0.1, 0.0]))
        vis.get_view_control().set_up(np.array([-0.560, 0.620, 0.550]))
        vis.get_view_control().set_zoom(0.2)
    else:
        vis.get_view_control().set_front(view_point['front'])
        vis.get_view_control().set_lookat(view_point['lookat'])
        vis.get_view_control().set_up(view_point['up'])
        vis.get_view_control().set_zoom(view_point['zoom'])

    # cd = os.path.dirname(os.path.realpath(sys.argv[0]))
    # path = os.path.join(cd, '..', 'figures', 'images', f'{title}_{datetime.now().strftime("%b-%d-%H:%M:%S")}.png')

    if len(path) > 0:
        vis.capture_screen_image(path, True)
        vis.destroy_window()
    else:
        vis.run()
        
        







    
    
def filter_pcd_with_mask(pcd, mask):
    """
    Filters a point cloud using a 2D mask.
    Assumes mask and point cloud are aligned and have the same FOV.
    """
    points = np.asarray(pcd)

    # Assuming the point cloud is already cropped to match the mask's FOV
    mask_indices = np.where(mask.flatten() > 0)[0]
    filtered_points = points[mask_indices]

    # Create a new point cloud object
    filtered_pcd = o3d.geometry.PointCloud()
    filtered_pcd.points = o3d.utility.Vector3dVector(filtered_points)

    return filtered_pcd







def visualize_pcd(points):
    print("Processing input for visualization...")
    
    if isinstance(points, tuple):
        print("Input type: Tuple (xyz, rgb)")
        xyz, rgb = points[0], points[1]
        if np.max(rgb) > 1.0:
            rgb = rgb / 255.0
        print(f"XYZ shape: {xyz.shape}, RGB shape: {rgb.shape}")
    elif isinstance(points, np.ndarray):
        print("Input type: NumPy array")
        if points.shape[0] == 3 or points.shape[0] == 6:
            xyz = points[:3, :].T
            rgb = points[3:6, :].T  if points.shape[0] == 6 else None
        elif points.shape[1] == 3 or points.shape[1] == 6:
            xyz = points[:, :3]
            rgb = points[:, 3:6] if points.shape[1] == 6 else None
        else:
            raise ValueError("Array shape not recognized. Expected (N, 3), (3, N), (N, 6), or (6, N).")
        print(f"XYZ shape: {xyz.shape}, RGB shape: {'N/A' if rgb is None else rgb.shape}")
    elif isinstance(points, o3d.geometry.PointCloud):
        print("Input type: Open3D PointCloud")
        pcd = points
        o3d.visualization.draw_geometries([pcd])
        return
    else:
        raise ValueError("Invalid input type. Must be a NumPy array, tuple, or Open3D point cloud object.")
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    if rgb is not None:
        pcd.colors = o3d.utility.Vector3dVector(rgb)
    
    print(f"Visualizing point cloud with {len(pcd.points)} points.")
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
    o3d.visualization.draw_geometries([pcd, coordinate_frame])
