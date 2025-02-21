import matplotlib.pyplot as plt
import torch
import math


def matrix_to_rpy_distance(matrices, output_in_degrees=True):
    """
    Convert a batch of transformation matrices to roll, pitch, yaw (RPY) angles and distances.
    
    Args:
        matrices (torch.Tensor): A tensor of shape [N, 4, 4], where N is the batch size and 
                                 each matrix is a 4x4 transformation matrix.
        output_in_degrees (bool): If True, output angles will be in degrees. If False, output 
                                  will be in radians. Default is True.
    
    Returns:
        roll (torch.Tensor): Tensor of shape [N], containing roll angles for each matrix.
        pitch (torch.Tensor): Tensor of shape [N], containing pitch angles for each matrix.
        yaw (torch.Tensor): Tensor of shape [N], containing yaw angles for each matrix.
        distances (torch.Tensor): Tensor of shape [N], containing distances for each matrix.
    """
    # Extract rotation matrices and translation vectors from the input matrices
    rotation_matrices = matrices[:, :3, :3]  # Shape [N, 3, 3]
    translation_vectors = matrices[:, :3, 3]  # Shape [N, 3]

    # Calculate Euclidean distance for each translation vector
    distances = torch.norm(translation_vectors, dim=1)  # Shape [N]

    # Extract roll, pitch, and yaw using vectorized operations
    pitch = -torch.asin(rotation_matrices[:, 2, 0])  # Shape [N]
    roll = torch.atan2(rotation_matrices[:, 2, 1], rotation_matrices[:, 2, 2])  # Shape [N]
    yaw = torch.atan2(rotation_matrices[:, 1, 0], rotation_matrices[:, 0, 0])  # Shape [N]

    # Convert angles to degrees if required
    if output_in_degrees:
        roll = torch.rad2deg(roll)
        pitch = torch.rad2deg(pitch)
        yaw = torch.rad2deg(yaw)

    return roll, pitch, yaw, distances

def rpy_distance_to_matrix(roll, pitch, yaw, distance, world_center=torch.tensor([0.0, 0.0, 0.0])):
    # Convert from degrees to radians
    roll = math.radians(roll)
    pitch = math.radians(pitch)
    yaw = math.radians(yaw)

    # Rotation matrices for roll, pitch, and yaw
    R_roll = torch.tensor([[1, 0, 0],
                           [0, math.cos(roll), -math.sin(roll)],
                           [0, math.sin(roll), math.cos(roll)]], dtype=torch.float32)

    R_pitch = torch.tensor([[math.cos(pitch), 0, math.sin(pitch)],
                            [0, 1, 0],
                            [-math.sin(pitch), 0, math.cos(pitch)]], dtype=torch.float32)

    R_yaw = torch.tensor([[math.cos(yaw), -math.sin(yaw), 0],
                          [math.sin(yaw), math.cos(yaw), 0],
                          [0, 0, 1]], dtype=torch.float32)

    # Combine the rotations
    rotation_matrix = R_yaw @ R_pitch @ R_roll

    # Camera's position relative to the world center
    forward_vector = torch.tensor([0.0, 0.0, -1.0], dtype=torch.float32)
    camera_position = distance * (rotation_matrix @ forward_vector)
    camera_position_world = camera_position + world_center

    # Create the transformation matrix
    matrix = torch.eye(4, dtype=torch.float32)
    matrix[:3, :3] = rotation_matrix
    matrix[:3, 3] = camera_position_world

    return matrix


def interpolate_camera_matrices(w2c_start, w2c_end, num_steps):
    """
    Interpolates between two camera-to-world matrices.

    Args:
        w2c_start (torch.Tensor): The starting world-to-camera matrix.
        w2c_end (torch.Tensor): The ending world-to-camera matrix.
        num_steps (int): Number of interpolation steps.

    Returns:
        torch.Tensor: Interpolated world-to-camera matrices.
    """
    return torch.stack([(1 - alpha) * w2c_start + alpha * w2c_end for alpha in torch.linspace(0, 1, num_steps)])

def save_images(imgs, prefix="image"):
    """
    Saves a batch of images to disk with a given prefix.
    
    Args:
        imgs (torch.Tensor): Batch of images (assumed to be in [0, 1] range).
        prefix (str): Prefix for the saved filenames.
    """
    for i, img in enumerate(imgs):
        # Detach the tensor and move it to CPU, then convert to NumPy
        img_clamped = torch.clamp(img, 0, 1).detach().cpu().numpy()
        
        # Save the image using plt.imsave
        plt.imsave(f'{prefix}_{i}.png', img_clamped)
        print(f"Saved: {prefix}_{i}.png")
        
        
        
def extract_rpy_distance_range(c2w_matrices):
    # in degree
    roll, pitch, yaw, distance = matrix_to_rpy_distance(c2w_matrices, output_in_degrees=True)
    roll_range = (roll.min().item()-1, roll.max().item()+1)
    pitch_range = (pitch.min().item()-1, pitch.max().item()+1)
    yaw_range = (yaw.min().item()-1, yaw.max().item()+1)
    distance_range = (distance.min().item()-0.1, distance.max().item()+0.1)
    
    return roll_range, pitch_range, yaw_range, distance_range


