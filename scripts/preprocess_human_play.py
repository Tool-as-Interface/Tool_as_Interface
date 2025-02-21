import os
import pathlib
import warnings
from glob import glob
import click
import numpy as np
import torch
import tqdm
import cv2
import sys
from einops import rearrange
import omegaconf

from ti.common.replay_buffer import ReplayBuffer
from ti.model.common.rotation_transformer import RotationTransformer
from ti.utils.segmentation import get_segmentation_mask, initialize_segmentation_models


sys.path.append(os.path.join(os.path.dirname(__file__), '../third_party'))

original_dir = os.getcwd()

os.chdir(os.path.join(os.path.dirname(__file__), '../third_party/FoundationPose'))
from FoundationPose.estimater import *
from FoundationPose.datareader import *

os.chdir(original_dir)


def track_tool_poses(video, depth_video, seg_save_path, detect_save_path, intrinsics, output_video_path, **params):
    segmentation_labels = params.get('segmentation_labels', [])
    to_origin = params['pose_model_info'].get('to_origin', np.eye(4))  
    bbox = params['pose_model_info'].get('bbox', [0, 0, 0, 0])

    base_pose_in_world = params.get('base_pose_in_world', np.eye(4))
    cam_extrinsics = params.get('cam_extrinsics', np.eye(4))

    pose_model = params.get('pose_model', None)
    rotation_transformer = params.get('rotation_transformer', None)
    
    cam_in_base = np.linalg.inv(base_pose_in_world)@cam_extrinsics
    T, H, W, C= video.shape

    rgb_video = video

    video = torch.from_numpy(video).cuda().float()
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Define the codec, e.g., 'mp4v' for .mp4 format
    fps = 15 
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (W, H))
    
    pose_matrices = np.empty((T, 4, 4))  # T x 4 x 4 matrix
    for i, (rgb, depth) in enumerate(zip(rgb_video, depth_video)):
        if i==0:
            tool_mask, _ = get_segmentation_mask(video[i].cpu().numpy()[:,:,::-1].astype(np.uint8), segmentation_labels, seg_save_path, params['grounding_dino_model'], params['sam_predictor'])
            pose = pose_model.register(K=intrinsics, rgb=rgb, depth=depth, ob_mask=tool_mask, iteration=10)

        else:
            pose = pose_model.track_one(rgb=rgb, depth=depth, K=intrinsics, iteration=5)
            
        pose_matrices[i] = cam_in_base@pose
        
        center_pose = pose@np.linalg.inv(to_origin)
        vis = draw_posed_3d_box(intrinsics, img=rgb, ob_in_cam=center_pose, bbox=bbox)
        vis = draw_xyz_axis(rgb, ob_in_cam=center_pose, scale=0.1, K=intrinsics, thickness=3, transparency=0, is_input_rgb=True)
        
        vis_bgr = vis[..., ::-1]
        video_writer.write(vis_bgr)

    video_writer.release()

    trans = pose_matrices[:, :3, 3]  # Extract translation vectors
    rot_mat = pose_matrices[:, :3, :3]    # Extract rotation matrices
    rot_vec = rotation_transformer.forward(rot_mat)  # Convert rotation matrices to rotation vectors
    pose_vec = np.concatenate([
        trans, rot_vec
    ], axis=-1).astype(np.float32)

    return pose_vec

def collect_states_from_demo(target_dir,episode_ends,episode_lengths,
                             demos_group, demo_k, view_names,
                             **params):

    img_size = params.get('img_size', (128,128))

    seg_save_path = os.path.join(target_dir, f"segmentations/{demo_k}.png")
    detect_save_path = os.path.join(target_dir, f"detections/{demo_k}.png")
    episode_slice = slice(episode_ends[demo_k] - episode_lengths[demo_k], episode_ends[demo_k])

    def update_zarr_dataset(new_data_key, new_data, observations_group, chunks=True):
        try:
            if new_data_key in observations_group:
                # If the dataset exists, resize and append the new data
                array = observations_group[new_data_key]
                new_len = array.shape[0] + new_data.shape[0]
                new_shape = (new_len,) + array.shape[1:]
                array.resize(new_shape)
                array[-new_data.shape[0]:] = new_data
            else:
                # If the dataset does not exist, create it
                observations_group.create_dataset(new_data_key, data=new_data, chunks=chunks, overwrite=False)
        except Exception as e:
            print(f"Error adding data to {new_data_key}: {e}")

    for view in view_names[:1]:
        rgb = demos_group[f'observations/images/{view}_color'][episode_slice].copy()
        resize_rgb = np.array([cv2.resize(img, img_size, interpolation=cv2.INTER_AREA) for img in rgb])
        resize_rgb = rearrange(resize_rgb, "t h w c -> t c h w")

        depth = demos_group[f'observations/images/{view}_depth'][episode_slice].copy()

        depth = depth.astype(np.float64)
        if depth.max() > 10:
            depth = depth / 1000.0
        max_thresh=1.5

        resize_depth = np.array([cv2.resize(d, img_size, interpolation=cv2.INTER_AREA)[..., None] for d in depth])
        resize_depth = rearrange(resize_depth, "t h w c -> t c h w")
        for t in range(resize_depth.shape[0]):
            frame = resize_depth[t, 0, :, :] 
            invalid_mask = (frame > max_thresh)
            for i in range(frame.shape[0]):
                for j in range(frame.shape[1]):
                    if invalid_mask[i, j]:
                        if j < frame.shape[1] - 1:  
                            frame[i, j] = frame[i, j + 1]
                        else: 
                            frame[i, j] = frame[i, j - 1]
            resize_depth[t, 0, :, :] = frame

        intrinsics = demos_group[f'observations/images/{view}_intrinsics'][0].copy().astype(np.float64)


        pose_video_path = os.path.join(target_dir, "videos", f"{demo_k}_{view}_pose.mp4")
        poses=track_tool_poses(rgb, depth, seg_save_path, detect_save_path, intrinsics, pose_video_path, **params)
        update_zarr_dataset(f'robot_eef_pose', poses, demos_group['observations'])
        


def generate_data(source_data_path, target_dir, views=[],**params):

    cont_from = params.get('cont_from', None)

    input_path = pathlib.Path(source_data_path).expanduser()
    in_zarr_path = input_path.joinpath('replay_buffer.zarr')
    assert in_zarr_path.is_dir(), f"{in_zarr_path} does not exist"
    in_replay_buffer = ReplayBuffer.create_from_path(str(in_zarr_path.absolute()), mode='a')
    demos = in_replay_buffer
    demo_keys = range(cont_from, len(in_replay_buffer.episode_ends))
    camera_keys = list(in_replay_buffer['observations/images'].keys()) 
    
    episode_ends = in_replay_buffer.episode_ends
    episode_lengths = in_replay_buffer.episode_lengths
        
    # setup visualization class
    video_path = os.path.join(target_dir, 'videos')

    if not os.path.exists(video_path):
        os.makedirs(video_path, exist_ok=True)
    
    # Create directories for saving segmentations and detections if they don't exist
    seg_save_dir = os.path.join(target_dir, "segmentations")
    detect_save_dir = os.path.join(target_dir, "detections")
    for directory in [seg_save_dir, detect_save_dir]:
        os.makedirs(directory, exist_ok=True)
        
    with torch.no_grad():
        if f'observations/robot_eef_pose' not in demos:
            print("Tool poses not found in the dataset. Initiating tool pose tracking...")
            for idx, demo_k in enumerate(tqdm.tqdm(demo_keys)):
                collect_states_from_demo(target_dir,episode_ends,episode_lengths, demos, demo_k, views, **params)
        else:
            print("Tool poses already present in the dataset. Skipping tracking step.")


@click.command()
@click.option("--save", type=str, default="./data/preprocessed/")
@click.option("--views", type=list, default=['camera_3'], help="List of camera views to process")
@click.option("--cont_from", type=int, default=0, help="Continue from a specific episode")
@click.option("--task_conf_path", type=str, default='conf/train_bc/task/real_hammer_human.yaml')
def main(save, views, cont_from,task_conf_path):
    """
    save: str, target directory to save the preprocessed data
    skip_exist: bool, whether to skip the existing preprocessed h5df file
    """
    code_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    task_conf_path = os.path.join(code_dir, task_conf_path)
    task_cfg = omegaconf.OmegaConf.load(task_conf_path)
    data_path = task_cfg.dataset_path
    
        
    mesh = trimesh.load(task_cfg.mesh_file)
    to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
    bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2,3)
    pose_model_info = {}
    scorer = ScorePredictor()
    refiner = PoseRefinePredictor()
    glctx = dr.RasterizeCudaContext()
    est = FoundationPose(model_pts=mesh.vertices, model_normals=mesh.vertex_normals, mesh=mesh, scorer=scorer, refiner=refiner, debug_dir=os.path.join(code_dir,'debug'), debug=1, glctx=glctx)
        
    pose_model_info.update({
        'to_origin': to_origin,
        'bbox': bbox
    })
    

    grounding_dino_model, sam_predictor = initialize_segmentation_models()

    np.random.seed(seed=0)
    set_seed(0)
    rotation_transformer = RotationTransformer(from_rep='matrix', to_rep='axis_angle')
    
    
    robot_side = task_cfg.robot_side
    cam_serials = task_cfg.cam_serials

    robot_extrinsics_path=os.path.join(code_dir, f'ti/real_world/robot_extrinsics/{robot_side}_base_pose_in_world.npy')
    camera_extrinsics_path=os.path.join(code_dir, f'ti/real_world/cam_extrinsics/{cam_serials[views[0]]}.npy')
    
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
        
        
    # load task name embeddings
    # task_bert_embs_dict = get_task_bert_embs(root, data_path, zarr_format)
    params = {
        'pose_model': est,
        'pose_model_info': pose_model_info,
        'cont_from': cont_from,
        'segmentation_labels': task_cfg.segmentation_labels,
        'rotation_transformer': rotation_transformer,
        'grounding_dino_model': grounding_dino_model,
        'sam_predictor': sam_predictor,
        'img_size': (128,128),
        'base_pose_in_world': base_pose_in_world,
        'cam_extrinsics': cam_extrinsics,
    }
    file_name = os.path.basename(data_path)
    source_data_path = data_path

    
    save_dir = os.path.join(save, file_name)
    os.makedirs(save_dir, exist_ok=True)
    generate_data(source_data_path, 
                  save_dir, 
                  views=views,
                  **params)
            


if __name__ == "__main__":
    main()