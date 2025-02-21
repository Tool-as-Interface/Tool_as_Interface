from typing import Tuple
import math
import cv2
import numpy as np
from scipy import ndimage
from matplotlib import colors
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import os
import time
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
import PIL
from PIL import Image
import torchvision.transforms as tvf
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
from PIL.ImageOps import exif_transpose
import dask.array as da
import torch
from ti.model.common.rotation_transformer import RotationTransformer
try:
    from pillow_heif import register_heif_opener  # noqa
    register_heif_opener()
    heif_support_enabled = True
except ImportError:
    heif_support_enabled = False

ImgNorm = tvf.Compose([tvf.ToTensor(), tvf.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
TorchNorm = tvf.Compose([tvf.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

ImgNorm_inv = tvf.Compose([tvf.Normalize((-0.5/0.5, -0.5/0.5, -0.5/0.5), (1/0.5, 1/0.5, 1/0.5))])

def draw_reticle(img, u, v, label_color):
    """
    Draws a reticle (cross-hair) on the image at the given position on top of
    the original image.
    @param img (In/Out) uint8 3 channel image
    @param u X coordinate (width)
    @param v Y coordinate (height)
    @param label_color tuple of 3 ints for RGB color used for drawing.
    """
    # Cast to int.
    u = int(u)
    v = int(v)

    white = (255, 255, 255)
    cv2.circle(img, (u, v), 10, label_color, 1)
    cv2.circle(img, (u, v), 11, white, 1)
    cv2.circle(img, (u, v), 12, label_color, 1)
    cv2.line(img, (u, v + 1), (u, v + 3), white, 1)
    cv2.line(img, (u + 1, v), (u + 3, v), white, 1)
    cv2.line(img, (u, v - 1), (u, v - 3), white, 1)
    cv2.line(img, (u - 1, v), (u - 3, v), white, 1)


def draw_text(
    img,
    *,
    text,
    uv_top_left,
    color=(255, 255, 255),
    fontScale=0.5,
    thickness=1,
    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
    outline_color=(0, 0, 0),
    line_spacing=1.5,
):
    """
    Draws multiline with an outline.
    """
    assert isinstance(text, str)

    uv_top_left = np.array(uv_top_left, dtype=float)
    assert uv_top_left.shape == (2,)

    for line in text.splitlines():
        (w, h), _ = cv2.getTextSize(
            text=line,
            fontFace=fontFace,
            fontScale=fontScale,
            thickness=thickness,
        )
        uv_bottom_left_i = uv_top_left + [0, h]
        org = tuple(uv_bottom_left_i.astype(int))

        if outline_color is not None:
            cv2.putText(
                img,
                text=line,
                org=org,
                fontFace=fontFace,
                fontScale=fontScale,
                color=outline_color,
                thickness=thickness * 3,
                lineType=cv2.LINE_AA,
            )
        cv2.putText(
            img,
            text=line,
            org=org,
            fontFace=fontFace,
            fontScale=fontScale,
            color=color,
            thickness=thickness,
            lineType=cv2.LINE_AA,
        )

        uv_top_left += [0, h * line_spacing]


def get_image_transform(
        input_res: Tuple[int,int]=(1280,720), 
        output_res: Tuple[int,int]=(640,480), 
        bgr_to_rgb: bool=False):

    iw, ih = input_res
    ow, oh = output_res
    rw, rh = None, None
    interp_method = cv2.INTER_AREA

    if (iw/ih) >= (ow/oh):
        # input is wider
        rh = oh
        rw = math.ceil(rh / ih * iw)
        if oh > ih:
            interp_method = cv2.INTER_LINEAR
    else:
        rw = ow
        rh = math.ceil(rw / iw * ih)
        if ow > iw:
            interp_method = cv2.INTER_LINEAR
    
    w_slice_start = (rw - ow) // 2
    w_slice = slice(w_slice_start, w_slice_start + ow)
    h_slice_start = (rh - oh) // 2
    h_slice = slice(h_slice_start, h_slice_start + oh)
    c_slice = slice(None)
    if bgr_to_rgb:
        c_slice = slice(None, None, -1)

    def transform(img: np.ndarray):
        assert img.shape == ((ih,iw,3))
        # resize
        img = cv2.resize(img, (rw, rh), interpolation=interp_method)
        # crop
        img = img[h_slice, w_slice, c_slice]
        return img
    return transform

def optimal_row_cols(
        n_cameras,
        in_wh_ratio,
        max_resolution=(1920, 1080)
    ):
    out_w, out_h = max_resolution
    out_wh_ratio = out_w / out_h
    
    n_rows = np.arange(n_cameras,dtype=np.int64) + 1
    n_cols = np.ceil(n_cameras / n_rows).astype(np.int64)
    cat_wh_ratio = in_wh_ratio * (n_cols / n_rows)
    ratio_diff = np.abs(out_wh_ratio - cat_wh_ratio)
    best_idx = np.argmin(ratio_diff)
    best_n_row = n_rows[best_idx]
    best_n_col = n_cols[best_idx]
    best_cat_wh_ratio = cat_wh_ratio[best_idx]

    rw, rh = None, None
    if best_cat_wh_ratio >= out_wh_ratio:
        # cat is wider
        rw = math.floor(out_w / best_n_col)
        rh = math.floor(rw / in_wh_ratio)
    else:
        rh = math.floor(out_h / best_n_row)
        rw = math.floor(rh * in_wh_ratio)
    
    # crop_resolution = (rw, rh)
    return rw, rh, best_n_col, best_n_row



def refine_depth_image(depth_image, gaussian_filter_std=2):
    """
    Refines a depth image or a batch of depth images by filling missing values and smoothing.

    Parameters:
    - depth_image: A numpy array representing the depth image or a batch of depth images.
    - gaussian_filter_std: Standard deviation for Gaussian filter.

    Returns:
    - Refined depth image or batch of depth images.
    """

    def refine_single_image(img):
        # Replace NaN values with zero
        depth_refine = np.nan_to_num(img)

        # Create a mask where depth values are zero
        mask = (depth_refine == 0).astype(np.uint8)

        # Convert depth image to float32 for inpainting
        depth_refine = depth_refine.astype(np.float32)

        # Inpaint to fill holes in the depth image
        depth_refine = cv2.inpaint(depth_refine, mask, 2, cv2.INPAINT_NS)

        # Convert back to original depth image type (float)
        depth_refine = depth_refine.astype(np.float)

        # Apply Gaussian filter for smoothing
        depth_refine = ndimage.gaussian_filter(depth_refine, gaussian_filter_std)

        return depth_refine

    # Determine if input is a single image or a batch
    if depth_image.ndim == 2:  # Single image
        return refine_single_image(depth_image)
    elif depth_image.ndim == 3:  # Batch of images
        # Process each image in the batch
        # Note: Inpainting cannot be vectorized and is applied per image
        return np.array([refine_single_image(img) for img in depth_image])
    else:
        raise ValueError("Input must be a 2D or 3D numpy array")
    
    
def visualize_hsv_channels(img, color_space='rgb'):
    if color_space == 'rgb':
        # Convert the image to HSV
        hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    elif color_space == 'bgr':
        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Split the HSV image into separate channels
    h, s, v = cv2.split(hsv_img)

    # Flatten the channels for scatter plot
    h_flat = h.flatten()
    s_flat = s.flatten()
    v_flat = v.flatten()

    # Convert the HSV image to RGB for color representation in the scatter plot
    rgb_img = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2RGB)
    pixel_colors = rgb_img.reshape((np.shape(rgb_img)[0]*np.shape(rgb_img)[1], 3))
    norm = colors.Normalize(vmin=0., vmax=255.)
    pixel_colors = norm(pixel_colors).tolist()
    
    
    # Visualize each channel using matplotlib
    plt.figure(figsize=(16, 12))

    plt.subplot(2, 2, 1)
    plt.imshow(rgb_img)
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(2, 2, 2)
    plt.imshow(h, cmap='hsv')
    plt.colorbar()
    plt.title('Hue Channel')
    plt.axis('off')

    plt.subplot(2, 2, 3)
    plt.imshow(s, cmap='gray')
    plt.colorbar()
    plt.title('Saturation Channel')
    plt.axis('off')

    plt.subplot(2, 2, 4)
    plt.imshow(v, cmap='gray')
    plt.colorbar()
    plt.title('Value Channel')
    plt.axis('off')

    # Create a 3D scatter plot
    fig = plt.figure()
    axis = fig.add_subplot(1, 1, 1, projection="3d")
    axis.scatter(h_flat, s_flat, v_flat, facecolors=pixel_colors, marker=".")
    axis.set_xlabel("Hue")
    axis.set_ylabel("Saturation")
    axis.set_zlabel("Value")

    plt.show()
    
    
    
# def draw_keypoints_on_image( camera_idx, keypoints_world, colors, intrinsics, extrinsics, dist_coeffs):
#     """Draws keypoints on a given image."""
#     cam_extrinsics = extrinsics[camera_idx]
    
#     # Extract rotation and translation vectors
#     rvec_obj, tvec_obj = cam_extrinsics[:3, :3], cam_extrinsics[:3, 3]
#     # Convert rotation matrix to vector
#     rvec_obj, _ = cv2.Rodrigues(rvec_obj)

#     image_points, _ = cv2.projectPoints(keypoints_world, rvec_obj, tvec_obj, intrinsics[camera_idx], dist_coeffs[camera_idx])

#     # Draw the points on the image    
#     color_frame = colors[camera_idx]
#     # Ensure frame is contiguous for modification
#     contiguous_frame = np.ascontiguousarray(color_frame) if not color_frame.flags['C_CONTIGUOUS'] else color_frame

#     for point in image_points:
#         x, y = int(point[0][0]), int(point[0][1])
#         cv2.circle(contiguous_frame, (x, y), radius=3, color=(0, 255, 0), thickness=-1)  # blue color in RGB

#     if not color_frame.flags['C_CONTIGUOUS']:
#         color_frame[:] = contiguous_frame
        




def generate_color_palette(num_colors):
    """Generate a more distinct color palette using HSV color space."""
    palette = []
    for i in range(num_colors):
        hue = int(360 * i / num_colors)  # Hue value from 0 to 359
        saturation = 255 if i % 3 == 0 else 200  # Alternate between full and slightly reduced saturation
        value = 255 if i % 2 == 0 else 180  # Alternate between full and slightly reduced value
        color = np.zeros((1, 1, 3), dtype=np.uint8)
        color[0, 0] = [hue, saturation, value]  # Set HSV values
        color_bgr = cv2.cvtColor(color, cv2.COLOR_HSV2BGR)
        palette.append(tuple(int(c) for c in color_bgr[0, 0]))
    return palette

def generate_spectral_palette(num_colors):
    """Generate a color palette using the 'Spectral' colormap from Seaborn."""
    import seaborn as sns
    cmap = sns.color_palette("Spectral", num_colors)
    palette = [(int(r*255), int(g*255), int(b*255)) for r, g, b in cmap]
    return palette

def generate_rainbow_palette(num_colors):
    """Generate a rainbow-like gradient color palette using HSV color space."""
    palette = []
    for i in range(num_colors):
        hue = int(164* i / (num_colors - 1))  # Hue value from 0 to 240 (red to violet)
        saturation = 255  # Full saturation
        value = 255  # Full value
        color = np.zeros((1, 1, 3), dtype=np.uint8)
        color[0, 0] = [hue, saturation, value]  # Set HSV values
        color_bgr = cv2.cvtColor(color, cv2.COLOR_HSV2BGR)
        palette.append(tuple(int(c) for c in color_bgr[0, 0]))
    return palette


# Define a larger color palette with smooth transitions
color_palette = generate_rainbow_palette(25)


def draw_keypoints_on_image(camera_idx, keypoints_world, colors, intrinsics, extrinsics, dist_coeffs, dif_color=False):
    """Draws keypoints on a given image."""
    cam_extrinsics = extrinsics[camera_idx]
    
    # Extract rotation and translation vectors
    rvec_obj, tvec_obj = cam_extrinsics[:3, :3], cam_extrinsics[:3, 3]
    # Convert rotation matrix to vector
    rvec_obj, _ = cv2.Rodrigues(rvec_obj)

    # Project points (for all keypoints at once)
    image_points, _ = cv2.projectPoints(keypoints_world, rvec_obj, tvec_obj, intrinsics[camera_idx], dist_coeffs[camera_idx])

    # Draw the points on the image
    color_frame = colors[camera_idx]

    # Convert image points to integer coordinates
    image_points = image_points.squeeze().astype(int)

    # Draw all points using vectorized operations
    for idx, point in enumerate(image_points):
        # cv2.circle(color_frame, tuple(point), radius=3, color=(0, 255, 0), thickness=-1)  # Green color in BGR
        if dif_color:
            color = color_palette[idx % len(color_palette)]
        else:
            color = (0, 255, 0)  # Default green color in BGR

        cv2.circle(color_frame, tuple(point), radius=1, color=color, thickness=-1)


    return color_frame  # Return the modified frame


def visualize_keypoints_in_workspace(keypoints,
                                     only_last_frame=False,
                                     actions=None,
                                     workspace_size=(512, 512), 
                                     real_boundaries=None,
                                     trans_real_in_sim=None,
                                     separate_plots=True,
                                     hide_boundaries=False):
    """
    Visualize 2D keypoints within a specified workspace size with (0,0) at the upper left corner.

    Parameters:
    keypoints (numpy.ndarray): Numpy array of keypoints with shape (num_frames, num_keypoints * 2).
    workspace_size (tuple): Size of the workspace (width, height).
    separate_plots (bool): If True, each frame's keypoints are plotted separately. 
                           If False, all frames are plotted on the same plot.
    """
    num_frames = keypoints.shape[0]
    num_keypoints = keypoints.shape[1] // 2
    if only_last_frame:
        num_frames = 1
        keypoints = keypoints[-1:]
        
    save_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'data/visualizations', 'keypoint')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if real_boundaries is not None and trans_real_in_sim is not None:
        # Convert boundaries to simulation coordinates
        real_boundaries_corners = np.array([
            [real_boundaries['x_lower'], real_boundaries['y_lower']],
            [real_boundaries['x_upper'], real_boundaries['y_lower']],
            [real_boundaries['x_upper'], real_boundaries['y_upper']],
            [real_boundaries['x_lower'], real_boundaries['y_upper']],
            [real_boundaries['x_lower'], real_boundaries['y_lower']]  # Close the rectangle
        ])

        real_boundaries_corners_homogeneous = np.hstack([real_boundaries_corners, np.ones((real_boundaries_corners.shape[0], 1))])
        sim_boundaries_corners = np.dot(trans_real_in_sim, real_boundaries_corners_homogeneous.T).T[:, :2]
   
    # Turn on interactive mode
    fig, axes = plt.subplots(1, num_frames, figsize=(6 * num_frames, 6))
    if num_frames == 1:
        axes = [axes]
    for i, ax in enumerate(axes):
        if not hide_boundaries:
            ax.plot(sim_boundaries_corners[:, 0], sim_boundaries_corners[:, 1], color='red')  # Draw boundaries

        x = keypoints[i, 0:num_keypoints*2:2]
        y = keypoints[i, 1:num_keypoints*2:2]
        ax.scatter(x[:-1], y[:-1], color=(180/255, 119/255, 31/255))  # Convert RGB to [0, 1] range
        ax.scatter(x[-1:], y[-1:], color='blue')  # Plot the last keypoint in blue
        if actions is not None:
            ax.scatter(actions[:, 0], actions[:, 1], color='green', marker='x')  # Draw actions

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title(f'Frame {i+1}')
        # ax.set_xlim(workspace_size[0], 0)  # Invert x-axis
        ax.set_xlim(0, workspace_size[0])
        ax.set_ylim(workspace_size[1], 0)  # Invert y-axis
        ax.set_aspect('equal', adjustable='box')  # Set equal unit lengths for x and y axes
        ax.grid(False)

        # Convert plot to image
        canvas = FigureCanvas(fig)
        canvas.draw()
        img = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        # Display using OpenCV
        cv2.imshow('Keypoints Visualization', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        cv2.waitKey(1)
        
        # Save the figure
        # save_path = os.path.join(save_dir, f'frame_{time.time()}.png')
        # cv2.imwrite(save_path, img)

        plt.close(fig)  # Close the figure to free memory

        plt.close(fig)  # Close the figure to free memory


class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)


def visualize_6d_trajectory(action):
    """
    Visualizes a trajectory in 3D space with rotation vectors shown as arrows.

    Parameters:
    - action (np.ndarray): An array of shape (N, D) where N is the number of points in the trajectory.
                           D can be 3 (positions only), 6 (positions + rotation vectors), or 9 (positions + 6D rotation representation).
    """
    # Validate input dimensions
    if action.shape[1] not in [3, 6, 9]:
        raise ValueError("Invalid action shape. Must be (N, 3), (N, 6), or (N, 9)")

    positions = action[:, :3]  # Always extract the first three columns for position

    # Determine what additional data is included
    if action.shape[1] == 3:
        rotations = None
    elif action.shape[1] == 6:
        rotations = action[:, 3:6]
    elif action.shape[1] == 9:
        rotation_transformer = RotationTransformer(from_rep='rotation_6d', to_rep='axis_angle')
        rotations = rotation_transformer.forward(action[:, 3:9])

    # Creating the figure
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plotting the trajectory
    ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], 'b-', marker='o', label='Trajectory')

    # Visualizing rotations with arrows if they exist
    arrow_scale = 0.01  # Smaller value for smaller arrows
    if rotations is not None:
        for i in range(len(positions)):
            # Normalizing rotation vector for consistent arrow length
            norm = np.linalg.norm(rotations[i])
            if norm != 0:
                rotation_norm = rotations[i] / norm
                ax.quiver(positions[i, 0], positions[i, 1], positions[i, 2], 
                          rotation_norm[0], rotation_norm[1], rotation_norm[2], 
                          color='r', length=arrow_scale, normalize=True)

    # Optionally, add starting and ending points in different colors
    ax.scatter(positions[0, 0], positions[0, 1], positions[0, 2], color='green', s=100, label='Start')
    ax.scatter(positions[-1, 0], positions[-1, 1], positions[-1, 2], color='red', s=100, label='End')

    # Setting labels and title for clarity
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Trajectory Visualization with Optional Rotation')

    # Legend
    ax.legend()

    # Show plot
    plt.show()

def _resize_pil_image(img, long_edge_size):
    S = max(img.size)
    if S > long_edge_size:
        interp = PIL.Image.LANCZOS
    elif S <= long_edge_size:
        interp = PIL.Image.BICUBIC
    new_size = tuple(int(round(x*long_edge_size/S)) for x in img.size)
    return img.resize(new_size, interp)

def load_images_mast3r(folder_or_list, size, square_ok=False, verbose=True, device='cuda'):
    """ open and convert all images in a list or folder to proper input format for DUSt3R
    """
    if isinstance(folder_or_list, str):
        if verbose:
            print(f'>> Loading images from {folder_or_list}')
        root, folder_content = folder_or_list, sorted(os.listdir(folder_or_list))

    elif isinstance(folder_or_list, list):
        if verbose:
            print(f'>> Loading a list of {len(folder_or_list)} images')
        root, folder_content = '', folder_or_list
    elif isinstance(folder_or_list, np.ndarray):
        # Check if the NumPy array is a batch of images (N, W, H, C)
        if len(folder_or_list.shape) == 4:  # Batch of images
            N, W, H, C = folder_or_list.shape
            if verbose:
                print(f'>> Loading a batch of {N} images from a numpy array')
            folder_content = folder_or_list
            root = ''
        else:
            raise ValueError(f'Invalid numpy array shape: {folder_or_list.shape}. Expected (N, W, H, C).')
    elif isinstance(folder_or_list, da.Array):
        # Handle Dask array (batch of images)
        if len(folder_or_list.shape) == 4:  # Batch of images
            N, W, H, C = folder_or_list.shape
            if verbose:
                print(f'>> Loading a batch of {N} images from a Dask array')
            folder_content = folder_or_list
            root = ''
        else:
            raise ValueError(f'Invalid Dask array shape: {folder_or_list.shape}. Expected (N, W, H, C).')
    elif isinstance(folder_or_list, torch.Tensor):
        # If it's a PyTorch tensor, skip resizing and cropping
        if len(folder_or_list.shape) == 4:
            N, C, H, W = folder_or_list.shape
            if verbose:
                print(f'>> Loading a batch of {N} images from a torch Tensor')
            folder_content = folder_or_list
            root = ''
        else:
            raise ValueError(f'Invalid tensor shape: {folder_or_list.shape}. Expected (N, C, H, W).')
    else:
        raise ValueError(f'bad {folder_or_list=} ({type(folder_or_list)})')

    supported_images_extensions = ['.jpg', '.jpeg', '.png']
    if heif_support_enabled:
        supported_images_extensions += ['.heic', '.heif']
    supported_images_extensions = tuple(supported_images_extensions)

    imgs = []
    for path_or_img in folder_content:
        if isinstance(path_or_img, str) and not path_or_img.lower().endswith(supported_images_extensions):
            # Skip non-image files if input is a folder
            continue
        
        # Load the image from file or use the preloaded NumPy array
        if isinstance(path_or_img, str):
            img = exif_transpose(PIL.Image.open(os.path.join(root, path_or_img))).convert('RGB')
        elif isinstance(path_or_img, np.ndarray):
            # Convert the NumPy array (assuming it has shape HxWx3) to a PIL image
            img = PIL.Image.fromarray(path_or_img.astype(np.uint8))
        elif isinstance(folder_or_list, da.Array):
            # Convert the Dask array slice to a NumPy array (using .compute()) and then to a PIL image
            img = PIL.Image.fromarray(path_or_img.compute().astype(np.uint8))
        elif isinstance(path_or_img, torch.Tensor):
            img = path_or_img
        else:
            raise ValueError(f'Unsupported image type: {type(path_or_img)}')
        
        if isinstance(path_or_img, torch.Tensor):
            imgs.append(dict(img=TorchNorm(img)[None].to(device), true_shape=np.int32(
                [img.shape[-2:]]), idx=len(imgs), instance=str(len(imgs))))

        else:
            W1, H1 = img.size
            if size == 224:
                # resize short side to 224 (then crop)
                img = _resize_pil_image(img, round(size * max(W1/H1, H1/W1)))
            else:
                # resize long side to 512
                img = _resize_pil_image(img, size)
            W, H = img.size
            cx, cy = W//2, H//2
            if size == 224:
                half = min(cx, cy)
                img = img.crop((cx-half, cy-half, cx+half, cy+half))
            else:
                halfw, halfh = ((2*cx)//16)*8, ((2*cy)//16)*8
                if not (square_ok) and W == H:
                    halfh = 3*halfw/4
                img = img.crop((cx-halfw, cy-halfh, cx+halfw, cy+halfh))

            W2, H2 = img.size
            if verbose:
                print(f' - adding image with resolution {W1}x{H1} --> {W2}x{H2}')
                
            imgs.append(dict(img=ImgNorm(img)[None].to(device), true_shape=np.int32(
                [img.size[::-1]]), idx=len(imgs), instance=str(len(imgs))))

    assert imgs, f'No images found in {root}'
    if verbose:
        print(f' (Found {len(imgs)} images)')
    return imgs




