import numpy as np
import cv2

def get_marker_pose(image, matrix_coefficients, distortion_coefficients=None, 
                    aruco_dict_type=cv2.aruco.DICT_5X5_100, marker_id=24, marker_size=0.094):
    """
    Detects a specific ArUco marker by ID, draws it on the image, and returns the pose.

    Parameters:
    - image: np.ndarray, the RGB image in which to detect the marker.
    - matrix_coefficients: np.ndarray, the camera's intrinsic calibration matrix.
    - distortion_coefficients: np.ndarray, the camera's distortion coefficients (default is None).
    - aruco_dict_type: cv2.aruco.Dictionary, type of ArUco dictionary (default is DICT_5X5_100).
    - marker_id: int, the ID of the marker to detect (default is 24).
    - marker_size: float, the physical size of the marker in meters (default is 0.94 m).

    Returns:
    - pose: dict, containing the rotation vector `rvec`, translation vector `tvec`, and the updated image.
    - image_with_marker: np.ndarray, the image with the marker and axes drawn on it.
    """
    if image.dtype != np.uint8:
        image = image.astype(np.uint8)
    image = np.ascontiguousarray(image)

    # Convert image to grayscale for marker detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Load the dictionary and parameters for the specific ArUco marker type
    aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict_type)
    parameters = cv2.aruco.DetectorParameters()
    
    # Detect markers in the image
    corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
    
    if ids is not None:
        for i, marker_id_detected in enumerate(ids.flatten()):
            if marker_id_detected == marker_id:
                # Estimate pose for the specific marker
                rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(
                    corners[i], marker_size, matrix_coefficients, distortion_coefficients
                )
                
                # Convert rotation vector to rotation matrix
                rotation_matrix, _ = cv2.Rodrigues(rvec)
                
                # Create the homogeneous transformation matrix
                homogeneous_matrix = np.eye(4)
                homogeneous_matrix[:3, :3] = rotation_matrix
                homogeneous_matrix[:3, 3] = tvec.flatten()

                # Draw the detected marker and its axes on the image
                cv2.aruco.drawDetectedMarkers(image, [corners[i]])
                cv2.drawFrameAxes(image, matrix_coefficients, distortion_coefficients or np.zeros((5,)), rvec, tvec, marker_size * 0.5)
                
                # Return the pose information and the updated image
                pose = {
                    "rvec": rvec.flatten(),  # Rotation vector
                    "tvec": tvec.flatten(),  # Translation vector
                    "homogeneous_matrix": homogeneous_matrix  # Homogeneous transformation matrix
                }
                return pose, image
    
    # Return None if the marker is not found
    return None, image
