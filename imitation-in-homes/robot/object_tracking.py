import numpy as np
from scipy.spatial.transform import Rotation as R

K = np.array([
        [720, 0, 480], 
        [0, 720, 360], 
        [0, 0, 1]
    ])

def get_transformation_matrix(xyz, quats):
    # Convert quaternions to rotation matrix
    # w, x, y, z = quats # TODO: we fixed this, remove in future code
    rotation_matrix = R.from_quat(quats).as_matrix()
    
    # Create translation matrix
    T = np.eye(4)
    # T[:3, 3] = xyz
    T[:3, 3] = xyz

    T[:3, :3] = rotation_matrix

    return T

def get_camera_z_from_euclidean(u, v, D, K):
    pixel = np.array([u, v, 1.0])
    ray = np.linalg.inv(K) @ pixel
    direction = ray / np.linalg.norm(ray)
    Z = D * direction[2]  # camera Z component
    return Z

def pixel_to_3d_from_camera_z(u, v, Z, K):
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy

    return np.array([X, Y, Z])

def project_3d_to_2d(K, point_3d):
    X, Y, Z = point_3d
    
    point_cam = np.array([X, Y, Z])
    
    projected = K @ (point_cam / Z)
    
    u, v = projected[:2]
    
    return u, v

def get_new_2d_point(object_x, object_y, object_depth, relative_transformation_matrix):
    object_x = int(object_x * 960)
    object_y = int(object_y * 720)

    # Convert Euclidean depth to camera-Z
    camera_z = get_camera_z_from_euclidean(object_x, object_y, object_depth, K)

    # Back-project using Z
    bread_3d = pixel_to_3d_from_camera_z(object_x, object_y, camera_z, K)

    if relative_transformation_matrix is None:
        return bread_3d

    x, y, z = bread_3d # TODO: make proper transformation

    new_bread_3d = np.dot(np.linalg.inv(relative_transformation_matrix), np.append([x, -y, -z], 1))[:3] # converting point from canonical frame to r3d frame (x, -y, -z)

    x, y, z = new_bread_3d
    canonical_point_3d = [x, -y, -z]
    new_bread_2d = project_3d_to_2d(K, canonical_point_3d) # TODO: make proper transformation; converting from r3d frame to canonical frame (x, -y, -z)

    return new_bread_2d, canonical_point_3d

def transformation_matrix_to_relative_translation(transformation_matrix):
    rotation_matrix = transformation_matrix[:3, :3]
    translation_vector = transformation_matrix[:3, 3]

    euler_angles = R.from_matrix(rotation_matrix).as_euler('xyz', degrees=False)

    return euler_angles, translation_vector