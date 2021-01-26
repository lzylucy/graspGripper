from numba import njit, prange
import numpy as np

def transform_is_valid(t, tolerance=1e-3):
    '''
    In:
        t: Numpy array [4x4] that is an transform candidate.
        tolerance: maximum absolute difference for two numbers to be considered close enough to each other.
    Out: 
        bool: True if array is a valid transform else False.
    Purpose:
        Check if array is a valid transform.
    '''
    if t.shape != (4,4):
        return False

    rtr = np.matmul(t[:3, :3].T, t[:3, :3])
    rrt = np.matmul(t[:3, :3], t[:3, :3].T)

    inverse_check = np.isclose(np.eye(3), rtr, atol=tolerance).all() and np.isclose(np.eye(3), rrt, atol=tolerance).all()
    det_check = np.isclose(np.linalg.det(t[:3, :3]), 1.0, atol=tolerance).all()
    last_row_check = np.isclose(t[3, :3], np.zeros((1, 3)), atol=tolerance).all() and np.isclose(t[3, 3], 1.0, atol=tolerance).all()

    return inverse_check and det_check and last_row_check

def transform_concat(t1, t2):
    '''
    In: 
        t1: Numpy array [4x4], left transform.
        t2: Numpy array [4x4], right transform.
    Out: 
        t1 * t2 as a numpy arrays [4x4].
    Purpose:
        Concatenate transforms.
    '''
    if not transform_is_valid(t1):
        raise ValueError('Invalid input transform t1')
    if not transform_is_valid(t2):
        raise ValueError('Invalid input transform t2')

    return np.matmul(t1, t2)

def transform_point3s(t, ps):
    '''
    In:
        t: Numpy array [4x4] to represent a transform
        ps: point3s represented as a numpy array [Nx3], where each row is a point.
    Out:
        Transformed point3s as a numpy array [Nx3].
    Purpose:
        Transfrom point from one space to another.
    '''
    if not transform_is_valid(t):
        raise ValueError('Invalid input transform t')

    if len(ps.shape) != 2 or ps.shape[1] != 3:
        raise ValueError('Invalid input points p')

    # convert to homogeneous
    ps_homogeneous = np.hstack([ps, np.ones((len(ps), 1), dtype=np.float32)])
    ps_transformed = np.dot(t, ps_homogeneous.T).T

    return ps_transformed[:, :3]

def transform_inverse(t):
    '''
    In:
        t: Numpy array [4x4] to represent a transform.
    Out:
        The inverse of the transform.
    Purpose:
        Find the inverse of the transfom.
    '''
    if not transform_is_valid(t):
        raise ValueError('Invalid input transform t')

    return np.linalg.inv(t)

@njit(parallel=True)
def camera_to_image(intrinsics, camera_points):
    '''
    In:
        intrinsics: Numpy array [3x3] containing camera pinhole intrinsics.
        p_camera: Numpy array [Nx3] representing point3s in camera coordinates.
    Out:
        Numpy array [Nx2] representing the projection of the point3 on the image plane.
    Purpose:
        Project a point3 in camera space to the image plane.
    '''
    if intrinsics.shape != (3, 3):
        raise ValueError('Invalid input intrinsics')
    if len(camera_points.shape) != 2 or camera_points.shape[1] != 3:
        raise ValueError('Invalid camera point')

    u0 = intrinsics[0, 2]
    v0 = intrinsics[1, 2]
    fx = intrinsics[0, 0]
    fy = intrinsics[1, 1]

    image_coordinates = np.empty((camera_points.shape[0], 2), dtype=np.int64)
    for i in prange(camera_points.shape[0]):
        image_coordinates[i, 0] = int(np.round((camera_points[i, 0] * fx / camera_points[i, 2]) + u0))
        image_coordinates[i, 1] = int(np.round((camera_points[i, 1] * fy / camera_points[i, 2]) + v0))

    return image_coordinates

def depth_to_point_cloud(intrinsics, depth_image):
    '''
    In: 
        intrinsics:  Numpy array [3x3] given as [[fx, 0, u0], [0, fy, v0], [0, 0, 1]]
        depth_image: Numpy array [lxw] where each value is the z-depth value
    Out:
        point_cloud: list of numpy arrays [1x3]
    Purpose:
        Back project a depth image to a point cloud.
    '''
    u0 = intrinsics[0, 2]
    v0 = intrinsics[1, 2]
    fx = intrinsics[0, 0]
    fy = intrinsics[1, 1]

    point_cloud = []

    for v in range(depth_image.shape[0]):
        for u in range(depth_image.shape[1]):
            if depth_image[v, u] <= 0:
                continue
            point_cloud.append(
                np.array([
                    (u - u0) * depth_image[v, u] / fx,
                    (v - v0) * depth_image[v, u] / fy,
                    depth_image[v, u]]))

    return point_cloud
