import cv2
import numpy as np
import os
import trimesh
from camera import *
import image, transforms, ply


def mesh2pts(mesh_path, N):
    """
    Sample a pointcloud from a mesh.
    :param mesh_path: string, path of the mesh
    :param N: int, number of points to sample
    :return pts: Numpy array [Nx3], sampled point cloud
    """
    mesh = trimesh.load(mesh_path)
    pts, _ = trimesh.sample.sample_surface(mesh, count=N)
    return pts


def obj_mesh2pts(objID, point_num):
    """
    :param objID:
    :param point_num: int, number of points to sample
    :return: pts: Numpy array [Nx3], sampled point cloud
    """
    obj_filename = './YCB_subsubset/' + list_obj_foldername[objID - 1] + '/model_com.obj'  # objects ID start from 1
    pts = mesh2pts(mesh_path=obj_filename, N=point_num)
    return pts


def gen_obj_depth(objID, depth, mask):
    """
    Generate depth image for a specific object.
    :param objID: int, generate depth for all objects when objID == -1
    :param depth: Numpy array [HxW]
    :param mask: Numpy array [HxW]
    :return obj_depth: Numpy array [HxW]
    """
    if objID == -1:
        obj_mask = np.where(mask == 0, 0, 1)
    else:
        obj_mask = np.where(mask == objID, 1, 0)
    obj_depth = depth * obj_mask
    return obj_depth


def obj_depth2pts(objID, depth, mask, camera, view_matrix):
    """
    Generate pointcloud projected from depth of specific object in world coordinate.
    :param objID: int
    :param depth: Numpy array [HxW]
    :param mask: Numpy array [HxW]
    :param camera: Camera object
    :param view_matrix: 4x4 matrix, stored as a list of 16 floats
    :return world_pts: Numpy array [Nx3]
    """
    obj_depth = gen_obj_depth(objID, depth, mask)
    cam_pts = np.asarray(transforms.depth_to_point_cloud(camera.intrinsic_matrix, obj_depth))
    if len(cam_pts) == 0:
        return
    else:
        world_pts = transforms.transform_point3s(cam_view2pose(view_matrix), cam_pts)
    return world_pts


def align_pts(pts_a, pts_b, max_iterations, threshold):
    init_matrix, transformed, cost = trimesh.registration.procrustes(pts_a,
                                                                     pts_b,
                                                                     reflection=False,
                                                                     translation=True,
                                                                     scale=False,
                                                                     return_cost=True)
    matrix, transformed, cost = trimesh.registration.icp(pts_a,
                                                         pts_b,
                                                         initial=init_matrix,
                                                         max_iterations=max_iterations,
                                                         scale=False,
                                                         threshold=threshold, )
    return matrix, transformed


def export_ply(pts, path, pts_type):
    """
    :param pts: Numpy array [Nx3], point cloud
    :param path: path to be saved, without suffix
    :param pts_type: string, used to specify suffix and point color
    :return: None
    """
    suffix = "_" + pts_type + ".ply"
    color_switcher = {
        "gt": [0, 0, 0],  # Black
        "gtmask": [0, 255, 0],  # Green
        "gtmask_transformed": [0, 0, 255],  # Blue
        "predmask": [255, 0, 0],  # Red
        "predmask_transformed": [255, 255, 0],  # Yellow
    }
    color = color_switcher[pts_type]
    # To be updated after Samir
    # ptcloud = ply.Ply(points=pts, colors=color)
    # ptcloud.write(path)
    ptcloud = trimesh.points.PointCloud(vertices=pts, colors=color)
    ptcloud.show()
    ptcloud.export(path + suffix)
    return


def estimate_pose(test_ID, depth, mask, camera, view_matrix, pts_type):
    pts_transformed = np.empty([0, 3])
    for objID in range(1, 6):
        pts_depth = obj_depth2pts(objID, depth, mask, camera, view_matrix)
        if pts_depth is not None:
            # In order to use procrustes(require input pointclouds to have the same number of points)
            # have to sample the same .obj in each test case to match the number of points in projected point cloud
            pts_mesh = obj_mesh2pts(objID, point_num=pts_depth.shape[0])
            transform, transformed = align_pts(pts_mesh,
                                               pts_depth,
                                               max_iterations=1000,
                                               threshold=1e-07)
            # Save predicted pose matrix
            np.save(dataset_dir + "pred_pose/" + pts_type[:-12] + "/" + str(test_ID) + "_" + str(objID), transform)
            pts_transformed = np.concatenate((pts_transformed, transformed), axis=0)
    export_ply(pts=pts_transformed,
               path=dataset_dir + "exported_ply/" + str(test_ID),
               pts_type=pts_type)


def denoise_mask(mask):
    kernel = np.ones((5, 5), np.uint8)
    for i in range(3):
        mask = cv2.dilate(mask, kernel, iterations=1 * i)
        mask = cv2.erode(mask, kernel, iterations=2 * i)
        mask = cv2.dilate(mask, kernel, iterations=1 * i)
    return mask


if __name__ == "__main__":
    # Setup camera -- to recover coordinate, keep consistency with that in gen_dataset.py
    my_camera = Camera(
        image_size=(480, 640),
        near=0.01,
        far=10.0,
        fov_w=69.40
    )

    dataset_dir = "./dataset/test/"
    if not os.path.exists(dataset_dir + "exported_ply/"):
        os.makedirs(dataset_dir + "exported_ply/")
    if not os.path.exists(dataset_dir + "pred_pose/"):
        os.makedirs(dataset_dir + "pred_pose/")
        os.makedirs(dataset_dir + "pred_pose/gtmask/")
        os.makedirs(dataset_dir + "pred_pose/predmask/")

    list_obj_foldername = [
        "003_cracker_box",
        "004_sugar_box",
        "005_tomato_soup_can",
        "006_mustard_bottle",
        "007_tuna_fish_can",
    ]

    for test_ID in range(10):
        view_matrix = np.load(dataset_dir + "view_matrix/" + str(test_ID) + ".npy")
        depth = image.read_depth(dataset_dir + "depth/" + str(test_ID) + "_depth.png")
        gt_mask = image.read_mask(dataset_dir + "gt/" + str(test_ID) + "_gt.png")
        pred_mask = denoise_mask(image.read_mask(dataset_dir + "pred/" + str(test_ID) + "_pred.png"))
        pts_gtmask = obj_depth2pts(-1, depth, gt_mask, my_camera, view_matrix)
        export_ply(pts=pts_gtmask,
                path=dataset_dir + "exported_ply/" + str(test_ID),
                pts_type="gtmask")
        estimate_pose(test_ID, depth, gt_mask, my_camera, view_matrix, pts_type="gtmask_transformed")
        estimate_pose(test_ID, depth, pred_mask, my_camera, view_matrix, pts_type="predmask_transformed")
        break
