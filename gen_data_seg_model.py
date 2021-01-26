import sim
from random import seed
import os
import camera
import pybullet as p
import numpy as np
import image
from tqdm import tqdm


if __name__ == "__main__":
    seed(0)
    np.random.seed(0)

    # setup an appropriate data size
    #   (appropriate == which you think is good enough for training and testing)
    # ==================================================================================
    dataset_size = 300
    # ===============================================================================

    object_shapes = [
        "assets/objects/cube.urdf",
        "assets/objects/rod.urdf",
        "assets/objects/custom.urdf",
    ]

    env = sim.PyBulletSim(object_shapes = object_shapes, gui=False)

    # setup camera
    my_camera = camera.Camera(
        image_size=(480, 640),
        near=0.01,
        far=10.0,
        fov_w=50
    )
    camera_target_position = (env._workspace1_bounds[:, 0] + env._workspace1_bounds[:, 1]) / 2
    camera_target_position[2] = 0
    camera_distance = np.sqrt(((np.array([0.5, -0.5, 0.8]) - camera_target_position)**2).sum())
    view_matrix = p.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=camera_target_position,
        distance=camera_distance,
        yaw=90,
        pitch=-60,
        roll=0,
        upAxisIndex=2,
    )

    obj_ids = env._objects_body_ids  # everything else will be treated as background

    # Create the dataset
    dataset_dir = "dataset"
    rgb_dir = os.path.join(dataset_dir, "rgb")
    if not os.path.exists(rgb_dir):
        os.makedirs(rgb_dir)
    depth_dir = os.path.join(dataset_dir, "depth")
    if not os.path.exists(depth_dir):
        os.makedirs(depth_dir)
    mask_dir = os.path.join(dataset_dir, "gt")
    if not os.path.exists(mask_dir):
        os.makedirs(mask_dir)

    for i in tqdm(range(dataset_size)):
        rgb_obs, depth_obs, mask_obs = camera.make_obs(my_camera, view_matrix)
        rgb_name = f"{dataset_dir}/rgb/{i}_rgb.png"
        depth_name = f"{dataset_dir}/depth/{i}_depth.png"
        mask_name = f"{dataset_dir}/gt/{i}_gt.png"
        image.write_rgb(rgb_obs.astype(np.uint8), rgb_name)
        image.write_depth(depth_obs, depth_name)

        # process mask
        indices_covered = np.zeros_like(mask_obs, dtype=np.bool)
        for obj_index, obj_id in enumerate(obj_ids):
            obj_pixel_indices = (mask_obs == obj_id)
            mask_obs[obj_pixel_indices] = obj_index + 1
            indices_covered |= obj_pixel_indices
        mask_obs[~indices_covered] = 0
        image.write_mask(mask_obs, mask_name)
        env.reset_objects()