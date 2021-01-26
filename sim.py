import pybullet as p
import pybullet_data
import numpy as np
import time


class PyBulletSim:
    """
    PyBulletSim: Implements two tote UR5 simulation environment with obstacles for grasping 
        and manipulation
    """
    def __init__(self, use_random_objects=False, object_shapes=None, gui=True):
        # 3D workspace for tote 1
        self._workspace1_bounds = np.array([
            [0.38, 0.62],  # 3x2 rows: x,y,z cols: min,max
            [-0.22, 0.22],
            [0.00, 0.5]
        ])
        # 3D workspace for tote 2
        self._workspace2_bounds = np.copy(self._workspace1_bounds)
        self._workspace2_bounds[0, :] = - self._workspace2_bounds[0, ::-1]

        # load environment
        if gui:
            p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self._plane_id = p.loadURDF("plane.urdf")
        p.setGravity(0, 0, -9.8)

        # load UR5 robot
        self.robot_body_id = p.loadURDF(
            "assets/ur5/ur5.urdf", [0, 0, 0.4], p.getQuaternionFromEuler([0, 0, 0]))
        self._mount_body_id = p.loadURDF(
            "assets/ur5/mount.urdf", [0, 0, 0.2], p.getQuaternionFromEuler([0, 0, 0]))

        # Placeholder for gripper body id
        self._gripper_body_id = None
        self.robot_end_effector_link_index = 9
        self._robot_tool_offset = [0, 0, -0.05]
        # Distance between tool tip and end-effector joint
        self._tool_tip_to_ee_joint = np.array([0, 0, 0.15])

        # Get revolute joint indices of robot (skip fixed joints)
        robot_joint_info = [p.getJointInfo(self.robot_body_id, i) for i in range(
            p.getNumJoints(self.robot_body_id))]
        self._robot_joint_indices = [
            x[0] for x in robot_joint_info if x[2] == p.JOINT_REVOLUTE]

        # joint position threshold in radians (i.e. move until joint difference < epsilon)
        self._joint_epsilon = 1e-3

        # Robot home joint configuration (over tote 1)
        self.robot_home_joint_config = [-np.pi, -
                                        np.pi/2, np.pi/2, -np.pi/2, -np.pi/2, 0]
        # Robot goal joint configuration (over tote 2)
        self.robot_goal_joint_config = [
            0, -np.pi/2, np.pi/2, -np.pi/2, -np.pi/2, 0]

        self.move_joints(self.robot_home_joint_config, speed=1.0)


        # Load totes and fix them to their position
        self._tote1_position = (
            self._workspace1_bounds[:, 0] + self._workspace1_bounds[:, 1]) / 2
        self._tote1_position[2] = 0.01
        self._tote1_body_id = p.loadURDF(
            "assets/tote/toteA_large.urdf", self._tote1_position, p.getQuaternionFromEuler([0, 0, 0]), useFixedBase=True)

        self._tote2_position = (
            self._workspace2_bounds[:, 0] + self._workspace2_bounds[:, 1]) / 2
        self._tote2_position[2] = 0.01
        self._tote2_body_id = p.loadURDF(
            "assets/tote/toteA_large.urdf", self._tote2_position, p.getQuaternionFromEuler([0, 0, 0]), useFixedBase=True)

        # Load objects
        # - possible object colors
        self._object_colors = get_tableau_palette()

        # - Define possible object shapes
        if object_shapes is not None:
            self._object_shapes = object_shapes
        else:
            self._object_shapes = [
                "assets/objects/cube.urdf",
                "assets/objects/rod.urdf",
                "assets/objects/custom.urdf",
            ]
        self._num_objects = len(self._object_shapes)
        self._object_shape_ids = [
            i % len(self._object_shapes) for i in range(self._num_objects)]
        self._objects_body_ids = []
        for i in range(self._num_objects):
            object_body_id = p.loadURDF(self._object_shapes[i], [ 0.5, 0.1, 0.1], p.getQuaternionFromEuler([0, 0, 0]))
            self._objects_body_ids.append(object_body_id)
            p.changeVisualShape(object_body_id, -1, rgbaColor=[*self._object_colors[i], 1])
        self.reset_objects()

        # Add obstacles
        self.obstacles = [
            p.loadURDF('assets/obstacles/block.urdf',
                       basePosition=[0, 0.65, 0.9],
                       useFixedBase=True
                       ),
            p.loadURDF('assets/obstacles/block.urdf',
                       basePosition=[0, 0.65, 0.3],
                       useFixedBase=True
                       ),
            p.loadURDF('assets/obstacles/block.urdf',
                       basePosition=[0, -0.65, 0.9],
                       useFixedBase=True
                       ),
            p.loadURDF('assets/obstacles/block.urdf',
                       basePosition=[0, -0.65, 0.3],
                       useFixedBase=True
                       ),
            p.loadURDF('assets/obstacles/block.urdf',
                       basePosition=[0, 0, 1.5],
                       useFixedBase=True
                       ),
        ]
        self.obstacles.extend(
            [self._plane_id, self._tote1_body_id, self._tote2_body_id])

    def load_gripper(self):
        if self._gripper_body_id is not None:
            print("Gripper already loaded")
            return

        # Attach robotiq gripper to UR5 robot
        # - We use createConstraint to add a fixed constraint between the ur5 robot and gripper.
        self._gripper_body_id = p.loadURDF("assets/gripper/robotiq_2f_85.urdf")
        p.resetBasePositionAndOrientation(self._gripper_body_id, [
                                          0.5, 0.1, 0.2], p.getQuaternionFromEuler([np.pi, 0, 0]))

        p.createConstraint(self.robot_body_id, self.robot_end_effector_link_index, self._gripper_body_id, 0, jointType=p.JOINT_FIXED, jointAxis=[
                           0, 0, 0], parentFramePosition=[0, 0, 0], childFramePosition=self._robot_tool_offset, childFrameOrientation=p.getQuaternionFromEuler([0, 0, np.pi/2]))

        # Set friction coefficients for gripper fingers
        for i in range(p.getNumJoints(self._gripper_body_id)):
            p.changeDynamics(self._gripper_body_id, i, lateralFriction=1.0, spinningFriction=1.0,
                             rollingFriction=0.0001, frictionAnchor=True)
        self.step_simulation(1e3)

    def move_joints(self, target_joint_state, speed=0.03):
        """
            Move robot arm to specified joint configuration by appropriate motor control
        """
        assert len(self._robot_joint_indices) == len(target_joint_state)
        p.setJointMotorControlArray(
            self.robot_body_id, self._robot_joint_indices,
            p.POSITION_CONTROL, target_joint_state,
            positionGains=speed * np.ones(len(self._robot_joint_indices))
        )

        timeout_t0 = time.time()
        while True:
            # Keep moving until joints reach the target configuration
            current_joint_state = [
                p.getJointState(self.robot_body_id, i)[0]
                for i in self._robot_joint_indices
            ]
            if all([
                np.abs(
                    current_joint_state[i] - target_joint_state[i]) < self._joint_epsilon
                for i in range(len(self._robot_joint_indices))
            ]):
                break
            if time.time()-timeout_t0 > 10:
                print(
                    "Timeout: robot is taking longer than 10s to reach the target joint state. Skipping...")
                p.setJointMotorControlArray(
                    self.robot_body_id, self._robot_joint_indices,
                    p.POSITION_CONTROL, self.robot_home_joint_config,
                    positionGains=np.ones(len(self._robot_joint_indices))
                )
                break
            self.step_simulation(1)

    def move_tool(self, position, orientation, speed=0.03):
        """
            Move robot tool (end-effector) to a specified pose
            @param position: Target position of the end-effector link
            @param orientation: Target orientation of the end-effector link
        """
        target_joint_state = np.zeros((6,))  # this should contain appropriate joint angle values
        # ========= Part 1 ========
        # Using inverse kinematics (p.calculateInverseKinematics), find out the target joint configuration of the robot
        # in order to reach the desired end_effector position and orientation
        # HINT: p.calculateInverseKinematics takes in the end effector **link index** and not the **joint index**. You can use 
        #   self.robot_end_effector_link_index for this 
        # HINT: You might want to tune optional parameters of p.calculateInverseKinematics for better performance
        # ===============================
        target_joint_state = p.calculateInverseKinematics(self.robot_body_id,
                                                          self.robot_end_effector_link_index,
                                                          position, orientation,
                                                          maxNumIterations=100, residualThreshold=1e-4)
        self.move_joints(target_joint_state)

    def robot_go_home(self, speed=0.1):
        self.move_joints(self.robot_home_joint_config, speed)

    def close_gripper(self):
        p.setJointMotorControl2(
            self._gripper_body_id, 1, p.VELOCITY_CONTROL, targetVelocity=5, force=10000)
        self.step_simulation(4e2)

    def open_gripper(self):
        p.setJointMotorControl2(
            self._gripper_body_id, 1, p.VELOCITY_CONTROL, targetVelocity=-5, force=10000)
        self.step_simulation(4e2)

    def check_grasp_success(self):
        return p.getJointState(self._gripper_body_id, 1)[0] < 0.834 - 0.001

    def execute_grasp(self, grasp_position, grasp_angle):
        """
            Execute grasp sequence
            @param: grasp_position: 3d position of place where the gripper jaws will be closed
            @param: grasp_angle: angle of gripper before executing grasp from positive x axis in radians 
        """
        # Adjust grasp_position to account for end-effector length
        grasp_position = grasp_position + self._tool_tip_to_ee_joint
        gripper_orientation = p.getQuaternionFromEuler(
            [np.pi, 0, grasp_angle])
        pre_grasp_position_over_bin = grasp_position+np.array([0, 0, 0.3])
        pre_grasp_position_over_object = grasp_position+np.array([0, 0, 0.1])
        post_grasp_position = grasp_position+np.array([0, 0, 0.3])
        grasp_success = False
        # ========= PART 2============
        # Implement the following grasp sequence:
        # 1. open gripper
        # 2. Move gripper to pre_grasp_position_over_bin
        # 3. Move gripper to pre_grasp_position_over_object
        # 4. Move gripper to grasp_position
        # 5. Close gripper
        # 6. Move gripper to post_grasp_position
        # 7. Move robot to robot_home_joint_config
        # 8. Detect whether or not the object was grasped and return grasp_success
        # ============================
        self.open_gripper()
        self.move_tool(pre_grasp_position_over_bin, None)
        self.move_tool(pre_grasp_position_over_object, gripper_orientation)
        self.move_tool(grasp_position, gripper_orientation)
        self.close_gripper()
        self.move_tool(post_grasp_position, None)
        self.robot_go_home(speed=0.03)
        grasp_success = self.check_grasp_success()
        return grasp_success

    def execute_place(self, place_angle=90.):
        gripper_orientation = p.getQuaternionFromEuler(
            [np.pi, 0, ((place_angle+180.) % 360.-180.)*np.pi/180.])
        place_position = np.array([0.4, -0.65, 0.4])
        self.move_tool(place_position, gripper_orientation, speed=0.01)
        self.open_gripper()

    def step_simulation(self, num_steps):
        for i in range(int(num_steps)):
            p.stepSimulation()
            if self._gripper_body_id is not None:
                # Constraints
                gripper_joint_positions = np.array([p.getJointState(self._gripper_body_id, i)[
                                                0] for i in range(p.getNumJoints(self._gripper_body_id))])
                p.setJointMotorControlArray(
                    self._gripper_body_id, [6, 3, 8, 5, 10], p.POSITION_CONTROL,
                    [
                        gripper_joint_positions[1], -gripper_joint_positions[1], 
                        -gripper_joint_positions[1], gripper_joint_positions[1],
                        gripper_joint_positions[1]
                    ],
                    positionGains=np.ones(5)
                )
            # time.sleep(1e-3)

    def reset_objects(self):
        for object_body_id in self._objects_body_ids:
            random_position = np.random.random_sample((3))*(self._workspace1_bounds[:, 1]-(
                self._workspace1_bounds[:, 0]+0.1))+self._workspace1_bounds[:, 0]+0.1
            random_orientation = np.random.random_sample((3))*2*np.pi-np.pi
            p.resetBasePositionAndOrientation(
                object_body_id, random_position, p.getQuaternionFromEuler(random_orientation))
        self.step_simulation(2e2)

    def set_joint_positions(self, values):
        assert len(self._robot_joint_indices) == len(values)
        for joint, value in zip(self._robot_joint_indices, values):
            p.resetJointState(self.robot_body_id, joint, value)

    def check_collision(self, q, distance=0.18):
        self.set_joint_positions(q)
        for obstacle_id in self.obstacles:
            closest_points = p.getClosestPoints(
                self.robot_body_id, obstacle_id, distance)
            if closest_points is not None and len(closest_points) != 0:
                return True
        return False


class SphereMarker:
    def __init__(self, position, radius=0.05, rgba_color=(1, 0, 0, 0.8), text=None, orientation=None, p_id=0):
        self.p_id = p_id
        position = np.array(position)
        vs_id = p.createVisualShape(
            p.GEOM_SPHERE, radius=radius, rgbaColor=rgba_color, physicsClientId=self.p_id)

        self.marker_id = p.createMultiBody(
            baseMass=0,
            baseInertialFramePosition=[0, 0, 0],
            baseCollisionShapeIndex=-1,
            baseVisualShapeIndex=vs_id,
            basePosition=position,
            useMaximalCoordinates=False
        )

        self.debug_item_ids = list()
        if text is not None:
            self.debug_item_ids.append(
                p.addUserDebugText(text, position + radius)
            )
        
        if orientation is not None:
            # x axis
            axis_size = 2 * radius
            rotation_mat = np.asarray(p.getMatrixFromQuaternion(orientation)).reshape(3,3)

            # x axis
            x_end = np.array([[axis_size, 0, 0]]).transpose()
            x_end = np.matmul(rotation_mat, x_end)
            x_end = position + x_end[:, 0]
            self.debug_item_ids.append(
                p.addUserDebugLine(position, x_end, lineColorRGB=(1, 0, 0))
            )
            # y axis
            y_end = np.array([[0, axis_size, 0]]).transpose()
            y_end = np.matmul(rotation_mat, y_end)
            y_end = position + y_end[:, 0]
            self.debug_item_ids.append(
                p.addUserDebugLine(position, y_end, lineColorRGB=(0, 1, 0))
            )
            # z axis
            z_end = np.array([[0, 0, axis_size]]).transpose()
            z_end = np.matmul(rotation_mat, z_end)
            z_end = position + z_end[:, 0]
            self.debug_item_ids.append(
                p.addUserDebugLine(position, z_end, lineColorRGB=(0, 0, 1))
            )

    def __del__(self):
        p.removeBody(self.marker_id, physicsClientId=self.p_id)
        for debug_item_id in self.debug_item_ids:
            p.removeUserDebugItem(debug_item_id)


def get_tableau_palette():
    """
    returns a beautiful color palette
    :return palette (np.array object): np array of rgb colors in range [0, 1]
    """
    palette = np.array(
        [
            [89, 169, 79],  # green
            [156, 117, 95],  # brown
            [237, 201, 72],  # yellow
            [78, 121, 167],  # blue
            [255, 87, 89],  # red
            [242, 142, 43],  # orange
            [176, 122, 161],  # purple
            [255, 157, 167],  # pink
            [118, 183, 178],  # cyan
            [186, 176, 172]  # gray
        ],
        dtype=np.float
    )
    return palette / 255.

