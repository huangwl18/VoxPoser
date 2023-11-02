import os
import numpy as np
import open3d as o3d
import json
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import ArmActionMode, EndEffectorPoseViaPlanning
from rlbench.action_modes.gripper_action_modes import Discrete, GripperActionMode
from rlbench.environment import Environment
import rlbench.tasks as tasks
from pyrep.const import ObjectType
from utils import normalize_vector, bcolors

class CustomMoveArmThenGripper(MoveArmThenGripper):
    """
    A potential workaround for the default MoveArmThenGripper as we frequently run into zero division errors and failed path.
    TODO: check the root cause of it.
    Ignore arm action if it fails.

    Attributes:
        _prev_arm_action (numpy.ndarray): Stores the previous arm action.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._prev_arm_action = None

    def action(self, scene, action):
        arm_act_size = np.prod(self.arm_action_mode.action_shape(scene))
        arm_action = np.array(action[:arm_act_size])
        ee_action = np.array(action[arm_act_size:])
        # if the arm action is the same as the previous action, skip it
        if self._prev_arm_action is not None and np.allclose(arm_action, self._prev_arm_action):
            self.gripper_action_mode.action(scene, ee_action)
        else:
            try:
                self.arm_action_mode.action(scene, arm_action)
            except Exception as e:
                print(f'{bcolors.FAIL}[rlbench_env.py] Ignoring failed arm action; Exception: "{str(e)}"{bcolors.ENDC}')
            self.gripper_action_mode.action(scene, ee_action)
        self._prev_arm_action = arm_action.copy()

class VoxPoserRLBench():
    def __init__(self, visualizer=None):
        """
        Initializes the VoxPoserRLBench environment.

        Args:
            visualizer: Visualization interface, optional.
        """
        action_mode = CustomMoveArmThenGripper(arm_action_mode=EndEffectorPoseViaPlanning(),
                                        gripper_action_mode=Discrete())
        self.rlbench_env = Environment(action_mode)
        self.rlbench_env.launch()
        self.task = None

        self.workspace_bounds_min = np.array([self.rlbench_env._scene._workspace_minx, self.rlbench_env._scene._workspace_miny, self.rlbench_env._scene._workspace_minz])
        self.workspace_bounds_max = np.array([self.rlbench_env._scene._workspace_maxx, self.rlbench_env._scene._workspace_maxy, self.rlbench_env._scene._workspace_maxz])
        self.visualizer = visualizer
        if self.visualizer is not None:
            self.visualizer.update_bounds(self.workspace_bounds_min, self.workspace_bounds_max)
        self.camera_names = ['front', 'left_shoulder', 'right_shoulder', 'overhead', 'wrist']
        # calculate lookat vector for all cameras (for normal estimation)
        name2cam = {
            'front': self.rlbench_env._scene._cam_front,
            'left_shoulder': self.rlbench_env._scene._cam_over_shoulder_left,
            'right_shoulder': self.rlbench_env._scene._cam_over_shoulder_right,
            'overhead': self.rlbench_env._scene._cam_overhead,
            'wrist': self.rlbench_env._scene._cam_wrist,
        }
        forward_vector = np.array([0, 0, 1])
        self.lookat_vectors = {}
        for cam_name in self.camera_names:
            extrinsics = name2cam[cam_name].get_matrix()
            lookat = extrinsics[:3, :3] @ forward_vector
            self.lookat_vectors[cam_name] = normalize_vector(lookat)
        # load file containing object names for each task
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'task_object_names.json')
        with open(path, 'r') as f:
            self.task_object_names = json.load(f)

        self._reset_task_variables()

    def get_object_names(self):
        """
        Returns the names of all objects in the current task environment.

        Returns:
            list: A list of object names.
        """
        name_mapping = self.task_object_names[self.task.get_name()]
        exposed_names = [names[0] for names in name_mapping]
        return exposed_names

    def load_task(self, task):
        """
        Loads a new task into the environment and resets task-related variables.
        Records the mask IDs of the robot, gripper, and objects in the scene.

        Args:
            task (str or rlbench.tasks.Task): Name of the task class or a task object.
        """
        self._reset_task_variables()
        if isinstance(task, str):
            task = getattr(tasks, task)
        self.task = self.rlbench_env.get_task(task)
        self.arm_mask_ids = [obj.get_handle() for obj in self.task._robot.arm.get_objects_in_tree(exclude_base=False)]
        self.gripper_mask_ids = [obj.get_handle() for obj in self.task._robot.gripper.get_objects_in_tree(exclude_base=False)]
        self.robot_mask_ids = self.arm_mask_ids + self.gripper_mask_ids
        self.obj_mask_ids = [obj.get_handle() for obj in self.task._task.get_base().get_objects_in_tree(exclude_base=False)]
        # store (object name <-> object id) mapping for relevant task objects
        try:
            name_mapping = self.task_object_names[self.task.get_name()]
        except KeyError:
            raise KeyError(f'Task {self.task.get_name()} not found in "envs/task_object_names.json" (hint: make sure the task and the corresponding object names are added to the file)')
        exposed_names = [names[0] for names in name_mapping]
        internal_names = [names[1] for names in name_mapping]
        scene_objs = self.task._task.get_base().get_objects_in_tree(object_type=ObjectType.SHAPE,
                                                                      exclude_base=False,
                                                                      first_generation_only=False)
        for scene_obj in scene_objs:
            if scene_obj.get_name() in internal_names:
                exposed_name = exposed_names[internal_names.index(scene_obj.get_name())]
                self.name2ids[exposed_name] = [scene_obj.get_handle()]
                self.id2name[scene_obj.get_handle()] = exposed_name
                for child in scene_obj.get_objects_in_tree():
                    self.name2ids[exposed_name].append(child.get_handle())
                    self.id2name[child.get_handle()] = exposed_name

    def get_3d_obs_by_name(self, query_name):
        """
        Retrieves 3D point cloud observations and normals of an object by its name.

        Args:
            query_name (str): The name of the object to query.

        Returns:
            tuple: A tuple containing object points and object normals.
        """
        assert query_name in self.name2ids, f"Unknown object name: {query_name}"
        obj_ids = self.name2ids[query_name]
        # gather points and masks from all cameras
        points, masks, normals = [], [], []
        for cam in self.camera_names:
            points.append(getattr(self.latest_obs, f"{cam}_point_cloud").reshape(-1, 3))
            masks.append(getattr(self.latest_obs, f"{cam}_mask").reshape(-1))
            # estimate normals using o3d
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points[-1])
            pcd.estimate_normals()
            cam_normals = np.asarray(pcd.normals)
            # use lookat vector to adjust normal vectors
            flip_indices = np.dot(cam_normals, self.lookat_vectors[cam]) > 0
            cam_normals[flip_indices] *= -1
            normals.append(cam_normals)
        points = np.concatenate(points, axis=0)
        masks = np.concatenate(masks, axis=0)
        normals = np.concatenate(normals, axis=0)
        # get object points
        obj_points = points[np.isin(masks, obj_ids)]
        if len(obj_points) == 0:
            raise ValueError(f"Object {query_name} not found in the scene")
        obj_normals = normals[np.isin(masks, obj_ids)]
        # voxel downsample using o3d
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(obj_points)
        pcd.normals = o3d.utility.Vector3dVector(obj_normals)
        pcd_downsampled = pcd.voxel_down_sample(voxel_size=0.001)
        obj_points = np.asarray(pcd_downsampled.points)
        obj_normals = np.asarray(pcd_downsampled.normals)
        return obj_points, obj_normals

    def get_scene_3d_obs(self, ignore_robot=False, ignore_grasped_obj=False):
        """
        Retrieves the entire scene's 3D point cloud observations and colors.

        Args:
            ignore_robot (bool): Whether to ignore points corresponding to the robot.
            ignore_grasped_obj (bool): Whether to ignore points corresponding to grasped objects.

        Returns:
            tuple: A tuple containing scene points and colors.
        """
        points, colors, masks = [], [], []
        for cam in self.camera_names:
            points.append(getattr(self.latest_obs, f"{cam}_point_cloud").reshape(-1, 3))
            colors.append(getattr(self.latest_obs, f"{cam}_rgb").reshape(-1, 3))
            masks.append(getattr(self.latest_obs, f"{cam}_mask").reshape(-1))
        points = np.concatenate(points, axis=0)
        colors = np.concatenate(colors, axis=0)
        masks = np.concatenate(masks, axis=0)

        # only keep points within workspace
        chosen_idx_x = (points[:, 0] > self.workspace_bounds_min[0]) & (points[:, 0] < self.workspace_bounds_max[0])
        chosen_idx_y = (points[:, 1] > self.workspace_bounds_min[1]) & (points[:, 1] < self.workspace_bounds_max[1])
        chosen_idx_z = (points[:, 2] > self.workspace_bounds_min[2]) & (points[:, 2] < self.workspace_bounds_max[2])
        points = points[(chosen_idx_x & chosen_idx_y & chosen_idx_z)]
        colors = colors[(chosen_idx_x & chosen_idx_y & chosen_idx_z)]
        masks = masks[(chosen_idx_x & chosen_idx_y & chosen_idx_z)]

        if ignore_robot:
            robot_mask = np.isin(masks, self.robot_mask_ids)
            points = points[~robot_mask]
            colors = colors[~robot_mask]
            masks = masks[~robot_mask]
        if self.grasped_obj_ids and ignore_grasped_obj:
            grasped_mask = np.isin(masks, self.grasped_obj_ids)
            points = points[~grasped_mask]
            colors = colors[~grasped_mask]
            masks = masks[~grasped_mask]

        # voxel downsample using o3d
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        pcd_downsampled = pcd.voxel_down_sample(voxel_size=0.001)
        points = np.asarray(pcd_downsampled.points)
        colors = np.asarray(pcd_downsampled.colors).astype(np.uint8)

        return points, colors

    def reset(self):
        """
        Resets the environment and the task. Also updates the visualizer.

        Returns:
            tuple: A tuple containing task descriptions and initial observations.
        """
        assert self.task is not None, "Please load a task first"
        self.task.sample_variation()
        descriptions, obs = self.task.reset()
        obs = self._process_obs(obs)
        self.init_obs = obs
        self.latest_obs = obs
        self._update_visualizer()
        return descriptions, obs

    def apply_action(self, action):
        """
        Applies an action in the environment and updates the state.

        Args:
            action: The action to apply.

        Returns:
            tuple: A tuple containing the latest observations, reward, and termination flag.
        """
        assert self.task is not None, "Please load a task first"
        action = self._process_action(action)
        obs, reward, terminate = self.task.step(action)
        obs = self._process_obs(obs)
        self.latest_obs = obs
        self.latest_reward = reward
        self.latest_terminate = terminate
        self.latest_action = action
        self._update_visualizer()
        grasped_objects = self.rlbench_env._scene.robot.gripper.get_grasped_objects()
        if len(grasped_objects) > 0:
            self.grasped_obj_ids = [obj.get_handle() for obj in grasped_objects]
        return obs, reward, terminate

    def move_to_pose(self, pose, speed=None):
        """
        Moves the robot arm to a specific pose.

        Args:
            pose: The target pose.
            speed: The speed at which to move the arm. Currently not implemented.

        Returns:
            tuple: A tuple containing the latest observations, reward, and termination flag.
        """
        if self.latest_action is None:
            action = np.concatenate([pose, [self.init_obs.gripper_open]])
        else:
            action = np.concatenate([pose, [self.latest_action[-1]]])
        return self.apply_action(action)
    
    def open_gripper(self):
        """
        Opens the gripper of the robot.
        """
        action = np.concatenate([self.latest_obs.gripper_pose, [1.0]])
        return self.apply_action(action)

    def close_gripper(self):
        """
        Closes the gripper of the robot.
        """
        action = np.concatenate([self.latest_obs.gripper_pose, [0.0]])
        return self.apply_action(action)

    def set_gripper_state(self, gripper_state):
        """
        Sets the state of the gripper.

        Args:
            gripper_state: The target state for the gripper.

        Returns:
            tuple: A tuple containing the latest observations, reward, and termination flag.
        """
        action = np.concatenate([self.latest_obs.gripper_pose, [gripper_state]])
        return self.apply_action(action)

    def reset_to_default_pose(self):
        """
        Resets the robot arm to its default pose.

        Returns:
            tuple: A tuple containing the latest observations, reward, and termination flag.
        """
        if self.latest_action is None:
            action = np.concatenate([self.init_obs.gripper_pose, [self.init_obs.gripper_open]])
        else:
            action = np.concatenate([self.init_obs.gripper_pose, [self.latest_action[-1]]])
        return self.apply_action(action)

    def get_ee_pose(self):
        assert self.latest_obs is not None, "Please reset the environment first"
        return self.latest_obs.gripper_pose

    def get_ee_pos(self):
        return self.get_ee_pose()[:3]

    def get_ee_quat(self):
        return self.get_ee_pose()[3:]

    def get_last_gripper_action(self):
        """
        Returns the last gripper action.

        Returns:
            float: The last gripper action.
        """
        if self.latest_action is not None:
            return self.latest_action[-1]
        else:
            return self.init_obs.gripper_open

    def _reset_task_variables(self):
        """
        Resets variables related to the current task in the environment.

        Note: This function is generally called internally.
        """
        self.init_obs = None
        self.latest_obs = None
        self.latest_reward = None
        self.latest_terminate = None
        self.latest_action = None
        self.grasped_obj_ids = None
        # scene-specific helper variables
        self.arm_mask_ids = None
        self.gripper_mask_ids = None
        self.robot_mask_ids = None
        self.obj_mask_ids = None
        self.name2ids = {}  # first_generation name -> list of ids of the tree
        self.id2name = {}  # any node id -> first_generation name
   
    def _update_visualizer(self):
        """
        Updates the scene in the visualizer with the latest observations.

        Note: This function is generally called internally.
        """
        if self.visualizer is not None:
            points, colors = self.get_scene_3d_obs(ignore_robot=False, ignore_grasped_obj=False)
            self.visualizer.update_scene_points(points, colors)
    
    def _process_obs(self, obs):
        """
        Processes the observations, specifically converts quaternion format from xyzw to wxyz.

        Args:
            obs: The observation to process.

        Returns:
            The processed observation.
        """
        quat_xyzw = obs.gripper_pose[3:]
        quat_wxyz = np.concatenate([quat_xyzw[-1:], quat_xyzw[:-1]])
        obs.gripper_pose[3:] = quat_wxyz
        return obs

    def _process_action(self, action):
        """
        Processes the action, specifically converts quaternion format from wxyz to xyzw.

        Args:
            action: The action to process.

        Returns:
            The processed action.
        """
        quat_wxyz = action[3:7]
        quat_xyzw = np.concatenate([quat_wxyz[1:], quat_wxyz[:1]])
        action[3:7] = quat_xyzw
        return action