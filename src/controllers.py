import numpy as np
from transforms3d.quaternions import mat2quat
from utils import normalize_vector
import copy
import time
from dynamics_models import PushingDynamicsModel

# creating some aliases for end effector and table in case LLMs refer to them differently
EE_ALIAS = ['ee', 'endeffector', 'end_effector', 'end effector', 'gripper', 'hand']

class Controller:
    def __init__(self, env, config):
        self.config = config
        self.env = env
        self.dynamics_model = PushingDynamicsModel()
    
    def _calculate_ee_rot(self, pushing_dir):
        """
        Given a pushing direction, calculate the rotation matrix for the end effector
        It is offsetted such that it doesn't exactly point towards the direction but slanted towards table, so it's safer
        """
        pushing_dir = normalize_vector(pushing_dir)
        desired_dir = pushing_dir + np.array([0, 0, -np.linalg.norm(pushing_dir)])
        desired_dir = normalize_vector(desired_dir)
        left = np.cross(pushing_dir, desired_dir)
        left = normalize_vector(left)
        up = np.array(desired_dir, dtype=np.float32)
        up = normalize_vector(up)
        forward = np.cross(left, up)
        forward = normalize_vector(forward)
        rotmat = np.eye(3).astype(np.float32)
        rotmat[:3, 0] = forward
        rotmat[:3, 1] = left
        rotmat[:3, 2] = up
        quat_wxyz = mat2quat(rotmat)
        return quat_wxyz
    
    def _apply_mpc_control(self, control, target_velocity=1):
        """
        apply control to the object; depending on different control type
        """
        # calculate start and final ee pose
        contact_position = control[:3]  # [3]
        pushing_dir = control[3:6]  # [3]
        pushing_dist = control[6]  # [1]
        # calculate a safe end effector rotation
        ee_quat = self._calculate_ee_rot(pushing_dir)
        # calculate translation
        start_dist = 0.08
        t_start = contact_position - pushing_dir * start_dist
        t_interact = contact_position + pushing_dir * pushing_dist
        t_rest = contact_position - pushing_dir * start_dist * 0.8

        # apply control
        self.env.close_gripper()
        # move to start pose
        self.env.move_to_pose(np.concatenate([t_start, ee_quat]), speed=target_velocity)
        print('[controllers.py] moved to start pose', end='; ')
        # move to interact pose
        self.env.move_to_pose(np.concatenate([t_interact, ee_quat]), speed=target_velocity * 0.2)
        print('[controllers.py] moved to final pose', end='; ')
        # back to rest pose
        self.env.move_to_pose(np.concatenate([t_rest, ee_quat]), speed=target_velocity * 0.33)
        print('[controllers.py] back to release pose', end='; ')
        self.env.reset_to_default_pose()
        print('[controllers.py] back togenerate_random_control default pose', end='')
        print()

    def execute(self, movable_obs, waypoint):
        """
        execute a waypoint
        If movable is "end effector", then do not consider object interaction (no dynamics considered)
        If movable is "object", then consider object interaction (use heuristics-based dynamics model)

        :param movable_obs: observation dict of the object to be moved
        :param waypoint: list, [target_xyz, target_rotation, target_velocity, target_gripper], target_xyz is for movable in world frame
        :return: None
        """
        info = dict()
        target_xyz, target_rotation, target_velocity, target_gripper = waypoint
        object_centric = (not movable_obs['name'].lower() in EE_ALIAS)
        # move to target pose directly
        if not object_centric:
            target_pose = np.concatenate([target_xyz, target_rotation])
            result = self.env.apply_action(np.concatenate([target_pose, [target_gripper]]))
            info['mp_info'] = result
        # optimize through dynamics model to obtain robot actions
        else:
            start = time.time()
            # sample control sequence using MPC
            movable_obs = {key: value for key, value in movable_obs.items() if key in ['_point_cloud_world']}
            best_control, self.mpc_info = self.random_shooting_MPC(movable_obs, target_xyz)  # [7]
            print('[controllers.py] mpc search completed in {} seconds with {} samples'.format(time.time() - start, self.config.num_samples))
            # apply first control in the sequence
            self.mpc_velocity = target_velocity
            self._apply_mpc_control(best_control[0])
            print(f'[controllers.py] applied control (pos: {best_control[0][:3].round(4)}, dir: {best_control[0][3:6].round(4)}, dist: {best_control[0][6:].round(4)})')
            info['mpc_info'] = self.mpc_info
            info['mpc_control'] = best_control[0]
        return info

    def random_shooting_MPC(self, start_obs, target):
        # Initialize empty list to store the control sequence and corresponding cost
        obs_sequences = []
        controls_sequences = []
        costs = []
        info = dict()
        # repeat the observation for the number of samples (non-batched -> batched)
        batched_start_obs = {}
        for key, value in start_obs.items():
            batched_start_obs[key] = np.repeat(value[None, ...], self.config.num_samples, axis=0)
        obs_sequences.append(batched_start_obs)
        # Generate random control sequences
        for t in range(self.config.horizon_length):
            curr_obs = copy.deepcopy(obs_sequences[-1])
            controls = self.generate_random_control(curr_obs, target)
            # Simulate the system using the model and the generated control sequence
            pred_next_obs = self.forward_step(curr_obs, controls)
            # record the control sequence and the resulting observation
            obs_sequences.append(pred_next_obs)  # obs_sequences: [T, N, obs_dim]
            controls_sequences.append(controls)  # controls_sequences: [T, N, control_dim]
        # Calculate the cost of the generated control sequence
        costs = self.calculate_cost(obs_sequences, controls_sequences, target)  # [N]
        # Find the control sequence with the lowest cost
        best_traj_idx = np.argmin(costs)
        best_controls_sequence = np.array([control_per_step[best_traj_idx] for control_per_step in controls_sequences])  # [T, control_dim]
        # log info
        info['best_controls_sequence'] = best_controls_sequence
        info['best_cost'] = costs[best_traj_idx]
        info['costs'] = costs
        info['controls_sequences'] = controls_sequences
        info['obs_sequences'] = obs_sequences
        return best_controls_sequence, info

    def forward_step(self, obs, controls):
        """
        obs: dict including point cloud [B, N, obs_dim]
        controls: batched control sequences [B, control_dim]
        returns: resulting point cloud [B, N, obs_dim]
        """
        # forward dynamics
        pcs = obs['_point_cloud_world']  # [B, N, 3]
        contact_position = controls[:, :3]  # [B, 3]
        pushing_dir = controls[:, 3:6]  # [B, 3]
        pushing_dist = controls[:, 6:]  # [B, 1]
        inputs = (pcs, contact_position, pushing_dir, pushing_dist)
        next_pcs = self.dynamics_model.forward(inputs)  # [B, N, 3]
        # assemble next_obs
        next_obs = copy.deepcopy(obs)
        next_obs['_point_cloud_world'] = next_pcs
        return next_obs

    def generate_random_control(self, obs, target):
        """
        the function samples the following:
        1) contact_position [B, 3]: uniform sample randomly from object point cloud
        2) pushing_dir [B, 3]: fixed to be the direction from contact_position to target
        3) pushing_dist [B, 1]: uniform sampling from some range

        returns: batched control sequences [B, 7] (3 for contact position, 3 for gripper direction, 1 for gripper moving distance)
        """
        pcs = obs['_point_cloud_world']  # [B, N, 3]
        num_samples, num_points, _ = pcs.shape
        # sample contact position randomly on point cloud
        points_idx = np.random.randint(0, num_points, num_samples)
        contact_positions = pcs[np.arange(num_samples), points_idx]  # [B, 3]
        # sample pushing_dir
        pushing_dirs = target - contact_positions  # [B, 3]
        pushing_dirs = normalize_vector(pushing_dirs)
        # sample pushing_dist
        pushing_dist = np.random.uniform(-0.02, 0.09, size=(num_samples, 1))  # [B, 1]
        # assemble control sequences
        controls = np.concatenate([contact_positions, pushing_dirs, pushing_dist], axis=1)  # [B, 7]
        return controls

    def calculate_cost(self, obs_sequences, controls_sequences, target_xyz):
        """
        Calculate the cost of the generated control sequence

        inputs:
        obs_sequences: batched observation sequences [T, B, N, 3]
        controls_sequences: batched control sequences [T, B, 7]

        returns: cost [B, 1]
        """
        num_samples, _, _ = obs_sequences[0]['_point_cloud_world'].shape
        last_obs = obs_sequences[-1]
        costs = []
        for i in range(num_samples):
            last_pc = last_obs['_point_cloud_world'][i]  # [N, 3]
            last_position = np.mean(last_pc, axis=0)  # [3]
            cost = np.linalg.norm(last_position - target_xyz)
            costs.append(cost)
        costs = np.array(costs)  # [B]
        return costs