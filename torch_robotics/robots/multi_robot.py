import matplotlib.pyplot as plt
import numpy as np
import torch
from itertools import chain
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from torch_robotics import environments, robots
from torch_robotics.environments.primitives import plot_sphere
from torch_robotics.robots.robot_base import RobotBase
from torch_robotics.robots.robot_point_mass import RobotPointMass
from torch_robotics.torch_utils.torch_utils import to_numpy, to_torch

import matplotlib.collections as mcoll
from scipy.spatial.transform import Rotation as R
from corallab_sim.utilities.spatial import get_transform, transform_point
from copy import deepcopy


def apply_transform(pose, points, tensor_args):
    if pose is None:
        return points
    else:
        cpu_pose = pose.clone().cpu()
        t = torch.tensor(get_transform(pos=cpu_pose[:3], rotq=cpu_pose[3:]), **tensor_args)
        homogenous_points = torch.nn.functional.pad(points, (0, 1), mode="constant", value=1)
        homogenous_points_prime = torch.matmul(t, homogenous_points.mT)
        points_prime = homogenous_points_prime.mT[..., :3]
        return points_prime


class MultiRobot(RobotBase):

    def __init__(self,
                 name='MultiRobot',
                 subrobot_id=None,
                 subrobot_count=1,
                 subrobots=[],
                 include_base_poses=False,
                 tensor_args=None,
                 **kwargs):

        if subrobot_id is not None:
            subrobot_class = getattr(robots, subrobot_id)
            self.subrobots = [subrobot_class(tensor_args=tensor_args) for _ in range(subrobot_count)]
        elif len(subrobots) >= 0:
            self.subrobots = subrobots
        else:
            raise ValueError("Need to provide subrobot_id or subrobots")

        q_limits = torch.hstack([robot.q_limits for robot in self.subrobots])
        self.n_subrobots = len(self.subrobots)

        self.include_base_poses = include_base_poses

        # default base poses from which FK originates
        default_pose = None
        self.base_poses = [default_pose] * self.n_subrobots
        self.base_pose_q_dim = 4

        # free q selector
        free_q_selector_offset = 0
        free_q_selector_elements = []
        fixed_q_selector_offset = 0
        fixed_q_selector_elements = []

        for robot in self.subrobots:
            fixed_q_selector_length = fixed_q_selector_offset + self.base_pose_q_dim
            fixed_q_selector_elements.extend(range(fixed_q_selector_offset, fixed_q_selector_length))
            fixed_q_selector_offset += fixed_q_selector_length + robot.q_dim

            free_q_selector_offset += self.base_pose_q_dim
            free_q_selector_length = free_q_selector_offset + robot.q_dim
            free_q_selector_elements.extend(range(free_q_selector_offset, free_q_selector_length))
            free_q_selector_offset = free_q_selector_length

        self.free_q_selector = torch.tensor(free_q_selector_elements)
        self.fixed_q_selector = torch.tensor(fixed_q_selector_elements)
        self.fixed_and_free_q_dim = free_q_selector_length

        # collision checking
        link_names_for_object_collision_checking = []
        for i, robot in enumerate(self.subrobots):
            link_names_for_object_collision_checking.extend(
                [f"r{i}_{name}" for name in robot.link_names_for_object_collision_checking]
            )

        link_margins_for_object_collision_checking = []
        for robot in self.subrobots:
            link_margins_for_object_collision_checking.extend(
                robot.link_margins_for_object_collision_checking
            )

        link_margins_for_self_collision_checking = []
        for robot in self.subrobots:
            link_margins_for_self_collision_checking.extend(
                robot.link_margins_for_self_collision_checking
            )

        link_idx_offset = 0
        link_idxs_for_object_collision_checking = []
        for robot in self.subrobots:
            link_idxs_for_object_collision_checking.extend(
                [link_idx_offset + idx for idx in robot.link_idxs_for_object_collision_checking]
            )
            link_idx_offset = len(link_idxs_for_object_collision_checking)

        # TODO: Figure out
        num_interpolated_points_for_object_collision_checking = len(link_names_for_object_collision_checking)

        # TODO: Self Collision Checking

        # A multi robot should always have self collision checking because it is
        # necessarily contains multiple links.

        link_names_for_self_collision_checking = []
        link_names_pairs_for_self_collision_checking = {}
        link_names_for_all_pairs = []
        link_idxs_for_self_collision_checking = []
        link_idx_offset = 0
        link_names_for_self_collision_checking_with_grasped_object = []
        self_collision_margin_robot = 0.05

        for i, robot in enumerate(self.subrobots):
            if robot.df_collision_self is None:
                # If the subrobot has no self collision field, it is only a
                # single link. Then the multi robot's self collision field
                # should consider it.
                new_links = [f"r{i}_{name}" for name in robot.link_names_for_object_collision_checking]
                link_names_for_self_collision_checking.extend(new_links)

                if len(link_names_for_all_pairs) > 0:
                    for link in new_links:
                        link_names_pairs_for_self_collision_checking[link] = deepcopy(link_names_for_all_pairs)

                link_names_for_all_pairs.extend(new_links)

                new_link_idxs = [link_idx_offset + idx for idx in robot.link_idxs_for_object_collision_checking]
                link_idxs_for_self_collision_checking.extend(new_link_idxs)

                # TODO: Intelligently determine self collision margin from subrobots
                self_collision_margin_robot = max(self_collision_margin_robot, 0.1)
            else:
                # If the subrobot has a self collision field, then it should
                # incorporate that subrobot's own self collision info into the
                # new field, in addition to considering collisions with the
                # other subrobots.

                new_links = [f"r{i}_{name}" for name in robot.link_names_for_self_collision_checking]
                link_names_for_self_collision_checking.extend(new_links)

                for k, v in robot.link_names_pairs_for_self_collision_checking.items():
                    r_k = f"r{i}_{k}"
                    r_vs = [f"r{i}_{link_name}" for link_name in v]

                    link_names_pairs_for_self_collision_checking[r_k] = [*r_vs, *link_names_for_all_pairs]

                link_names_for_all_pairs.extend(new_links)

                new_link_idxs = [link_idx_offset + idx for idx in robot.link_idxs_for_self_collision_checking]
                link_idxs_for_self_collision_checking.extend(new_link_idxs)

                self_collision_margin_robot = max(self_collision_margin_robot, 0.15)

            link_idx_offset += robot.get_num_links()

        # TODO: Figure out
        num_interpolated_points_for_self_collision_checking = len(link_names_for_self_collision_checking)

        self.total_num_links = link_idx_offset

        super().__init__(
            name=name,
            q_limits=q_limits,
            # object collision
            link_names_for_object_collision_checking=link_names_for_object_collision_checking,
            link_margins_for_object_collision_checking=link_margins_for_object_collision_checking,
            link_idxs_for_object_collision_checking=link_idxs_for_object_collision_checking,
            num_interpolated_points_for_object_collision_checking=num_interpolated_points_for_object_collision_checking,
            # self collision
            link_names_for_self_collision_checking=link_names_for_self_collision_checking,
            link_margins_for_self_collision_checking=link_margins_for_self_collision_checking,
            link_names_pairs_for_self_collision_checking=link_names_pairs_for_self_collision_checking,
            link_idxs_for_self_collision_checking=link_idxs_for_self_collision_checking,
            num_interpolated_points_for_self_collision_checking=num_interpolated_points_for_self_collision_checking,
            self_collision_margin_robot=self_collision_margin_robot,
            # misc
            tensor_args=tensor_args,
            **kwargs
        )

    def fk_map_collision_impl(self, q, **kwargs):
        q_offset = 0
        subrobot_fks = []

        q = self.safe_select_free_q(q)

        for base_pose, robot in zip(self.base_poses, self.subrobots):
            subrobot_q = q[..., q_offset:q_offset+robot.q_dim]
            subrobot_fk = robot.fk_map_collision_impl(subrobot_q, **kwargs)
            subrobot_fk = apply_transform(base_pose, subrobot_fk, self.tensor_args)
            subrobot_fks.append(subrobot_fk)
            q_offset += robot.q_dim

        # Second to last dim across links
        res = torch.concatenate(subrobot_fks, axis=-2)
        return res

    def get_num_links(self):
        return self.total_num_links

    def render(self, ax, q=None, color='blue', cmap='Blues', margin_multiplier=1., **kwargs):
        q_offset = 0
        subrobot_fks = []
        for robot in self.subrobots:
            subrobot_q = q[..., q_offset:q_offset+robot.q_dim]
            robot.render(ax, q=subrobot_q)
            q_offset += robot.q_dim

    def render_trajectories(
            self, ax, trajs=None, start_state=None, goal_state=None, colors=['blue'],
            linestyle='solid', **kwargs):

        # trajs = trajs.clone().cpu()
        # start_state = start_state.clone().cpu()
        # goal_state = goal_state.clone().cpu()

        has_velocity = False
        trajs = self.safe_select_free_q(trajs)

        if trajs.shape[-1] < self.q_dim:
            raise Exception("Invalid trajectory dimension (too small)")
        elif trajs.shape[-1] == self.q_dim:
            has_velocity = False
        elif trajs.shape[-1] == 2 * self.q_dim:
            has_velocity = True
        else:
            raise Exception("Invalid trajectory dimension (too large)")

        q_offset = 0
        subrobot_fks = []
        for robot in self.subrobots:
            # Select subset of trajectory for current robot
            position_selector = np.arange(q_offset, q_offset+robot.q_dim)
            if has_velocity:
                velocity_selector = np.arange(self.q_dim + q_offset, self.q_dim + q_offset+robot.q_dim)
                subrobot_selector = np.hstack((position_selector, velocity_selector))
            else:
                subrobot_selector = position_selector

            subrobot_trajs = trajs[..., subrobot_selector]

            # Select subset of start state for current robot
            if start_state is not None:
                subrobot_start_state = start_state[position_selector]
            else:
                subrobot_start_state = None

            # Select subset of goal state for current robot
            if goal_state is not None:
                subrobot_goal_state = goal_state[position_selector]
            else:
                subrobot_goal_state = None

            colors = colors * trajs.shape[0]

            robot.render_trajectories(ax, trajs=subrobot_trajs,
                                      start_state=subrobot_start_state,
                                      goal_state=subrobot_goal_state,
                                      colors=colors,
                                      linestyle=linestyle)

            q_offset += robot.q_dim

    def update_base_poses(self, base_poses):
        """Update bases for every robot panda subrobot."""
        self.base_poses = base_poses

    def get_base_poses(self):
        """Get bases for every robot panda subrobot."""
        return self.base_poses

    def insert_base_pose_qs(self, qs, has_velocity=True):
        if has_velocity:
            new_q_shape = (*qs.shape[:-1], 2 * self.fixed_and_free_q_dim)
        else:
            new_q_shape = (*qs.shape[:-1], self.fixed_and_free_q_dim)

        new_q_arr = torch.zeros(new_q_shape, **self.tensor_args)

        base_pose_qs_list = [self.q_from_pose(pose) for pose in self.get_base_poses()]
        base_pose_qs = torch.concatenate(base_pose_qs_list)

        new_q_arr[..., self.fixed_q_selector] = base_pose_qs

        if has_velocity:
            velocity_selector = self.fixed_and_free_q_dim + self.free_q_selector
            selector = torch.concatenate((self.free_q_selector, velocity_selector))
        else:
            selector = self.free_q_selector

        new_q_arr[..., selector] = qs
        return new_q_arr

    def remove_base_pose_qs(self, qs, has_velocity=True):
        if has_velocity:
            velocity_selector = self.fixed_and_free_q_dim + self.free_q_selector
            selector = torch.concatenate((self.free_q_selector, velocity_selector))
        else:
            selector = self.free_q_selector

        return qs[..., selector]


    def full_q_idx_to_free_q_idx(self, i):
        if i in self.fixed_q_selector:
            return None
        else:
            return self.free_q_selector.tolist().index(i)

    def q_from_pose(self, pose):
        position = pose[:3]
        theta = torch.tensor(R.from_quat(pose[3:].cpu()).as_euler("xyz")[2:], **self.tensor_args)
        return torch.concatenate((position, theta))

    def pose_from_q(self, q):
        position = q[:3]
        theta = q[3].cpu().item()
        quat = R.from_euler("xyz", [0., 0., theta]).as_quat()
        quat_tensor = torch.tensor(quat, **self.tensor_args)
        return torch.concatenate((position, quat_tensor))

    def all_poses_from_q(self, q):
        """Get bases for every robot panda subrobot."""
        poses = []
        base_q = q[self.fixed_q_selector]
        offset = 0

        for robot in self.subrobots:
            subrobot_base_q = base_q[offset:offset+self.base_pose_q_dim]
            pose = self.pose_from_q(subrobot_base_q)
            poses.append(pose)
            offset += self.base_pose_q_dim

        return poses

    # overrides
    def get_position(self, x):
        if self.include_base_poses:
            return x[..., :self.fixed_and_free_q_dim]
        else:
            return x[..., :self.q_dim]

    def get_velocity(self, x):
        if self.include_base_poses:
            return x[..., self.fixed_and_free_q_dim:]
        else:
            return x[..., self.q_dim:]

    def safe_select_free_q(self, q):
        if self.include_base_poses:
            if q.shape[-1] == self.fixed_and_free_q_dim:
                return self.remove_base_pose_qs(q, has_velocity=False)
            elif q.shape[-1] == 2 * self.fixed_and_free_q_dim:
                return self.remove_base_pose_qs(q, has_velocity=True)
            elif q.shape[-1] == self.q_dim:
                return q
            else:
                raise Exception("Input has unexpected dimensions")
        else:
            return q
