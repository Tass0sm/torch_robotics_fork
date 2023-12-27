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


class MultiRobot(RobotBase):

    def __init__(self,
                 name='MultiRobot',
                 subrobot_id=None,
                 subrobot_count=1,
                 subrobots=[],
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

        link_idx_offset = 0
        link_idxs_for_object_collision_checking = []
        for robot in self.subrobots:
            link_idxs_for_object_collision_checking.extend(
                [link_idx_offset + idx for idx in robot.link_idxs_for_object_collision_checking]
            )
            link_idx_offset = len(link_idxs_for_object_collision_checking)

        # TODO: Figure out
        num_interpolated_points_for_object_collision_checking = len(link_names_for_object_collision_checking)

        super().__init__(
            name=name,
            q_limits=q_limits,
            link_names_for_object_collision_checking=link_names_for_object_collision_checking,
            link_margins_for_object_collision_checking=link_margins_for_object_collision_checking,
            link_idxs_for_object_collision_checking=link_idxs_for_object_collision_checking,
            num_interpolated_points_for_object_collision_checking=num_interpolated_points_for_object_collision_checking,
            tensor_args=tensor_args,
            **kwargs
        )

    def fk_map_collision_impl(self, q, **kwargs):
        q_offset = 0
        subrobot_fks = []
        for robot in self.subrobots:
            subrobot_q = q[..., q_offset:q_offset+robot.q_dim]
            subrobot_fk = robot.fk_map_collision_impl(subrobot_q, **kwargs)
            subrobot_fks.append(subrobot_fk)
            q_offset += robot.q_dim

        # Second to last dim across links
        res = torch.concatenate(subrobot_fks, axis=-2)
        return res

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

        q_offset = 0
        subrobot_fks = []
        for robot in self.subrobots:
            # Select subset of trajectory for current robot
            position_selector = np.arange(q_offset, q_offset+robot.q_dim)
            velocity_selector = np.arange(self.q_dim + q_offset, self.q_dim + q_offset+robot.q_dim)
            subrobot_selector = np.hstack((position_selector, velocity_selector))
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

    # Updating Base Poses
    def update_base_poses(self, base_poses):
        """Update bases for every robot panda subrobot."""

        for robot, base_pose in zip(self.subrobots, base_poses):
            assert robot.name == "RobotPanda"
            robot.diff_panda.update_base_pose(base_pose)
