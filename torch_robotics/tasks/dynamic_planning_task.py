import einops
import numpy as np
import torch

from torch_robotics.torch_utils.torch_utils import to_torch
from torch_robotics.torch_planning_objectives.fields.dynamic_distance_fields import CollisionDynamicObjectDistanceField
from .tasks import PlanningTask


class DynamicPlanningTask(PlanningTask):

    def __init__(
            self,
            t_max : float = 64.0,
            v_max : float = 0.5,
            dynamic_obstacle_cutoff_margin : float = 0.01,
            **kwargs
    ):
        super().__init__(**kwargs)

        self.t_max = t_max
        self.v_max = v_max

        self.df_collision_dynamic_objects = CollisionDynamicObjectDistanceField(
            self.robot,
            link_idxs_for_collision_checking=self.robot.link_idxs_for_object_collision_checking,
            num_interpolated_points=self.robot.num_interpolated_points_for_object_collision_checking,
            link_margins_for_object_collision_checking_tensor=self.robot.link_margins_for_object_collision_checking_tensor,
            cutoff_margin=dynamic_obstacle_cutoff_margin,
            clamp_sdf=True,
            tensor_args=self.tensor_args
        )

    def add_dynamic_obstacle(self, robot, traj, time):
        traj = to_torch(traj, **self.tensor_args)
        if traj.ndim == 2:
            traj = traj.unsqueeze(0)

        time = to_torch(time, **self.tensor_args)
        self.df_collision_dynamic_objects.add_dynamic_obstacle(robot, traj, time)

    def clear_dynamic_obstacles(self):
        self.df_collision_dynamic_objects.clear_dynamic_obstacles()

    def random_coll_free_q(self, n_samples=1, max_samples=1000, max_tries=1000, time=0, randomize_base=False, **kwargs):
        # Random position in configuration space not in collision
        reject = True
        samples = torch.zeros((n_samples, self.robot.q_dim), **self.tensor_args)
        idx_begin = 0

        if isinstance(time, int):
            time = to_torch(time, **self.tensor_args).unsqueeze(0)

        if randomize_base:
            assert self.robot.name == "MultiRobot"
            base_poses = self.env.random_poses(self.robot.n_subrobots)
            self.robot.update_base_poses(base_poses)

        for i in range(max_tries):
            qs = self.robot.random_q(max_samples)
            in_collision = self.compute_collision(time, qs, **kwargs).squeeze()
            idxs_not_in_collision = torch.argwhere(in_collision == False).squeeze()
            if idxs_not_in_collision.nelement() == 0:
                # all points are in collision
                continue
            if idxs_not_in_collision.nelement() == 1:
                idxs_not_in_collision = [idxs_not_in_collision]
            idx_random = torch.randperm(len(idxs_not_in_collision))[:n_samples]
            free_qs = qs[idxs_not_in_collision[idx_random]]
            idx_end = min(idx_begin + free_qs.shape[0], samples.shape[0])
            samples[idx_begin:idx_end] = free_qs[:idx_end - idx_begin]
            idx_begin = idx_end
            if idx_end >= n_samples:
                reject = False
                break

        if reject:
            sys.exit("Could not find a collision free configuration")

        return samples.squeeze(), None

    def static_compute_collision(self, q, **kwargs):
        q_pos = self.robot.get_position(q)
        return self._compute_collision_or_cost(None, q_pos, field_type='occupancy', static=True, **kwargs)

    def compute_collision(self, time, q, **kwargs):
        q_pos = self.robot.get_position(q)
        return self._compute_collision_or_cost(time, q_pos, field_type='occupancy', **kwargs)

    def compute_collision_cost(self, time, q, **kwargs):
        q_pos = self.robot.get_position(q)
        return self._compute_collision_or_cost(time, q_pos, field_type='sdf', **kwargs)

    def _compute_collision_or_cost(self, time, q, field_type='occupancy', static=False, debug=False, **kwargs):
        # q.shape needs to be reshaped to (batch, horizon, q_dim)
        q_original_shape = q.shape
        b = 1
        h = 1
        collisions = None
        if q.ndim == 1:
            q = q.unsqueeze(0).unsqueeze(0)  # add batch and horizon dimensions for interface
            collisions = torch.ones((1, ), **self.tensor_args)
        elif q.ndim == 2:
            b = q.shape[0]
            q = q.unsqueeze(1)  # add horizon dimension for interface
            collisions = torch.ones((b, 1), **self.tensor_args)  # (batch, 1)
        elif q.ndim == 3:
            b = q.shape[0]
            h = q.shape[1]
            collisions = torch.ones((b, h), **self.tensor_args)  # (batch, horizon)
        elif q.ndim > 3:
            raise NotImplementedError

        if not static and time.ndim == 0:
            time = time.unsqueeze(0)
            assert time.shape[0] == q.shape[1]

        if self.use_occupancy_map:
            raise NotImplementedError
            # ---------------------------------- For occupancy maps ----------------------------------
            ########################################
            # Configuration space boundaries
            idxs_coll_free = torch.argwhere(torch.all(
                torch.logical_and(torch.greater_equal(q, self.robot.q_min), torch.less_equal(q, self.robot.q_max)),
                dim=-1))  # I, 2

            # check if all points are out of bounds (in collision)
            if idxs_coll_free.nelement() == 0:
                return collisions

            ########################################
            # Task space collisions
            # forward kinematics
            q_try = q[idxs_coll_free[:, 0], idxs_coll_free[:, 1]]  # I, q_dim
            x_pos = self.robot.fk_map_collision(q_try, pos_only=True)  # I, taskspaces, x_dim

            # workspace boundaries
            # configuration is not valid if any points in the tasks spaces is out of workspace boundaries
            idxs_ws_in_boundaries = torch.argwhere(torch.all(torch.all(torch.logical_and(
                torch.greater_equal(x_pos, self.ws_min), torch.less_equal(x_pos, self.ws_max)), dim=-1),
                dim=-1)).squeeze()  # I_ws

            idxs_coll_free = idxs_coll_free[idxs_ws_in_boundaries].view(-1, 2)

            # collision in tasks space
            x_pos_in_ws = x_pos[idxs_ws_in_boundaries]  # I_ws, x_dim
            collisions_pos_x = self.env.occupancy_map.get_collisions(x_pos_in_ws, **kwargs)
            if len(collisions_pos_x.shape) == 1:
                collisions_pos_x = collisions_pos_x.view(1, -1)
            idxs_taskspace = torch.argwhere(torch.all(collisions_pos_x == 0, dim=-1)).squeeze()

            idxs_coll_free = idxs_coll_free[idxs_taskspace].view(-1, 2)

            # filter collisions
            if len(collisions) == 1:
                collisions[idxs_coll_free[:, 0]] = 0
            else:
                collisions[idxs_coll_free[:, 0], idxs_coll_free[:, 1]] = 0
        else:
            # ---------------------------------- For distance fields ----------------------------------
            ########################################
            # For distance fields

            # forward kinematics
            fk_collision_pos = self.robot.fk_map_collision(q)  # batch, horizon, taskspaces, x_dim

            # Self collision
            if self.df_collision_self is not None:
                cost_collision_self, info1 = self.df_collision_self.compute_cost(q, fk_collision_pos, field_type=field_type, debug=debug, **kwargs)
            else:
                cost_collision_self = 0
                info1 = {}

            # Object collision
            if self.df_collision_objects is not None:
                cost_collision_objects, info2 = self.df_collision_objects.compute_cost(q, fk_collision_pos, field_type=field_type, **kwargs)
            else:
                cost_collision_objects = 0
                info2 = {}

            # Workspace boundaries
            if self.df_collision_ws_boundaries is not None:
                cost_collision_border, info3 = self.df_collision_ws_boundaries.compute_cost(q, fk_collision_pos, field_type=field_type, **kwargs)
            else:
                cost_collision_border = 0
                info3 = {}

            ########################
            # Dynamic obstacles collision
            if not self.df_collision_dynamic_objects.is_empty() and not static:
                cost_collision_dynamic_obs, info4 = self.df_collision_dynamic_objects.compute_cost(time, q, fk_collision_pos, field_type=field_type, debug=debug, **kwargs)
            else:
                cost_collision_dynamic_obs = 0
                info4 = {}


            if field_type == 'occupancy':
                collisions = cost_collision_self | cost_collision_objects | cost_collision_border | cost_collision_dynamic_obs
            else:
                collisions = cost_collision_self + cost_collision_objects + cost_collision_border + cost_collision_dynamic_obs

            info = dict(**info1, **info2, **info3, **info4)
            info["cost_collision_self"] = cost_collision_self
            info["cost_collision_objects"] = cost_collision_objects
            info["cost_collision_border"] = cost_collision_border
            info["cost_collision_dynamic_obs"] = cost_collision_dynamic_obs

        return collisions
