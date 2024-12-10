from abc import ABC, abstractmethod

import einops
import torch
from matplotlib import pyplot as plt

from torch_robotics.torch_utils.torch_utils import DEFAULT_TENSOR_ARGS, to_torch, to_numpy
from torch_robotics.torch_kinematics_tree.geometrics.utils import SE3_distance
from torch_robotics.visualizers.planning_visualizer import create_fig_and_axes
import torch.nn.functional as Functional

import scipy.interpolate

from .distance_fields import *


class DynamicObstacle:

    def __init__(
            self,
            entity,
            traj,
            timesteps,
            tensor_args : dict = {}
    ):
        assert timesteps.shape[0] == traj.shape[1]

        self.entity = entity
        self.traj = traj
        self.timesteps = timesteps
        self.tensor_args = tensor_args

    def dynamic_state(
            self,
            time : torch.Tensor
    ):
        time = time.clamp(self.timesteps[0], self.timesteps[-1])

        timesteps_cpu = self.timesteps.cpu()
        traj_cpu = self.traj.cpu().squeeze()

        if traj_cpu.ndim == 1:
            traj_cpu = traj_cpu.unsqueeze(0)

        interpolator = scipy.interpolate.interp1d(
            timesteps_cpu,
            traj_cpu,
            axis=0, bounds_error=False, fill_value=(traj_cpu[0], traj_cpu[-1])
        )

        traj_at_time = interpolator(time.cpu())
        traj_at_time = to_torch(traj_at_time, **self.tensor_args)

        if traj_at_time.nelement() == 0:
            raise NotImplementedError

        return traj_at_time.unsqueeze(0)

    def dynamic_fk(
            self,
            time : int,
            **kwargs
    ):
        q = self.dynamic_state(time)
        return self.entity.fk_map_collision(q, **kwargs)

    def get_margins(self):
        margins = to_torch(self.entity.link_margins_for_object_collision_checking, **self.tensor_args)
        return margins


class DynamicDistanceField(ABC):
    def __init__(self, tensor_args=None):
        self.tensor_args = tensor_args

    def distances(self):
        pass

    def compute_collision(self):
        pass

    @abstractmethod
    def compute_distance(self, *args, **kwargs):
        pass

    def compute_cost(self, time, q, link_pos, *args, **kwargs):
        time_orig_shape = time.shape
        q_orig_shape = q.shape
        link_orig_shape = link_pos.shape
        if len(link_orig_shape) == 2:
            h = 1
            b, d = link_orig_shape
            link_pos = einops.rearrange(link_pos, "b d -> b 1 1 d")  # add dimension of task space link
        elif len(link_orig_shape) == 3:
            h = 1
            b, t, d = link_orig_shape
            link_pos = einops.rearrange(link_pos, "b t d -> b 1 t d")
        elif len(link_orig_shape) == 4:  # batch, horizon, num_links, 3  # position tensor
            b, h, t, d = link_orig_shape
            link_pos = einops.rearrange(link_pos, "b h t d -> b h t d")
        elif len(link_orig_shape) == 5:  # batch, horizon, num_links, 4, 4  # homogeneous transform tensor
            b, h, t, d, d = link_orig_shape
            link_pos = einops.rearrange(link_pos, "b h t d d -> b h t d d")
        else:
            raise NotImplementedError

        assert time.shape[0] == link_pos.shape[1]

        # link_tensor_pos
        # position: batch x horizon x num_links x 2/3
        cost = self.compute_costs_impl(time, q, link_pos, *args, **kwargs)

        return cost

    @abstractmethod
    def compute_costs_impl(self, *args, **kwargs):
        pass

    @abstractmethod
    def zero_grad(self):
        pass


class DynamicEmbodimentDistanceFieldBase(DynamicDistanceField):

    def __init__(self,
                 robot,
                 link_idxs_for_collision_checking=None,
                 num_interpolated_points=30,
                 collision_margins=0.,
                 cutoff_margin=0.001,
                 field_type='sdf', clamp_sdf=False,
                 interpolate_link_pos=False,
                 **kwargs):
        super().__init__(**kwargs)
        assert robot is not None, "You need to pass a robot instance to the embodiment distance fields"
        self.robot = robot
        self.link_idxs_for_collision_checking = link_idxs_for_collision_checking
        self.num_interpolated_points = num_interpolated_points
        self.collision_margins = collision_margins
        self.cutoff_margin = cutoff_margin
        self.field_type = field_type
        self.clamp_sdf = clamp_sdf
        self.interpolate_link_pos = interpolate_link_pos

    def compute_costs_impl(self, time, q, link_pos, **kwargs):
        # position link_pos tensor # batch x time x num_links x 3
        # interpolate to approximate link spheres
        if self.robot.grasped_object is not None:
            n_grasped_object_points = self.robot.grasped_object.n_base_points_for_collision
            link_pos_robot = link_pos[..., :-n_grasped_object_points, :]
            link_pos_grasped_object = link_pos[..., -n_grasped_object_points:, :]
        else:
            link_pos_robot = link_pos

        link_pos_robot = link_pos_robot[..., self.link_idxs_for_collision_checking, :]
        if self.interpolate_link_pos:
            # select the robot links used for collision checking
            link_pos = interpolate_points_v1(link_pos_robot, self.num_interpolated_points)

        # stack collision points from grasped object
        # these points do not need to be interpolated
        if self.robot.grasped_object is not None:
            link_pos = torch.cat((link_pos, link_pos_grasped_object), dim=-2)

        embodiment_cost = self.compute_embodiment_cost(time, q, link_pos, **kwargs)
        return embodiment_cost

    def compute_embodiment_cost(self, time, q, link_pos, field_type=None, **kwargs):  # position tensor
        if field_type is None:
            field_type = self.field_type

        if field_type == 'rbf':
            return self.compute_embodiment_rbf_distances(link_pos, **kwargs).sum((-1, -2))
        elif field_type == 'sdf':  # this computes the negative cost from the DISTANCE FUNCTION
            cutoff_margin = kwargs.get("margin", self.cutoff_margin)
            margin = self.collision_margins + cutoff_margin
            # returns all distances from each link to the environment
            link_pos = link_pos[..., self.link_idxs_for_collision_checking, :]
            margin_minus_sdf = -(self.compute_embodiment_signed_distances(time, q, link_pos, **kwargs) - margin)

            if self.clamp_sdf:
                clamped_sdf = torch.relu(margin_minus_sdf)
            else:
                clamped_sdf = margin_minus_sdf

            if len(clamped_sdf.shape) == 3:  # cover the multiple objects case
                clamped_sdf = clamped_sdf.max(-2)[0]

            # sum over link points for gradient computation
            return clamped_sdf.sum(-1)
        elif field_type == 'occupancy':
            return self.compute_embodiment_collision(time, q, link_pos, **kwargs)
            # distances = self.self_distances(link_pos, **kwargs)  # batch_dim x (links * (links - 1) / 2)
            # return (distances < margin).sum(-1)
        else:
            raise NotImplementedError('field_type {} not implemented'.format(field_type))

    def compute_distance(self, time, q, link_pos, **kwargs):
        # raise NotImplementedError
        link_pos = interpolate_points_v1(link_pos, self.num_interpolated_points)
        self_distances = self.compute_embodiment_signed_distances(time, q, link_pos, **kwargs).min(-1)[0]  # batch_dim
        return self_distances

    def zero_grad(self):
        pass
        # raise NotImplementedError

    @abstractmethod
    def compute_embodiment_rbf_distances(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def compute_embodiment_signed_distances(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def compute_embodiment_collision(self, *args, **kwargs):
        raise NotImplementedError


class CollisionDynamicObjectBase(DynamicEmbodimentDistanceFieldBase):

    def __init__(self, *args, link_margins_for_object_collision_checking_tensor=None, **kwargs):
        super().__init__(*args, collision_margins=link_margins_for_object_collision_checking_tensor, **kwargs)

    def compute_embodiment_rbf_distances(self, time, link_pos, **kwargs):  # position tensor
        raise NotImplementedError
        margin = kwargs.get('margin', self.margin)
        rbf_distance = torch.exp(torch.square(self.object_signed_distances(time, link_pos, **kwargs)) / (-margin ** 2 * 2))
        return rbf_distance

    def compute_embodiment_signed_distances(self, time, q, link_pos, **kwargs):
        return self.object_signed_distances(time, link_pos, **kwargs)

    def compute_embodiment_collision(self, time, q, link_pos, **kwargs):
        info = {}

        # position tensor
        margin = kwargs.get('margin', self.collision_margins + self.cutoff_margin)
        link_pos = link_pos[..., self.link_idxs_for_collision_checking, :]
        signed_distances = self.object_signed_distances(time, link_pos, **kwargs)
        collisions = signed_distances < margin

        # reduce over links (dim -1) and over objects (dim -2)
        any_collision = torch.any(torch.any(collisions, dim=-1), dim=-1)
        return any_collision, info

    @abstractmethod
    def object_signed_distances(self, *args, **kwargs):
        raise NotImplementedError


class CollisionDynamicObjectDistanceField(CollisionDynamicObjectBase):

    def __init__(self,
                 *args,
                 dynamic_obstacles : list = [],
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.dynamic_obstacles = dynamic_obstacles

    def add_dynamic_obstacle(self, entity, traj, time):
        obstacle = DynamicObstacle(entity, traj, time, tensor_args=self.tensor_args)
        self.dynamic_obstacles.append(obstacle)

    def clear_dynamic_obstacles(self):
        self.dynamic_obstacles = []

    def is_empty(self):
        return len(self.dynamic_obstacles) == 0

    def get_dynamic_spheres(self, time):
        """
        returns
        - centers: batch x time x 2/3
        - radii: batch x time x 1
        """

        links = []
        margins = []

        for obstacle in self.dynamic_obstacles:
            links_i = obstacle.dynamic_fk(time)
            if links_i is None:
                continue

            b, h, t, d = links_i.shape
            margins_i = obstacle.get_margins()
            margins_i = margins_i.expand((b, h, t))

            links.append(links_i)
            margins.append(margins_i)

        centers = torch.stack(links, dim=2) # batch x horizon x obs x links x dim
        radii = torch.stack(margins, dim=2)

        return centers, radii

    def object_signed_distances(self, time, link_pos, **kwargs):
        if len(self.dynamic_obstacles) == 0:
            return torch.inf

        centers, radii = self.get_dynamic_spheres(time)

        # unsqueeze for broadcasting over obstacles
        # batches x horizon x objects x num_links x dim
        link_pos = link_pos.unsqueeze(2)

        # batch x horizon x objects x num_links
        try:
            distance_to_centers = torch.norm(link_pos - centers, dim=-1)
        except RuntimeError:
            breakpoint()

        # batch x horizon x num_links
        distance_to_spheres = distance_to_centers - radii

        # batch x horizon
        # sdf = torch.min(distance_to_spheres, dim=-1)[0].unsqueeze(-2)

        return distance_to_spheres
