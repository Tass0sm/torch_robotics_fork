import matplotlib.pyplot as plt
import numpy as np
import torch
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from torch_robotics.environments.primitives import plot_sphere
from torch_robotics.robots.robot_point_mass import RobotPointMass
from torch_robotics.torch_utils.torch_utils import to_numpy, to_torch

import matplotlib.collections as mcoll


class RobotDisk(RobotPointMass):

    def __init__(self,
                 name='RobotDisk',
                 q_limits=torch.tensor([[-1, -1], [1, 1]]),  # configuration space limits
                 radius=0.10,
                 **kwargs):
        super(RobotPointMass, self).__init__(
            name=name,
            q_limits=to_torch(q_limits, **kwargs['tensor_args']),
            link_names_for_object_collision_checking=['link_0'],
            link_margins_for_object_collision_checking=[radius],
            link_idxs_for_object_collision_checking=[0],
            num_interpolated_points_for_object_collision_checking=1,
            **kwargs
        )
