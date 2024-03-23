import numpy as np
import torch
from matplotlib import pyplot as plt
import pybullet as p

from torch_robotics.environments.env_base import EnvBase
from torch_robotics.torch_utils.torch_utils import DEFAULT_TENSOR_ARGS
from torch_robotics.visualizers.planning_visualizer import create_fig_and_axes


class EnvOpen3D(EnvBase):

    def __init__(self, name='EnvOpen3D', tensor_args=None, **kwargs):
        super().__init__(
            name=name,
            limits=torch.tensor([[-2, -2, -0.25], [2, 2, 2]], **tensor_args),  # environments limits
            tensor_args=tensor_args,
            **kwargs
        )

    def setup_pybullet_env(self):
        self.pb_objs = []

        plane_id = p.loadURDF("plane.urdf", basePosition=[0, 0, -0.05])
        self.pb_objs.append(plane_id)

    def get_gpmp2_params(self, robot=None):
        params = dict(
            opt_iters=250,
            num_samples=64,
            sigma_start=1e-3,
            sigma_gp=1e-1,
            sigma_goal_prior=1e-3,
            sigma_coll=1e-3,
            step_size=1e-2,
            sigma_start_init=1e-4,
            sigma_goal_init=1e-4,
            sigma_gp_init=0.1,
            sigma_start_sample=1e-3,
            sigma_goal_sample=1e-3,
            solver_params={
                'delta': 1e-2,
                'trust_region': True,
                'sparse_computation': False,
                'sparse_computation_block_diag': False,
                'method': 'cholesky',
                # 'method': 'cholesky-sparse',
                # 'method': 'inverse',
            },
            stop_criteria=0.1,
        )
        if robot.name == "RobotPanda":
            return params
        elif robot.name == "MultiRobot":
            return params
        else:
            raise NotImplementedError

    def get_rrt_connect_params(self, robot=None):
        params = dict(
            n_iters=10000,
            step_size=torch.pi/30,
            n_radius=torch.pi/4,
            n_pre_samples=50000,

            max_time=90
        )
        if robot.name == "RobotPanda":
            return params
        elif robot.name == "MultiRobot":
            return params
        else:
            raise NotImplementedError


if __name__ == '__main__':
    env = EnvOpen3D(
        precompute_sdf_obj_fixed=True,
        sdf_cell_size=0.01,
        tensor_args=DEFAULT_TENSOR_ARGS
    )
    fig, ax = create_fig_and_axes(env.dim)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    env.render(ax)
    fig.savefig("env-open-3d-render.png")
    plt.close(fig)

    # Render sdf
    fig, ax = create_fig_and_axes(env.dim)
    env.render_sdf(ax, fig)

    # Render gradient of sdf
    env.render_grad_sdf(ax, fig)
    fig.savefig("env-open-3d-sdf.png")
    plt.close(fig)
