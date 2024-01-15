from collections import OrderedDict

import einops
import numpy as np
import torch

from torch_robotics.environments.primitives import MultiSphereField
from torch_robotics.robots.robot_base import RobotBase
from torch_robotics.torch_kinematics_tree.geometrics.frame import Frame
from torch_robotics.torch_kinematics_tree.geometrics.quaternion import q_convert_wxyz
from torch_robotics.torch_kinematics_tree.geometrics.skeleton import get_skeleton_from_model
from torch_robotics.torch_kinematics_tree.geometrics.utils import link_pos_from_link_tensor, link_rot_from_link_tensor, \
    link_quat_from_link_tensor
from torch_robotics.torch_kinematics_tree.models.robot_tree import convert_link_dict_to_tensor
from torch_robotics.torch_kinematics_tree.models.robots import DifferentiableUR5
from torch_robotics.torch_planning_objectives.fields.distance_fields import interpolate_points_v1, CollisionSelfFieldWrapperSTORM
from torch_robotics.torch_utils.torch_utils import to_numpy
from torch_robotics.visualizers.plot_utils import plot_coordinate_frame


class RobotUR5(RobotBase):

    def __init__(self,
                 use_self_collision_storm=False,
                 grasped_object=None,
                 tensor_args=None,
                 **kwargs):

        self.gripper = False

        #############################################
        # Differentiable robots model
        self.link_name_ee = 'ee_link'
        self.link_name_grasped_object = 'grasped_object'

        self.diff_ur5 = DifferentiableUR5(device=tensor_args['device'])

        self.jl_lower, self.jl_upper, _, _ = self.diff_ur5.get_joint_limit_array()
        q_limits = torch.tensor(np.array([self.jl_lower, self.jl_upper]), **tensor_args)

        #############################################
        # Robot collision model for object avoidance
        # https://media.cheggcdn.com/media%2Fce1%2Fce100d57-2fdf-4cd7-8f4a-111a156d6339%2Fphp1EL2S4.png
        # https://www.researchgate.net/profile/Jesse-Haviland/publication/361785335/figure/fig1/AS:1174695604953098@1657080665902/The-Elementary-Transform-Sequence-of-the-7-degree-offreedom-Franka-Emika-Panda.png
        link_names_for_object_collision_checking = [
            # 'base_link',
            # 'shoulder_link',
            'upper_arm_link',
            'forearm_link',
            'wrist_1_link',
            'wrist_2_link',
            'wrist_3_link',
            self.link_name_ee,
        ]
        # these margins correspond to link_names_for_collision_checking
        link_margins_for_object_collision_checking = [
            # 0.1,
            # 0.1,
            0.125,
            0.125,
            0.13,
            0.1,
            0.08,
            0.025,
        ]
        assert len(link_names_for_object_collision_checking) == len(link_margins_for_object_collision_checking)

        link_idxs_for_object_collision_checking = []
        for link_name in link_names_for_object_collision_checking:
            idx = self.diff_ur5._name_to_idx_map[link_name]
            link_idxs_for_object_collision_checking.append(idx)

        #############################################
        # Robot collision model for self collision
        link_names_pairs_for_self_collision_checking = OrderedDict({
            'wrist_1_link': ['shoulder_link'],
            'wrist_2_link': ['base_link', 'shoulder_link', 'upper_arm_link'],
            'wrist_3_link': ['base_link', 'shoulder_link', 'upper_arm_link'],
            self.link_name_ee: ['base_link', 'shoulder_link', 'upper_arm_link'],
        })

        # self collision due to grasped object
        link_names_for_self_collision_checking_with_grasped_object = [
            'base_link', 'shoulder_link', 'upper_arm_link', 'forearm_link',
            # 'wrist_1_link'
        ]

        # retrieve unique names
        # link_names_for_self_collision_checking = self.diff_ur5.get_link_names()
        link_names_for_self_collision_checking = []
        for k, v in link_names_pairs_for_self_collision_checking.items():
            link_names_for_self_collision_checking.append(k)
            link_names_for_self_collision_checking.extend(v)
        link_names_for_self_collision_checking.extend(link_names_for_self_collision_checking_with_grasped_object)
        link_names_for_self_collision_checking = sorted(list(set(link_names_for_self_collision_checking)))

        link_idxs_for_self_collision_checking = []
        for link_name in link_names_for_self_collision_checking:
            idx = self.diff_ur5._name_to_idx_map[link_name]
            link_idxs_for_self_collision_checking.append(idx)

        link_margins_for_self_collision_checking = [
            0.1, # base_link
            0.08, # ee_link
            0.1, # forearm_link
            0.125, # shoulder_link
            0.125, # upper_arm_link
            0.075, # wrist_1_link
            0.13, # wrist_2_link
            0.1, # wrist_3_link
        ]
        assert len(link_names_for_self_collision_checking) == len(link_margins_for_self_collision_checking)

        #############################################
        super().__init__(
            name='RobotUR5',
            q_limits=q_limits,
            grasped_object=grasped_object,
            link_names_for_object_collision_checking=link_names_for_object_collision_checking,
            link_margins_for_object_collision_checking=link_margins_for_object_collision_checking,
            link_idxs_for_object_collision_checking=link_idxs_for_object_collision_checking,
            margin_for_grasped_object_collision_checking=0.001,  # small margin for object placement
            num_interpolated_points_for_object_collision_checking=len(link_names_for_object_collision_checking),
            link_names_for_self_collision_checking=link_names_for_self_collision_checking,
            link_margins_for_self_collision_checking=link_margins_for_self_collision_checking,
            link_names_pairs_for_self_collision_checking=link_names_pairs_for_self_collision_checking,
            link_idxs_for_self_collision_checking=link_idxs_for_self_collision_checking,
            num_interpolated_points_for_self_collision_checking=len(link_names_for_self_collision_checking),
            self_collision_margin_robot=0.05,
            link_names_for_self_collision_checking_with_grasped_object=link_names_for_self_collision_checking_with_grasped_object,
            self_collision_margin_grasped_object=0.05,
            tensor_args=tensor_args,
            **kwargs
        )

        #############################################
        # Override self collision distance field with the one from STORM - https://arxiv.org/abs/2104.13542
        if use_self_collision_storm:
            assert grasped_object is None, ("STORM self collision model does not work if objects are grasped. "
                                            "Learn a self collision model of the robots grasping the object "
                                            "(e.g. using the object mesh).")
            self.df_collision_self = CollisionSelfFieldWrapperSTORM(
                self, 'robot_self/franka_self_sdf.pt', self.q_dim, tensor_args=self.tensor_args)

    def fk_map_collision_impl(self, q, **kwargs):
        q_orig_shape = q.shape
        if len(q_orig_shape) == 3:
            b, h, d = q_orig_shape
            q = einops.rearrange(q, 'b h d -> (b h) d')
        elif len(q_orig_shape) == 2:
            h = 1
            b, d = q_orig_shape
        else:
            raise NotImplementedError

        link_pose_dict = self.diff_ur5.compute_forward_kinematics_all_links(q, return_dict=True)
        # link_tensor = convert_link_dict_to_tensor(link_pose_dict, self.link_names_for_object_collision_checking)
        link_tensor = convert_link_dict_to_tensor(link_pose_dict, self.diff_ur5.get_link_names())

        # Transform collision points of the grasp object with the forward kinematics
        grasped_object_points_in_robot_base_frame = None
        if self.grasped_object:
            grasped_object_points_in_object_frame = self.grasped_object.base_points_for_collision
            frame_grasped_object = link_pose_dict[self.link_name_grasped_object]
            # TODO - by default assumes that world frame is the robots base frame
            grasped_object_points_in_robot_base_frame = frame_grasped_object.transform_point(grasped_object_points_in_object_frame)

        if len(q_orig_shape) == 3:
            link_tensor = einops.rearrange(link_tensor, "(b h) t d1 d2 -> b h t d1 d2", b=b, h=h)

        link_pos = link_pos_from_link_tensor(link_tensor)  # (batch horizon), taskspaces, x_dim
        if grasped_object_points_in_robot_base_frame is not None:
            if len(q_orig_shape) == 3:
                grasped_object_points_in_robot_base_frame = einops.rearrange(grasped_object_points_in_robot_base_frame, "(b h) d1 d2 -> b h d1 d2", b=b, h=h)
            link_pos = torch.cat((link_pos, grasped_object_points_in_robot_base_frame), dim=-2)

        return link_pos

    def get_EE_pose(self, q):
        return self.diff_ur5.compute_forward_kinematics_all_links(q, link_list=[self.link_name_ee])

    def get_EE_position(self, q):
        ee_pose = self.get_EE_pose(q)
        return link_pos_from_link_tensor(ee_pose)

    def get_EE_orientation(self, q, rotation_matrix=True):
        ee_pose = self.get_EE_pose(q)
        if rotation_matrix:
            return link_rot_from_link_tensor(ee_pose)
        else:
            return link_quat_from_link_tensor(ee_pose)

    def get_model_path(self):
        return self.diff_ur5.model_path

    def get_num_links(self):
        return len(self.diff_ur5._model.links)

    def render(self, ax, q=None, color='blue', arrow_length=0.15, arrow_alpha=1.0, arrow_linewidth=2.0,
               draw_links_spheres=False, **kwargs):
        # draw skeleton
        skeleton = get_skeleton_from_model(self.diff_ur5, q, self.diff_ur5.get_link_names())
        skeleton.draw_skeleton(ax=ax, color=color)

        # forward kinematics
        fks_dict = self.diff_ur5.compute_forward_kinematics_all_links(q.unsqueeze(0), return_dict=True)

        # draw link collision points
        if draw_links_spheres:
            link_tensor = convert_link_dict_to_tensor(fks_dict, self.link_names_for_object_collision_checking)
            link_pos = link_pos_from_link_tensor(link_tensor)
            link_pos = interpolate_points_v1(link_pos, self.num_interpolated_points_for_object_collision_checking).squeeze(0)
            spheres = MultiSphereField(
                link_pos,
                self.link_margins_for_object_collision_checking_robot_tensor.view(-1, 1),
                tensor_args=self.tensor_args)
            spheres.render(ax, color='red', cmap='Reds', **kwargs)

        # draw EE frame
        frame_EE = fks_dict[self.link_name_ee]
        plot_coordinate_frame(
            ax, frame_EE, tensor_args=self.tensor_args,
            arrow_length=arrow_length, arrow_alpha=arrow_alpha, arrow_linewidth=arrow_linewidth
        )

        # draw grasped object
        if self.grasped_object is not None:
            frame_grasped_object = fks_dict[self.link_name_grasped_object]

            # draw object
            pos = frame_grasped_object.translation.squeeze()
            ori = q_convert_wxyz(frame_grasped_object.get_quaternion().squeeze())
            self.grasped_object.render(ax, pos=pos, ori=ori, color=color)

            # draw object collision points
            points_in_object_frame = self.grasped_object.base_points_for_collision
            points_in_robot_base_frame = frame_grasped_object.transform_point(points_in_object_frame).squeeze()
            points_in_robot_base_frame_np = to_numpy(points_in_robot_base_frame)
            ax.scatter(
                points_in_robot_base_frame_np[:, 0],
                points_in_robot_base_frame_np[:, 1],
                points_in_robot_base_frame_np[:, 2],
                color=color
            )

    def render_trajectories(self, ax, trajs=None, start_state=None, goal_state=None, colors=['gray'], **kwargs):
        if trajs is not None:
            trajs_pos = self.get_position(trajs)
            for traj, color in zip(trajs_pos, colors):
                for t in range(traj.shape[0]):
                    q = traj[t]
                    self.render(ax, q, color, **kwargs, arrow_length=0.1, arrow_alpha=0.5, arrow_linewidth=1.)
            if start_state is not None:
                self.render(ax, start_state, color='green')
            if goal_state is not None:
                self.render(ax, goal_state, color='purple')
