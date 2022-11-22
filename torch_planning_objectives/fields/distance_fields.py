from abc import ABC, abstractmethod
import torch
import copy
import yaml

from torch_kinematics_tree.geometrics.utils import transform_point
from torch_kinematics_tree.utils.files import get_configs_path
from torch_planning_objectives.fields.utils.geom_types import tensor_sphere
from torch_planning_objectives.fields.utils.distance import find_link_distance, find_obstacle_distance


class DistanceField(ABC):
    def __init__(self, ):
        pass

    @abstractmethod
    def compute_distance(self):
        pass

    @abstractmethod
    def zero_grad(self):
        pass 



class SphereDistanceField(DistanceField):
    """ This class holds a batched collision model where the robot is represented as spheres.
        All points are stored in the world reference frame, obtained by using update_pose calls.
    """

    def __init__(self, robot_collision_params, batch_size=1, device='cpu'):
        """ Initialize with robot collision parameters, look at franka_reacher.py for an example.
        Args:
            robot_collision_params (Dict): collision model parameters
            batch_size (int, optional): Batch size of parallel sdf computation. Defaults to 1.
            tensor_args (dict, optional): compute device and data type. Defaults to {'device':"cpu", 'dtype':torch.float32}.
        """        
        self.batch_dim = [batch_size, ]
        self.device = device

        self._link_spheres = None
        self._batch_link_spheres = None
        self.w_batch_link_spheres = None

        self.robot_collision_params = robot_collision_params
        self.load_robot_collision_model(robot_collision_params)

        self.self_dist = None
        self.obst_dist = None

    def load_robot_collision_model(self, robot_collision_params):
        """Load robot collision model, called from constructor
        Args:
            robot_collision_params (Dict): loaded from yml file
        """        
        self.robot_links = robot_collision_params['link_objs']

        # load collision file:
        coll_yml = (get_configs_path() / robot_collision_params['collision_spheres']).as_posix()
        with open(coll_yml) as file:
            coll_params = yaml.load(file, Loader=yaml.FullLoader)

        self._link_spheres = []
        # we store as [n_link, n_dim]

        for j_idx, j in enumerate(self.robot_links):
            n_spheres = len(coll_params[j])
            link_spheres = torch.zeros((n_spheres, 4), device=self.device)
            for i in range(n_spheres):
                link_spheres[i, :] = tensor_sphere(coll_params[j][i][:3], coll_params[j][i][3], device=self.device, tensor=link_spheres[i])
            self._link_spheres.append(link_spheres)

    def build_batch_features(self, clone_objs=False, batch_dim=None):
        """clones poses/object instances for computing across batch. Use this once per batch size change to avoid re-initialization over repeated calls.
        Args:
            clone_objs (bool, optional): clones objects. Defaults to False.
            batch_size ([type], optional): batch_size to clone. Defaults to None.
        """

        if(batch_dim is not None):
            self.batch_dim = batch_dim
        if(clone_objs):
            self._batch_link_spheres = []
            for i in range(len(self._link_spheres)):
                _batch_link_i = self._link_spheres[i].view(tuple([1] * len(self.batch_dim) + list(self._link_spheres[i].shape)))
                _batch_link_i = _batch_link_i.repeat(tuple(self.batch_dim + [1, 1]))
                self._batch_link_spheres.append(_batch_link_i)
        self.w_batch_link_spheres = copy.deepcopy(self._batch_link_spheres)

    def update_batch_robot_collision_objs(self, links_dict):
        '''update pose of link spheres
        Args:
        links_pos: bxnx3
        links_rot: bxnx3x3
        '''

        for i in range(len(self.robot_links)):
            link_H = links_dict[self.robot_links[i]].get_transform_matrix()
            link_pos, link_rot = link_H[..., :-1, -1], link_H[..., :3, :3]
            self.w_batch_link_spheres[i][..., :3] = transform_point(self._batch_link_spheres[i][..., :3], link_rot, link_pos.unsqueeze(-2))

    def check_collisions(self, obstacle_spheres=None):
        """Analytic method to compute signed distance between links.
        Args:
            link_trans ([tensor]): link translation as batch [b, 3]
            link_rot ([type]): link rotation as batch [b, 3, 3]
        Returns:
            [tensor]: signed distance [b, 1]
        """        
        n_links = len(self.w_batch_link_spheres)
        if self.self_dist is None:
            self.self_dist = torch.zeros(self.batch_dim + [n_links, n_links], device=self.device) - 100.0
        if self.obst_dist is None:
            self.obst_dist = torch.zeros(self.batch_dim + [n_links,], device=self.device) - 100.0
        dist = self.self_dist
        dist = find_link_distance(self.w_batch_link_spheres, dist)
        total_dist = dist.max(1)[0]
        if obstacle_spheres is not None:
            obst_dist = self.obst_dist
            obst_dist = find_obstacle_distance(obstacle_spheres, self.w_batch_link_spheres, obst_dist)
            total_dist += obst_dist
        return dist

    def compute_distance(self, links_dict, obstacle_spheres=None):
        self.update_batch_robot_collision_objs(links_dict)
        return self.check_collisions(obstacle_spheres).max(1)[0]

    def get_batch_robot_link_spheres(self):
        return self.w_batch_link_spheres

    def zero_grad(self):
        self.self_dist.detach_()
        self.self_dist.grad = None
        self.obst_dist.detach_()
        self.obst_dist.grad = None
        for i in range(len(self.robot_links)):
            self.w_batch_link_spheres[i].detach_()
            self.w_batch_link_spheres[i].grad = None
            self._batch_link_spheres[i].detach_()
            self._batch_link_spheres[i].grad = None