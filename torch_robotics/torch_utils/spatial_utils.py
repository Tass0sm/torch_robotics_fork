import numpy as np
from scipy.spatial.transform import Rotation as R


def get_transform(rotq=None, euler=None, rotvec=None, matrix=None, pos=np.array([[0, 0, 0]])):
    """utility function to create transformation matrix from different input forms"""
    if pos.ndim == 1:
        bs = 1
    else:
        bs = pos.shape[0]

    trans = np.zeros((bs, 4, 4))
    trans[:] = np.eye(4)

    if rotq is not None:
        trans[:, :-1, :-1] = R.from_quat(rotq).as_matrix()
    elif euler is not None:
        trans[:, :-1, :-1] = R.from_euler("xyz", euler).as_matrix()
    elif rotvec is not None:
        trans[:, :-1, :-1] = R.from_rotvec(rotvec).as_matrix()
    elif matrix is not None:
        trans[:, :-1, :-1] = matrix

    trans[:, :-1, -1:] = pos.reshape(bs, -1, 1)

    return trans
