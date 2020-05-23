import numpy as np
import numba as nb
from mocap.math.quaternion import qfix


@nb.jit(nb.float32[:, :, :](
    nb.float32[:, :, :]
), nopython=True, nogil=True)
def reflect_over_x(seq):
    """ reflect sequence over x-y (exchange left-right)
    INPLACE
    """
    I = np.array([
        [-1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
        ], np.float32)
    # ensure we do not fuck up memory
    for frame in range(len(seq)):
        person = seq[frame]
        for jid in range(len(person)):
            pt3d = np.ascontiguousarray(person[jid])
            seq[frame, jid] = I @ pt3d
    return seq


def mirror_p3d(seq):
    """
    :param seq: [n_frames, 24*3]
    """
    if len(seq.shape) == 2:
        n_joints = seq.shape[1]//3
    elif len(seq.shape) == 3:
        n_joints = seq.shape[1]
    else:
        raise ValueError("incorrect shape:" + str(seq.shape))
    
    assert n_joints in [24], 'wrong joint number:' + str(n_joints)

    if n_joints == 24:
        LS = [1, 4, 7, 10, 13, 16, 18, 20, 22]
        RS = [2, 5, 8, 11, 14, 17, 19, 21, 23]

    lr = np.array(LS + RS)
    rl = np.array(RS + LS)
    n_frames = len(seq)
    x = seq.reshape((n_frames, -1, 3))
    x_copy = x.copy()
    x = reflect_over_x(x_copy)
    x[:, lr] = x[:, rl]
    return x
