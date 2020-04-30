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
    :param seq: [n_frames, 32*3]
    """
    if len(seq.shape) == 2:
        n_joints = seq.shape[1]//3
    elif len(seq.shape) == 3:
        n_joints = seq.shape[1]
    else:
        raise ValueError("incorrect shape:" + str(seq.shape))
    
    assert n_joints in [32, 17], 'wrong joint number:' + str(n_joints)

    if n_joints == 32:
        LS = [6, 7, 8, 9, 10, 16, 17, 18, 19, 20, 21, 22, 23]
        RS = [1, 2, 3, 4, 5, 24, 25, 26, 27, 28, 29, 30, 31]
    elif n_joints == 17:
        LS = [4, 5, 6, 11, 12, 13]
        RS = [1, 2, 3, 14, 15, 16]

    lr = np.array(LS + RS)
    rl = np.array(RS + LS)
    n_frames = len(seq)
    x = seq.reshape((n_frames, -1, 3))
    x_copy = x.copy()
    x = reflect_over_x(x_copy)
    x[:, lr] = x[:, rl]
    return x


def mirror_quaternion(seq):
    """
    :param seq: [n_frames x 32*4]
    :return:
    """
    flatten = False
    n_frames = seq.shape[0]
    if len(seq.shape) == 2:
        flatten = True
        seq = seq.reshape((n_frames, 32, 4))
    joints_left = [1, 2, 3, 4, 5, 24, 25, 26, 27, 28, 29, 30, 31]
    joints_right = [6, 7, 8, 9, 10, 16, 17, 18, 19, 20, 21, 22, 23]
    seq_mirror = seq.copy()
    seq_mirror[:, joints_left] = seq[:, joints_right]
    seq_mirror[:, joints_right] = seq[:, joints_left]
    seq_mirror[:, :, [2, 3]] *= -1
    seq_mirror = qfix(seq_mirror)
    if flatten:
        seq_mirror = seq_mirror.reshape((n_frames, -1))
    return seq_mirror
