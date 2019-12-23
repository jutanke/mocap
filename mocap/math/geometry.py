import numpy as np
import numba as nb
import numpy.linalg as la
from math import pi


@nb.jit(nb.float32[:, :](
    nb.float32
), nopython=True, nogil=True )
def rot2d(alpha):
    """ get a 2d rotation matrix
    :param alpha: in radians
    :returns rotation matrix
    """
    R = np.array([
        [np.cos(alpha), -np.sin(alpha)],
        [np.sin(alpha), np.cos(alpha)]
    ], np.float32)
    return np.ascontiguousarray(R)


@nb.jit(nb.float32[:, ](
    nb.float32[:, ], nb.float32[:, ]
), nopython=True, nogil=True)
def get_2d_normal(left, right):
    """ gets the 2d normal
    :param left: [x, y]
    :param right: [x, y]
    """
    lr = left - right
    lr = lr/la.norm(lr)
    R = rot2d(-pi/2)
    R = np.ascontiguousarray(R)
    n = R @ lr
    return n


@nb.jit(nb.float32(
    nb.float32[:, ]
), nopython=True, nogil=True)
def get_2d_rotation_for_upward(p):
    """ gets alpha to rotate point p to [0, 1]
    :param p: [x, y]
    :return angle in radians
    """
    p = p / la.norm(p)  # normalize (just to be sure)
    up = np.array([0, 1], np.float32)

    alpha = np.arccos(up @ p)
    if p[0] < 0:
        alpha *= -1

    return alpha

# === 3 dimensions ===


def distances(human1, human2):
    """ calculate distances between two humans for each joint
    :param human1: [ (x, y, z), ... ]
    :param human2: [ (x, y, z), ... ]
    """
    J = len(human1)
    assert len(human2) == J
    return _distances(human1, human2)


@nb.jit(nb.float32[:, ](
    nb.float32[:, :], nb.float32[:, :]
), nopython=True, nogil=True)
def _distances(human1, human2):
    """ calculate distances between two humans for each joint
    :param human1: [ (x, y, z), ... ]
    :param human2: [ (x, y, z), ... ]
    """
    J = len(human1)
    results = np.empty((J, ), np.float32)
    for jid in range(J):
        a = human1[jid]
        b = human2[jid]
        results[jid] = la.norm(a - b)

    return results


@nb.jit(nb.float32[:, ](
    nb.float32[:, ], nb.float32[:, ], nb.int32[:, ]
), nopython=True, nogil=True)
def get_3d_normal_on_plane(left, right, plane):
    """ calculate 3d normal on defined plane
    :param left: (x, y, z)
    :param right: (x, y, z)
    :return
    """
    oo_plane_dim = -1  # find out-of-plane dimension
    for i in range(3):
        i_is_not_in_plane = True
        for j in plane:
            if i == j:
                i_is_not_in_plane = False

        if i_is_not_in_plane:
            oo_plane_dim = i
            break

    oo_mean = (left[oo_plane_dim] + right[oo_plane_dim]) / 2

    left = left[plane]
    right = right[plane]

    n_2d = get_2d_normal(left, right)

    result = np.empty((3, ), np.float32)
    result[plane[0]] = n_2d[0]
    result[plane[1]] = n_2d[1]
    result[oo_plane_dim] = oo_mean

    return result


@nb.jit(nb.float32[:, :](
    nb.float32, nb.float32, nb.float32
), nopython=True, nogil=True)
def rot3d(a, b, c):
    """
    """
    Rx = np.array([
        [1., 0., 0.],
        [0, np.cos(a), -np.sin(a)],
        [0, np.sin(a), np.cos(a)]
    ], np.float32)
    Ry = np.array([
        [np.cos(b), 0, np.sin(b)],
        [0., 1., 0.],
        [-np.sin(b), 0, np.cos(b)]
    ], np.float32)
    Rz = np.array([
        [np.cos(c), -np.sin(c), 0],
        [np.sin(c), np.cos(c), 0],
        [0., 0., 1.]
    ], np.float32)
    return np.ascontiguousarray(Rx @ Ry @ Rz)


@nb.jit(nb.float32[:, :](
    nb.float32[:, ], nb.float32[:, ]
), nopython=True, nogil=True)
def get_3d_rotation_to_align_horizontally(left, right):
    """ aligns left and right so that both touch the x-axis
    :param left: (x, y, z)
    :param right: (x, y, z)
    :return R
    """
    plane = np.array([0, 2], np.int32)
    x, y, z = get_3d_normal_on_plane(left, right, plane)
    n = np.array([x, z])
    alpha = -get_2d_rotation_for_upward(n)
    R = rot3d(0., alpha, 0.)
    return R


@nb.jit(nb.float32[:, :](
    nb.float32[:, ], nb.float32[:, ]
), nopython=True, nogil=True)
def get_3d_rotation_to_face_forward(left, right):
    """ align skeleton so that it faces forward in positive
        y direction (0, 1, 0)
    :param left: (x, y, z)
    :param right: (x, y, z)
    :return R
    """
    plane = np.array([0, 1], np.int32)
    x, y, z = get_3d_normal_on_plane(left, right, plane)
    n = np.array([x, y])
    alpha = get_2d_rotation_for_upward(n)
    R = rot3d(0., 0., alpha)
    return R


def apply_rotation(person, R):
    """
    :param person: { J x 3 }
    :param R: { 3 x 3 }
    :return:
    """
    person = np.ascontiguousarray(person)
    R = np.ascontiguousarray(R)
    person_mu = person - np.mean(person, axis=0)
    person_R = person_mu @ R.T
    return person_R + np.mean(person, axis=0)