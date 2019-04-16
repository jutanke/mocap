import numpy as np
import numba as nb
import mocap.processing.normalize as norm
import mocap.math.geometry as gm
import math


@nb.njit(
    nb.float32[:, :](
        nb.float32[:, :, :]
        , nb.int64, nb.int64, nb.int64, nb.int64
    ), nogil=True)
def global2relative(seq, J, j_root, j_left, j_right):
    """
    :param seq:
    :param J:
    :param j_root:
    :param j_left:
    :param j_right:
    :return:
    """
    n_frames = len(seq)
    result = np.empty((n_frames, (J * 3 + 1)), np.float32)

    for frame in range(0, n_frames):
        human = seq[frame]
        left = human[j_left]
        right = human[j_right]
        t = -human[j_root]
        plane = np.array([0, 1], np.int32)
        x, y, z = gm.get_3d_normal_on_plane(left, right, plane)
        n = np.array([x, y])
        alpha = gm.get_2d_rotation_for_upward(n)
        R = gm.rot3d(0., 0., alpha)
        human_norm = norm.apply_euclidean_transform(human, R, t)

        for jid in range(J):
            if jid == j_root:  # insert displacement vector
                displacement = np.zeros((3, ), np.float32)
                if frame > 0:
                    prev_human = seq[frame - 1]
                    a = prev_human[j_root]
                    b = human[j_root]
                    displacement = b - a

                result[frame, jid * 3: (jid+1) * 3] = displacement
            else:
                result[frame, jid * 3: (jid+1) * 3] = human_norm[jid]

        result[frame, J * 3] = alpha
    return result


@nb.njit(
    nb.float32[:, :, :](
        nb.float32[:, :],
        nb.int64, nb.int64
    ), nogil=True
)
def relative2global(seq, J, j_root):
    """
    :param seq: {n x J * 3 + 1}
    :param J:
    :param j_root:
    :return:
    """
    n_frames = len(seq)
    true_J = J
    has_fake_root = False
    if j_root == -1:
        has_fake_root = True
        true_J = J - 1  # remove fake root
        j_root = true_J  # by convention the last joint is the root

    result = np.empty((n_frames, true_J, 3), np.float32)

    global_displacement = np.zeros((3, ), np.float32)

    for frame in range(0, n_frames):
        human = np.copy(seq[frame, :-1]).reshape((J, 3))

        displacement = np.copy(human[j_root])
        global_displacement += displacement
        new_human = human.copy()
        new_human[j_root] *= 0

        if frame > 0:
            alpha = seq[frame, -1]
            R = gm.rot3d(0, 0, alpha)
            R = np.transpose(R)  # inverse R to recover

            norm.apply_rot_inplace(new_human, R)
            new_human += global_displacement

        if has_fake_root:
            for jid in range(true_J):  # implicit: last jid is fake root
                result[frame, jid] = new_human[jid]
        else:
            result[frame] = new_human

    return result


class Transformer:
    """
    Apply the following normalization scheme:

        Input: sequence in global coordinates {n x J x 3}
        [ x x x x x .. x ]
        [ y y y y y .. y ]
        [ z z z z z .. z ]

        Output: normalized sequence: {n x (J-1 * 3 + 3 + 1)}
            * each frame is normalized at the origin
            * each frame the person faces forward (based on hip)
            * memorize displacement vector to previous frame
            * memorize rotation over z-axis to previous frame

    """

    def __init__(self, j_root, j_left, j_right):
        """
        :param root_node: the node which will be used
            for normalization
        """
        self.j_root = j_root
        self.j_left = j_left
        self.j_right = j_right

    def global2relative(self, seq):
        """ converts a sequence in global coordinates to a relative
            one
        :param seq: {n x J x 3}
        :return:
        """
        if len(seq.shape) == 2:
            n = len(seq)
            seq = seq.reshape((n, -1, 3))
        _, J, _ = seq.shape

        j_root = self.j_root
        j_left = self.j_left
        j_right = self.j_right
        if j_root == -1:
            seq = norm.insert_root_node_as_avg(seq, j_left, j_right)
            j_root = J  # last joint is the root node
            J += 1

        return global2relative(seq, J,
                               j_root, j_left, j_right)

    def relative2global(self, seq):
        """ converts a sequence from relative coordinates to global
            one
        :param seq: {n x J * 3 + 1}
        :return:
        """
        assert len(seq.shape) == 2, "<shape mismatch>:" + str(seq.shape)
        j_root = self.j_root

        n, dim = seq.shape
        J = (dim - 1) / 3
        assert int(math.ceil(J)) == int(math.floor(J)), "J=" + str(J)

        return relative2global(seq, J, j_root)
