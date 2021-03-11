import numpy as np
import numba as nb
import mocap.math.geometry as gm


def get_euclidean_transform(human,
                            j_root=0, j_left=6, j_right=1,
                            only_xy=True):
    """ normalize person at 0,0,0 facing 0,1,0
    :param human: {32x3}
    :param j_root:
    :param j_left:
    :param j_right:
    :param only_xy: {Boolean} if True we only transfrom on xy plane
    """
    R, t = _get_euclidean_transform(human, j_root, j_left, j_right)
    if not only_xy:
        human_transformed = apply_euclidean_transform(human, R, t)
        left = human_transformed[j_left]
        right = human_transformed[j_right]
        R_ = gm.get_3d_rotation_to_align_horizontally(left, right)
        R = R_ @ R
    return R, t


@nb.jit(nb.types.Tuple((nb.float32[:, :], nb.float32[:,]))(
    nb.float32[:, :], nb.int64, nb.int64, nb.int64
), nopython=True, nogil=True)
def _get_euclidean_transform(human, j_root, j_left, j_right):
    """ normalize person at 0,0,0 facing (0,1,0)
    :param human: {32x3}
    :param j_root: {int} if -1, use centre of left-right
    :param j_left: {int}
    :param j_right: {int}
    """
    left = human[j_left]
    right = human[j_right]
    if j_root >= 0:
        t = -human[j_root]
    else:
        # use centre between l/r
        t = -(left + right)/2

    R = gm.get_3d_rotation_to_face_forward(left, right)
    return R, t


@nb.jit(nb.float32[:, :](
    nb.float32[:, :], nb.float32[:, :], nb.float32[:, ]
), nopython=True, nogil=True)
def apply_euclidean_transform(human, R, t):
    """ applies rigid transform
    :param human: {32x3}
    :param R: {3x3} in SO(3)
    :param t: {3x1} in |R^3
    """
    new_human = human.copy() + t
    R = np.ascontiguousarray(R)
    for i in range(len(new_human)):
        p = new_human[i]
        new_human[i] = R @ p
    return new_human


@nb.jit(nb.float32[:, :, :](
    nb.float32[:, :, :], nb.float32[:, :], nb.float32[:, ]
), nopython=True, nogil=True)
def apply_euclidean_transform_to_sequence(seq, R, t):
    """ applies rigid transform
    :param seq:
    :param R:
    :param t:
    :return:
    """
    result = np.empty(seq.shape, np.float32)
    for i in range(len(seq)):
        human = seq[i]
        result[i] = apply_euclidean_transform(human, R, t)
    return result


def normalize_sequence_at_frame(seq, frame,
        j_root=0, j_left=6, j_right=1):
    """ Normalize sequence at {frame}
    """
    assert len(seq) > frame
    if len(seq.shape) == 2:
        n_frames = len(seq)
        seq = seq.reshape((n_frames, -1, 3))
    return _normalize_sequence_at_frame(seq, frame,
                                        j_root, j_left, j_right)


@nb.jit(nb.float32[:, :, :](
    nb.float32[:, :, :], nb.int64, nb.int64, nb.int64, nb.int64
), nopython=True, nogil=True)
def _normalize_sequence_at_frame(seq, frame, j_root, j_left, j_right):
    """ Normalize the sequence at {frame}
    :param seq: Nx32x3
    """
    human = seq[frame]
    R, t = _get_euclidean_transform(human, j_root, j_left, j_right)
    result = np.empty(seq.shape, np.float32)
    for i in range(len(seq)):
        human = seq[i]

        result[i] = apply_euclidean_transform(human, R, t)
    return result


@nb.jit(nb.float32[:, :, :](
    nb.float32[:, :, :], nb.int64, nb.int64, nb.int64, nb.int64, nb.float32[:, :], nb.float32[:, ]
), nopython=True, nogil=True)
def _normalize_sequence_at_frame_return_transforms(seq, frame, j_root, j_left, j_right, R_out, t_out):
    """ Normalize the sequence at {frame}
    :param seq: Nx32x3
    """
    human = seq[frame]
    R, t = _get_euclidean_transform(human, j_root, j_left, j_right)
    for i in range(3):
        for j in range(3):
            R_out[i,j] = R[i,j]
        t_out[i] = t[i]
    result = np.empty(seq.shape, np.float32)
    for i in range(len(seq)):
        human = seq[i]
        result[i] = apply_euclidean_transform(human, R, t)
    return result


def remove_global_translation(seq, j_root=0):
    unflattend = False
    if len(seq.shape) == 2:
        unflattend = True
        n_frames = len(seq)
        seq = seq.reshape((n_frames, -1, 3))
    T = np.expand_dims(seq[:, j_root, :], axis=1)
    if unflattend:
        seq = seq.reshape((n_frames, -1))
    return np.ascontiguousarray(seq - T)


def remove_rotation_and_translation(seq, j_root=0, j_left=6, j_right=1):
    unflattend = False
    if len(seq.shape) == 2:
        unflattend = True
        n_frames = len(seq)
        seq = seq.reshape((n_frames, -1, 3))
    seq = _remove_rotation_and_translation(seq, j_root, j_left, j_right)
    if unflattend:
        seq = seq.reshape((n_frames, -1))
    return seq


@nb.jit(nb.float32[:, :, :](
    nb.float32[:, :, :], nb.int64, nb.int64, nb.int64
), nopython=True, nogil=True)
def _remove_rotation_and_translation(seq, j_root, j_left, j_right):
    """
    :param seq:
    :param j_root:
    :param j_left:
    :param j_right:
    :return:
    """
    n, J, dim = seq.shape
    result = np.empty((n, J, dim), np.float32)
    for i in range(n):
        human = seq[i]
        R, t = _get_euclidean_transform(human, j_root, j_left, j_right)
        result[i] = apply_euclidean_transform(human, R, t)

    return result


@nb.jit(nb.float32[:, :, :](
    nb.float32[:, :, :], nb.int64, nb.int64, nb.int64
), nopython=True, nogil=True)
def remove_translation(seq, j_root, j_left, j_right):
    R = np.eye(3, dtype=np.float32)
    result = np.empty(seq.shape, np.float32)
    for frame in range(len(seq)):
        human = seq[frame]

        left = human[j_left]
        right = human[j_right]
        if j_root >= 0:
            t = -human[j_root]
        else:
            # use centre between l/r
            t = -(left + right) / 2

        result[frame] = apply_euclidean_transform(human, R, t)
    return result


@nb.njit(nb.float32[:, :, :](
    nb.float32[:, :, :], nb.int64, nb.int64
), nogil=True)
def insert_root_node_as_avg(seq, j_left, j_right):
    """
    :param seq: [n, J, 3]
    :param j_left:
    :param j_right:
    :return:
    """
    n, J, dim = seq.shape
    result = np.empty((n, J + 1, dim), np.float32)
    result[:, :J, :] = seq
    for t in range(n):
        human = seq[t]
        left = human[j_left]
        right = human[j_right]
        root = (left + right)/2
        result[t, J, :] = root

    return result


def batch_normalize_sequence_at_frame(seqs, frame,
                                      j_root=0, j_left=6, j_right=1,
                                      return_transforms=False,
                                      flatten_output=True):
    """
    :param seqs: [bs x n x 54 ]
    :param frame:
    :param j_root:
    :param j_left:
    :param j_right:
    :return:
    """
    n_batch = seqs.shape[0]
    n_frames = seqs.shape[1]
    if len(seqs.shape) == 3:
        n_joints = seqs.shape[2]//3
        seqs = seqs.reshape((n_batch, n_frames, n_joints, 3))
    else:
        n_joints = seqs.shape[2]

    assert len(seqs.shape) == 4, str(seqs.shape)

    if return_transforms:
        
        R = np.empty((n_batch, 3, 3), dtype=np.float32)
        T = np.empty((n_batch, 3), dtype=np.float32)
        seqs = _batch_normalize_sequence_at_frame_return_transforms(
            seqs, frame, j_root=j_root, j_left=j_left, j_right=j_right,
            R_out=R, t_out=T
        )
        if flatten_output:
            seqs = seqs.reshape((n_batch, n_frames, n_joints*3))
        return seqs, R, T
    else:
        seqs = _batch_normalize_sequence_at_frame(seqs, frame,
                                                  j_root=j_root,
                                                  j_left=j_left,
                                                  j_right=j_right)
        if flatten_output:
            seqs = seqs.reshape((n_batch, n_frames, n_joints*3))
        return seqs


@nb.njit(nb.float32[:, :, :, :](
    nb.float32[:, :, :, :], nb.int64, nb.int64, nb.int64, nb.int64
), nogil=True)
def _batch_normalize_sequence_at_frame(seqs, frame, j_root, j_left, j_right):
    """
    :param seqs: [bs x n x J x 3 ]
    :param frame:
    :param j_root:
    :param j_left:
    :param j_right:
    :return:
    """
    bs = len(seqs)
    for i in range(bs):
        seqs[i] = _normalize_sequence_at_frame(seqs[i], frame,
                                               j_root, j_left, j_right)
    return seqs


@nb.njit(nb.float32[:, :, :, :](
    nb.float32[:, :, :, :], nb.int64, nb.int64, nb.int64, nb.int64, nb.float32[:, :, :], nb.float32[:, :]
), nogil=True)
def _batch_normalize_sequence_at_frame_return_transforms(seqs, frame, j_root, j_left, j_right, R_out, t_out):
    """
    :param seqs: [bs x n x J x 3 ]
    :param frame:
    :param j_root:
    :param j_left:
    :param j_right:
    :return:
    """
    bs = len(seqs)
    for i in range(bs):
        seqs[i] = _normalize_sequence_at_frame_return_transforms(
            seqs[i], frame, j_root, j_left, j_right, R_out[i], t_out[i])
    return seqs

