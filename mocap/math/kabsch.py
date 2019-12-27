import numpy as np
import numba as nb
import numpy.linalg as la


def kabsch(P, Q):
    """
    :param P: {n_joints x 3}
    :param Q: {n_joints x 3}
    """
    if len(P.shape) == 1:
        P = P.reshape((-1, 3))
    if len(Q.shape) == 1:
        Q = Q.reshape((-1, 3))
    assert len(P.shape) == 2 and len(Q.shape) == 2, str(P.shape) + ' & ' + str(Q.shape)
    assert P.shape == Q.shape, str(P.shape) + ' & ' + str(Q.shape)

    if P.dtype != np.float32:
        P = P.astype(np.float32)
    if Q.dtype != np.float32:
        Q = Q.astype(np.float32)

    return _kabsch(P, Q)


@nb.njit(nb.float32[:, :](
    nb.float32[:, :], nb.float32[:, :]
), nogil=True)
def _kabsch(P, Q):
    """
    :param P: {n_joints x 3}
    :param Q: {n_joints x 3}
    """
    n_joints = P.shape[0]
    P_centroid = np.sum(P, axis=0) / n_joints
    Q_centroid = np.sum(Q, axis=0) / n_joints

    P = P - P_centroid
    Q = Q - Q_centroid

    H = np.transpose(P) @ Q

    u, _, vh = la.svd(H)
    v = np.transpose(vh)
    uh = np.transpose(u)

    d = la.det(v @ uh)

    D = np.array([1, 0, 0, 
                  0, 1, 0,
                  0, 0, d], dtype=np.float32).reshape((3, 3))
    R = v @ D @ uh
    return R

