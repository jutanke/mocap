import numpy as np


def expmap2euler(seq):
    """
    :param seq: {n x 99}
    """
    n_frames, dim = seq.shape
    assert dim == 99, 'dim is ' + str(dim)
    euler = np.copy(seq)
    for j in range(n_frames):
        for k in np.arange(0, 97, 3):
            idx = [k, k+1, k+2]
            R = expmap2rotmat(seq[j, idx])
            euler[j, idx] = rotmat2euler(R.T)
    euler[:, 0:6] = 0
    return euler[:, 3:]


def expmap2rotmat(r):
    """
    Converts an exponential map angle to a rotation matrix
    Matlab port to python for evaluation purposes
    I believe this is also called Rodrigues' formula
    https://github.com/asheshjain399/RNNexp/blob/srnn/structural_rnn/CRFProblems/H3.6m/mhmublv/Motion/expmap2rotmat.m
    Args
    r: 1x3 exponential map
    Returns
    R: 3x3 rotation matrix
    """
    theta = np.linalg.norm( r )
    r0  = np.divide( r, max(theta, np.finfo(np.float32).eps) )
    r0x = np.array([0, -r0[2], r0[1], 0, 0, -r0[0], 0, 0, 0]).reshape(3,3)
    r0x = r0x - r0x.T
    R = np.eye(3,3) + np.sin(theta)*r0x + (1-np.cos(theta))*(r0x).dot(r0x)
    return R


def rotmat2euler(R):
    """
    Converts a rotation matrix to Euler angles
    Matlab port to python for evaluation purposes
    https://github.com/asheshjain399/RNNexp/blob/srnn/structural_rnn/CRFProblems/H3.6m/mhmublv/Motion/RotMat2Euler.m#L1
    Args
    R: a 3x3 rotation matrix
    Returns
    eul: a 3x1 Euler angle representation of R
    """
    if R[0,2] == 1 or R[0,2] == -1:
    # special case
        E3   = 0 # set arbitrarily
        dlta = np.arctan2( R[0,1], R[0,2] )
        if R[0,2] == -1:
            E2 = np.pi/2
            E1 = E3 + dlta
        else:
            E2 = -np.pi/2
            E1 = -E3 + dlta
    else:
        E2 = -np.arcsin( R[0,2] )
        E1 = np.arctan2( R[1,2]/np.cos(E2), R[2,2]/np.cos(E2) )
        E3 = np.arctan2( R[0,1]/np.cos(E2), R[0,0]/np.cos(E2) )
    eul = np.array([E1, E2, E3])
    return eul