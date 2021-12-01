import numpy as np


def angular2euclidean(seq):
    """
    """
    n_frames = len(seq)
    out = []
    for t in range(n_frames):
        posexyz = fkl(seq[t])
        out.append(posexyz)
    return (np.array(out) * 0.056444).astype('float32')

def fkl( angles ):
    """
    Convert joint angles and bone lenghts into the 3d points of a person.
    Based on expmap2xyz.m, available at
    https://github.com/asheshjain399/RNNexp/blob/7fc5a53292dc0f232867beb66c3a9ef845d705cb/structural_rnn/CRFProblems/H3.6m/mhmublv/Motion/exp2xyz.m
    Args
    angles: 99-long vector with 3d position and 3d joint angles in expmap format
    parent: 32-long vector with parent-child relationships in the kinematic tree
    offset: 96-long vector with bone lenghts
    rotInd: 32-long list with indices into angles
    expmapInd: 32-long list with indices into expmap angles
    Returns
    xyz: 32x3 3d points that represent a person in 3d space
    """
    parent, offset, posInd, expmapInd = _some_variables()

    assert len(angles) == 117

    # Structure that indicates parents for each joint
    njoints   = 38
    xyzStruct = [dict() for x in range(njoints)]

    for i in np.arange( njoints ):

    # try:
    #     if not rotInd[i] : # If the list is empty
    #       xangle, yangle, zangle = 0, 0, 0
    #     else:
    #       xangle = angles[ rotInd[i][2]-1 ]
    #       yangle = angles[ rotInd[i][1]-1 ]
    #       zangle = angles[ rotInd[i][0]-1 ]
    # except:
    #    print (i)

        try:
            if not posInd[i] : # If the list is empty
                xangle, yangle, zangle = 0, 0, 0
            else:
                xangle = angles[ posInd[i][2]-1 ]
                yangle = angles[ posInd[i][1]-1 ]
                zangle = angles[ posInd[i][0]-1 ]
        except:
            print (i)

        r = angles[ expmapInd[i] ]

        thisRotation = expmap2rotmat(r)
        thisPosition = np.array([zangle, yangle, xangle])

        if parent[i] == -1: # Root node
            xyzStruct[i]['rotation'] = thisRotation
            xyzStruct[i]['xyz']      = np.reshape(offset[i,:], (1,3)) + thisPosition
        else:
            xyzStruct[i]['xyz'] = (offset[i,:] + thisPosition).dot( xyzStruct[ parent[i] ]['rotation'] ) + xyzStruct[ parent[i] ]['xyz']
            xyzStruct[i]['rotation'] = thisRotation.dot( xyzStruct[ parent[i] ]['rotation'] )

    xyz = [xyzStruct[i]['xyz'] for i in range(njoints)]
    xyz = np.array( xyz ).squeeze()
    xyz = xyz[:,[0,2,1]]

    return np.reshape( xyz, [-1] )


def _some_variables():
    """
    We define some variables that are useful to run the kinematic tree
    Args
    None
    Returns
    parent: 32-long vector with parent-child relationships in the kinematic tree
    offset: 96-long vector with bone lenghts
    rotInd: 32-long list with indices into angles
    expmapInd: 32-long list with indices into expmap angles
    """

    parent = np.array([0, 1, 2, 3, 4, 5, 6,1, 8, 9,10, 11,12, 1, 14,15,16,17,18,19, 16,
                    21,22,23,24,25,26,24,28,16,30,31,32,33,34,35,33,37])-1

    # inch_to_mm = 0.056444
    inch_to_mm = 1
    offset = inch_to_mm * np.array([0,0	,0	,0,	0,	0,	1.65674000000000,	-1.80282000000000,	0.624770000000000,	2.59720000000000,	-7.13576000000000,	0,	2.49236000000000,	-6.84770000000000,	0,	0.197040000000000,	-0.541360000000000,	2.14581000000000,	0,	0,	1.11249000000000,	0,	0,	0,	-1.61070000000000,	-1.80282000000000,	0.624760000000000,	-2.59502000000000,	-7.12977000000000,	0,	-2.46780000000000,	-6.78024000000000,	0,	-0.230240000000000,	-0.632580000000000,	2.13368000000000,	0,	0,	1.11569000000000,	0,	0,	0,	0.0196100000000000,	2.05450000000000,	-0.141120000000000,	0.0102100000000000,	2.06436000000000,	-0.0592100000000000,	0,	0,0,	0.00713000000000000,	1.56711000000000,	0.149680000000000,	0.0342900000000000,	1.56041000000000,	-0.100060000000000,	0.0130500000000000,	1.62560000000000,	-0.0526500000000000,	0,	0,	0,	3.54205000000000,	0.904360000000000,	-0.173640000000000,	4.86513000000000,	0,	0,	3.35554000000000,	0,	0	,0	,0	,0	,0.661170000000000,	0,	0,	0.533060000000000,	0,	0	,0	,0	,0	,0.541200000000000	,0	,0.541200000000000,	0	,0	,0	,-3.49802000000000,	0.759940000000000,	-0.326160000000000,	-5.02649000000000	,0	,0,	-3.36431000000000,	0,0,	0,	0	,0	,-0.730410000000000,	0,	0	,-0.588870000000000,0	,0,	0,	0	,0	,-0.597860000000000	,0	,0.597860000000000])
    offset = offset.reshape(-1,3)

    rotInd = [[6, 5, 4],
            [9, 8, 7],
            [12, 11, 10],
            [15, 14, 13],
            [18, 17, 16],
            [21, 20, 19],
            [],
            [24, 23, 22],
            [27, 26, 25],
            [30, 29, 28],
            [33, 32, 31],
            [36, 35, 34],
            [],
            [39, 38, 37],
            [42, 41, 40],
            [45, 44, 43],
            [48, 47, 46],
            [51, 50, 49],
            [54, 53, 52],
            [],
            [57, 56, 55],
            [60, 59, 58],
            [63, 62, 61],
            [66, 65, 64],
            [69, 68, 67],
            [72, 71, 70],
            [],
            [75, 74, 73],
            [],
            [78, 77, 76],
            [81, 80, 79],
            [84, 83, 82],
            [87, 86, 85],
            [90, 89, 88],
            [93, 92, 91],
            [],
            [96, 95, 94],
            []]
    posInd=[]
    for ii in np.arange(38):
        if ii==0:
            posInd.append([1,2,3])
        else:
            posInd.append([])


    expmapInd = np.split(np.arange(4,118)-1,38)

    return parent, offset, posInd, expmapInd



def rotmat2euler( R ):
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
    dlta = np.arctan2( R[0,1], R[0,2] );

    if R[0,2] == -1:
      E2 = np.pi/2;
      E1 = E3 + dlta;
    else:
      E2 = -np.pi/2;
      E1 = -E3 + dlta;

  else:
    E2 = -np.arcsin( R[0,2] )
    E1 = np.arctan2( R[1,2]/np.cos(E2), R[2,2]/np.cos(E2) )
    E3 = np.arctan2( R[0,1]/np.cos(E2), R[0,0]/np.cos(E2) )

  eul = np.array([E1, E2, E3]);
  return eul


def quat2expmap(q):
  """
  Converts a quaternion to an exponential map
  Matlab port to python for evaluation purposes
  https://github.com/asheshjain399/RNNexp/blob/srnn/structural_rnn/CRFProblems/H3.6m/mhmublv/Motion/quat2expmap.m#L1
  Args
    q: 1x4 quaternion
  Returns
    r: 1x3 exponential map
  Raises
    ValueError if the l2 norm of the quaternion is not close to 1
  """
  if (np.abs(np.linalg.norm(q)-1)>1e-3):
    print(np.linalg.norm(q))
    raise(ValueError, "quat2expmap: input quaternion is not norm 1")

  sinhalftheta = np.linalg.norm(q[1:])
  coshalftheta = q[0]

  r0    = np.divide( q[1:], (np.linalg.norm(q[1:]) + np.finfo(np.float32).eps));
  theta = 2 * np.arctan2( sinhalftheta, coshalftheta )
  theta = np.mod( theta + 2*np.pi, 2*np.pi )

  if theta > np.pi:
    theta =  2 * np.pi - theta
    r0    = -r0

  r = r0 * theta
  return r

def rotmat2quat(R):
  """
  Converts a rotation matrix to a quaternion
  Matlab port to python for evaluation purposes
  https://github.com/asheshjain399/RNNexp/blob/srnn/structural_rnn/CRFProblems/H3.6m/mhmublv/Motion/rotmat2quat.m#L4
  Args
    R: 3x3 rotation matrix
  Returns
    q: 1x4 quaternion
  """
  rotdiff = R - R.T;

  r = np.zeros(3)
  r[0] = -rotdiff[1,2]
  r[1] =  rotdiff[0,2]
  r[2] = -rotdiff[0,1]
  sintheta = np.linalg.norm(r) / 2;
  r0 = np.divide(r, np.linalg.norm(r) + np.finfo(np.float32).eps );

  costheta = (np.trace(R)-1) / 2;

  theta = np.arctan2( sintheta, costheta );

  q      = np.zeros(4)
  q[0]   = np.cos(theta/2)
  q[1:] = r0*np.sin(theta/2)
  return q

def rotmat2expmap(R):
  return quat2expmap( rotmat2quat(R) );

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
  R = np.eye(3,3) + np.sin(theta)*r0x + (1-np.cos(theta))*(r0x).dot(r0x);
  return R


def unNormalizeData(normalizedData, data_mean, data_std, dimensions_to_ignore, actions, one_hot ):
  """Borrowed from SRNN code. Reads a csv file and returns a float32 matrix.
  https://github.com/asheshjain399/RNNexp/blob/srnn/structural_rnn/CRFProblems/H3.6m/generateMotionData.py#L12
  Args
    normalizedData: nxd matrix with normalized data
    data_mean: vector of mean used to normalize the data
    data_std: vector of standard deviation used to normalize the data
    dimensions_to_ignore: vector with dimensions not used by the model
    actions: list of strings with the encoded actions
    one_hot: whether the data comes with one-hot encoding
  Returns
    origData: data originally used to
  """
  T = normalizedData.shape[0]
  D = data_mean.shape[0]

  origData = np.zeros((T, D), dtype=np.float32)
  dimensions_to_use = []
  for i in range(D):
    if i in dimensions_to_ignore:
      continue
    dimensions_to_use.append(i)
  dimensions_to_use = np.array(dimensions_to_use)

  if one_hot:
    origData[:, dimensions_to_use] = normalizedData[:, :-len(actions)]
  else:
    origData[:, dimensions_to_use] = normalizedData[:, :]

  origData = origData * np.expand_dims(data_std, 0) + np.expand_dims(data_mean, 0) 

  return origData


def revert_output_format(poses, data_mean, data_std, dim_to_ignore, actions, one_hot):
  """
  Converts the output of the neural network to a format that is more easy to
  manipulate for, e.g. conversion to other format or visualization
  Args
    poses: The output from the TF model. A list with (seq_length) entries,
    each with a (batch_size, dim) output
  Returns
    poses_out: A tensor of size (batch_size, seq_length, dim) output. Each
    batch is an n-by-d sequence of poses.
  """
  # seq_len = len(poses)
  # if seq_len == 0:
  #   return []
  #
  # batch_size, dim = poses[0].shape
  #
  # poses_out = np.concatenate(poses)
  # poses_out = np.reshape(poses_out, (seq_len, batch_size, dim))
  # poses_out = np.transpose(poses_out, [1, 0, 2])
  poses_out=poses
  poses_out_list = []
  for i in xrange(poses_out.shape[0]):
    poses_out_list.append(
      unNormalizeData(poses_out[i, :, :], data_mean, data_std, dim_to_ignore, actions, one_hot))

  return poses_out_list
