import numba as nb
import numpy as np
import matplotlib.pyplot as plt
from os.path import join, dirname, isdir, isfile
from os import listdir
from transforms3d.euler import euler2mat
from mpl_toolkits.mplot3d import Axes3D
import mocap.dataaquisition.cmu as CMU_DA
from mocap.datasets.dataset import DataSet


CMU_DA.aquire_cmumocap()  # load CMU data if needed

ALL_SUBJECTS = list(sorted(listdir(CMU_DA.CMU_DIR)))

def GET_ACTIONS(subject):
  subject_loc = join(CMU_DA.CMU_DIR, subject)
  assert isdir(subject_loc), subject_loc
  actions = [f[len(subject) + 1:-4] for f in list(sorted(listdir(subject_loc))) \
    if f.endswith('.amc')]
  return actions


def get(subject, action, store_binary=True, z_is_up=True):
    subject_loc = join(CMU_DA.CMU_DIR, subject)
    assert isdir(subject_loc), subject_loc

    if store_binary:
        file_meta = ''
        if z_is_up:
            file_meta = '_zup'
        npy_file = join(subject_loc, subject + '_' + action + file_meta + '.npy')
        if isfile(npy_file):
            points3d = np.load(npy_file).astype('float32')
            return points3d

    asf_file = join(subject_loc, subject + '.asf')
    amc_file = join(subject_loc, subject + '_' + action + '.amc')
    assert isfile(asf_file), asf_file
    assert isfile(amc_file), amc_file
    joints = parse_asf(asf_file)
    motions = parse_amc(amc_file)

    n_joints = 31
    n_frames = len(motions)

    points3d = np.empty((n_frames, n_joints, 3), np.float32)

    for frame, motion in enumerate(motions):
        joints['root'].set_motion(motion)
        for jid, j in enumerate(joints.values()):
            points3d[frame, jid] = np.squeeze(j.coordinate)

    if z_is_up:
        R = np.array([
            [1, 0, 0],
            [0, 0, 1],
            [0, -1, 0]
        ], np.float32)
        points3d = points3d @ R

    points3d = points3d * 0.056444  # convert inches to mm
    if store_binary:
        np.save(npy_file, points3d)  # file must not exist

    return points3d


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
    # seq_reflect = np.empty(seq.shape, np.float32)
    for frame in range(len(seq)):
        person = seq[frame]
        for jid in range(len(person)):
            pt3d = np.ascontiguousarray(person[jid])
            # seq_reflect[frame, jid] = I @ pt3d
            seq[frame, jid] = I @ pt3d

    # return seq_reflect
    return seq


def mirror(seq):
  """
  :param seq: [n_frames, 32*3]
  """
  if len(seq.shape) == 2:
      n_joints = seq.shape[1]//3
  elif len(seq.shape) == 3:
      n_joints = seq.shape[1]
  else:
      raise ValueError("incorrect shape:" + str(seq.shape))
  
  assert n_joints in [31], 'wrong joint number:' + str(n_joints)

  # left = [0, 1, 2, 3, 8, 9, 10, 11]
  # right = [4, 5, 6, 7, 12, 13, 14, 15]
  left =  [1, 2, 3, 4, 5,  17, 18, 19, 20, 21, 22, 23]
  right = [6, 7, 8, 9, 10, 24, 25, 26, 27, 28, 29, 30]
  lr = np.array(left + right, np.int64)
  rl = np.array(right + left, np.int64)

  seq_ = reflect_over_x(seq.copy())
  seq_[:, lr, :] = seq_[:, rl, :]
  return seq_


# ===================================

class CMU_DataSet(DataSet):

  def __init__(self, subjects,
               store_binary=True,
               z_is_up=True,
               iterate_with_framerate=False,
               iterate_with_keys=False):
    subjects_with_60fps = {'60', '61', '75', '87', '88', '89'}
    seqs = []
    keys = []
    framerates = []
    for subject in subjects:
      for action in GET_ACTIONS(subject):
        seq = get(subject, action, 
                  store_binary=store_binary,
                  z_is_up=z_is_up)
        seqs.append(seq)
        keys.append((subject, action))

        if subject in subjects_with_60fps:
          framerates.append(60)
        else:
          framerates.append(120)


    super().__init__([seqs], Keys=keys,
                     framerate=framerates,
                     iterate_with_framerate=iterate_with_framerate,
                     iterate_with_keys=iterate_with_keys,
                     j_root=-1, j_left=1, j_right=6,
                     mirror_fn=mirror)

# =====================================================================
# External code taken from: Yuxiao Zhou (https://calciferzh.github.io/)
# https://github.com/CalciferZh/AMCParser
# =====================================================================
class Joint:
  def __init__(self, name, direction, length, axis, dof, limits):
    """
    Definition of basic joint. The joint also contains the information of the
    bone between it's parent joint and itself. Refer
    [here](https://research.cs.wisc.edu/graphics/Courses/cs-838-1999/Jeff/ASF-AMC.html)
    for detailed description for asf files.
    Parameter
    ---------
    name: Name of the joint defined in the asf file. There should always be one
    root joint. String.
    direction: Default direction of the joint(bone). The motions are all defined
    based on this default pose.
    length: Length of the bone.
    axis: Axis of rotation for the bone.
    dof: Degree of freedom. Specifies the number of motion channels and in what
    order they appear in the AMC file.
    limits: Limits on each of the channels in the dof specification
    """
    self.name = name
    self.direction = np.reshape(direction, [3, 1])
    self.length = length
    axis = np.deg2rad(axis)
    self.C = euler2mat(*axis)
    self.Cinv = np.linalg.inv(self.C)
    self.limits = np.zeros([3, 2])
    self.dof = dof
    for lm, nm in zip(limits, dof):
      if nm == 'rx':
        self.limits[0] = lm
      elif nm == 'ry':
        self.limits[1] = lm
      else:
        self.limits[2] = lm
    self.parent = None
    self.children = []
    self.coordinate = None
    self.matrix = None

  def set_motion(self, motion):
    if self.name == 'root':
      self.coordinate = np.reshape(np.array(motion['root'][:3]), [3, 1])
      rotation = np.deg2rad(motion['root'][3:])
      self.rotation = rotation
      self.matrix = self.C.dot(euler2mat(*rotation)).dot(self.Cinv)
    else:
      idx = 0
      rotation = np.zeros(3)
      for axis, lm in enumerate(self.limits):
        if not np.array_equal(lm, np.zeros(2)):
          rotation[axis] = motion[self.name][idx]
          idx += 1
      rotation = np.deg2rad(rotation)
      self.rotation = rotation
      self.matrix = self.parent.matrix.dot(self.C).dot(euler2mat(*rotation)).dot(self.Cinv)
      self.coordinate = self.parent.coordinate + self.length * self.matrix.dot(self.direction)
    for child in self.children:
      child.set_motion(motion)

  def draw(self):
    joints = self.to_dict()
    fig = plt.figure()
    ax = Axes3D(fig)

    ax.set_xlim3d(-50, 10)
    ax.set_ylim3d(-20, 40)
    ax.set_zlim3d(-20, 40)

    xs, ys, zs = [], [], []
    for joint in joints.values():
      xs.append(joint.coordinate[0, 0])
      ys.append(joint.coordinate[1, 0])
      zs.append(joint.coordinate[2, 0])
    plt.plot(zs, xs, ys, 'b.')

    for joint in joints.values():
      child = joint
      if child.parent is not None:
        parent = child.parent
        xs = [child.coordinate[0, 0], parent.coordinate[0, 0]]
        ys = [child.coordinate[1, 0], parent.coordinate[1, 0]]
        zs = [child.coordinate[2, 0], parent.coordinate[2, 0]]
        plt.plot(zs, xs, ys, 'r')
    plt.show()

  def to_dict(self):
    ret = {self.name: self}
    for child in self.children:
      ret.update(child.to_dict())
    return ret

  def pretty_print(self):
    print('===================================')
    print('joint: %s' % self.name)
    print('direction:')
    print(self.direction)
    print('limits:', self.limits)
    print('parent:', self.parent)
    print('children:', self.children)


def read_line(stream, idx):
  if idx >= len(stream):
    return None, idx
  line = stream[idx].strip().split()
  idx += 1
  return line, idx


def parse_asf(file_path):
  '''read joint data only'''
  with open(file_path) as f:
    content = f.read().splitlines()

  for idx, line in enumerate(content):
    # meta infomation is ignored
    if line == ':bonedata':
      content = content[idx+1:]
      break

  # read joints
  joints = {'root': Joint('root', np.zeros(3), 0, np.zeros(3), [], [])}
  idx = 0
  while True:
    # the order of each section is hard-coded

    line, idx = read_line(content, idx)

    if line[0] == ':hierarchy':
      break

    assert line[0] == 'begin'

    line, idx = read_line(content, idx)
    assert line[0] == 'id'

    line, idx = read_line(content, idx)
    assert line[0] == 'name'
    name = line[1]

    line, idx = read_line(content, idx)
    assert line[0] == 'direction'
    direction = np.array([float(axis) for axis in line[1:]])

    # skip length
    line, idx = read_line(content, idx)
    assert line[0] == 'length'
    length = float(line[1])

    line, idx = read_line(content, idx)
    assert line[0] == 'axis'
    assert line[4] == 'XYZ'

    axis = np.array([float(axis) for axis in line[1:-1]])

    dof = []
    limits = []

    line, idx = read_line(content, idx)
    if line[0] == 'dof':
      dof = line[1:]
      for i in range(len(dof)):
        line, idx = read_line(content, idx)
        if i == 0:
          assert line[0] == 'limits'
          line = line[1:]
        assert len(line) == 2
        mini = float(line[0][1:])
        maxi = float(line[1][:-1])
        limits.append((mini, maxi))

      line, idx = read_line(content, idx)

    assert line[0] == 'end'
    joints[name] = Joint(
      name,
      direction,
      length,
      axis,
      dof,
      limits
    )

  # read hierarchy
  assert line[0] == ':hierarchy'

  line, idx = read_line(content, idx)

  assert line[0] == 'begin'

  while True:
    line, idx = read_line(content, idx)
    if line[0] == 'end':
      break
    assert len(line) >= 2
    for joint_name in line[1:]:
      joints[line[0]].children.append(joints[joint_name])
    for nm in line[1:]:
      joints[nm].parent = joints[line[0]]

  return joints


def parse_amc(file_path):
  with open(file_path) as f:
    content = f.read().splitlines()

  for idx, line in enumerate(content):
    if line == ':DEGREES':
      content = content[idx+1:]
      break

  frames = []
  idx = 0
  line, idx = read_line(content, idx)
  assert line[0].isnumeric(), line
  EOF = False
  while not EOF:
    joint_degree = {}
    while True:
      line, idx = read_line(content, idx)
      if line is None:
        EOF = True
        break
      if line[0].isnumeric():
        break
      joint_degree[line[0]] = [float(deg) for deg in line[1:]]
    frames.append(joint_degree)
  return frames
