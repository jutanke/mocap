from mocap.datasets.dataset import DataSet
import numpy as np
import numba as nb
from os.path import isdir, isfile, join


@nb.njit(nb.float32[:, :](nb.int64[:], nb.int64), nogil=True)
def create_onehot(labels_as_class, n_activities):
  n_frames = len(labels_as_class)
  labels = np.zeros((n_frames, n_activities), dtype=np.float32)
  for t in range(n_frames):
    label = labels_as_class[t]
    labels[t, label] = 1.0
  return labels


class CustomActivities(DataSet):
  """ Requires a directory where labels are stored based on the key
  """

  def __init__(self, dataset, activity_dir, n_activities, 
               data_target=0, prefix='', postfix=''):
    Data = dataset.Data
    Keys = dataset.Keys
    assert isdir(activity_dir), activity_dir
     
    # handle labels
    Labels = []
    for sid in range(len(dataset)):
      seq = Data[data_target][sid]
      
      key = Keys[sid]
      if isinstance(key, str):
        fname = prefix + key + postfix + '.npy'
      else:
        fname = prefix + '_'.join([str(item) for item in key]) + postfix + '.npy'
      fname = join(activity_dir, fname)
      labels_as_class = np.load(fname)
      assert len(labels_as_class) == len(seq), str(seq.shape) + ' vs ' + str(labels_as_class.shape)
      n_frames = len(seq)
      labels = create_onehot(labels_as_class, n_activities)
      Labels.append(labels)
    
    uid = str(abs(hash(activity_dir)) % (10 ** 8))
    
    Data.append(Labels)
    super().__init__(Data, Keys=Keys,
                     framerate=dataset.framerate,
                     iterate_with_framerate=dataset.iterate_with_framerate,
                     iterate_with_keys=dataset.iterate_with_keys,
                     j_root=dataset.j_root, 
                     j_left=dataset.j_left, 
                     j_right=dataset.j_right,
                     n_joints=dataset.n_joints, 
                     name=dataset.name + '_' + uid,
                     mirror_fn=dataset.mirror_fn,
                     joints_per_limb=dataset.joints_per_limb)
    
