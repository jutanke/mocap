import numba as nb
import numpy as np
from os import listdir
from os.path import join, dirname, isdir, isfile
from tqdm import tqdm
from zipfile import ZipFile
from mocap.datasets.dataset import DataSet
import mocap.dataaquisition.h36m as H36M_DA

data_dir = join(dirname(__file__), '../data/h36m')
password_file = join(dirname(__file__), '../data/password.txt')
assert isdir(data_dir), data_dir

# -- check if we need to extract the zip files --
for subdir in ['fixed_skeleton', 'labels']:
    zip_files = [f for f in listdir(join(data_dir, subdir)) if f.endswith('.zip')]
    txt_files = [f for f in listdir(join(data_dir, subdir)) if f.endswith('.txt')]
    if len(zip_files) > len(txt_files):
        print('\n[Human3.6M] decompress data.. ->', subdir)

        assert isfile(password_file), 'could not find ' + password_file + '!!'
        password = open(password_file, 'r').read()

        for zfile in tqdm(zip_files):
            zfile = join(join(data_dir, subdir), zfile)
            zip_obj = ZipFile(zfile)
            zip_obj.extractall(join(data_dir, subdir), pwd=password.encode('utf-8'))
        print()


def get3d(actor, action, sid):
    fname = join(join(data_dir, 'fixed_skeleton'), actor + '_' + action + '_' + str(sid) + '.txt')
    seq = np.loadtxt(fname, dtype=np.float32)
    return seq


def get_expmap(actor, action, sid):
    H36M_DA.aquire_expmap()
    fname = join(join(join(data_dir, 'expmap/h3.6m/dataset'), actor), action + '_' + str(sid) + '.txt')
    seq = np.loadtxt(fname, delimiter=',', dtype=np.float32)
    return seq


def get_euler(actor, action, sid):
    H36M_DA.aquire_euler()
    fname = join(join(data_dir, 'euler'), actor + '_' + action + '_' + str(sid) + '.npy')
    seq = np.load(fname)
    return seq


def get_labels(actor, action, sid):
    fname = join(join(data_dir, 'labels'), actor + '_' + action + '_' + str(sid) + '_label.txt')
    seq = np.loadtxt(fname, dtype=np.float32)
    return seq


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
    LS = [6, 7, 8, 9, 10, 16, 17, 18, 19, 20, 21, 22, 23]
    RS = [1, 2, 3, 4, 5, 24, 25, 26, 27, 28, 29, 30, 31]
    lr = np.array(LS + RS)
    rl = np.array(RS + LS)
    n_frames = len(seq)
    x = seq.reshape((n_frames, -1, 3))
    x_copy = x.copy()
    x = reflect_over_x(x_copy)
    x[:, lr] = x[:, rl]
    return x


# =======================
# D A T A S E T S
# =======================

ACTIONS = [
    'directions',
    'discussion',
    'eating',
    'greeting',
    'phoning',
    'posing',
    'purchases',
    'sitting',
    'sittingdown',
    'smoking',
    'takingphoto',
    'waiting',
    'walking',
    'walkingdog',
    'walkingtogether'
]

ACTORS = ['S1', 'S5', 'S6', 'S7', 'S8', 'S9', 'S11']


class H36M_Simplified(DataSet):

    def __init__(self, dataset, data_target=0):
        """
        :param dataset: {mocap.datasets.dataset.DataSet}
        """
        assert data_target < dataset.n_data_entries
        used_joints = [0, 1, 2, 3, 6, 7, 8, 12, 13, 14, 15, 17, 18, 19, 25, 26, 27]
        Keys = dataset.Keys
        Data_new = []
        for did, data in enumerate(dataset.Data):
            if did == data_target:
                seqs = []
                for seq in data:
                    flattend = False
                    n_frames = len(seq)
                    if len(seq.shape) == 2:
                        flattend = True
                        seq = seq.reshape((n_frames, 32, 3))
                    seq = seq[:, used_joints, :]
                    if flattend:
                        seq = seq.reshape((n_frames, -1))
                    seqs.append(seq)
                Data_new.append(seqs)
            else:
                Data_new.append(data)
        
        super().__init__(Data_new, Keys=Keys, 
                         framerate=dataset.framerate,
                         iterate_with_framerate=dataset.iterate_with_framerate,
                         iterate_with_keys=dataset.iterate_with_keys,
                         j_root=0, j_left=4, j_right=1)


class H36M_FixedSkeleton(DataSet):

    def __init__(self, actors, actions=ACTIONS,
                 iterate_with_framerate=False,
                 iterate_with_keys=False):
        seqs = []
        keys = []
        for actor in actors:
            for action in actions:
                for sid in [1, 2]:
                    seq = get3d(actor, action, sid)
                    seqs.append(seq)
                    keys.append((actor, action, sid))
        super().__init__([seqs], Keys=keys, framerate=50, 
                         iterate_with_framerate=iterate_with_framerate,
                         iterate_with_keys=iterate_with_keys,
                         j_root=0, j_left=6, j_right=1)


class H36M_FixedSkeleton_withActivities(DataSet):

    def __init__(self, actors, actions=ACTIONS,
                 iterate_with_framerate=False,
                 iterate_with_keys=False):
        seqs = []
        labels = []
        keys = []
        for actor in actors:
            for action in actions:
                for sid in [1, 2]:
                    seq = get3d(actor, action, sid)
                    label = get_labels(actor, action, sid)
                    seqs.append(seq)
                    labels.append(label)
                    keys.append((actor, action, sid))
        super().__init__([seqs, labels], Keys=keys, framerate=50,
                         iterate_with_framerate=iterate_with_framerate,
                         iterate_with_keys=iterate_with_keys,
                         j_root=0, j_left=6, j_right=1)


