import numba as nb
import numpy as np
from os import listdir, makedirs
from os.path import join, dirname, isdir, isfile
from tqdm import tqdm
from zipfile import ZipFile
from mocap.datasets.dataset import DataSet
import mocap.math.fk as FK
import mocap.dataaquisition.h36m as H36M_DA
import mocap.processing.normalize as norm

local_data_dir = join(dirname(__file__), '../data/h36m')
data_dir = H36M_DA.DATA_DIR
password_file = join(dirname(__file__), '../data/password.txt')
assert isdir(local_data_dir), local_data_dir
if not isdir(data_dir):
    makedirs(data_dir)

CACHE_get3d_fixed_from_rotation = {}

# -- check if we need to extract the zip files --
for subdir, needs_password in zip(['labels'], [True]):

    if not isdir(join(data_dir, subdir)):
        makedirs(join(data_dir, subdir))

    zip_files = [f for f in listdir(join(local_data_dir, subdir)) if f.endswith('.zip')]
    txt_files = [f for f in listdir(join(data_dir, subdir)) if f.endswith('.txt')]

    if len(zip_files) > len(txt_files):
        print('\n[mocap][Human3.6M] decompress data.. ->', subdir)

        if needs_password:
            if not isfile(password_file):
                print("\t\n\033[1m\033[93mCannot find password file... skipping decompression\033[0m")
                print()
                continue
            assert isfile(password_file), 'could not find ' + password_file + '!!'
            password = open(password_file, 'r').read()

        for zfile in tqdm(zip_files):
            zfile = join(join(local_data_dir, subdir), zfile)
            zip_obj = ZipFile(zfile)
            if needs_password:
                zip_obj.extractall(join(data_dir, subdir), pwd=password.encode('utf-8'))
            else:
                zip_obj.extractall(join(data_dir, subdir))
        print()


def get3d(actor, action, sid):
    H36M_DA.acquire_h36m()
    fname = join(join(data_dir, 'p3d'), actor + '_' + action + '_' + str(sid) + '.npy')
    seq = np.load(fname)
    return seq


def get3d_fixed(actor, action, sid):
    H36M_DA.acquire_fixed_skeleton()
    fname_binary = join(join(data_dir, 'fixed_skeleton'), actor + '_' + action + '_' + str(sid) + '.npy')
    if isfile(fname_binary):
        seq = np.load(fname_binary)
    else:
        fname = join(join(data_dir, 'fixed_skeleton'), actor + '_' + action + '_' + str(sid) + '.txt')
        seq = np.loadtxt(fname, dtype=np.float32)
        np.save(fname_binary, seq)
    return seq


def get3d_fixed_from_rotation(actor, action, sid):
    loc = join(data_dir, 'fixed_skeleton_from_rotation')
    fname = join(loc, actor + '_' + action + '_' + str(sid) + '.txt')
    if isfile(fname):
        seq = np.load(fname)
        n_frames = len(seq)
        seq = seq.reshape((n_frames, -1))
        return seq
    else:
        if not isdir(loc):
            makedirs(loc)
        global CACHE_get3d_fixed_from_rotation
        if (actor, action, sid) not in CACHE_get3d_fixed_from_rotation:
            seq = get_euler(actor, action, sid)
            seq = FK.euler_fk(seq)
            seq = reflect_over_x(seq)
            seq = mirror_p3d(seq)  # there are some mirroring issues in the original rotational data:
            # https://github.com/una-dinosauria/human-motion-prediction/issues/46
            seq = seq.astype('float32')
            np.save(fname, seq)
            n_frames = len(seq)
            seq = seq.reshape((n_frames, -1))
            CACHE_get3d_fixed_from_rotation[actor, action, sid] = seq
        return CACHE_get3d_fixed_from_rotation[actor, action, sid]


def get_expmap(actor, action, sid):
    H36M_DA.acquire_expmap()
    fname = join(join(join(data_dir, 'expmap/h3.6m/dataset'), actor), action + '_' + str(sid) + '.txt')
    seq = np.loadtxt(fname, delimiter=',', dtype=np.float32)
    return seq


def get_euler(actor, action, sid):
    H36M_DA.acquire_euler()
    fname = join(join(data_dir, 'euler'), actor + '_' + action + '_' + str(sid) + '.npy')
    seq = np.load(fname)
    return seq


@nb.njit(nb.float32[:, :, :](
    nb.float32[:, :, :]
), nogil=True)
def remap_labels(Labels):
    """
    :param Labels: {n_frames x 11}
    :return: {n_frames x 10}
    """
    # 'kneeling',  # 0
    # 'kneeling down',  # 1
    # 'leaning down',  # 2
    # 'sitting chair',  # 3
    # 'sitting down',  # 4
    # 'sitting floor',  # 5
    # 'squatting',  # 6
    # 'standing',  # 7
    # 'standing up',  # 8
    # 'steps',  # 9
    # 'walking']  # 10
    mapping = [
        5,  # 0 => kneeling --> sitting midway
        2,  # 1 => kneeling down --> sitting down
        7,  # 2 => leaning down --> leaning down
        3,  # 3 => sitting chair --> sitting chair
        2,  # 4 => sitting down --> sitting down
        4,  # 5 => sitting floor --> sitting floor
        5,  # 6 => squatting --> sitting midway
        0,  # 7 => standing --> standing
        6,  # 8 => standing up --> standing up
        0,  # 9 => steps --> standing
        1]  # 10=> walking --> walking
    n_frames = Labels.shape[0]
    result = np.zeros((n_frames, 8), np.float32)
    for t in range(n_frames):
        src_label = np.argmax(Labels[t])
        target_label = mapping[src_label]
        result[t, target_label] = 1.0
    return result


def get_labels(actor, action, sid):
    fname = join(join(data_dir, 'labels'), actor + '_' + action + '_' + str(sid) + '_label.txt')
    seq = np.loadtxt(fname, dtype=np.float32)
    return seq


def get_simplified_labels(actor, action, sid):
    fname = join(join(data_dir, 'labels_simple'), actor + '_' + action + '_' + str(sid) + '_label.txt')
    if not isfile(fname):
        seq = get_labels(actor, action, sid)
        seq = remap_labels(seq)
        np.savetxt(fname, seq, fmt='d')
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

LABEL_NAMES = [
    'kneeling',         # 0
    'kneeling down',    # 1
    'leaning down',     # 2
    'sitting chair',    # 3
    'sitting down',     # 4
    'sitting floor',    # 5
    'squatting',        # 6
    'standing',         # 7
    'standing up',      # 8
    'steps',            # 9
    'walking']          # 10


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
                         j_root=0, j_left=4, j_right=1,
                         n_joints=17,
                         mirror_fn=mirror_p3d)


class H36M(DataSet):

    def __init__(self, actors, actions=ACTIONS,
                 iterate_with_framerate=False,
                 iterate_with_keys=False,
                 remove_global_Rt=False):
        seqs = []
        keys = []
        for actor in actors:
            for action in actions:
                for sid in [1, 2]:
                    seq = get3d(actor, action, sid)
                    if remove_global_Rt:
                        seq = norm.remove_rotation_and_translation(seq, j_root=0, j_left=6, j_right=1)
                    seqs.append(seq)
                    keys.append((actor, action, sid))
        super().__init__([seqs], Keys=keys, framerate=50,
                         iterate_with_framerate=iterate_with_framerate,
                         iterate_with_keys=iterate_with_keys,
                         j_root=0, j_left=6, j_right=1,
                         n_joints=32,
                         mirror_fn=mirror_p3d)


class H36M_FixedSkeleton(DataSet):

    def __init__(self, actors, actions=ACTIONS,
                 iterate_with_framerate=False,
                 iterate_with_keys=False,
                 remove_global_Rt=False):
        seqs = []
        keys = []
        for actor in actors:
            for action in actions:
                for sid in [1, 2]:
                    seq = get3d_fixed(actor, action, sid)
                    if remove_global_Rt:
                        seq = norm.remove_rotation_and_translation(seq, j_root=0, j_left=6, j_right=1)
                    seqs.append(seq)
                    keys.append((actor, action, sid))
        super().__init__([seqs], Keys=keys, framerate=50, 
                         iterate_with_framerate=iterate_with_framerate,
                         iterate_with_keys=iterate_with_keys,
                         j_root=0, j_left=6, j_right=1,
                         n_joints=32,
                         mirror_fn=mirror_p3d)


class H36M_FixedSkeleton_withActivities(DataSet):

    def __init__(self, actors, actions=ACTIONS,
                 iterate_with_framerate=False,
                 iterate_with_keys=False,
                 remove_global_Rt=False):
        seqs = []
        labels = []
        keys = []
        for actor in actors:
            for action in actions:
                for sid in [1, 2]:
                    seq = get3d_fixed(actor, action, sid)
                    if remove_global_Rt:
                        seq = norm.remove_rotation_and_translation(seq, j_root=0, j_left=6, j_right=1)
                    label = get_labels(actor, action, sid)
                    seqs.append(seq)
                    labels.append(label)
                    keys.append((actor, action, sid))
        super().__init__([seqs, labels], Keys=keys, framerate=50,
                         iterate_with_framerate=iterate_with_framerate,
                         iterate_with_keys=iterate_with_keys,
                         j_root=0, j_left=6, j_right=1,
                         n_joints=32,
                         mirror_fn=mirror_p3d)


class H36M_FixedSkeleton_withSimplifiedActivities(DataSet):

    def __init__(self, actors, actions=ACTIONS,
                 iterate_with_framerate=False,
                 iterate_with_keys=False,
                 remove_global_Rt=False):
        seqs = []
        labels = []
        keys = []
        for actor in actors:
            for action in actions:
                for sid in [1, 2]:
                    seq = get3d_fixed(actor, action, sid)
                    if remove_global_Rt:
                        seq = norm.remove_rotation_and_translation(seq, j_root=0, j_left=6, j_right=1)
                    label = get_simplified_labels(actor, action, sid)
                    seqs.append(seq)
                    labels.append(label)
                    keys.append((actor, action, sid))
        super().__init__([seqs, labels], Keys=keys, framerate=50,
                         iterate_with_framerate=iterate_with_framerate,
                         iterate_with_keys=iterate_with_keys,
                         j_root=0, j_left=6, j_right=1,
                         n_joints=32,
                         mirror_fn=mirror_p3d)


class H36M_withActivities(DataSet):

    def __init__(self, actors, actions=ACTIONS,
                 iterate_with_framerate=False,
                 iterate_with_keys=False,
                 remove_global_Rt=False):
        seqs = []
        labels = []
        keys = []
        for actor in actors:
            for action in actions:
                for sid in [1, 2]:
                    seq = get3d(actor, action, sid)
                    label = get_labels(actor, action, sid)
                    seqs.append(seq)
                    if remove_global_Rt:
                        seq = norm.remove_rotation_and_translation(seq, j_root=0, j_left=6, j_right=1)
                    labels.append(label)
                    keys.append((actor, action, sid))
        super().__init__([seqs, labels], Keys=keys, framerate=50,
                         iterate_with_framerate=iterate_with_framerate,
                         iterate_with_keys=iterate_with_keys,
                         j_root=0, j_left=6, j_right=1,
                         n_joints=32,
                         mirror_fn=mirror_p3d)


class H36M_withSimplifiedActivities(DataSet):

    def __init__(self, actors, actions=ACTIONS,
                 iterate_with_framerate=False,
                 iterate_with_keys=False,
                 remove_global_Rt=False):
        seqs = []
        labels = []
        keys = []
        for actor in actors:
            for action in actions:
                for sid in [1, 2]:
                    seq = get3d(actor, action, sid)
                    label = get_simplified_labels(actor, action, sid)
                    if remove_global_Rt:
                        seq = norm.remove_rotation_and_translation(seq, j_root=0, j_left=6, j_right=1)
                    seqs.append(seq)
                    labels.append(label)
                    keys.append((actor, action, sid))
        super().__init__([seqs, labels], Keys=keys, framerate=50,
                         iterate_with_framerate=iterate_with_framerate,
                         iterate_with_keys=iterate_with_keys,
                         j_root=0, j_left=6, j_right=1,
                         n_joints=32,
                         mirror_fn=mirror_p3d)


class H36M_FixedSkeletonFromRotation(DataSet):

    def __init__(self, actors, actions=ACTIONS,
                 iterate_with_framerate=False,
                 iterate_with_keys=False,
                 remove_global_Rt=False):
        seqs = []
        keys = []
        for actor in actors:
            for action in actions:
                for sid in [1, 2]:
                    seq = get3d_fixed_from_rotation(actor, action, sid)
                    if remove_global_Rt:
                        seq = norm.remove_rotation_and_translation(seq, j_root=0, j_left=6, j_right=1)
                    seqs.append(seq)
                    keys.append((actor, action, sid))
        super().__init__([seqs], Keys=keys, framerate=50,
                         iterate_with_framerate=iterate_with_framerate,
                         iterate_with_keys=iterate_with_keys,
                         j_root=0, j_left=6, j_right=1,
                         n_joints=32,
                         mirror_fn=mirror_p3d)


class H36M_FixedSkeletonFromRotation_withActivities(DataSet):

    def __init__(self, actors, actions=ACTIONS,
                 iterate_with_framerate=False,
                 iterate_with_keys=False):
        seqs = []
        labels = []
        keys = []
        for actor in actors:
            for action in actions:
                for sid in [1, 2]:
                    seq = get3d_fixed_from_rotation(actor, action, sid)
                    label = get_labels(actor, action, sid)
                    seqs.append(seq)
                    labels.append(label)
                    keys.append((actor, action, sid))
        super().__init__([seqs, labels], Keys=keys, framerate=50,
                         iterate_with_framerate=iterate_with_framerate,
                         iterate_with_keys=iterate_with_keys,
                         j_root=0, j_left=6, j_right=1,
                         n_joints=32,
                         mirror_fn=mirror_p3d)


class H36M_FixedSkeletonFromRotation_withSimplifiedActivities(DataSet):

    def __init__(self, actors, actions=ACTIONS,
                 iterate_with_framerate=False,
                 iterate_with_keys=False):
        seqs = []
        labels = []
        keys = []
        for actor in actors:
            for action in actions:
                for sid in [1, 2]:
                    seq = get3d_fixed_from_rotation(actor, action, sid)
                    label = get_simplified_labels(actor, action, sid)
                    seqs.append(seq)
                    labels.append(label)
                    keys.append((actor, action, sid))
        super().__init__([seqs, labels], Keys=keys, framerate=50,
                         iterate_with_framerate=iterate_with_framerate,
                         iterate_with_keys=iterate_with_keys,
                         j_root=0, j_left=6, j_right=1,
                         n_joints=32,
                         mirror_fn=mirror_p3d)
