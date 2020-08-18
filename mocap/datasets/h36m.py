import numba as nb
import numpy as np
from os import listdir, makedirs
from os.path import join, dirname, isdir, isfile
from tqdm import tqdm
from zipfile import ZipFile
from mocap.datasets.dataset import DataSet, Limb
import mocap.math.fk as FK
import mocap.math.kabsch as KB
import mocap.processing.conversion as conv
from mocap.datasets.h36m_constants import ACTORS, LABEL_NAMES, ACTIONS, EXP_MEANPOSE, EXP_STDPOSE, EXP_DIMS2IGNORE, EXP_DIMS2USE
from mocap.dataaquisition.h36m import acquire_expmap, acquire_h36m, DATA_DIR
import mocap.processing.normalize as norm
from mocap.math.mirror_h36m import reflect_over_x, mirror_p3d, mirror_quaternion
from mocap.math.quaternion import expmap_to_quaternion, qfix

local_data_dir = join(dirname(__file__), '../data/h36m')
data_dir = DATA_DIR
password_file = join(dirname(__file__), '../data/password.txt')
assert isdir(local_data_dir), local_data_dir
if not isdir(data_dir):
    makedirs(data_dir)


def batch_remove_duplicate_joints(seq):
    """
    :param seq: [n_batch x n_frames x 96]
    """
    n_batch = seq.shape[0]
    n_frames = seq.shape[1]
    seq = seq.reshape((n_batch * n_frames, -1))
    assert seq.shape[1] == 96, str(seq.shape)
    return remove_duplicate_joints(seq).reshape((n_batch, n_frames, -1))

def remove_duplicate_joints(seq):
    """
    :param seq: [n_frames x 96]
    """
    n_frames = len(seq)
    if len(seq.shape) == 2:
        assert seq.shape[1] == 96, str(seq.shape)
        seq = seq.reshape((n_frames, 32, 3))
    assert len(seq.shape) == 3, str(seq.shape)
    valid_jids = [
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
        12, # 11
        13, # 12
        14, # 13
        15, # 14
        17, # 15
        18, # 16
        19, # 17
        21, # 18
        22, # 19
        25, # 20
        26, # 21
        27, # 22
        29, # 23
        30  # 24
    ]
    result = np.empty((n_frames, 25, 3), dtype=np.float32)
    for i, j in enumerate(valid_jids):
        result[:, i] = seq[:, j]
    return result.reshape((n_frames, -1))


def batch_recover_duplicate_joints(seq):
    """
    :param seq: [n_batch x n_frames x 75]
    """
    n_batch = seq.shape[0]
    n_frames = seq.shape[1]
    seq = seq.reshape((n_batch * n_frames, -1))
    assert seq.shape[1] == 75, str(seq.shape)
    return recover_duplicate_joints(seq).reshape((n_batch, n_frames, -1))


def recover_duplicate_joints(seq):
    """
    :param seq: [n_batch x 75]
    """
    n_frames = len(seq)
    if len(seq.shape) == 2:
        assert seq.shape[1] == 75, str(seq.shape)
        seq = seq.reshape((n_frames, 25, 3))
    assert len(seq.shape) == 3, str(seq.shape)

    jid_map = [
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
        0, # 11
        11, # 12
        12, # 13
        13, # 14
        14, # 15
        12, # 16
        15, # 17
        16, # 18
        17, # 19,
        17, # 20,
        18, # 21
        19, # 22
        19, # 23
        12, # 24
        20, # 25
        21, # 26
        22, # 27
        22, # 28
        23, # 29
        24, # 30
        24  # 31
    ]

    result = np.empty((n_frames, 32, 3), dtype=np.float32)
    for i, j in enumerate(jid_map):
        result[:, i] = seq[:, j]
    return result.reshape((n_frames, -1))


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
    acquire_h36m()
    fname = join(join(data_dir, 'p3d'), actor + '_' + action + '_' + str(sid) + '.npy')
    seq = np.load(fname)
    return seq


def get3d_fixed(actor, action, sid):
    acquire_fixed_skeleton()
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
    acquire_expmap()
    fname = join(join(join(data_dir, 'expmap/h3.6m/dataset'), actor), action + '_' + str(sid) + '.txt')
    seq = np.loadtxt(fname, delimiter=',', dtype=np.float32)
    return seq


def get_reduced_expmap(actor, action, sid):
    global EXP_MEANPOSE, EXP_DIMS2IGNORE, EXP_STDPOSE, EXP_DIMS2USE
    seq = get_expmap(actor, action, sid)
    seq = (seq - EXP_MEANPOSE) / EXP_STDPOSE
    seq = np.ascontiguousarray(seq[:, EXP_DIMS2USE])
    return seq


def batch_recover_reduced_expmap(Seq):
    """
    :param seq: n_batch x n_frames x 54
    :return:
    """
    assert len(Seq.shape) == 3 and Seq.shape[2] == 54, str(Seq.shape)
    n_batch = Seq.shape[0]
    n_frames = Seq.shape[1]
    Seq = np.reshape(Seq, (n_batch * n_frames, 54))
    Seq = np.reshape(recover_reduced_expmap(Seq), (n_batch, n_frames, 99))
    return  Seq


def recover_reduced_expmap(seq):
    """
    :param seq: n_frames x 54
    :return:
    """
    global EXP_MEANPOSE, EXP_DIMS2IGNORE, EXP_STDPOSE, EXP_DIMS2USE
    data_mean = np.expand_dims(EXP_MEANPOSE, axis=0)
    data_std = np.expand_dims(EXP_STDPOSE, axis=0)
    n_frames = seq.shape[0]
    dim_reduced = seq.shape[1]
    assert len(seq.shape) == 2 and dim_reduced == 54, 'nope..' + str(seq.shape)
    recovered = np.zeros((n_frames, 99), dtype=np.float32)
    recovered[:, EXP_DIMS2USE] = seq
    recovered = recovered * data_std + data_mean
    # recovered = recovered + data_mean
    return recovered


def get_euler(actor, action, sid):
    acquire_euler()
    fname = join(join(data_dir, 'euler'), actor + '_' + action + '_' + str(sid) + '.npy')
    seq = np.load(fname)
    return seq


def get_quaternion(actor, action, sid):
    quat_dir = join(data_dir, 'quaternions')
    if not isdir(quat_dir):
        makedirs(quat_dir)
    fname = join(quat_dir, actor + '_' + action + '_' + str(sid) + '.npy')
    if not isfile(fname):
        seq_exp = get_expmap(actor, action, sid)
        n_frames = len(seq_exp)
        seq_exp = seq_exp.reshape((n_frames, -1, 3))
        seq_exp = seq_exp[:, 1:]  # discard first entry which represents translation
        seq_quat = qfix(expmap_to_quaternion(seq_exp))
        seq_quat = seq_quat.reshape((n_frames, -1))
        np.save(fname, seq_quat)
    else:
        seq_quat = np.load(fname)
    seq_quat[:, 3:4] = 0  # ignore global rotation
    seq_quat[:, 0] = 1
    return seq_quat


@nb.njit(nb.float32[:, :](
    nb.float32[:, :]
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
    sl_dir = join(data_dir, 'labels_simple')
    fname = join(sl_dir, actor + '_' + action + '_' + str(sid) + '_label.txt')
    if not isdir(sl_dir):
        makedirs(sl_dir)
    if not isfile(fname):
        seq = get_labels(actor, action, sid)
        seq = remap_labels(seq)
        np.savetxt(fname, seq, fmt='%d')
    seq = np.loadtxt(fname, dtype=np.float32)
    return seq

# =======================
# D A T A  A Q U I S I T I O N
# =======================
#TODO this is an AWFUL fix

def acquire_fixed_skeleton():
    acquire_euler()
    global DATA_DIR
    data_dir = join(DATA_DIR, 'fixed_skeleton')
    if not isdir(data_dir):
        print('[mocap][Human3.6M] generate fixed skeletons:', data_dir)
        makedirs(data_dir)
        for actor in ACTORS:
            print('\thandle actor ', actor)
            for action in tqdm(ACTIONS):
                for sid in [1, 2]:
                    fname = join(data_dir, actor + '_' + action + '_' + str(sid) + '.txt')
                    seq1 = get3d_fixed_from_rotation(actor, action, sid)
                    seq2 = get3d(actor, action, sid)
                    assert len(seq1) == len(seq2), actor + ' ' + action + ' -> ' + str(seq1.shape) + '|' + str(seq2.shape)

                    seq1_ = []
                    for p, q in zip(seq1, seq2):
                        p = KB.rotate_P_to_Q(p, q)
                        seq1_.append(p)
                    n_frames = len(seq1)
                    seq1 = np.reshape(seq1_, (n_frames, -1))
                    np.savetxt(fname, seq1)


def acquire_fixed_skeleton_from_rotation():
    acquire_euler()
    global DATA_DIR
    data_dir = join(DATA_DIR, 'fixed_skeleton_from_rotation')
    if not isdir(data_dir):
        print('[mocap][Human3.6M] generate fixed skeletons from rotation:', data_dir)
        makedirs(data_dir)
        for actor in ACTORS:
            print('\thandle actor ', actor)
            for action in tqdm(ACTIONS):
                for sid in [1, 2]:
                    fname = join(data_dir, actor + '_' + action + '_' + str(sid) + '.npy')
                    seq1 = get3d_fixed_from_rotation(actor, action, sid)
                    np.save(fname, seq1)


def acquire_euler():
    acquire_expmap()
    global DATA_DIR
    euler_dir = join(DATA_DIR, 'euler')
    if not isdir(euler_dir):
        makedirs(euler_dir)
        for actor in ACTORS:
            for action in ACTIONS:
                for sid in [1, 2]:
                    fname = join(euler_dir, actor + '_' + action + '_' + str(sid) + '.npy')
                    if not isfile(fname):
                        print('[data aquisition] - h36m - extract euler ', (actor, action, sid))
                        exp_seq = get_expmap(actor, action, sid)
                        euler_seq = conv.expmap2euler(exp_seq).astype('float32')
                        np.save(fname, euler_seq)


# =======================
# D A T A S E T S
# =======================


class H36M_Simplified(DataSet):

    def __init__(self, dataset, data_target=0):
        """
        :param dataset: {mocap.datasets.dataset.DataSet}
        """
        joints_per_limb = {
            Limb.HEAD: [8, 9, 10],
            Limb.LEFT_ARM: [11, 12, 13],
            Limb.LEFT_LEG: [4, 5, 6],
            Limb.RIGHT_ARM: [14, 15, 16],
            Limb.RIGHT_LEG: [1, 2, 3],
            Limb.BODY: [14, 8, 7, 11, 1, 4]
        }
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
                         n_joints=17, name=dataset.name + '_sfied',
                         mirror_fn=mirror_p3d,
                         joints_per_limb=joints_per_limb)


class H36M(DataSet):

    def __init__(self, actors, actions=ACTIONS,
                 iterate_with_framerate=False,
                 iterate_with_keys=False,
                 remove_global_Rt=False):
        joints_per_limb = {
            Limb.HEAD: [15, 14, 16],
            Limb.LEFT_ARM: [17, 18, 19, 21, 22],
            Limb.LEFT_LEG: [6, 7, 8, 9, 10],
            Limb.RIGHT_ARM: [25, 26, 27, 29, 30],
            Limb.RIGHT_LEG: [1, 2, 3, 4, 5],
            Limb.BODY: [0, 1, 6, 17, 25, 12]
        }
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
                         n_joints=32, name='h36m',
                         mirror_fn=mirror_p3d,
                         joints_per_limb=joints_per_limb)


class H36M_FixedSkeleton(DataSet):

    def __init__(self, actors, actions=ACTIONS,
                 iterate_with_framerate=False,
                 iterate_with_keys=False,
                 remove_global_Rt=False):
        joints_per_limb = {
            Limb.HEAD: [15, 14, 16],
            Limb.LEFT_ARM: [17, 18, 19, 21, 22],
            Limb.LEFT_LEG: [6, 7, 8, 9, 10],
            Limb.RIGHT_ARM: [25, 26, 27, 29, 30],
            Limb.RIGHT_LEG: [1, 2, 3, 4, 5],
            Limb.BODY: [0, 1, 6, 17, 25, 12]
        }
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
                         n_joints=32, name='h36mf',
                         mirror_fn=mirror_p3d,
                         joints_per_limb=joints_per_limb)


class H36M_FixedSkeleton_withActivities(DataSet):

    def __init__(self, actors, actions=ACTIONS,
                 iterate_with_framerate=False,
                 iterate_with_keys=False,
                 remove_global_Rt=False):
        joints_per_limb = {
            Limb.HEAD: [15, 14, 16],
            Limb.LEFT_ARM: [17, 18, 19, 21, 22],
            Limb.LEFT_LEG: [6, 7, 8, 9, 10],
            Limb.RIGHT_ARM: [25, 26, 27, 29, 30],
            Limb.RIGHT_LEG: [1, 2, 3, 4, 5],
            Limb.BODY: [0, 1, 6, 17, 25, 12]
        }
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
                         n_joints=32, name='h36mf_wa',
                         mirror_fn=mirror_p3d,
                         joints_per_limb=joints_per_limb)


class H36M_ReducedExp_withSimplifiedActivities(DataSet):
    """
    Reduced as at Martinz
    """

    def __init__(self, actors, actions=ACTIONS,
                 iterate_with_framerate=False,
                 iterate_with_keys=False):
        seqs = []
        labels = []
        keys = []
        for actor in actors:
            for action in actions:
                for sid in [1, 2]:
                    seq = get_reduced_expmap(actor, action, sid)
                    label = get_simplified_labels(actor, action, sid)
                    seqs.append(seq)
                    labels.append(label)
                    keys.append((actor, action, sid))
        super().__init__([seqs, labels], Keys=keys, framerate=50,
                         iterate_with_framerate=iterate_with_framerate,
                         iterate_with_keys=iterate_with_keys,
                         j_root=0, j_left=6, j_right=1,
                         n_joints=33, name='h36mexpred_sa',
                         mirror_fn=None,
                         joints_per_limb=None)


class H36M_ReducedExp_withActivities(DataSet):
    """
    Reduced as at Martinz
    """

    def __init__(self, actors, actions=ACTIONS,
                 iterate_with_framerate=False,
                 iterate_with_keys=False):
        seqs = []
        labels = []
        keys = []
        for actor in actors:
            for action in actions:
                for sid in [1, 2]:
                    seq = get_reduced_expmap(actor, action, sid)
                    label = get_labels(actor, action, sid)
                    seqs.append(seq)
                    labels.append(label)
                    keys.append((actor, action, sid))
        super().__init__([seqs, labels], Keys=keys, framerate=50,
                         iterate_with_framerate=iterate_with_framerate,
                         iterate_with_keys=iterate_with_keys,
                         j_root=0, j_left=6, j_right=1,
                         n_joints=33, name='h36mexpred_a',
                         mirror_fn=None,
                         joints_per_limb=None)


class H36M_Exp_withSimplifiedActivities(DataSet):

    def __init__(self, actors, actions=ACTIONS,
                 iterate_with_framerate=False,
                 iterate_with_keys=False):
        joints_per_limb = {
            Limb.HEAD: [15, 14, 16],
            Limb.LEFT_ARM: [17, 18, 19, 21, 22],
            Limb.LEFT_LEG: [6, 7, 8, 9, 10],
            Limb.RIGHT_ARM: [25, 26, 27, 29, 30],
            Limb.RIGHT_LEG: [1, 2, 3, 4, 5],
            Limb.BODY: [0, 1, 6, 17, 25, 12]
        }
        seqs = []
        labels = []
        keys = []
        for actor in actors:
            for action in actions:
                for sid in [1, 2]:
                    seq = get_expmap(actor, action, sid)
                    label = get_simplified_labels(actor, action, sid)
                    seqs.append(seq)
                    labels.append(label)
                    keys.append((actor, action, sid))
        super().__init__([seqs, labels], Keys=keys, framerate=50,
                         iterate_with_framerate=iterate_with_framerate,
                         iterate_with_keys=iterate_with_keys,
                         j_root=0, j_left=6, j_right=1,
                         n_joints=33, name='h36mexp_sa',
                         mirror_fn=mirror_p3d,
                         joints_per_limb=joints_per_limb)

        
class H36M_Exp_withActivities(DataSet):

    def __init__(self, actors, actions=ACTIONS,
                 iterate_with_framerate=False,
                 iterate_with_keys=False):
        joints_per_limb = {
            Limb.HEAD: [15, 14, 16],
            Limb.LEFT_ARM: [17, 18, 19, 21, 22],
            Limb.LEFT_LEG: [6, 7, 8, 9, 10],
            Limb.RIGHT_ARM: [25, 26, 27, 29, 30],
            Limb.RIGHT_LEG: [1, 2, 3, 4, 5],
            Limb.BODY: [0, 1, 6, 17, 25, 12]
        }
        seqs = []
        labels = []
        keys = []
        for actor in actors:
            for action in actions:
                for sid in [1, 2]:
                    seq = get_expmap(actor, action, sid)
                    label = get_labels(actor, action, sid)
                    seqs.append(seq)
                    labels.append(label)
                    keys.append((actor, action, sid))
        super().__init__([seqs, labels], Keys=keys, framerate=50,
                         iterate_with_framerate=iterate_with_framerate,
                         iterate_with_keys=iterate_with_keys,
                         j_root=0, j_left=6, j_right=1,
                         n_joints=33, name='h36mexp_a',
                         mirror_fn=mirror_p3d,
                         joints_per_limb=joints_per_limb)
        

class H36M_FixedSkeleton_withSimplifiedActivities(DataSet):

    def __init__(self, actors, actions=ACTIONS,
                 iterate_with_framerate=False,
                 iterate_with_keys=False,
                 remove_global_Rt=False):
        joints_per_limb = {
            Limb.HEAD: [15, 14, 16],
            Limb.LEFT_ARM: [17, 18, 19, 21, 22],
            Limb.LEFT_LEG: [6, 7, 8, 9, 10],
            Limb.RIGHT_ARM: [25, 26, 27, 29, 30],
            Limb.RIGHT_LEG: [1, 2, 3, 4, 5],
            Limb.BODY: [0, 1, 6, 17, 25, 12]
        }
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
                         n_joints=32, name='h36mf_sa',
                         mirror_fn=mirror_p3d,
                         joints_per_limb=joints_per_limb)


class H36M_withActivities(DataSet):

    def __init__(self, actors, actions=ACTIONS,
                 iterate_with_framerate=False,
                 iterate_with_keys=False,
                 remove_global_Rt=False):
        joints_per_limb = {
            Limb.HEAD: [15, 14, 16],
            Limb.LEFT_ARM: [17, 18, 19, 21, 22],
            Limb.LEFT_LEG: [6, 7, 8, 9, 10],
            Limb.RIGHT_ARM: [25, 26, 27, 29, 30],
            Limb.RIGHT_LEG: [1, 2, 3, 4, 5],
            Limb.BODY: [0, 1, 6, 17, 25, 12]
        }
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
                         n_joints=32, name='h36m_a',
                         mirror_fn=mirror_p3d,
                         joints_per_limb=joints_per_limb)


class H36M_withSimplifiedActivities(DataSet):

    def __init__(self, actors, actions=ACTIONS,
                 iterate_with_framerate=False,
                 iterate_with_keys=False,
                 remove_global_Rt=False):
        joints_per_limb = {
            Limb.HEAD: [15, 14, 16],
            Limb.LEFT_ARM: [17, 18, 19, 21, 22],
            Limb.LEFT_LEG: [6, 7, 8, 9, 10],
            Limb.RIGHT_ARM: [25, 26, 27, 29, 30],
            Limb.RIGHT_LEG: [1, 2, 3, 4, 5],
            Limb.BODY: [0, 1, 6, 17, 25, 12]
        }
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
                         n_joints=32, name='h36m_sa',
                         mirror_fn=mirror_p3d,
                         joints_per_limb=joints_per_limb)


class H36M_FixedSkeletonFromRotation(DataSet):

    def __init__(self, actors, actions=ACTIONS,
                 iterate_with_framerate=False,
                 iterate_with_keys=False,
                 remove_global_Rt=False):
        joints_per_limb = {
            Limb.HEAD: [15, 14, 16],
            Limb.LEFT_ARM: [17, 18, 19, 21, 22],
            Limb.LEFT_LEG: [6, 7, 8, 9, 10],
            Limb.RIGHT_ARM: [25, 26, 27, 29, 30],
            Limb.RIGHT_LEG: [1, 2, 3, 4, 5],
            Limb.BODY: [0, 1, 6, 17, 25, 12]
        }
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
                         n_joints=32, name='h36mffr',
                         mirror_fn=mirror_p3d,
                         joints_per_limb=joints_per_limb)


class H36M_FixedSkeletonFromRotation_withActivities(DataSet):

    def __init__(self, actors, actions=ACTIONS,
                 iterate_with_framerate=False,
                 iterate_with_keys=False):
        joints_per_limb = {
            Limb.HEAD: [15, 14, 16],
            Limb.LEFT_ARM: [17, 18, 19, 21, 22],
            Limb.LEFT_LEG: [6, 7, 8, 9, 10],
            Limb.RIGHT_ARM: [25, 26, 27, 29, 30],
            Limb.RIGHT_LEG: [1, 2, 3, 4, 5],
            Limb.BODY: [0, 1, 6, 17, 25, 12]
        }
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
                         n_joints=32, name='h36mffr_a',
                         mirror_fn=mirror_p3d,
                         joints_per_limb=joints_per_limb)


class H36M_FixedSkeletonFromRotation_withSimplifiedActivities(DataSet):

    def __init__(self, actors, actions=ACTIONS,
                 iterate_with_framerate=False,
                 iterate_with_keys=False):
        joints_per_limb = {
            Limb.HEAD: [15, 14, 16],
            Limb.LEFT_ARM: [17, 18, 19, 21, 22],
            Limb.LEFT_LEG: [6, 7, 8, 9, 10],
            Limb.RIGHT_ARM: [25, 26, 27, 29, 30],
            Limb.RIGHT_LEG: [1, 2, 3, 4, 5],
            Limb.BODY: [0, 1, 6, 17, 25, 12]
        }
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
                         n_joints=32, name='h36mffr_sa',
                         mirror_fn=mirror_p3d,
                         joints_per_limb=joints_per_limb)


class H36M_Quaternions(DataSet):

    def __init__(self, actors, actions=ACTIONS,
                 iterate_with_framerate=False,
                 iterate_with_keys=False):
        joints_per_limb = {
            Limb.HEAD: [15, 14, 16],
            Limb.LEFT_ARM: [17, 18, 19, 21, 22],
            Limb.LEFT_LEG: [6, 7, 8, 9, 10],
            Limb.RIGHT_ARM: [25, 26, 27, 29, 30],
            Limb.RIGHT_LEG: [1, 2, 3, 4, 5],
            Limb.BODY: [0, 1, 6, 17, 25, 12]
        }
        seqs = []
        keys = []
        for actor in actors:
            for action in actions:
                for sid in [1, 2]:
                    seq = get_quaternion(actor, action, sid)
                    n_frames = len(seq)
                    seq = seq.reshape((n_frames, -1))
                    seqs.append(seq)
                    keys.append((actor, action, sid))
        super().__init__([seqs], Keys=keys, framerate=50,
                         iterate_with_framerate=iterate_with_framerate,
                         iterate_with_keys=iterate_with_keys,
                         j_root=0, j_left=6, j_right=1,
                         n_joints=32, name='h36mq',
                         mirror_fn=mirror_quaternion,
                         joints_per_limb=joints_per_limb)


class H36M_Quaternions_withSimplifiedActivities(DataSet):

    def __init__(self, actors, actions=ACTIONS,
                 iterate_with_framerate=False,
                 iterate_with_keys=False):
        joints_per_limb = {
            Limb.HEAD: [15, 14, 16],
            Limb.LEFT_ARM: [17, 18, 19, 21, 22],
            Limb.LEFT_LEG: [6, 7, 8, 9, 10],
            Limb.RIGHT_ARM: [25, 26, 27, 29, 30],
            Limb.RIGHT_LEG: [1, 2, 3, 4, 5],
            Limb.BODY: [0, 1, 6, 17, 25, 12]
        }
        seqs = []
        labels = []
        keys = []
        for actor in actors:
            for action in actions:
                for sid in [1, 2]:
                    seq = get_quaternion(actor, action, sid)
                    n_frames = len(seq)
                    seq = seq.reshape((n_frames, -1))
                    label = get_simplified_labels(actor, action, sid)
                    seqs.append(seq)
                    labels.append(label)
                    keys.append((actor, action, sid))
        super().__init__([seqs, labels], Keys=keys, framerate=50,
                         iterate_with_framerate=iterate_with_framerate,
                         iterate_with_keys=iterate_with_keys,
                         j_root=0, j_left=6, j_right=1,
                         n_joints=32, name='h36mq_sa',
                         mirror_fn=mirror_quaternion,
                         joints_per_limb=joints_per_limb)




def mirror_p3d_reduced(seq):
    """
    :param seq: {n_frames x 14*3}
    :return:
    """
    assert len(seq.shape) == 2, str(seq.shape)
    n_frames = len(seq)
    LS = [6, 7, 8, 9, 10, 15, 16, 17, 18, 19]
    RS = [1, 2, 3, 4,  5, 20, 21, 22, 23, 24]
    lr = np.array(LS + RS)
    rl = np.array(RS + LS)
    x = seq.reshape((n_frames, -1, 3))
    x_copy = x.copy()
    x = reflect_over_x(x_copy)
    x[:, lr] = x[:, rl]
    return x.reshape((n_frames, -1))


class H36M_Reduced(DataSet):

    def __init__(self, dataset, data_target=0, force_flatten=True):
        """
        :param dataset: {mocap.datasets.dataset.DataSet}
        :param force_flatten: {boolean} if True we flatten the poses
                            into vectors
        """
        joints_per_limb = {
            Limb.HEAD: [12, 13, 14],
            Limb.LEFT_ARM: [15, 16, 17, 18, 19],
            Limb.LEFT_LEG: [6, 7, 8, 9, 10],
            Limb.RIGHT_ARM: [20, 21, 22, 23, 24],
            Limb.RIGHT_LEG: [1, 2, 3, 4, 5],
            Limb.BODY: [1, 0, 6, 11, 15, 20, 12]
        }
        assert dataset.n_joints == 32
        assert data_target < dataset.n_data_entries

        Data_new = []
        Keys = dataset.Keys

        for did, data in enumerate(dataset.Data):
            if did == data_target:
                seqs = []
                for seq in data:
                    n_frames = len(seq)
                    new_seq = remove_duplicate_joints(seq)
                    if force_flatten:
                        new_seq = new_seq.reshape((n_frames, -1))
                    seqs.append(new_seq)
                Data_new.append(seqs)
            else:
                Data_new.append(data)
        
        super().__init__(Data_new, Keys=Keys,
                         framerate=dataset.framerate,
                         iterate_with_framerate=dataset.iterate_with_framerate,
                         iterate_with_keys=dataset.iterate_with_keys,
                         j_root=0, j_left=6, j_right=1,
                         n_joints=25, name=dataset.name + '_red',
                         mirror_fn=mirror_p3d_reduced,
                         joints_per_limb=joints_per_limb)
