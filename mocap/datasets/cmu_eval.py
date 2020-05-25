import numba as nb
import numpy as np
import matplotlib.pyplot as plt
from os.path import join, dirname, isdir, isfile, abspath
from os import listdir, makedirs
from transforms3d.euler import euler2mat
from mpl_toolkits.mplot3d import Axes3D
from mocap.math.fk_cmueval import angular2euclidean
from mocap.datasets.dataset import DataSet
import mocap.processing.normalize as norm
from enum import IntEnum

class DataType(IntEnum):
    TRAIN = 1
    TEST = 2

ACTIVITES = [
    'basketball', 'basketball_signal', 'directing_traffic',
    'jumping', 'running', 'soccer', 'walking', 'walking_extra',
    'washwindow'
]


class CMUEval(DataSet):

    def __init__(self, activities, datatype):
        local_data_dir = abspath(join(dirname(__file__), '../data/cmu_eval'))
        if datatype == DataType.TEST:
            local_data_dir = join(local_data_dir, 'test')
        else:
            local_data_dir = join(local_data_dir, 'train')
        
        seqs = []
        keys = []
        for activity in activities:
            loc = join(local_data_dir, activity)
            assert isdir(loc), loc

            files = sorted([f for f in listdir(loc) if f.endswith('.txt')])
            for f in files:
                fname = join(loc, f)
                seq = np.loadtxt(fname, delimiter=',')
                keys.append(fname)
                seqs.append(seq.astype('float32'))
        
        super().__init__(
            [seqs], Keys=keys,
            framerate=120,
            iterate_with_framerate=False,
            iterate_with_keys=False,
            j_root=0, j_left=0, j_right=0,
            n_joints=38,
            mirror_fn=None)
    

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
        assert seq.shape[1] == 114, str(seq.shape)
        seq = seq.reshape((n_frames, 38, 3))
    assert len(seq.shape) == 3, str(seq.shape)
    valid_jids = [
        0,  # 0
        2,  # 1
        3,  # 2
        4,  # 3
        5,  # 4
        6,  # 5
        8,  # 6
        9,  # 7
        10, # 8
        11, # 9
        12, # 10
        14, # 11
        15, # 12
        17, # 13
        18, # 14
        19, # 15
        21, # 16
        22, # 17
        23, # 18
        25, # 19
        26, # 20
        28, # 21
        30, # 22
        31, # 23
        32, # 24
        34, # 25
        35, # 26
        37, # 27
    ]
    result = np.empty((n_frames, 28, 3), dtype=np.float32)
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
        assert seq.shape[1] == 84, str(seq.shape)
        seq = seq.reshape((n_frames, 28, 3))
    assert len(seq.shape) == 3, str(seq.shape)

    jid_map = [
        0,  # 0
        0,  # 1
        1,  # 2
        2,  # 3
        3,  # 4
        4,  # 5
        5,  # 6
        0,  # 7
        6,  # 8
        7,  # 9
        8,  # 10
        9,  # 11
        10, # 12
        0,  # 13
        11, # 14
        12, # 15
        12, # 16
        13, # 17
        14, # 18
        15, # 19
        12, # 20
        16, # 21
        17, # 22
        18, # 23
        18, # 24
        19, # 25
        20, # 26
        18, # 27
        21, # 28
        12, # 29
        22, # 30
        23, # 31
        24, # 32
        24, # 33
        25, # 34
        26, # 35
        24, # 36
        27, # 37
    ]

    result = np.empty((n_frames, 38, 3), dtype=np.float32)
    for i, j in enumerate(jid_map):
        result[:, i] = seq[:, j]
    return result.reshape((n_frames, -1))


class CMUEval3D(DataSet):

    def __init__(self, activities, datatype, data_storage_dir='/tmp'):
        local_data_dir = abspath(join(dirname(__file__), '../data/cmu_eval'))
        data_storage_dir = join(data_storage_dir, 'cmueval')
        if datatype == DataType.TEST:
            data_storage_dir = join(data_storage_dir, 'test')
            local_data_dir = join(local_data_dir, 'test')
        else:
            data_storage_dir = join(data_storage_dir, 'train')
            local_data_dir = join(local_data_dir, 'train')

        seqs = []
        keys = []
        for activity in activities:
            loc = join(local_data_dir, activity)
            assert isdir(loc), loc

            loc_npy = join(data_storage_dir, activity)
            if not isdir(loc_npy):
                makedirs(loc_npy)

            files = sorted([f for f in listdir(loc) if f.endswith('.txt')])
            for f in files:
                f_npy = join(loc_npy, f) + '.npy'
                fname = join(loc, f)
                if isfile(f_npy):
                    seq = np.load(f_npy)
                else:
                    seq = np.loadtxt(fname, delimiter=',').astype('float32')
                    seq = angular2euclidean(seq).astype('float32')
                    np.save(f_npy, seq)
                keys.append(fname)
                seqs.append(seq.astype('float32'))
        
        super().__init__(
            [seqs], Keys=keys,
            framerate=120,
            iterate_with_framerate=False,
            iterate_with_keys=False,
            j_root=-1, j_left=8, j_right=2,
            n_joints=38,
            mirror_fn=None)





