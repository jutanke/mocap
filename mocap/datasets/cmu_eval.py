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





