import numba as nb
import numpy as np
import matplotlib.pyplot as plt
from os.path import join, dirname, isdir, isfile, abspath
from os import listdir
from transforms3d.euler import euler2mat
from mpl_toolkits.mplot3d import Axes3D
import mocap.dataaquisition.cmu as CMU_DA
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





