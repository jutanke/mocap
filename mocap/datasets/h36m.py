import numba as nb
import numpy as np
from os import listdir
from os.path import join, dirname, isdir, isfile
from tqdm import tqdm
from zipfile import ZipFile

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


def get_labels(actor, action, sid):
    fname = join(join(data_dir, 'labels'), actor + '_' + action + '_' + str(sid) + '.txt')
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
