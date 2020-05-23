from mocap.datasets.dataset import DataSet
from os.path import isdir, join, isfile, abspath, dirname
from os import makedirs
import pickle as pkl
import numpy as np
from mocap.processing.normalize import remove_rotation_and_translation
from tqdm.auto import tqdm
from mocap.math.amass_fk import rotmat2euclidean
from mocap.math.mirror_smpl import mirror_p3d


class AMASS_SMPL3d(DataSet):
    def __init__(self, files, data_loc,
                 output_dir='/tmp/amass'):
        """
        :param data_loc: location where the original data is stored
        :param output_dir: location where we store preprocessed files for
            faster access
        """
        j_root = 0
        j_left = 1
        j_right = 2
        seqs_rot, keys = get_seqs_and_keys_rotmat(files, data_loc, output_dir)
        seqs = []
        for seq in seqs_rot:
            seq = rotmat2euclidean(seq)
            seq = remove_rotation_and_translation(
                seq, j_root=j_root, j_left=j_left, j_right=j_right
            )
            seqs.append(seq)
        super().__init__(
            Data=[seqs], Keys=keys, framerate=60,
            iterate_with_framerate=False,
            iterate_with_keys=False,
            j_root=j_root, j_left=j_left, j_right=j_right,
            n_joints=24, mirror_fn=mirror_p3d
        )
        


class AMASS(DataSet):
    def __init__(self, files, data_loc,
                 output_dir='/tmp/amass'):
        """
        :param data_loc: location where the original data is stored
        :param output_dir: location where we store preprocessed files for
            faster access
        """
        seqs, keys = get_seqs_and_keys_rotmat(files, data_loc, output_dir)
        super().__init__(
            Data=[seqs], Keys=keys, framerate=60,
            iterate_with_framerate=False,
            iterate_with_keys=False,
            j_root=0, j_left=False, j_right=False,
            n_joints=15, mirror_fn=None
        )
        

def get_seqs_and_keys_rotmat(files, data_loc, output_dir):
    assert isdir(data_loc)
    if not isdir(output_dir):
        makedirs(output_dir)

    data_loc = join(data_loc, 'synthetic60FPS/Synthetic_60FPS')
    assert isdir(data_loc)

    seqs = []
    keys = []
    for file in tqdm(files):
        keys.append(file)
        fname_prep = join(output_dir, file) + '.npy'
        if isfile(fname_prep):
            seq = np.load(fname_prep)
        else:
            loc_prep = dirname(abspath(fname_prep))
            if not isdir(loc_prep):
                makedirs(loc_prep)
            
            fname = join(data_loc, file)
            assert isfile(fname)

            # HEAVILY inspired by https://github.com/eth-ait/spl/blob/master/preprocessing/preprocess_dip.py
            # compute normalization stats online
            with open(fname, 'rb') as f:
                data = pkl.load(f, encoding='latin1')
            seq = np.array(data['poses'])  # shape (seq_length, 135)
            assert len(seq) > 0, 'file is empty'

            print('save', fname_prep)
            np.save(fname_prep, seq)
        seqs.append(seq)
    return seqs, keys