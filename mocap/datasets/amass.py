from mocap.datasets.dataset import DataSet
from os.path import isdir, join, isfile, abspath, dirname
from os import makedirs
import pickle as pkl
import numpy as np
from tqdm.auto import tqdm


class AMASS(DataSet):
    def __init__(self, files, data_loc,
                 output_dir='/tmp/amass'):
        """
        :param data_loc: location where the original data is stored
        :param output_dir: location where we store preprocessed files for
            faster access
        """
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
                # n_all, mean_all, var_all, m2_all = 0.0, 0.0, 0.0, 0.0
                # n_channel, mean_channel, var_channel, m2_channel = 0.0, 0.0, 0.0, 0.0
                # min_all, max_all = np.inf, -np.inf
                # min_seq_len, max_seq_len = np.inf, -np.inf
                with open(fname, 'rb') as f:
                    data = pkl.load(f, encoding='latin1')
                seq = np.array(data['poses'])  # shape (seq_length, 135)
                assert len(seq) > 0, 'file is empty'

                print('save', fname_prep)
                np.save(fname_prep, seq)
            seqs.append(seq)

        super().__init__(
            Data=[seqs], Keys=keys, framerate=60,
            iterate_with_framerate=False,
            iterate_with_keys=False,
            j_root=0, j_left=False, j_right=False,
            n_joints=15, mirror_fn=None
        )

        

