from mocap.datasets.dataset import DataSet
from os.path import isdir, join, isfile, abspath, dirname
from os import makedirs
import pickle as pkl
import numpy as np
from mocap.processing.normalize import remove_rotation_and_translation
from tqdm.auto import tqdm
from mocap.math.amass_fk import rotmat2euclidean
from mocap.math.mirror_smpl import mirror_p3d
import quaternion
import cv2


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
            iterate_with_keys=False, name='amass_smpl3d',
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
            iterate_with_keys=False, name='amass',
            j_root=0, j_left=False, j_right=False,
            n_joints=15, mirror_fn=None
        )


class AMASS_QUAT(DataSet):
    def __init__(self, files, data_loc,
                 output_dir='/tmp/amass'):
        """
        :param data_loc: location where the original data is stored
        :param output_dir: location where we store preprocessed files for
            faster access
        """
        seqs_rot, keys = get_seqs_and_keys_rotmat(files, data_loc, output_dir)
        
        seqs = []
        for key, seq_rot in tqdm(zip(keys, seqs_rot), total=len(seqs_rot)):
            fname = join(output_dir, key) + '_quat.npy'
            if isfile(fname):
                seq = np.load(fname)
            else:
                seq = rotmat2quat(seq_rot)
                np.save(fname, seq)
            seqs.append(seq)
        del seqs_rot

        super().__init__(
            Data=[seqs], Keys=keys, framerate=60,
            iterate_with_framerate=False,
            iterate_with_keys=False, name='amassq',
            j_root=0, j_left=False, j_right=False,
            n_joints=15, mirror_fn=None
        )


class AMASS_EXP(DataSet):
    def __init__(self, files, data_loc,
                 output_dir='/tmp/amass'):
        """
        :param data_loc: location where the original data is stored
        :param output_dir: location where we store preprocessed files for
            faster access
        """
        seqs_rot, keys = get_seqs_and_keys_rotmat(files, data_loc, output_dir)
        
        seqs = []
        for key, seq_rot in tqdm(zip(keys, seqs_rot), total=len(seqs_rot)):
            fname = join(output_dir, key) + '_exp.npy'
            if isfile(fname):
                seq = np.load(fname)
            else:
                seq = rotmat2aa(seq_rot).astype('float32')
                np.save(fname, seq)
            seqs.append(seq)
        del seqs_rot

        super().__init__(
            Data=[seqs], Keys=keys, framerate=60,
            iterate_with_framerate=False,
            iterate_with_keys=False, name='amassexp',
            j_root=0, j_left=False, j_right=False,
            n_joints=15, mirror_fn=None
        )


def rotmat2aa(rotmats):
    """
    Convert rotation matrices to angle-axis format.
    Args:
        oris: np array of shape (seq_length, n_joints*9).
    Returns: np array of shape (seq_length, n_joints*3)
    """
    seq_length = rotmats.shape[0]
    assert rotmats.shape[1] % 9 == 0
    n_joints = rotmats.shape[1] // 9
    ori = np.reshape(rotmats, [seq_length*n_joints, 3, 3])
    aas = np.zeros([seq_length*n_joints, 3])
    for i in range(ori.shape[0]):
        aas[i] = np.squeeze(cv2.Rodrigues(ori[i])[0])
    return np.reshape(aas, [seq_length, n_joints*3])
    

def rotmat2quat(rotmats):
    """
    Convert rotation matrices to quaternions. It ensures that there's no switch to the antipodal representation
    within this sequence of rotations.
    Args:
        oris: np array of shape (seq_length, n_joints*9).
    Returns: np array of shape (seq_length, n_joints*4)
    """
    seq_length = rotmats.shape[0]
    assert rotmats.shape[1] % 9 == 0
    ori = np.reshape(rotmats, [seq_length, -1, 3, 3])
    ori_q = quaternion.as_float_array(quaternion.from_rotation_matrix(ori))
    ori_qc = correct_antipodal_quaternions(ori_q)
    ori_qc = np.reshape(ori_qc, [seq_length, -1])
    return ori_qc


def correct_antipodal_quaternions(quat):
    """
    Removes discontinuities coming from antipodal representation of quaternions. At time step t it checks which
    representation, q or -q, is closer to time step t-1 and chooses the closest one.
    Args:
        quat: numpy array of shape (N, K, 4) where N is the number of frames and K the number of joints. K is optional,
          i.e. can be 0.
    Returns: numpy array of shape (N, K, 4) with fixed antipodal representation
    """
    assert len(quat.shape) == 3 or len(quat.shape) == 2
    assert quat.shape[-1] == 4

    if len(quat.shape) == 2:
        quat_r = quat[:, np.newaxis].copy()
    else:
        quat_r = quat.copy()

    def dist(x, y):
        return np.sqrt(np.sum((x - y) ** 2, axis=-1))

    # Naive implementation looping over all time steps sequentially.
    # For a faster implementation check the QuaterNet paper.
    quat_corrected = np.zeros_like(quat_r)
    quat_corrected[0] = quat_r[0]
    for t in range(1, quat.shape[0]):
        diff_to_plus = dist(quat_r[t], quat_corrected[t - 1])
        diff_to_neg = dist(-quat_r[t], quat_corrected[t - 1])

        # diffs are vectors
        qc = quat_r[t]
        swap_idx = np.where(diff_to_neg < diff_to_plus)
        qc[swap_idx] = -quat_r[t, swap_idx]
        quat_corrected[t] = qc
    quat_corrected = np.squeeze(quat_corrected)
    return quat_corrected

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
            seq = np.array(data['poses']).astype('float32')  # shape (seq_length, 135)
            assert len(seq) > 0, 'file is empty'
            np.save(fname_prep, seq)
        seqs.append(seq)
    return seqs, keys