from mocap.data.mocap import MocapHandler
import numpy as np
import numba as nb
import hashlib
from pak.datasets.CMU_MoCap import CMU_MoCap


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
    # seq_reflect = np.empty(seq.shape, np.float32)
    for frame in range(len(seq)):
        person = seq[frame]
        for jid in range(len(person)):
            pt3d = person[jid]
            # seq_reflect[frame, jid] = I @ pt3d
            seq[frame, jid] = I @ pt3d

    # return seq_reflect
    return seq


left = [0, 1, 2, 3, 8, 9, 10, 11]
right = [4, 5, 6, 7, 12, 13, 14, 15]
lr = np.array(left + right, np.int64)
rl = np.array(right + left, np.int64)


@nb.njit(nb.float32[:, :, :](
    nb.float32[:, :, :]
), nogil=True)
def switch_lr(seq):
    """ switches left-right
    :param seq:
    :return:
    """
    global lr, rl
    seq = reflect_over_x(seq)
    seq[:, lr, :] = seq[:, rl, :]
    return seq


class CMUHandler(MocapHandler):

    def __init__(self, data_root, subjects, allowed_actions=None, cherrypicking=None):
        """
        :param data_root:
        :param subjects:
        :param allowed_actions:
        :param cherrypicking
        """
        self.data_root = data_root
        cmu = CMU_MoCap(data_root, store_binary=True)
        self.cmu = cmu
        self.allowed_actions = allowed_actions
        self.subjects = subjects
        self.cherrypicking = cherrypicking

        if cherrypicking is None:
            if allowed_actions is None:
                pairs = [(s, a) for s in subjects
                         for a in cmu.get_actions(s)]
            else:
                pairs = [(s, a) for s in subjects
                         for a in cmu.get_actions(s) if a in allowed_actions]
        else:  # cherry picking!
            assert allowed_actions is None
            pairs = [(s, a) for s in subjects for a in cmu.get_actions(s)]
            pairs += cherrypicking
            # pairs = [(s, a) for s in subjects for a in cmu.get_actions(s) if (s, a) in cherrypicking]

        good_joints = np.array([
            1, 2, 3, 4,
            6, 7, 8, 9,
            17, 18, 19, 20,
            24, 25, 26, 27,
            14, 16
        ])

        # TODO use unified
        sequences = {}
        for s, a in pairs:
            seq = cmu.get(s, a)
            seq = seq[:, good_joints, :]
            seq *= 0.056444  # convert inches to mm
            sequences[s, a] = seq.astype('float32')

        J = 18
        super().__init__(
            sequences=sequences,
            J=J,
            j_root=-1,
            j_left=0,
            j_right=4,
            cherrypicking=cherrypicking
        )

    def get_unique_identifier(self):
        cherrypicking = self.cherrypicking
        subjects = self.subjects
        if cherrypicking is None:
            allowed_actions = self.allowed_actions
            str_actors = "_actors" + '_'.join(sorted(subjects))
            if allowed_actions is None:
                str_actions = ''
            else:
                str_actions = '_'.join(sorted(allowed_actions))
            return 'cmu_' + str_actors + str_actions
        else:  # cherry picking!
            cherrypicking = list(cherrypicking)
            str_actors = "_actors" + '_'.join(sorted(subjects)) + '_'
            str_cherry = sorted(['val' + "_".join(v) for v in cherrypicking])
            str_cherry = '__'.join(str_cherry).encode('utf-8')
            hash = hashlib.sha256(str_cherry).hexdigest()
            return 'cmu_' + str_actors + hash

    def flip_lr(self, seq):
        return switch_lr(seq.copy())

    def get_framerate(self, item):
        """
        :param item: [subject, action]
        :return:
        """
        subject, _ = item
        subjects_with_60fps = {'60', '61', '75', '87', '88', '89'}
        if subject in subjects_with_60fps:
            return 60
        else:
            return 120
