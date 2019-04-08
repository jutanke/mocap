import numpy as np
import numba as nb
from mocap.data.mocap import MocapHandler
from mocap.data.cmu import reflect_over_x
from pak.datasets.human36m import Human36m


left = [4, 5, 6, 11, 12, 13]
right = [1, 2, 3, 14, 15, 16]
lr = np.array(left + right, np.int64)
rl = np.array(right + left, np.int64)


@nb.njit(nb.float32[:, :, :](
    nb.float32[:, :, :]
), nogil=True)
def switch_lr(seq):
    """
    :param seq:
    :return:
    """
    global lr, rl
    seq = reflect_over_x(seq.copy())
    seq[:, lr, :] = seq[:, rl, :]
    return seq


class Human36mHandler(MocapHandler):

    @staticmethod
    def J():
        return 17

    def __init__(self, data_root, actors, allowed_actions=None, cherrypicking=None):
        assert cherrypicking is None, "no impl for this yet!"
        h36m = Human36m(data_root)
        self.allowed_actions = allowed_actions
        self.actors = actors

        if allowed_actions is None:
            pairs = [(s, a, sub_a) for s in actors for sub_a in [0, 1]
                     for a in h36m.actions]
        else:
            pairs = [(s, a, sub_a) for s in actors for sub_a in [0, 1]
                     for a in h36m.actions if a in allowed_actions]

        valid_ids = np.array([
            0, 1, 2, 3, 6, 7, 8, 12, 14, 15, 13,
            17, 18, 19, 25, 26, 27
        ], np.int32)

        sequences = {}
        for actor, action, subaction in pairs:
            seq = h36m.get_3d(actor, action, subaction)
            seq = seq[:, valid_ids, :].astype('float32') / 1000  # to mm
            sequences[actor, action, subaction] = seq

        J = Human36mHandler.J()
        super().__init__(
            sequences=sequences,
            J=J,
            j_root=0,
            j_left=4,
            j_right=1,
            cherrypicking=cherrypicking
        )

    def get_unique_identifier(self):
        allowed_actions = self.allowed_actions
        subjects = self.actors
        str_actors = "_actors" + '_'.join(sorted(subjects))
        if allowed_actions is None:
            str_actions = ''
        else:
            str_actions = '_'.join(sorted(allowed_actions))
        return 'h36m_' + str_actors + str_actions

    def flip_lr(self, seq):
        return switch_lr(seq.copy())

    def get_framerate(self, item):
        """
        :param item:
        :return:
        """
        return 50
