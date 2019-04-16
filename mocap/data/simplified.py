import numpy as np
import numba as nb
from mocap.data.mocap import MocapHandler
from mocap.data.h36m import Human36mHandler
from mocap.data.cmu import CMUHandler, reflect_over_x


left = [0, 1, 2, 6, 7, 8]
right = [3, 4, 5, 9, 10, 11]
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


class Simplified(MocapHandler):
    """

    """

    def __init__(self, proxy_mocap):
        """
        :param proxy_mocap: {MocapHandler}
        """
        self.J = 14
        self.proxy_mocap = proxy_mocap

        sequences = {}
        proxy_sequences = proxy_mocap.sequences

        if isinstance(proxy_mocap, CMUHandler):
            simplified_to_proxy = np.array([
                (0, 0), (1, 1), (2, 2),
                (3, 4), (4, 5), (5, 6),
                (6, 8), (7, 9), (8, 10),
                (9, 12), (10, 13), (11, 14),
                (12, 16), (13, 17)
            ], np.int64)
        elif isinstance(proxy_mocap, Human36mHandler):
            simplified_to_proxy = np.array([
                (0, 4), (1, 5), (2, 6),
                (3, 1), (4, 2), (5, 3),
                (6, 11), (7, 12), (8, 13),
                (9, 14), (10, 15), (11, 16),
                (12, 10), (13, 9)
            ], np.int64)
        else:
            raise NotImplementedError

        a = simplified_to_proxy[:, 0]
        b = simplified_to_proxy[:, 1]
        for key, seq_ in proxy_sequences.items():
            n = len(seq_)
            seq = np.empty((n, self.J, 3), np.float32)
            seq[:, a, :] = seq_[:, b, :].copy()
            sequences[key] = seq

        del proxy_mocap.sequences  # free memory

        super().__init__(
            sequences=sequences,
            J=self.J,
            j_root=-1,
            j_left=0,
            j_right=4
        )

    def get_framerate(self, item):
        """
        :param item:
        :return:
        """
        return self.proxy_mocap.get_framerate(item)

    def flip_lr(self, seq):
        """
        :param seq:
        :return:
        """
        return switch_lr(seq)

    def get_unique_identifier(self):
        """
        :return:
        """
        return "simple_" + self.proxy_mocap.get_unique_identifier()
