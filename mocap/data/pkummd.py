import numpy as np
import numba as nb
import hashlib
from mocap.data.mocap import MocapHandler
from mocap.data.cmu import reflect_over_x
from pak.datasets.PKU_MMD import PKU_MMD


class PKUMMDHandler(MocapHandler):

    @staticmethod
    def J():
        return 25

    def __init__(self, data_root, pids):
        """

        :param data_root:
        :param pids: actors
        """
        self.pids = sorted(self.pids)
        txt = '++'.join(self.pids).encode('utf-8')
        hash_object = hashlib.sha384(txt)
        self.unique_id = hash_object.hexdigest()

        data = PKU_MMD(data_root)

    def get_unique_identifier(self):
        return self.unique_id
