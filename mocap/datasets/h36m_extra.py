from mocap.datasets.dataset import DataSet, Limb
from mocap.datasets.h36m import reflect_over_x
import numpy as np


def mirror_p3d(seq):
    """
    :param seq: {n_frames x 17*3}
    :return:
    """
    assert len(seq.shape) == 2, str(seq.shape)
    n_frames = len(seq)
    LS = [4, 5, 6, 11, 12, 13]
    RS = [1, 2, 3, 14, 15, 16]
    lr = np.array(LS + RS)
    rl = np.array(RS + LS)
    x = seq.reshape((n_frames, -1, 3))
    x_copy = x.copy()
    x = reflect_over_x(x_copy)
    x[:, lr] = x[:, rl]
    return x.reshape((n_frames, -1))


class MotionGAN17(DataSet):

    def __init__(self, dataset, data_target=0, force_flatten=True):
        """
        :param dataset: {mocap.datasets.dataset.DataSet}
        :param force_flatten: {boolean} if True we flatten the poses
                            into vectors
        """
        assert data_target < dataset.n_data_entries
        joints_per_limb = None
        assert dataset.n_joints == 32

        #[0, 1, 2, 3, 6, 7, 8, 12, 13, 14, 15, 17, 18, 19, 25, 26, 27]
        translate = [
            (0, 0),  # center
            (1, 1),  # r
            (2, 2),  # r
            (3, 3),  # r
            (6, 4),  # l
            (7, 5),  # l
            (8, 6),  # l
            (12, 7), # center
            (13, 8), # center
            (14, 9), # center
            (15, 10), # center
            (17, 11), # l
            (18, 12), # l
            (19, 13), # l
            (25, 14), # r
            (26, 15), # r
            (27, 16)  # r
        ]

        Data_new = []
        Keys = dataset.Keys

        for did, data in enumerate(dataset.Data):
            if did == data_target:
                seqs = []
                for seq in data:
                    flattend = False
                    n_frames = len(seq)
                    if len(seq.shape) == 2:
                        flattend = True
                        seq = seq.reshape((n_frames, -1, 3))
                    
                    new_seq = np.empty((n_frames, 17, 3), np.float32)

                    for jid_src, jid_tar in translate:
                        new_seq[:, jid_tar, :] = seq[:, jid_src, :]
                    
                    if flattend or force_flatten:
                        new_seq = new_seq.reshape((n_frames, -1))
                    seqs.append(new_seq)
                Data_new.append(seqs)
            else:
                Data_new.append(data)
        
        super().__init__(Data_new, Keys=Keys,
                         framerate=dataset.framerate,
                         iterate_with_framerate=dataset.iterate_with_framerate,
                         iterate_with_keys=dataset.iterate_with_keys,
                         j_root=0, j_left=1, j_right=4,
                         n_joints=17, name=dataset.name + '_mgan',
                         mirror_fn=mirror_p3d,
                         joints_per_limb=joints_per_limb)
