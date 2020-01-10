from mocap.datasets.dataset import DataSet
import numpy as np


class Combined(DataSet):

    def __init__(self, dataset, data_target=0):
        """
        :param dataset: {mocap.datasets.dataset.DataSet}
        """
        assert data_target < dataset.n_data_entries

        if dataset.n_joints == 31:  # cmu
            translate = [
                (6, 0),
                (7, 1),
                (8, 2),
                (1, 3),
                (2, 4),
                (3, 5),
                (24, 6),
                (25, 7),
                (26, 8),
                (17, 9),
                (18, 10),
                (19, 11),
                (14, 12),
                (16, 13)
            ]
        elif dataset.n_joints == 32:  # h36m
            translate = [
                (1, 0),
                (2, 1),
                (3, 2),
                (6, 3),
                (7, 4),
                (8, 5),
                (25, 6),
                (26, 7),
                (27, 8),
                (17, 9),
                (18, 10),
                (19, 11),
                (26, 12),
                (15, 13)
            ]
        elif dataset.n_joints == 17:  # h36m simplified
            translate = [
                (1, 0),
                (2, 1),
                (3, 2),
                (4, 3),
                (5, 4),
                (6, 5),
                (14, 6),
                (15, 7),
                (16, 8),
                (11, 9),
                (12, 10),
                (13, 11),
                (8,  12),
                (10, 13)
            ]
        else:
            raise ValueError('#joints->' + str(dataset.n_joints))

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
                    
                    new_seq = np.empty((n_frames, 14, 3), np.float32)

                    for jid_src, jid_tar in translate:
                        new_seq[:, jid_tar, :] = seq[:, jid_src, :]
                    
                    if flattend:
                        new_seq = new_seq.reshape((n_frames, -1))
                    seqs.append(new_seq)
            else:
                Data_new.append(data)
        
        super().__init__(Data_new, Keys=Keys,
                         framerate=dataset.framerate,
                         iterate_with_framerate=dataset.iterate_with_framerate,
                         iterate_with_keys=dataset.iterate_with_keys,
                         j_root=-1, j_left=0, j_right=3,
                         n_joints=14,
                         mirror_fn=None)
        