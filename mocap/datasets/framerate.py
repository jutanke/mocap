from mocap.datasets.dataset import DataSet
import numpy as np
import numba as nb
from math import ceil, floor
from os.path import isfile, isdir


@nb.njit(
    nb.float32[:, :](
        nb.float32[:, :], nb.float64, nb.float64),
        nogil=True)
def interpolate(seq, target_framerate, source_framerate):
    """
    :param seq: [n_frames, n_dim]
    :param target_framerate: {float}
    :param source_framerate: {float}
    """
    n_frames, n_dim = seq.shape
    new_n_frames = int(floor(n_frames * (target_framerate/source_framerate)))
    step_size = source_framerate/target_framerate
    new_seq = np.empty((new_n_frames, n_dim), dtype=np.float32)
    current_pointer = 0.0
    for t in range(new_n_frames):
        a = floor(current_pointer)
        b = ceil(current_pointer)
        perc = current_pointer - a
        perc_inv = 1.0 - perc
        a = int(a)
        b = int(b)
        pose_a = perc_inv * seq[a]
        pose_b = perc * seq[b]
        new_seq[t] = (pose_a + pose_b) / 2.0
        current_pointer += step_size
    return new_seq


class AdaptFramerate(DataSet):

    def __init__(self, dataset, target_framerate, storage=None):
        """
        :param dataset: {DataSet}
        :param target_framerate: {int}
        """
        if storage is not None:
            assert isdir(storage), storage
        Data_new = []
        Keys = dataset.Keys
        assert dataset.n_data_entries == 1

        for data in dataset.Data:
            new_data = []
            for seqid, seq in enumerate(data):
                if storage is not None:
                    fname = '_'.join([str(k) for k in Keys[seqid]]) + '_tf' + str(target_framerate) + '.npy'
                    fname = join(storage, fname)
                    if isfile(fname):
                        seq_new = np.load(fname)
                    else:
                        source_framerate = dataset.get_framerate(seqid)
                        seq_new = interpolate(seq, target_framerate, source_framerate)
                        np.save(fname, seq_new)
                else: 
                    source_framerate = dataset.get_framerate(seqid)
                    seq_new = interpolate(seq, target_framerate, source_framerate)
                new_data.append(seq_new)
            Data_new.append(new_data)
        
        super().__init__(
            Data=Data_new, Keys=Keys,
            framerate=target_framerate,
            iterate_with_framerate=dataset.iterate_with_framerate,
            iterate_with_keys=dataset.iterate_with_keys,
            j_root=dataset.j_root, j_left=dataset.j_left,
            j_right=dataset.j_left, n_joints=dataset.n_joints,
            name=dataset.name + '_fr' + str(target_framerate),
            joints_per_limb=dataset.joints_per_limb,
            mirror_fn=dataset.mirror_fn
        )
