import torch
import numpy as np
import numba as nb
from torch.utils.data import Dataset
import mocap.processing.normalize as norm

class PoseDataset(Dataset):

    def __init__(self, ds, n_frames, framerates, add_noise=False,
                 noise_var=0.001, mirror_data=False):
        self.n_frames = n_frames
        self.add_noise = add_noise
        self.noise_var = noise_var
        assert ds.n_data_entries == 2 or ds.n_data_entries == 1
        self.ds = ds
        meta = []
        for seqid in range(len(ds)):
            if ds.n_data_entries == 2:
                seq, _ = ds[seqid]
            else:
                seq = ds[seqid]
            Hrz = ds.get_framerate(seqid)
            n = len(seq)
            for fr in framerates:
                ss = int(round(Hrz / fr))
                min_length = n_frames * ss
                last_possible_start_frame = n - min_length
                if last_possible_start_frame > 0:
                    for t in range(last_possible_start_frame):
                        meta.append((seqid, t, ss, False))  # seq-id, t, ss, flip-left/right
                        if mirror_data:
                            meta.append((seqid, t, ss, True))
        self.meta = meta

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, item):
        n_frames = self.n_frames
        seqid, t, ss, flip_lr = self.meta[item]
        if self.ds.n_data_entries == 2:
            seq, labels = self.ds[seqid]
        else:
            seq = self.ds[seqid]
        seq = seq[t:t + n_frames * ss:ss]
        if flip_lr:
            seq = self.ds.mirror(seq)
            seq = np.reshape(seq, (n_frames, -1))
        if self.add_noise:
            stdvar = self.noise_var
            noise = np.random.normal(scale=stdvar, size=seq.shape)
            seq += noise
           
        if self.ds.n_data_entries == 2:
            labels = labels[t:t + n_frames * ss:ss]
            return seq, labels
        else:
            return seq
 
