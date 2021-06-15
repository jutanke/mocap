from mocap.datasets.dataset import DataSet, Limb
from os.path import isdir, isfile, join, isfile, abspath, dirname
from os import listdir
from os import makedirs
import numpy as np
from subprocess import PIPE, run
from tqdm import tqdm
from typing import List
import mocap.processing.normalize as norm


from mocap.visualization.sequence import SequenceVisualizer


def get_seqs_and_keys(amass_path, datasets, without_hands=False):
    """"""
    seqs = []
    keys = []
    for dataset in sorted(datasets):
        path = join(amass_path, dataset)
        assert isdir(path), path
        for subdir in [join(path, f) for f in sorted(listdir(path))]:
            for seq_fname in [join(subdir, f) for f in sorted(listdir(subdir))]:
                seq = np.load(seq_fname)
                if without_hands:
                    seq = seq[:, :22]
                n_frames = len(seq)
                seq = seq.reshape(n_frames, -1)
                seqs.append(seq)
                keys.append(seq_fname)
    return seqs, keys


def get_seqs_and_keys_from_exact_fnames(amass_path, exact_fnames, without_hands=False):
    """
    sample from a dataset only the given files
    """
    seqs = []
    keys = []
    for fname in sorted(exact_fnames):
        seq_fname = join(amass_path, fname)
        # assert isfile(seq_fname), seq_fname
        if isfile(seq_fname):
            seq = np.load(seq_fname)
            if without_hands:
                seq = seq[:, :22]
            n_frames = len(seq)
            seq = seq.reshape(n_frames, -1)
            seqs.append(seq)
            keys.append(seq_fname)
    return seqs, keys


class AMASS(DataSet):
    def __init__(
        self,
        amass_path: str,
        datasets: List[str],
        exact_files: List[str],
        remove_global_Rt=True,
    ):
        """"""
        assert isdir(amass_path)
        assert amass_path.split("/")[-1] == "amass2skel"

        seqs_ds = []
        keys_ds = []
        if len(datasets) > 0:
            seqs_ds, keys_ds = get_seqs_and_keys(
                amass_path, datasets, without_hands=True
            )
            if remove_global_Rt:
                seqs_ds = [
                    norm.remove_rotation_and_translation(
                        seq, j_root=0, j_left=1, j_right=2
                    )
                    for seq in seqs_ds
                ]
        seqs_ex = []
        keys_ex = []
        if len(exact_files) > 0:
            seqs_ex, keys_ex = get_seqs_and_keys_from_exact_fnames(
                amass_path, exact_files, without_hands=True
            )
            if remove_global_Rt:
                seqs_ex = [
                    norm.remove_rotation_and_translation(
                        seq, j_root=0, j_left=1, j_right=2
                    )
                    for seq in seqs_ex
                ]

        seqs = seqs_ds + seqs_ex
        keys = keys_ds + keys_ex

        super().__init__(
            Data=[seqs],
            Keys=keys,
            framerate=60,
            iterate_with_framerate=False,
            iterate_with_keys=False,
            name="amass_new",
            j_root=0,
            j_left=1,
            j_right=2,
            n_joints=22,
            joints_per_limb=None,
        )


class AMASS_withHands(DataSet):
    def __init__(self, amass_path: str, datasets: List[str]):
        """"""
        assert isdir(amass_path)
        assert amass_path.split("/")[-1] == "amass2skel"

        seqs, keys = get_seqs_and_keys(amass_path, datasets)

        super().__init__(
            Data=[seqs],
            Keys=keys,
            framerate=60,
            iterate_with_framerate=False,
            iterate_with_keys=False,
            name="amass_new_with_hands",
            j_root=0,
            j_left=1,
            j_right=2,
            n_joints=52,
            joints_per_limb=None,
        )
