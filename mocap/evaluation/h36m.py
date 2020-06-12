import numpy as np
import numba as nb
import math as m


def get_frames_for_short_term_evaluation_nout25_hz25():
    frames = [1, 3, 7, 9, 13, 24]
    return np.array(frames)


def find_indices_srnn(T1, T2, num_seeds):
    """ THIS METHOD IS TAKING FROM QUATERNET!
    This method replicates the behavior of the same method in
    https://github.com/una-dinosauria/human-motion-prediction
    """
    prefix_length = 50
    target_length = 100


    # Same seed as in https://github.com/asheshjain399/RNNexp/blob/srnn/structural_rnn/forecastTrajectories.py#L29
    rnd = np.random.RandomState(1234567890)

    idx = []
    for _ in range(num_seeds//2):
        idx.append(rnd.randint(16, T1-prefix_length-target_length))
        idx.append(rnd.randint(16, T2-prefix_length-target_length))
    return idx


def get(action, DS_class, actor='S5', Wrapper_class=None, Wrapper_fn=None, num_seeds=256, data_cbc=None):
    """
    :param action: {String} one of the 15 actions present in the h36m dataset
    :param DS_class: {mocap::datasets::h36m::*DataSet} any h36m dataset defined
        in this library
    :param Wrapper_class: {mocap::datasts::wrapper} any wrapper dataset, e.g. Combined
    :param Wrapper_fn: {function} takes as input the Dataset class and returns another dataset
    :param data_cbc: {function} callback with: def data_cbc(actor, action, sids, start_frames)
    returns:
    Evaluation sequence for Human36M
    """
    ds_test = DS_class(actors=[actor], actions=[action], 
                       remove_global_Rt=True)
    if Wrapper_class is not None:
        ds_test = Wrapper_class(ds_test)
    if Wrapper_fn is not None:
        ds_test = Wrapper_fn(ds_test)
    assert len(ds_test) == 2  # each action has two videos 
    if ds_test.n_data_entries == 1:
        seq1 = np.ascontiguousarray(ds_test[0][::2])  # sub-sample to 25Hz
        seq2 = np.ascontiguousarray(ds_test[1][::2])  # sub-sample to 25Hz
    else:
        seq1 = np.ascontiguousarray(ds_test[0][0][::2])  # sub-sample to 25Hz
        seq2 = np.ascontiguousarray(ds_test[1][0][::2])  # sub-sample to 25Hz
        labels1 = np.ascontiguousarray(ds_test[0][1][::2])
        labels2 = np.ascontiguousarray(ds_test[1][1][::2])

    T1 = len(seq1)
    T2 = len(seq2)

    idx = find_indices_srnn(T1, T2, num_seeds)
    
    if ds_test.n_data_entries == 2:
        Labels = []

    Seq = []
    Sids = []
    Frames = []
    for pos, t in enumerate(idx):
        Frames.append(t)
        if pos % 2 == 0:
            Sids.append(1)
            seq = seq1
            if ds_test.n_data_entries == 2:
                labels = labels1
        else:
            Sids.append(2)
            seq = seq2
            if ds_test.n_data_entries == 2:
                labels = labels2
        Seq.append(seq[t:t+150])
        if ds_test.n_data_entries == 2:
            Labels.append(labels[t:t+150])
    
    Seq = np.array(Seq)
    
    assert len(Frames) == len(Sids)
    if data_cbc is not None:
        data_cbc(actor, action, np.array(Sids), np.array(Frames))

    if ds_test.n_data_entries == 1:
        return Seq
    else:
        Labels = np.array(Labels)
        return Seq, Labels

    
@nb.njit(nb.float64[:](
    nb.float32[:, :, :], nb.float32[:, :, :]
), nogil=True)
def calculate_euclidean_distance(Y, Y_hat):
    """
    :param Y: [n_sequences, n_frames, 96]
    :param Y_hat: [n_sequences, n_frames, 96]
    :return : [n_frames]
    """
    n_sequences, n_frames, dim = Y.shape
    J = dim // 3
    result = np.zeros(shape=(n_frames,), dtype=np.float64)
    for s in range(n_sequences):
        for t in range(n_frames):
            total_euc_error = 0
            for jid in range(0, dim, 3):
                x_gt = Y[s, t, jid]
                y_gt = Y[s, t, jid + 1]
                z_gt = Y[s, t, jid + 2]
                x_pr = Y_hat[s, t, jid]
                y_pr = Y_hat[s, t, jid + 1]
                z_pr = Y_hat[s, t, jid + 2]
                val = (x_gt - x_pr) ** 2 + \
                      (y_gt - y_pr) ** 2 + \
                      (z_gt - z_pr) ** 2
                val = max(val, 0.00000001)
                euc = m.sqrt(val)
                total_euc_error += euc
            result[t] += (total_euc_error / J)
    result = result / n_sequences
    return result
