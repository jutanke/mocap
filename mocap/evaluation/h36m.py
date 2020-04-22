import numpy as np


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


def get(action, DS_class):
    """
    :param action: {String} one of the 15 actions present in the h36m dataset
    :param DS_class: {mocap::datasets::h36m::*DataSet} any h36m dataset defined
        in this library
    returns:
    Evaluation sequence for Human36M
    """
    ds_test = DS_class(actors=['S5'], actions=[action], 
                       remove_global_Rt=True)
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

    idx = find_indices_srnn(T1, T2, 256)
    
    if ds_test.n_data_entries == 2:
        Labels = []

    Seq = []
    for pos, t in enumerate(idx):
        if pos % 2 == 0:
            seq = seq1
            if ds_test.n_data_entries == 2:
                labels = labels1
        else:
            seq = seq2
            if ds_test.n_data_entries == 2:
                labels = labels2
        Seq.append(seq[t:t+150])
        if ds_test.n_data_entries == 2:
            Labels.append(labels[t:t+150])

    Seq = np.array(Seq)

    if ds_test.n_data_entries == 1:
        return Seq
    else:
        Labels = np.array(Labels)
        return Seq, Labels