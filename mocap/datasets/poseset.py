import numpy as np


class PoseSet:

    def __init__(self, dataset, data_target=0):
        """
        :param dataset: {mocap.datasets.dataset.DataSet}
        :param data_target: target data entry in the dataset
        """
        assert data_target < dataset.n_data_entries
        Data = []
        for data in dataset:
            seq = data[data_target]
            Data.append(seq)
        self.Data = np.concatenate(Data)

    def distance_to_dataset(self, pose):
        """ Calculate the distance to all elements of the dataset from {pose}.
            The distance is:
                min[ d(pose|posest), d(mirror(pose)|poseset) ]
        :param pose: {n_joints x 3} 
        """
        pass

    def distance_to_dataset_kabsch(self, pose):
        """ Calculate the distance to all elements of the dataset from {pose}.
            The distance is:
                min[ d(pose|posest), d(mirror(pose)|poseset) ]
        :param pose: {n_joints x 3} 
        """
        pass
