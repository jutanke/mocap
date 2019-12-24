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
        pass

    def distance_to_dataset_kabsch(self, pose):
        pass
