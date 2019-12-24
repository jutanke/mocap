import numpy as np


class DataSet:

    def __init__(self, Data, framerate):
        """
        :param Data: [data0, data1, ...] lists of sequences, all
            dataX must have the same length. This is a list so that
            multiple things can be associated with each other, e.g.
            human poses <--> activity labels
        :param framerate: framerate in Hz for each sequence
        """
        n_sequences = -1
        for data in Data:
            if n_sequences < 0:
                n_sequences = len(data)
            else:
                assert n_sequences == len(data), 'length mismatch:' + str(n_sequences) + ' vs ' + str(len(data))

        if not isinstance(framerate, int):
            assert len(framerate) == n_sequences
        self.Data = Data
        self.framerate = framerate
        self.n_data_entries = len(Data)
        self.n_sequences = n_sequences
    
    def get_framerate(self, index):
        """ return the framerate for the given sequence
        """
        assert isinstance(index, int)
        assert index >= 0 and index < self.n_sequences, 'out of bounds: ' + str(self.n_sequences) + ' vs ' + str(index)
        if isinstance(self.framerate, int):
            return self.framerate
        else:
            return self.framerate[index]

    def get_sequence(self, index):
        """ return all data entries for the given sequence
        """
        assert isinstance(index, int)
        assert index >= 0 and index < self.n_sequences, 'out of bounds: ' + str(self.n_sequences) + ' vs ' + str(index)
        if self.n_data_entries == 1:
            return self.Data[index][0]
        result = []
        for data in self.Data:
            result.append(data[index])
        return result
    

    def __len__(self):
        return len(self.Data[0])

    def __iter__(self):
        return DataSetIterator(self)

class DataSetIterator:

    def __init__(self, dataset):
        self.dataset = dataset
        self._index = 0
    
    def __next__(self):
        if self._index < len(self.dataset):
            result = []
            for data in self.dataset.Data:
                result.append(data[self._index])
            self._index += 1
            return result
        else:
            raise StopIteration

