import numpy as np
import hashlib


class DataSet:

    def __init__(self, Data, Keys, framerate, iterate_with_framerate,
                 iterate_with_keys, j_root, j_left, j_right,
                 n_joints,
                 mirror_fn=None):
        """
        :param Data: [data0, data1, ...] lists of sequences, all
            dataX must have the same length. This is a list so that
            multiple things can be associated with each other, e.g.
            human poses <--> activity labels
        :param Keys: key that uniquly identifies the video
        :param framerate: framerate in Hz for each sequence
        :param iterate_with_framerate: if True the iterator returns the framerate as well
        :param iterate_with_keys: if True the iterator returns the key as well
        :param mirror_fn: def mirror(seq): -->
        :param n_joints:
        """
        self.n_joints = n_joints
        self.iterate_with_framerate = iterate_with_framerate
        self.iterate_with_keys = iterate_with_keys
        self.j_root = j_root
        self.j_left = j_left
        self.j_right = j_right
        self.mirror_fn = mirror_fn
        n_sequences = -1
        for data in Data:
            if n_sequences < 0:
                n_sequences = len(data)
            else:
                assert n_sequences == len(data), 'length mismatch:' + str(n_sequences) + ' vs ' + str(len(data))
        assert n_sequences == len(Keys), str(n_sequences) + ' vs ' + str(len(Keys))
        if not isinstance(framerate, int):
            assert len(framerate) == n_sequences
        self.Data = Data
        self.Keys = Keys
        self.framerate = framerate
        self.n_data_entries = len(Data)
        self.n_sequences = n_sequences
    
    def mirror(self, seq):
        assert self.mirror_fn is not None
        return self.mirror_fn(seq)
    
    def get_unique_id(self):
        txt = ''
        for key in sorted(self.Keys):
            txt += str(key)
        return hashlib.sha256(txt.encode('utf-8')).hexdigest()
    
    def get_framerate(self, index):
        """ return the framerate for the given sequence
        """
        assert isinstance(index, int)
        assert index >= 0 and index < self.n_sequences, 'out of bounds: ' + str(self.n_sequences) + ' vs ' + str(index)
        if isinstance(self.framerate, int):
            return self.framerate
        else:
            return self.framerate[index]
    
    def normalize_per_joint(self, other_dataset=None, dataid=0):
        """ WARNING: this function overwrites 
        :param other_dataset: {Dataset} normalize based on other dataset (relevant for train/test splits)
        :param dataid: {int} in what "data folder" is the data stored?
        """

        for dd in self.Data[dataid]:
            print('d', len(dd))
        assert not self.is_joint_normalized
        data = np.concatenate(self.Data[dataid], axis=0)
        n_data = len(data)
        data = data.reshape((n_data, self.n_joints, -1))
        mu = np.expand_dims(np.mean(data, axis=0), axis=0)
        self.joint_mu = mu
        data = data - mu

        exit(1)

    def get_sequence(self, index):
        """ return all data entries for the given sequence
        """
        assert isinstance(index, int)
        assert index >= 0 and index < self.n_sequences, 'out of bounds: ' + str(self.n_sequences) + ' vs ' + str(index)
        if self.n_data_entries == 1:
            return self.Data[0][index]
        result = []
        for data in self.Data:
            result.append(data[index])
        return result
    
    def __getitem__(self, key):
        return self.get_sequence(key)

    def __len__(self):
        return len(self.Data[0])

    def __iter__(self):
        if self.iterate_with_framerate and self.iterate_with_keys:
            return DataSetWithFramerateWithKeysIterator(self)
        elif self.iterate_with_framerate:
            return DataSetWithFramerateIterator(self)
        elif self.iterate_with_keys:
            return DataSetWithKeysIterator(self)
        else:
            return DataSetIterator(self)


class Dataset_NormalizedJoints(DataSet):
    """
    Normalize data on joint level
    """

    def __init__(self, ds, dataid=0):
        Data = ds.Data
        indices = []
        for dd in Data[dataid]:
            indices.append(len(dd))
        data = np.concatenate(Data[dataid], axis=0)
        n_data = len(data)
        data = data.reshape((n_data, ds.n_joints, -1))
        mu = np.expand_dims(np.mean(data, axis=0), axis=0)
        data = data - mu
        self.mu = mu
        Seq_new = []
        start = 0
        end = 0
        for n_frames in indices:
            end += n_frames
            seq = data[start:end].reshape((n_frames, -1))
            Seq_new.append(seq)
            start += n_frames
        assert end == len(data)
        Data[dataid] = Seq_new
        super().__init__(
            Data=Data, Keys=ds.Keys,
            framerate=ds.framerate,
            iterate_with_framerate=ds.iterate_with_framerate,
            iterate_with_keys=ds.iterate_with_keys,
            j_root=ds.j_root, j_left=ds.j_left, j_right=ds.j_right,
            n_joints=ds.n_joints, mirror_fn=ds.mirror_fn
        )

    def normalize(self, seq):
        """
        :param seq: [n_frames x dim]
        """
        assert len(seq.shape) == 2, str(seq.shape)
        n_frames = len(seq)
        n_joints = self.n_joints
        seq = seq.reshape((n_frames, n_joints, -1))
        assert seq.shape[2] == 3 or seq.shape[2] == 4, str(seq.shape)
        seq = seq - self.mu
        seq = seq.reshape((n_frames, -1))
        return seq
    
    def denormalize(self, seq):
        """
        :param seq: [n_frames x dim]
        """
        assert len(seq.shape) == 2, str(seq.shape)
        n_frames = len(seq)
        n_joints = self.n_joints
        seq = seq.reshape((n_frames, n_joints, -1))
        assert seq.shape[2] == 3 or seq.shape[2] == 4, str(seq.shape)
        seq = seq + self.mu
        seq = seq.reshape((n_frames, -1))
        return seq


class Dataset_Normalized(DataSet):
    """
    Normalize data on joint level
    """

    def __init__(self, ds, dataid=0):
        Data = ds.Data
        indices = []
        for dd in Data[dataid]:
            indices.append(len(dd))
        data = np.concatenate(Data[dataid], axis=0)
        n_data = len(data)
        mu = np.expand_dims(np.mean(data, axis=0), axis=0)
        data = data - mu
        self.mu = mu
        Seq_new = []
        start = 0
        end = 0
        for n_frames in indices:
            end += n_frames
            seq = data[start:end]
            Seq_new.append(seq)
            start += n_frames
        assert end == len(data)
        Data[dataid] = Seq_new
        super().__init__(
            Data=Data, Keys=ds.Keys,
            framerate=ds.framerate,
            iterate_with_framerate=ds.iterate_with_framerate,
            iterate_with_keys=ds.iterate_with_keys,
            j_root=ds.j_root, j_left=ds.j_left, j_right=ds.j_right,
            n_joints=ds.n_joints, mirror_fn=ds.mirror_fn
        )

    def normalize(self, seq):
        """
        :param seq: [n_frames x dim]
        """
        assert len(seq.shape) == 2, str(seq.shape)
        seq = seq - self.mu
        return seq

    def denormalize(self, seq):
        """
        :param seq: [n_frames x dim]
        """
        assert len(seq.shape) == 2, str(seq.shape)
        seq = seq + self.mu
        return seq


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


class DataSetWithKeysIterator:

    def __init__(self, dataset):
        self.dataset = dataset
        self._index = 0
    
    def __next__(self):
        if self._index < len(self.dataset):
            result = []
            for data in self.dataset.Data:
                result.append(data[self._index])
            result.append(self.dataset.Keys[self._index])
            self._index += 1
            return result
        else:
            raise StopIteration


class DataSetWithFramerateIterator:

    def __init__(self, dataset):
        self.dataset = dataset
        self._index = 0
    
    def __next__(self):
        if self._index < len(self.dataset):
            result = []
            for data in self.dataset.Data:
                result.append(data[self._index])
            result.append(self.dataset.get_framerate(self._index))
            self._index += 1
            return result
        else:
            raise StopIteration


class DataSetWithFramerateWithKeysIterator:

    def __init__(self, dataset):
        self.dataset = dataset
        self._index = 0
    
    def __next__(self):
        if self._index < len(self.dataset):
            result = []
            for data in self.dataset.Data:
                result.append(data[self._index])
            result.append(self.dataset.get_framerate(self._index))
            result.append(self.dataset.Keys[self._index])
            self._index += 1
            return result
        else:
            raise StopIteration


