from abc import abstractmethod


class MocapHandler:

    def __init__(self, sequences, J, j_root, j_left,
                 j_right, cherrypicking=None, dim=None):
        """
        :param sequences: dict: { key_i: [ n_i x J x 3] }
        :param J:
        :param j_root:
        :param j_left:
        :param j_right:
        :param cherrypicking:
        :param dim: defines the dimension of a single person (e.g. (J x 3)
        """
        self.sequences = sequences
        self.sequences_as_list = list(sequences.values())
        self.sequences_meta = list(sequences.keys())
        self.J = J
        if dim is None:
            self.dim = (J, 3)
        else:
            self.dim = dim
        self.j_root = j_root
        self.j_left = j_left
        self.j_right = j_right

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, item):
        if isinstance(item, int):
            return self.sequences_as_list[item]
        else:
            return self.sequences[item]

    @abstractmethod
    def get_unique_identifier(self):
        raise NotImplementedError

    @abstractmethod
    def flip_lr(self, seq):
        """
        :param seq:
        :return:
        """
        raise NotImplementedError

    @abstractmethod
    def get_framerate(self, item):
        """
        :param item
        :return: framerate in Hz
        """
        raise NotImplementedError
