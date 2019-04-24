from mocap.data.mocap import MocapHandler
import mocap.processing.normalize as norm


class Normalized(MocapHandler):
    """

    """

    def __init__(self, proxy_mocap):
        """
        :param proxy_mocap:
        """
        self.proxy_mocap = proxy_mocap

        sequences = {}

        for key, seq in proxy_mocap.sequences.items():
            sequences[key] = norm.remove_translation(seq,
                                                     proxy_mocap.j_root,
                                                     proxy_mocap.j_left,
                                                     proxy_mocap.j_right)

        super().__init__(
            sequences=sequences,
            J=proxy_mocap.J,
            j_root=proxy_mocap.j_root,
            j_left=proxy_mocap.j_left,
            j_right=proxy_mocap.j_right
        )

    def get_unique_identifier(self):
        return "normalized_" + self.proxy_mocap.get_unique_identifier()

    def flip_lr(self, seq):
        return self.proxy_mocap.flip_lr(seq)

    def get_framerate(self, item):
        return self.proxy_mocap.get_framerate(item)

