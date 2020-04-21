import numpy as np
import numba as nb

"""
Various pre-processing functions for activities
"""

@nb.njit(nb.float32[:, :](
    nb.float32[:, :], nb.int64
), nogil=True)
def reshape_for_forecasting(labels, num_forecast):
    """
    This function takes a sequence of labels 
        [a, b, c, d, e, f, g, h]
    and stacks them per frame, given {num_forecast}
    (e.g. num_forecast = 3)
        [a, b, c, d, e, f, g, h]
        -->
        [abc, bce, cde, def, efg, fgh]
    the lenght of the new list is -> n_frames - num_forecast

    :param labels: [n_frames + num_forecast x label_dim]
    :param num_forecast: {int}
    returns
        [n_frames x label_dim * num_forecast]
    """
    n_frames_plus_num_forecast, label_dim = labels.shape
    n_frames = n_frames_plus_num_forecast - num_forecast
    result = np.empty((n_frames, label_dim * num_forecast), dtype=np.float32)
    for frame in range(n_frames):
        for step in range(num_forecast):
            offset_start = label_dim * step
            offset_end = label_dim * step + label_dim
            result[frame, offset_start:offset_end] = labels[frame+step]
    return result


@nb.njit(nb.float32[:, :, :](
    nb.float32[:, :, :], nb.int64
), nogil=True)
def batch_reshape_for_forecasting(Labels, num_forecast):
    """
    This function takes a sequence of labels 
        [a, b, c, d, e, f, g, h]
    and stacks them per frame, given {num_forecast}
    (e.g. num_forecast = 3)
        [a, b, c, d, e, f, g, h]
        -->
        [abc, bce, cde, def, efg, fgh]
    the lenght of the new list is -> n_frames - num_forecast

    :param labels: [n_batch, n_frames + num_forecast x label_dim]
    :param num_forecast: {int}
    returns
        [n_batch x n_frames x label_dim * num_forecast]
    """
    n_batch, n_frames_plus_num_forecast, label_dim = Labels.shape
    n_frames = n_frames_plus_num_forecast - num_forecast
    result = np.empty((n_batch, n_frames, label_dim * num_forecast), dtype=np.float32)
    for batch in range(n_batch):
        labels = Labels[batch]
        for frame in range(n_frames):
            for step in range(num_forecast):
                offset_start = label_dim * step
                offset_end = label_dim * step + label_dim
                result[batch, frame, offset_start:offset_end] = labels[frame+step]
    return result