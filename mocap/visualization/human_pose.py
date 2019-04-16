from mpl_toolkits.mplot3d import Axes3D
from pak.datasets.CMU_MoCap import plot as cmu_plot


def plot(ax, human, plot_jid=False, do_scatter=True, linewidth=2,
         lcolor="#3498db", rcolor="#e74c3c", alpha=0.85):
    """
    :param ax:
    :param human:
    :param plot_only_limbs:
    :param plot_jid:
    :param do_scatter:
    :param lcolor:
    :param rcolor:
    :param alpha:
    :return:
    """
    if len(human.shape) == 1:
        human = human.reshape((-1, 3))
    n_joints, n_channels = human.shape
    assert n_channels == 3
    if n_joints == 19:
        assert n_joints == 19

        connect = [
            (1, 2),
            (1, 3), (3, 4), (4, 5), (5, 6),
            (1, 7), (7, 8), (8, 9), (9, 10),
            (7, 15), (15, 16), (16, 17), (17, 18),
            (15, 11),
            (3, 11), (11, 12), (12, 13), (13, 14)
        ]

        LR = [
            False, False, False,
            True, True, True, True,
            False, False, False, False,
            True, True, True, True,
            False, False, False, False
        ]
    elif n_joints == 31:  # CMU
        cmu_plot(ax, human,
             lcolor=lcolor,
             rcolor=rcolor,
             alpha=alpha,
             plot_jid=plot_jid,
             do_scatter=do_scatter)
        return
    elif n_joints == 18:
        connect = [
            (0, 1), (1, 2), (2, 3),
            (0, 4),
            (4, 5), (5, 6), (6, 7),
            (8, 9), (9, 10), (10, 11),
            (12, 13), (13, 14), (14, 15),
            (8, 0), (12, 4),
            (8, 16), (16, 12),
            (16, 17)
        ]
        LR = [True, True, True, True,
              False, False, False, False,
              True, True, True, True,
              False, False, False, False,
              False, False]
    elif n_joints == 17:  # human 3.6m
        connect = [
            (0, 1), (1, 4), (1, 2), (2, 3),
            (4, 5), (5, 6),
            (1, 14), (4, 11), (14, 10), (10, 11),
            (11, 12), (12, 13), (14, 15), (15, 16),
            (10, 8), (8, 9),
            (14, 7), (7, 11)
        ]
        LR = [False, False, False, False, True,
              True, True, False, False, False,
              False, True, True, True, False,
              False, False]
    elif n_joints == 15:
        connect = [
            (0, 1), (1, 2), (2, 3),
            (4, 5), (5, 6), (6, 7),
            (8, 11), (0, 4), (8, 0), (11, 4),
            (11, 12), (12, 13),
            (8, 9), (9, 10), ((8, 11), 14)
        ]

        LR = [False, False, False, False,
              True, True, True, True,
              False, False, False,
              True, True, True,
              False]
    elif n_joints == 14:  # simplified
        connect = [
            (0, 1), (1, 2), (0, 3),
            (3, 4), (4, 5),
            (0, 6), (6, 7), (7, 8),
            (3, 9), (9, 10), (10, 11),
            (9, 12), (12, 6),
            (12, 13)
        ]
        LR = [
            True, True, True,
            False, False, False,
            True, True, True,
            False, False, False,
            False, False
        ]
    else:
        raise NotImplementedError()

    for a, b in connect:

        if isinstance(a, int):
            A = human[a]
        else:
            assert len(a) == 2
            left, right = a
            A = (human[left] + human[right]) / 2
            a = b  # make it integer

        B = human[b]
        is_left = LR[a] or LR[b]
        color = lcolor if is_left else rcolor

        ax.plot([A[0], B[0]], [A[1], B[1]], [A[2], B[2]],
                color=color, alpha=alpha, linewidth=linewidth)

    if do_scatter:
        ax.scatter(human[:, 0], human[:, 1], human[:, 2],
                   color='gray', alpha=0.4)

    if plot_jid:
        for i, (x, y, z) in enumerate(human):
            ax.text(x, y, z, str(i))
