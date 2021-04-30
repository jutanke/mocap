from mpl_toolkits.mplot3d import Axes3D


def plot(
    ax,
    human,
    plot_jid=False,
    do_scatter=True,
    linewidth=2,
    lcolor="#3498db",
    rcolor="#e74c3c",
    alpha=0.85,
):
    """
    :param ax:
    :param human:
    :param plot_jid:
    :param do_scatter:
    :param linewidth:
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

        # connect = [
        #     (0, 1), (1, 2), (2, 3),
        #     (0, 4), (4, 5), (5, 6),
        #     (1, 11), (15, 4), (15, 16), (16, 17),
        #     (11, 12), (12, 13), (11, 9), (9, 15),
        #     (9, 18)
        # ]

        connect = [
            (1, 2),
            (1, 3),
            (3, 4),
            (4, 5),
            (5, 6),
            (1, 7),
            (7, 8),
            (8, 9),
            (9, 10),
            (7, 15),
            (15, 16),
            (16, 17),
            (17, 18),
            (15, 11),
            (3, 11),
            (11, 12),
            (12, 13),
            (13, 14),
        ]

        LR = [
            False,
            False,
            False,
            True,
            True,
            True,
            True,
            False,
            False,
            False,
            False,
            True,
            True,
            True,
            True,
            False,
            False,
            False,
            False,
        ]
    # elif n_joints == 16:  # H36M reduced ExpMaps

    #     connect = [
    #         (0, 1), (0, 4),
    #         (1, 2), (2, 3), (4, 5), (5, 6),
    #         (1, 9), (4, 12),
    #         (9, 10), (10, 11),
    #         (12, 13), (13, 14),
    #         (9, 8), (8, 12), (8, 15)
    #     ]
    #     LR = [
    #         False, False, False, False, True,
    #         True, True, False, False, False,
    #         False, False, True, True, True,
    #         False
    #     ]

    elif n_joints == 38:  # CMU Eval
        connect = [
            (2, 8),
            (2, 3),
            (3, 4),
            (4, 5),
            (5, 6),
            (8, 9),
            (9, 10),
            (10, 11),
            (11, 12),
            (2, 21),
            (8, 30),
            (30, 31),
            (31, 32),
            (21, 22),
            (22, 23),
            (21, 17),
            (17, 30),
            (17, 18),
            (18, 19),
        ]
        LR = [
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            True,
            True,
            True,
            True,
            True,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
        ]
    elif n_joints == 28:  # cmu eval REDUCED
        connect = [
            (1, 6),
            (1, 2),
            (2, 3),
            (3, 4),
            (4, 5),
            (6, 7),
            (7, 8),
            (8, 9),
            (9, 10),
            (1, 16),
            (6, 22),
            (16, 13),
            (13, 22),
            (16, 17),
            (17, 18),
            (18, 19),
            (22, 23),
            (23, 24),
            (24, 25),
            (13, 14),
            (14, 15),
        ]
        LR = [
            False,
            False,
            False,
            False,
            False,
            False,
            True,
            True,
            True,
            True,
            True,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            True,
            True,
            True,
            True,
            True,
            True,
        ]
    elif n_joints == 25:  # h36m REMOVED DUPLICATES
        connect = [
            (0, 1),
            (0, 6),
            (1, 2),
            (2, 3),
            (3, 4),
            (4, 5),
            (6, 7),
            (7, 8),
            (8, 9),
            (9, 10),
            (1, 20),
            (6, 15),
            (15, 16),
            (16, 17),
            (17, 18),
            (18, 19),
            (20, 21),
            (21, 22),
            (22, 23),
            (23, 24),
            (20, 12),
            (12, 15),
            (12, 13),
            (13, 14),
        ]
        LR = [
            False,
            False,
            False,
            False,
            False,
            False,
            True,
            True,
            True,
            True,
            True,
            False,
            False,
            False,
            False,
            True,
            True,
            True,
            True,
            True,
            False,
            False,
            False,
            False,
            False,
        ]
    elif n_joints == 24:  # SMPL
        connect = [
            (0, 1),
            (0, 2),  # (0, 3),
            (1, 4),
            (5, 2),  # (3, 6),
            (7, 4),
            (8, 5),  # (6, 9),
            (7, 10),
            (8, 11),  # (9, 12),
            # (12, 13), (12, 14),
            (12, 15),
            # (13, 16), (12, 16), (14, 17), (12, 17),
            (12, 16),
            (12, 17),
            (16, 18),
            (19, 17),
            (20, 18),
            (21, 19),
            (22, 20),
            (23, 21),
            (1, 16),
            (2, 17),
        ]
        LR = [
            False,
            True,
            False,
            False,
            True,
            False,
            False,
            True,
            False,
            False,
            True,
            False,
            False,
            True,
            False,
            True,
            True,
            False,
            True,
            False,
            True,
            False,
            True,
            False,
            True,
            False,
        ]
    elif n_joints == 31:  # CMUmocap

        connect = [
            (11, 12),
            (12, 13),
            (13, 14),
            (14, 15),
            (15, 16),
            (1, 2),
            (2, 3),
            (3, 4),
            (4, 5),
            (6, 7),
            (7, 8),
            (8, 9),
            (9, 10),
            (1, 6),
            (11, 0),
            (0, 1),
            (0, 6),
            (17, 18),
            (18, 19),
            (19, 20),
            (20, 21),
            (21, 22),
            (22, 23),
            (17, 14),
            (14, 24),
            (24, 25),
            (25, 26),
            (26, 27),
            (27, 28),
            (28, 29),
            (29, 30),
        ]
        LR = [
            True,
            False,
            False,
            False,
            False,
            False,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
        ]

    elif n_joints == 32:  # h36m
        connect = [
            (1, 2),
            (2, 3),
            (3, 4),
            (4, 5),
            (6, 7),
            (7, 8),
            (8, 9),
            (9, 10),
            (0, 1),
            (0, 6),
            (6, 17),
            (17, 18),
            (18, 19),
            (19, 20),
            (20, 21),
            (21, 22),
            (1, 25),
            (25, 26),
            (26, 27),
            (27, 28),
            (28, 29),
            (29, 30),
            (24, 25),
            (24, 17),
            (24, 14),
            (14, 15),
        ]
        LR = [
            False,
            True,
            True,
            True,
            True,
            True,
            False,
            False,
            False,
            False,
            False,
            True,
            True,
            True,
            True,
            True,
            True,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            True,
            False,
            True,
            True,
            True,
            True,
            True,
            True,
        ]
    elif n_joints == 52 or n_joints == 22:  # AMASS
        connect = [
            (0, 1),
            (0, 2),
            (1, 4),
            (4, 7),
            (10, 7),
            (2, 5),
            (5, 8),
            (11, 8),
            (0, 3),
            (3, 6),
            (6, 9),
            (9, 14),
            (9, 13),
            (13, 14),
            (13, 16),
            (14, 17),
            (16, 18),
            (17, 19),
            (18, 20),
            (19, 21),
            (14, 12),
            (13, 12),
            (12, 15),
        ]
        LR = [True] * 52
        LR[0] = False
        LR[1] = False
        LR[4] = False
        LR[10] = False
        LR[7] = False
        LR[13] = False
        LR[16] = False
        LR[18] = False
        LR[20] = False
        LR[9] = False
        LR[12] = False

    elif n_joints == 18:
        connect = [
            (0, 1),
            (1, 2),
            (2, 3),
            (0, 4),
            (4, 5),
            (5, 6),
            (6, 7),
            (8, 9),
            (9, 10),
            (10, 11),
            (12, 13),
            (13, 14),
            (14, 15),
            (8, 0),
            (12, 4),
            (8, 16),
            (16, 12),
            (16, 17),
        ]
        LR = [
            True,
            True,
            True,
            True,
            False,
            False,
            False,
            False,
            True,
            True,
            True,
            True,
            False,
            False,
            False,
            False,
            False,
            False,
        ]
    elif n_joints == 17:  # human 3.6m
        connect = [
            (0, 1),
            (1, 4),
            (1, 2),
            (2, 3),
            (4, 5),
            (5, 6),
            (1, 14),
            (4, 11),
            (11, 12),
            (12, 13),
            (14, 15),
            (15, 16),
            (8, 9),
            (9, 10),
            (14, 7),
            (7, 11),
            (14, 8),
            (8, 11),
        ]
        LR = [
            False,
            False,
            False,
            False,
            True,
            True,
            True,
            False,
            False,
            False,
            False,
            True,
            True,
            True,
            False,
            False,
            False,
        ]
    elif n_joints == 15:
        connect = [
            (0, 1),
            (1, 2),
            (2, 3),
            (4, 5),
            (5, 6),
            (6, 7),
            (8, 11),
            (0, 4),
            (8, 0),
            (11, 4),
            (11, 12),
            (12, 13),
            (8, 9),
            (9, 10),
            ((8, 11), 14),
        ]

        LR = [
            False,
            False,
            False,
            False,
            True,
            True,
            True,
            True,
            False,
            False,
            False,
            True,
            True,
            True,
            False,
        ]
    elif n_joints == 14:  # simplified
        connect = [
            (0, 1),
            (1, 2),
            (0, 3),
            (3, 4),
            (4, 5),
            (0, 6),
            (6, 7),
            (7, 8),
            (3, 9),
            (9, 10),
            (10, 11),
            (9, 12),
            (12, 6),
            (12, 13),
        ]
        LR = [
            True,
            True,
            True,
            False,
            False,
            False,
            True,
            True,
            True,
            False,
            False,
            False,
            False,
            False,
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

        ax.plot(
            [A[0], B[0]],
            [A[1], B[1]],
            [A[2], B[2]],
            color=color,
            alpha=alpha,
            linewidth=linewidth,
        )

    if do_scatter:
        ax.scatter(human[:, 0], human[:, 1], human[:, 2], color="gray", alpha=0.4)

    if plot_jid:
        for i, (x, y, z) in enumerate(human):
            ax.text(x, y, z, str(i))
