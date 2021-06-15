from os.path import join


def unescape_spaces(txt):
    return (
        txt.replace("_SB_", " ")
        .replace("_SB2_", "-")
        .replace("dynamics.pkl", "poses.npz.npy")
    )


def convert_old_convention_to_new(fname):
    """"""
    head, tail = fname.split("/")
    if head == "ACCAD":
        try:
            idx = tail.index("c3d") + 3
        except:
            idx = 4
            subject = None
            for sub in ["s001", "s007", "s008", "s009", "s011"]:
                if tail.startswith(sub):
                    subject = sub
            if subject is None:
                raise NotImplementedError(f"[ACCAD] Cannot find subject {tail}")

        subject = tail[:idx]
        end = tail[idx:]
        end = unescape_spaces(end)

    elif head == "BioMotion":
        head = "BioMotionLab_NTroje"
        subject = tail[:6]
        end = tail[6:].replace("dynamics.pkl", "poses.npz.npy")
    elif head == "CMU":
        idx = tail.index("_")
        subject = tail[:idx]
        end = tail[idx + 1 :].replace(".pkl", "_poses.npz.npy")
    elif head == "Eyes":
        head = "Eyes_Japan_Dataset"
        idx = -1
        subject = None
        subjects = [
            "aita",
            "frederic",
            "hamada",
            "ichige",
            "kaiwa",
            "kanno",
            "kawaguchi",
            "kudo",
            "shiono",
            "takiguchi",
            "yamaoka",
            "yokoyama",
        ]
        for sub in subjects:
            if tail.startswith(sub):
                subject = sub
                idx = len(subject)
                break
        if subject is None:
            raise NotImplementedError(f"[EYES] Cannot find subject! {tail}")

        end = unescape_spaces(tail[idx:])
    elif head == "HDM05":
        head = "MPI_HDM05"
        subject = tail[:2]
        if subject not in ["bk", "dg", "mm", "tr"]:
            raise NotImplementedError(f"[HDM05] no subject {subject}")
        end = unescape_spaces(tail[2:])
    elif head == "HEva":
        head = "HumanEva"
        subject = tail[:2]
        end = tail[3:].replace(".pkl", "_poses.npz.npy")
        if subject not in ["S1", "S2", "S3"]:
            raise NotImplementedError(f"[HumanEva] no subject {subject}")
    elif head == "JointLimit":
        head = "MPI_Limits"
        subject = tail[:5]
        if subject not in ["03099", "03100", "03101"]:
            raise NotImplementedError(f"[MPI_Limits] no subject {subject}")
        end = tail[6:].replace(".pkl", "_poses.npz.npy")
    elif head == "Transition":
        head = "Transition_mocap"
        subject = tail[:9]
        end = tail[9:].replace(".pkl", "_poses.npz.npy")
    elif head == "SSM":
        head = "SSM_synced"
        subject = None
        for sub in ["20160330_03333", "20160930_50032", "20161014_50033"]:
            if tail.startswith(sub):
                subject = sub
                break
        if subject is None:
            raise NotImplementedError(f"[SSM] cannot find subject: {tail}")
        end = tail[len(subject) :].replace(".pkl", "_poses.npz.npy")
    else:
        raise NotImplementedError(f"Not implemented for head '{head}'")

    new_fname = join(head, join(subject, end))

    return new_fname
