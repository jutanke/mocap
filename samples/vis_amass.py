import sys

sys.path.insert(0, "../")
from mocap.datasets.amass import AMASS
import mocap.datasets.amass_constants.train as AMASS_TRAIN
import mocap.datasets.amass_constants.test as AMASS_TEST
from mocap.visualization.sequence import SequenceVisualizer
from os.path import isdir

datasets = ["BMLhandball"]
# datasets = ["ACCAD"]

path = "/mnt/Data/Dev/amass2skel/output/amass2skel"

train = AMASS_TRAIN.FILES
test = AMASS_TEST.FILES

train_leftover_datasets = [
    "BMLmovi",
    "EKUT",
    "KIT",
    "MPI_mosh",
    "TCD_handMocap",
    "SFU",
    "TotalCapture",
    "BMLhandball",
    "DFaust_67",
]


# ds_train = AMASS(path, datasets=train_leftover_datasets, exact_files=train)

ds = AMASS(path, datasets=[], exact_files=test)
seq = ds[10][::5]


vis_dir = "../output/"
if not isdir(vis_dir):
    makedirs(vis_dir)
vis = SequenceVisualizer(vis_dir, "vis_amass_new", to_file=True)

vis.plot(seq, create_video=True, plot_jid=False)
