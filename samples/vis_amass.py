import sys

sys.path.insert(0, "../")
from mocap.datasets.amass import AMASS
from mocap.visualization.sequence import SequenceVisualizer
from os.path import isdir

datasets = ["BMLhandball"]
# datasets = ["ACCAD"]

path = "/mnt/Data/Dev/amass2skel/output/amass2skel"
ds = AMASS(path, datasets)

print("len", len(ds))

seq = ds[10][::5]


vis_dir = "../output/"
if not isdir(vis_dir):
    makedirs(vis_dir)
vis = SequenceVisualizer(vis_dir, "vis_amass_new", to_file=True)

vis.plot(seq, create_video=True, plot_jid=False)
