import sys
sys.path.insert(0, './..')
import mocap.datasets.h36m as H36M
import mocap.datasets.cmu as CMU
from mocap.datasets.combined import Combined, mirror_p3d
from os.path import isdir
from os import makedirs
from mocap.visualization.sequence import SequenceVisualizer
import mocap.processing.normalize as norm
import random

vis_dir = '../output/'
if not isdir(vis_dir):
    makedirs(vis_dir)

vis = SequenceVisualizer(vis_dir, 'vis_combined',
                         to_file=True, mark_origin=False)

ds = H36M.H36M_withSimplifiedActivities(actors=['S5'], actions=['walking'],
                                        iterate_with_framerate=True,
                                        iterate_with_keys=True,
                                        remove_global_Rt=True)

ds = CMU.CMU(['01'], remove_global_Rt=True)

ds = Combined(ds)

# Seq, Labels = ds.get_sequence(0)
Seq = ds.get_sequence(0)

print("SEQ", Seq.shape)
# print("LAB", Labels.shape)

seq = Seq[200:300:2]

seq2 = mirror_p3d(seq)

views = [(45, 45)]
vis.plot(seq, seq2=seq2, parallel=True,
         create_video=True, plot_jid=False, noaxis=True,
         views=views)
