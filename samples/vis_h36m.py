import sys
sys.path.insert(0, './..')
import mocap.datasets.h36m as H36M
from os.path import isdir
from os import makedirs
from mocap.visualization.sequence import SequenceVisualizer

vis_dir = '../output/'
if not isdir(vis_dir):
    makedirs(vis_dir)

vis = SequenceVisualizer(vis_dir, 'vis_h36m', to_file=True, mark_origin=True)

seq = H36M.get3d('S1', 'walking', 1)

print('seq', seq.shape)

vis.plot(seq[0:250:4], 
    name='sample', 
    create_video=True, 
    plot_jid=True)
