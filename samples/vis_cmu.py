import sys
sys.path.insert(0, './..')
from os.path import isdir
from os import makedirs
import mocap.datasets.cmu as CMU
from mocap.visualization.sequence import SequenceVisualizer

vis_dir = '../output/'
if not isdir(vis_dir):
    makedirs(vis_dir)

print('go')

seq = CMU.get('01', '01')

print('seq', seq.shape)

vis = SequenceVisualizer(vis_dir, 'vis_cmu', to_file=True, mark_origin=False)

vis.plot(seq[0:200:5])
