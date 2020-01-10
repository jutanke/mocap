import sys
sys.path.insert(0, './..')
from os.path import isdir
from os import makedirs
import mocap.datasets.cmu as CMU
from mocap.visualization.sequence import SequenceVisualizer
import mocap.processing.normalize as norm

vis_dir = '../output/'
if not isdir(vis_dir):
    makedirs(vis_dir)

print('go')

seq = CMU.get('01', '01')
seq_norm = norm.normalize_sequence_at_frame(seq, 15, 
                                            j_root=-1,
                                            j_left=1,
                                            j_right=6)
seq_norm = norm.remove_rotation_and_translation(
    seq_norm, j_root=-1, j_left=1, j_right=6
)

print('seq', seq.shape)

vis = SequenceVisualizer(vis_dir, 'vis_cmu', 
                         to_file=True,
                         mark_origin=False)

views = [(0, 90)]
vis.plot(seq_norm[0:400:10], plot_jid=True, 
         noaxis=True, views=views)
