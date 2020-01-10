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

ds = CMU.CMU_DataSet(['01'])

print("len", len(ds))

seq = ds[0]
seq_norm = norm.normalize_sequence_at_frame(seq, 15, 
                                            j_root=ds.j_root,
                                            j_left=ds.j_left,
                                            j_right=ds.j_right)
seq_norm = norm.remove_rotation_and_translation(
    seq_norm, j_root=ds.j_root, 
              j_left=ds.j_left, 
              j_right=ds.j_right
)

seq_mirror = ds.mirror(seq_norm)

vis = SequenceVisualizer(vis_dir, 'vis_cmu', 
                         to_file=True,
                         mark_origin=False)

views = [(0, 90)]

vis.plot(seq_norm[0:400:20],
         parallel=False,
         plot_jid=True, 
         noaxis=True, views=views, 
         create_video=False)

vis.plot(seq_mirror[0:400:20],
         parallel=False,
         plot_jid=True, 
         noaxis=True, views=views, 
         create_video=False)


vis.plot(seq_norm[0:400:10], seq2=seq_mirror[0:400:10],
         parallel=True,
         plot_jid=False, 
         noaxis=True, views=views, 
         create_video=True)
