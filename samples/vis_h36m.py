import sys
sys.path.insert(0, './..')
import mocap.datasets.h36m as H36M
from os.path import isdir
from os import makedirs
from mocap.visualization.sequence import SequenceVisualizer
import mocap.processing.normalize as norm

vis_dir = '../output/'
if not isdir(vis_dir):
    makedirs(vis_dir)

vis = SequenceVisualizer(vis_dir, 'vis_h36m', to_file=True, mark_origin=True)

# Seq = H36M.get3d('S1', 'walking', 1)

ds = H36M.H36M_FixedSkeleton_withActivities(actors=['S1'], actions=['walking'], iterate_with_framerate=True)
Seq, Lab = ds.get_sequence(0)


seq = Seq[0:250:4]

for seq, lab, framerate in ds:
    print('seq', seq.shape)
    print('lab', lab.shape)
    print('framerate', framerate)
    print('')

exit(1)

vis.plot(seq, 
    name='sample', 
    create_video=True, 
    plot_jid=True)

seq_norm = norm.normalize_sequence_at_frame(seq, 15)
vis.plot(seq_norm, name='norm', create_video=True)

seq_noRt = norm.remove_rotation_and_translation(seq)
vis.plot(seq_noRt, name='remote_Rt', create_video=True)

