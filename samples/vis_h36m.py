import sys
sys.path.insert(0, './..')
import mocap.datasets.h36m as H36M
from os.path import isdir
from os import makedirs
from mocap.visualization.sequence import SequenceVisualizer
import mocap.processing.normalize as norm
import random

vis_dir = '../output/'
if not isdir(vis_dir):
    makedirs(vis_dir)

vis = SequenceVisualizer(vis_dir, 'vis_h36m', to_file=True, mark_origin=False)

ds = H36M.H36M_FixedSkeletonFromRotation(actors=['S5'], actions=['walking'],
                                         iterate_with_framerate=True,
                                         iterate_with_keys=True,
                                         remove_global_Rt=True)

ds = H36M.H36M_Simplified(ds)
Seq = ds.get_sequence(0)

start = random.randint(0, len(Seq) - 251)
seq = Seq[start:start+300:5]

print('seq', seq.shape)

seq = H36M.mirror_p3d(seq)

seq_norm = seq
# seq_norm = norm.normalize_sequence_at_frame(seq, 15,
#                                             j_root=ds.j_root,
#                                             j_left=ds.j_left,
#                                             j_right=ds.j_right)


# views = [(0, 90)]
views = [(45, 45)]
vis.plot(seq_norm, name='norm', create_video=False, plot_jid=False, views=views, noaxis=False)


exit(1)
ds = H36M.H36M_withActivities(actors=['S1'], actions=['walking'],
                              iterate_with_framerate=True,
                              iterate_with_keys=True)

ds = H36M.H36M_Simplified(ds)
Seq, Lab = ds.get_sequence(0)
seq = Seq[0:250:4]

print('seq', seq.shape)

seq = H36M.mirror_p3d(seq)

seq_norm = norm.normalize_sequence_at_frame(seq, 15, 
                                            j_root=ds.j_root,
                                            j_left=ds.j_left,
                                            j_right=ds.j_right)


views = [(0, 90)]
vis.plot(seq_norm, name='norm', create_video=True, plot_jid=False, views=views, noaxis=False)



# seq_noRt = norm.remove_rotation_and_translation(seq)
# vis.plot(seq_noRt, name='remote_Rt', create_video=True)

