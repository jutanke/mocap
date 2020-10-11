import sys
sys.path.insert(0, './..')

import mocap.settings as SET#
SET.set_data_path('/Users/work/Data/data')

import mocap.datasets.h36m as H36M
import mocap.datasets.h36m_extra as H36M_EXTRA
from os.path import isdir
from os import makedirs
from mocap.datasets.dataset import Dataset_NormalizedJoints, Dataset_Normalized
from mocap.visualization.sequence import SequenceVisualizer
import mocap.processing.normalize as norm
import mocap.processing.activities as act
import mocap.evaluation.h36m as Eval_h36m
import random
import numpy as np
import numpy.linalg as la




vis_dir = '../output/'
if not isdir(vis_dir):
    makedirs(vis_dir)

vis = SequenceVisualizer(vis_dir, 'vis_h36m_stitch', to_file=True, mark_origin=False)

ds = H36M.H36M_FixedSkeleton(actors=['S5'], actions=['walking'],
                                                      iterate_with_framerate=True,
                                                      iterate_with_keys=True,
                                                      remove_global_Rt=False)

seq = ds[1]


n_frames = len(seq)

seq = seq[100:500:2]
Seq = np.expand_dims(seq, axis=0)

n_frames = 25


Seq, R, t = norm.batch_normalize_sequence_at_frame(Seq, frame=n_frames-1, return_transforms=True)

seq_in = np.squeeze(Seq[:, :n_frames])

Rs = []
Ts = []
Seq_out = [Seq[:, n_frames:n_frames+n_frames]]


for t in range(n_frames, Seq.shape[1], n_frames):
    print(t)

    

# print("R", R.shape)
# print(R)

# print('Seq', Seq.shape)


seq_out = np.squeeze(np.concatenate(Seq_out))

print('seq_in', seq_in.shape)
print('Seq_out', seq_out.shape)
# exit(1)

# seq_out = np.squeeze(np.concatenate(Seq_out), axis=1)

vis.plot(
    seq1=seq_in,
    seq2=seq_out,
    name='walking', 
    parallel=False,
    create_video=True, plot_jid=False, noaxis=True)

exit(1)

seq = ds.denormalize(seq)

vis.plot(
    seq1=seq[50:200],
    name='walking', 
    parallel=False,
    create_video=True, plot_jid=False, noaxis=True)



print('labels', labels.shape)
print('seq', seq.shape)
exit(1)


Seq = Eval_h36m.get('walking', H36M.H36M_FixedSkeleton)
print('Walking:', Seq.shape)

# Seq_red = H36M.batch_remove_duplicate_joints(Seq)
# Seq_rec = H36M.batch_recover_duplicate_joints(Seq_red)

print('Seq_red', Seq_red.shape)

vis.plot(
    seq1=Seq_red[0],
    name='cmp', 
    parallel=False,
    create_video=True, plot_jid=True, noaxis=True)




# pose = Seq[0, 0].reshape((32, 3))

# SAME = []
# import numpy.linalg as la
# for j in range(32):
#     for i in range(j+1, 32):
#         dif = pose[i] - pose[j]
#         d = la.norm(pose[i] - pose[j])
#         if d < 0.01:
#             SAME.append((j, i))

# print('po', pose.shape)

# for jid in range(32):
#     d = la.norm(pose[jid] - pose[27])
#     print(str(jid) + ' --> ', d)

# print(SAME)
exit(1)


ds = H36M.H36M_Simplified(ds)
Seq, Labels = ds.get_sequence(0)

print('lab', Labels.shape)

Labels_ = act.reshape_for_forecasting(Labels, num_forecast=10)
# # print('lab2', Labels_.shape)
# exit(1)

print("SEQ", Seq.shape)

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
vis.plot(seq_norm, name='norm', create_video=False, plot_jid=False,
         views=views, noaxis=False)


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

