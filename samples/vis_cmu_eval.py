import sys
sys.path.insert(0, './..')
from os.path import isdir
from os import makedirs
from mocap.datasets.cmu_eval import CMUEval, CMUEval3D, ACTIVITES, DataType, remove_duplicate_joints, recover_duplicate_joints
from mocap.visualization.sequence import SequenceVisualizer
import mocap.processing.normalize as norm
import mocap.math.mirror_cmueval as MIR
import mocap.math.fk_cmueval as FK
from mocap.evaluation.npss import NPSS 
import numpy as np


vis_dir = '../output/'
if not isdir(vis_dir):
    makedirs(vis_dir)

ds = CMUEval3D(ACTIVITES, DataType.TEST)

seq = remove_duplicate_joints(ds[0][:50])
seq = norm.remove_rotation_and_translation(
    seq, j_root=-1, j_left=6, j_right=1)

seq_mir = MIR.mirror_p3d(seq)

vis = SequenceVisualizer(vis_dir, 'vis_cmueval', 
                         to_file=True,
                         vmin=-1, vmax=1,
                         mark_origin=False)

vis.plot(seq[0:50], create_video=True,
         noaxis=True,
         plot_jid=True)
vis.plot(seq_mir[0:50], create_video=True,
         noaxis=True,
         plot_jid=True)

# pose = seq[0].reshape(38, 3)

# print('pose', pose.shape)

# import numpy.linalg as la
# for i in range(37):
#     for j in range(i+1, 38):
#         a = pose[i]
#         b = pose[j]
#         dif = la.norm(a-b)
#         if dif < 0.0001:
#             print(str(i) + ', ' + str(j))

exit(1)

seq3d = FK.angular2euclidean(seq)
seq3d = norm.remove_rotation_and_translation(
    seq3d, j_root=-1, j_left=8, j_right=2)




exit(1)
a = np.reshape(seq3d[0:25], (1, 25, -1))
b = np.reshape(seq3d[1:26], (1, 25, -1))

print('a', a.shape)
score = NPSS(a, b)

print('score', score)

exit(1)


vis = SequenceVisualizer(vis_dir, 'vis_cmueval', 
                         to_file=True,
                         mark_origin=False)

vis.plot(seq3d, create_video=True,
         noaxis=True,
         plot_jid=True,
)