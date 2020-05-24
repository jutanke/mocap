import sys
sys.path.insert(0, './..')
from os.path import isdir
from os import makedirs
from mocap.datasets.cmu_eval import CMUEval, CMUEval3D, ACTIVITES, DataType
from mocap.visualization.sequence import SequenceVisualizer
import mocap.processing.normalize as norm
import mocap.math.fk_cmueval as FK
from mocap.evaluation.npss import NPSS 
import numpy as np


vis_dir = '../output/'
if not isdir(vis_dir):
    makedirs(vis_dir)

ds = CMUEval3D(ACTIVITES, DataType.TEST)

seq = ds[0][:50]
print('seq', seq.shape)


exit(1)

seq3d = FK.angular2euclidean(seq)
seq3d = norm.remove_rotation_and_translation(
    seq3d, j_root=-1, j_left=8, j_right=2)


a = np.reshape(seq3d[0:25], (1, 25, -1))
b = np.reshape(seq3d[1:26], (1, 25, -1))

print('a', a.shape)
score = NPSS(a, b)

print('score', score)

print('seq', seq3d.shape)
exit(1)


vis = SequenceVisualizer(vis_dir, 'vis_cmueval', 
                         to_file=True,
                         mark_origin=False)

vis.plot(seq3d, create_video=True,
         noaxis=True,
         plot_jid=True,
)