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

ACTIVITES = ['walking']
ds = CMUEval3D(ACTIVITES, DataType.TEST)

seq = remove_duplicate_joints(ds[0][10:160])
seq = norm.remove_rotation_and_translation(
    seq, j_root=-1, j_left=6, j_right=1)

# print('seq', seq.shape)

# a = np.reshape(seq[0:25], (1, 25, -1))
# b = np.reshape(seq[1:26], (1, 25, -1))

# print('a', a.shape)
# score = NPSS(a, b)

# print('score', score)

# exit(1)


vis = SequenceVisualizer(vis_dir, 'vis_cmueval', 
                         to_file=True,
                         mark_origin=False)

vis.plot(seq, create_video=True,
         noaxis=True,
         plot_jid=False,
)