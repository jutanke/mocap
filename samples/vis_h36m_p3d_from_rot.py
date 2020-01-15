import sys
sys.path.insert(0, './..')
import mocap.datasets.h36m as H36M
from os.path import isdir
from os import makedirs
from mocap.visualization.sequence import SequenceVisualizer
import mocap.processing.normalize as norm
import mocap.math.kabsch as KB
import  numpy as np

vis_dir = '../output/'
if not isdir(vis_dir):
    makedirs(vis_dir)

vis = SequenceVisualizer(vis_dir, 'vis_h36m_p3d_from_rot', to_file=True, mark_origin=True)


Seq1 = H36M.get3d_fixed_from_rotation('S1', 'walking', 1)[0:250:5]
Seq2 = H36M.get3d('S1', 'walking', 1)[0:250:5]

Seq1_ = []

for p, q in zip(Seq1, Seq2):
    p = KB.rotate_P_to_Q(p, q)
    Seq1_.append(p)
Seq1 = np.array(Seq1_)

vis.plot(Seq1, seq2=Seq2, parallel=True)
