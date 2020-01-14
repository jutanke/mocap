import sys
sys.path.insert(0, './..')
import mocap.datasets.h36m as H36M
from os.path import isdir
from os import makedirs
from mocap.visualization.sequence import SequenceVisualizer
import mocap.processing.normalize as norm
import mocap.math.fk as FK

vis_dir = '../output/'
if not isdir(vis_dir):
    makedirs(vis_dir)

vis = SequenceVisualizer(vis_dir, 'vis_h36m_p3d_from_rot', to_file=True, mark_origin=True)


Seq1 = norm.remove_rotation_and_translation(H36M.get3d_fixed_from_rotation('S1', 'walking', 1)[0:250:5])
Seq2 = norm.remove_rotation_and_translation(H36M.get3d('S1', 'walking', 1)[0:250:5])

vis.plot(Seq1, seq2=Seq2, parallel=True)
