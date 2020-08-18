import sys
sys.path.insert(0, './..')
import mocap.datasets.h36m as H36M
from os.path import isdir
from os import makedirs
from mocap.visualization.sequence import SequenceVisualizer
import mocap.processing.normalize as norm
import mocap.math.fk as FK
import mocap.processing.conversion as conv
import numpy as np

vis_dir = '../output/'
if not isdir(vis_dir):
    makedirs(vis_dir)

vis = SequenceVisualizer(vis_dir, 'vis_h36m_expmap', to_file=True, mark_origin=True)


Seq = H36M.get_euler('S1', 'walking', 1)[0:250:5]
Seq_xyz = FK.euler_fk(Seq)

ds = H36M.H36M_ReducedExp_withSimplifiedActivities(actors=['S1'])

seq, _ = ds[0]

seq = H36M.recover_reduced_expmap(seq)[200:350:3]

seq_euler = conv.expmap2euler(seq)
seq_xyz = FK.euler_fk(seq_euler)

Seq_xyz = norm.remove_rotation_and_translation(seq_xyz)


vis.plot(Seq_xyz, create_video=True)

