import sys
sys.path.insert(0, './..')
import mocap.datasets.h36m as H36M
from os.path import isdir
from os import makedirs
from mocap.visualization.sequence import SequenceVisualizer
import mocap.processing.normalize as norm
import mocap.math.fk as FK
import numpy as np

vis_dir = '../output/'
if not isdir(vis_dir):
    makedirs(vis_dir)

vis = SequenceVisualizer(vis_dir, 'vis_h36m_quaternion', to_file=True, mark_origin=True)

ds = H36M.H36M_Quaternions_withSimplifiedActivities(actors=['S1'])

seq, labels = ds[0]

Seq = H36M.get_quaternion('S1', 'walking', 1)[20:400:2]
Seq_mirror = H36M.mirror_quaternion(Seq)
Seq = np.expand_dims(Seq, axis=0)
Seq_mirror = np.expand_dims(Seq_mirror, axis=0)


print('seq', Seq.shape)

Seq_xyz = np.squeeze(FK.quaternion_fk(Seq))
Seq_mirror_xyz = np.squeeze(FK.quaternion_fk(Seq_mirror))

print('xyz', Seq_xyz.shape)


vis.plot(seq1=Seq_xyz, seq2=Seq_mirror_xyz, parallel=True, create_video=True)

