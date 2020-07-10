import sys
sys.path.insert(0, '../')
from mocap.settings import get_amass_validation_files, get_amass_test_files
from mocap.math.amass_fk import rotmat2euclidean, exp2euclidean
from mocap.visualization.sequence import SequenceVisualizer
from mocap.math.mirror_smpl import mirror_p3d
from mocap.datasets.dataset import Limb
from mocap.datasets.combined import Combined
from mocap.datasets.framerate import AdaptFramerate
import mocap.datasets.h36m as H36M
import numpy as np
import numpy.linalg as la


from mocap.datasets.amass import AMASS_SMPL3d, AMASS_QUAT, AMASS_EXP

data_loc = '/mnt/Data/datasets/amass'

val = get_amass_validation_files()
test = get_amass_test_files()

ds = AMASS_SMPL3d(val, data_loc=data_loc)
print(ds.get_joints_for_limb(Limb.LEFT_LEG))

ds = AdaptFramerate(Combined(ds), target_framerate=50)
print(ds.get_joints_for_limb(Limb.LEFT_LEG))


ds_h36m = Combined(H36M.H36M_FixedSkeleton(actors=['S5'], actions=['walking'], remove_global_Rt=True))

seq3d = ds[0]
seq3d_h36m = ds_h36m[0]

seq3d = seq3d[0:200].reshape((200, 14, 3))
seq3d_h36m = seq3d_h36m[0:200].reshape((200, 14, 3))


a = np.array([[[0.4, 0, 0]]])
b = np.array([[[-0.4, 0, 0]]])

seq3d += a
seq3d_h36m += b



vis_dir = '../output/'
vis = SequenceVisualizer(vis_dir, 'vis_amass_vs_h36m', 
                         to_file=True,
                         mark_origin=False)

vis.plot(seq1=seq3d, seq2=seq3d_h36m, parallel=True,
         create_video=True,
         noaxis=False,
         plot_jid=False,
)