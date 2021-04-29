import sys
sys.path.insert(0, '../')
from mocap.settings import get_amass_validation_files, get_amass_test_files
from mocap.math.amass_fk import rotmat2euclidean, exp2euclidean
from mocap.visualization.sequence import SequenceVisualizer
from mocap.math.mirror_smpl import mirror_p3d
from mocap.datasets.dataset import Limb
from mocap.datasets.combined import Combined
from mocap.datasets.framerate import AdaptFramerate
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

seq3d = ds[55]



vis_dir = '../output/'
vis = SequenceVisualizer(vis_dir, 'vis_amass', 
                         to_file=True,
                         mark_origin=False)

vis.plot(seq3d, create_video=True,
         noaxis=True,
         plot_jid=True,
)