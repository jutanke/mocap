import sys
sys.path.insert(0, '../')
from mocap.settings import get_amass_validation_files, get_amass_test_files
from mocap.math.amass_fk import rotmat2euclidean, exp2euclidean
from mocap.visualization.sequence import SequenceVisualizer
from mocap.math.mirror_smpl import mirror_p3d


from mocap.datasets.amass import AMASS_SMPL3d, AMASS_QUAT, AMASS_EXP

data_loc = '/mnt/Data/datasets/amass'

val = get_amass_validation_files()
test = get_amass_test_files()

ds = AMASS_EXP(val, data_loc=data_loc)


seq = ds[0]

seq3d = exp2euclidean(seq)

print('seq3d', seq3d.shape)


vis_dir = '../output/'
vis = SequenceVisualizer(vis_dir, 'vis_amass', 
                         to_file=True,
                         mark_origin=False)

vis.plot(seq3d, create_video=True,
         noaxis=True,
         plot_jid=True,
)