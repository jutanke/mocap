import sys
sys.path.insert(0, '../')
from mocap.settings import get_amass_validation_files, get_amass_test_files
from mocap.math.amass_fk import rotmat2euclidean
from mocap.visualization.sequence import SequenceVisualizer


from mocap.datasets.amass import AMASS

data_loc = '/mnt/Data/datasets/amass'

val = get_amass_validation_files()
test = get_amass_test_files()

ds = AMASS(val, data_loc=data_loc)
print('val', len(ds))


seq = ds[0][:10:2]

seq3d = rotmat2euclidean(seq)

print('seq3d', seq3d.shape)

vis_dir = '../output/'
vis = SequenceVisualizer(vis_dir, 'vis_amass', 
                         to_file=True,
                         mark_origin=False)

vis.plot(seq3d, create_video=True)
