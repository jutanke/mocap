import sys
sys.path.insert(0, './..')
from os.path import isdir
from os import makedirs
from mocap.datasets.cmu_eval import CMUEval, ACTIVITES, DataType
from mocap.visualization.sequence import SequenceVisualizer
import mocap.processing.normalize as norm
import mocap.math.fk_cmueval as FK


vis_dir = '../output/'
if not isdir(vis_dir):
    makedirs(vis_dir)

ds = CMUEval(ACTIVITES, DataType.TEST)

seq = ds[0][:50]
print('seq', seq.shape)

seq3d = FK.angular2euclidean(seq)
seq3d = norm.remove_rotation_and_translation(
    seq3d, j_root=-1, j_left=8, j_right=2)

print('seq', seq3d.shape)
exit(1)


vis = SequenceVisualizer(vis_dir, 'vis_cmueval', 
                         to_file=True,
                         mark_origin=False)

vis.plot(seq3d, create_video=True,
         noaxis=True,
         plot_jid=True,
)