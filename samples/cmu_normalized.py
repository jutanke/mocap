import json
import sys
from os.path import join, isdir
sys.path.insert(0, '../')
from mocap.visualization.sequence import SequenceVisualizer
from mocap.data.cmu import CMUHandler
from mocap.data.simplified import Simplified
from mocap.data.normalized import Normalized
import mocap.processing.normalize as norm

Settings = json.load(open('../settings.txt'))
root = join(Settings['data_root'], 'pak')
assert isdir(root), root

subsampling = 10
subjects = ['94']
cmu = CMUHandler(root, subjects)
handler = Normalized(Simplified(cmu))


viz = SequenceVisualizer(data_root=Settings['video_export'],
                         name='gt_cmu_norm',
                         vmax=1, vmin=-1,
                         subsampling=1,
                         with_pauses=False,
                         mark_origin=True,
                         to_file=False)

print('#videos', len(handler))

orig_seq = handler[0][0:1001]
orig_seq = norm.normalize_sequence_at_frame(orig_seq,
                                            frame=1000,
                                            j_root=handler.j_root,
                                            j_right=handler.j_right,
                                            j_left=handler.j_left)

viz.plot(orig_seq, noaxis=False, plot_jid=False)
