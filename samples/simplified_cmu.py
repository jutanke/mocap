import json
import sys
from os.path import join, isdir
sys.path.insert(0, '../')
from mocap.visualization.sequence import SequenceVisualizer
from mocap.data.cmu import CMUHandler
from mocap.data.simplified import Simplified

Settings = json.load(open('../settings.txt'))
root = join(Settings['data_root'], 'pak')
assert isdir(root), root

subsampling = 10
subjects = ['94']
cmu = CMUHandler(root, subjects)
handler = Simplified(cmu)

viz = SequenceVisualizer(data_root=Settings['video_export'],
                         name='gt_cmu', vmax=1, vmin=-1,
                         subsampling=10,
                         with_pauses=True,
                         mark_origin=False,
                         to_file=False)

print('#videos', len(handler))


seq = handler[0]

viz.plot(seq, noaxis=False, plot_jid=True)
