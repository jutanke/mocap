import json
import sys
from os.path import join, isdir
sys.path.insert(0, '../')
from mocap.visualization.sequence import SequenceVisualizer
from mocap.data.cmu import CMUHandler

Settings = json.load(open('../settings.txt'))
root = join(Settings['data_root'], 'pak')
assert isdir(root)

subsampling = 10
subjects = ['94']
cmu = CMUHandler(root, subjects)

viz = SequenceVisualizer(data_root=Settings['video_export'],
                         name='gt_cmu', vmax=2, vmin=-2,
                         subsampling=10,
                         with_pauses=True,
                         mark_origin=False,
                         to_file=False)

print('#videos', len(cmu))


seq = cmu[0]
seq = cmu.flip_lr(seq)

viz.plot(seq, noaxis=True)