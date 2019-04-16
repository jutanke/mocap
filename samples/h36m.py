import json
import sys
from os.path import isdir
sys.path.insert(0, '../')
from mocap.data.h36m import Human36mHandler
from mocap.visualization.sequence import SequenceVisualizer

Settings = json.load(open('../settings.txt'))
root = Settings['h36m_root']
assert isdir(root)

subsampling = 10
actors = ['S1']
h36m = Human36mHandler(root, actors)

viz = SequenceVisualizer(data_root=Settings['video_export'],
                         name='gt_h36m', vmax=2, vmin=-2,
                         subsampling=10,
                         to_file=False)

print('#videos', len(h36m))

seq = h36m[0]
print(seq.shape)

viz.plot(seq)
