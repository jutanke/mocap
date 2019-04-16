import json
import sys
from os.path import join, isdir
sys.path.insert(0, '../')
from mocap.visualization.sequence import SequenceVisualizer
from mocap.data.cmu import CMUHandler
from mocap.data.simplified import Simplified
from mocap.processing.relative import Transformer
import mocap.processing.normalize as norm

Settings = json.load(open('../settings.txt'))
root = join(Settings['data_root'], 'pak')
assert isdir(root), root

subsampling = 10
subjects = ['94']
cmu = CMUHandler(root, subjects)
handler = Simplified(cmu)

transformer = Transformer(j_root=handler.j_root,
                          j_left=handler.j_left,
                          j_right=handler.j_right)

viz = SequenceVisualizer(data_root=Settings['video_export'],
                         name='gt_cmu', vmax=2, vmin=-2,
                         subsampling=5,
                         with_pauses=False,
                         mark_origin=True,
                         to_file=False)

print('#videos', len(handler))


seq = handler[0]
seq = transformer.global2relative(seq)
seq = transformer.relative2global(seq)

orig_seq = handler[0]
orig_seq = norm.normalize_sequence_at_frame(orig_seq,
                                            frame=0,
                                            j_root=handler.j_root,
                                            j_right=handler.j_right,
                                            j_left=handler.j_left)

seq = norm.normalize_sequence_at_frame(seq,
                                       frame=0,
                                       j_root=handler.j_root,
                                       j_right=handler.j_right,
                                       j_left=handler.j_left)


viz.plot(seq, seq2=orig_seq, parallel=True,
         noaxis=False, plot_jid=False)
