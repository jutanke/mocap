import sys
sys.path.insert(0, './../../../')
from os.path import isfile, isdir, join
from os import listdir
from spacepy import pycdf
import numpy as np


assert len(sys.argv) == 2, str(len(sys.argv))

h36m_path = sys.argv[1]
assert isdir(h36m_path), h36m_path

print()
print('Human3.6M path:', h36m_path)
print()

ACTORS = ["S1", 'S5', 'S6', 'S7', 'S8', 'S9', 'S11']
ACTIONS = [
    'Directions',
    'Discussion',
    'Eating',
    'Greeting',
    'Phoning',
    'Posing',
    'Purchases',
    'Sitting',
    'SittingDown',
    'Smoking',
    'Photo',
    'Waiting',
    'Walking',
    'WalkingDog',
    'WalkTogether'
]

for actor in ACTORS:
    for action in ACTIONS:
        for sid in [0, 1]:
            # fix labeling... Human3.6M labeling is very messy and we need to fix it...
            if actor == 'S1' and action == 'Photo':
                action = 'TakingPhoto'
            if actor != 'S1' and action == 'WalkingDog':
                action = 'WalkDog'

            cdf_dir = join(join(h36m_path, actor), 'MyPoseFeatures')
            cdf_dir = join(cdf_dir, 'D3_Positions')

            videos = sorted(
                [f for f in listdir(cdf_dir) if f.startswith(action)])

            if (actor == 'S1' and action == 'Walking') or \
                    action == 'Sitting':
                # separate Walking from WalkingDog OR
                # separate Sitting from SittingDown
                assert len(videos) == 4
                videos = videos[0:2]

            assert len(videos) == 2, '# of videos:' + str(len(videos))
            a, b = videos
            if len(a) > len(b):  # ['xxx 9.cdf', 'xxx.cdf']
                videos = [b, a]
            else:
                assert len(a) == len(b)

            cdf_file = join(cdf_dir, videos[sid])
            assert isfile(cdf_file)

            cdf = pycdf.CDF(cdf_file)
            joints3d = np.squeeze(cdf['Pose']).reshape((-1, 32, 3))

            print("joints3d", joints3d.shape)
            exit(1)
