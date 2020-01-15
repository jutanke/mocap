import requests
import zipfile
from os.path import join, isdir, isfile
from os import makedirs
import mocap.datasets.h36m as H36M
import mocap.processing.conversion as conv
import mocap.processing.normalize as norm
import numpy as np
import mocap.settings as settings
import mocap.math.kabsch as KB
from tqdm import tqdm

DATA_DIR = join(settings.get_data_path(), 'h36m')


def acquire_fixed_skeleton():
    acquire_euler()
    global DATA_DIR
    data_dir = join(DATA_DIR, 'fixed_skeleton')
    if not isdir(data_dir):
        print('[mocap][Human3.6M] generate fixed skeletons:', data_dir)
        makedirs(data_dir)
        for actor in H36M.ACTORS:
            print('\thandle actor ', actor)
            for action in tqdm(H36M.ACTIONS):
                for sid in [1, 2]:
                    fname = join(data_dir, actor + '_' + action + '_' + str(sid) + '.txt')
                    # seq1 = norm.remove_rotation_and_translation(
                    #     H36M.get3d_fixed_from_rotation(actor, action, sid))
                    # seq2 = norm.remove_rotation_and_translation(
                    #     H36M.get3d(actor, action, sid))
                    seq1 = H36M.get3d_fixed_from_rotation(actor, action, sid)
                    seq2 = H36M.get3d(actor, action, sid)
                    assert len(seq1) == len(seq2), actor + ' ' + action + ' -> ' + str(seq1.shape) + '|' + str(seq2.shape)

                    seq1_ = []
                    for p, q in zip(seq1, seq2):
                        p = KB.rotate_P_to_Q(p, q)
                        seq1_.append(p)
                    n_frames = len(seq1)
                    seq1 = np.reshape(seq1_, (n_frames, -1))
                    np.savetxt(fname, seq1)


def acquire_euler():
    acquire_expmap()
    global DATA_DIR
    euler_dir = join(DATA_DIR, 'euler')
    if not isdir(euler_dir):
        makedirs(euler_dir)
        for actor in H36M.ACTORS:
            for action in H36M.ACTIONS:
                for sid in [1, 2]:
                    fname = join(euler_dir, actor + '_' + action + '_' + str(sid) + '.npy')
                    if not isfile(fname):
                        print('[data aquisition] - h36m - extract euler ', (actor, action, sid))
                        exp_seq = H36M.get_expmap(actor, action, sid)
                        euler_seq = conv.expmap2euler(exp_seq).astype('float32')
                        np.save(fname, euler_seq)


def acquire_expmap():
    global DATA_DIR
    exp_dir = join(DATA_DIR, 'expmap')
    zip_fname = join(exp_dir, 'h3.6m.zip')
    if not isdir(exp_dir):
        makedirs(exp_dir)
    
    if not isfile(zip_fname):
        print('[data aquisition] - h36m - download expmap data')
        r = requests.get('http://www.cs.stanford.edu/people/ashesh/h3.6m.zip')
        open(zip_fname, 'wb').write(r.content)
    
    exp_data_dir = join(exp_dir, 'h3.6m')
    if not isdir(exp_data_dir):
        print('[data aquisition] - h36m - extract exmap data')

        with zipfile.ZipFile(zip_fname, 'r') as zip_ref:
            zip_ref.extractall(exp_dir)





    