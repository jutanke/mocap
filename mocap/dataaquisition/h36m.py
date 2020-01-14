import requests
import zipfile
from os.path import join, dirname, isdir, isfile
from os import makedirs
import mocap.datasets.h36m as H36M
import mocap.processing.conversion as conv
import numpy as np

DATA_DIR = join(dirname(__file__), '../data/h36m')


def aquire_fixed_skeleton():
    aquire_euler()
    global DATA_DIR
    data_dir = join(DATA_DIR, 'fixed_skeleton')
    if not isdir(data_dir):
        makedirs(data_dir)
        for actor in H36M.ACTORS:
            for action in H36M.ACTIONS:
                for sid in [1, 2]:
                    fname = join(data_dir, actor + '_' + action + '_' + str(sid) + '.txt')


def aquire_euler():
    aquire_expmap()
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


def aquire_expmap():
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





    