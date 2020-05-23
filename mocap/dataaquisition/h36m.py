import requests
import zipfile
from os.path import join, isdir, isfile
from os import makedirs
from mocap.datasets.h36m_constants import ACTORS, ACTIONS
import mocap.processing.conversion as conv
import numpy as np
import mocap.settings as settings
import mocap.math.kabsch as KB
from tqdm import tqdm
from mocap.dataaquisition.get_h36m_skeleton import transform

DATA_DIR = join(settings.get_data_path(), 'h36m')


def acquire_h36m():
    target_dir = join(settings.get_data_path(), 'h36m/p3d')
    if not isdir(target_dir):
        transform(settings.get_h36m_path(), target_dir)

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
