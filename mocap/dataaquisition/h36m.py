import requests
import zipfile
from os.path import join, dirname, isdir, isfile
from os import makedirs

DATA_DIR = join(dirname(__file__), '../data/h36m')


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





    