import requests
import zipfile
from os.path import join, dirname, isdir, isfile
from os import makedirs
import mocap.datasets.h36m as H36M
import mocap.processing.conversion as conv
import numpy as np

location_file = join(dirname(__file__), '../data/cmu_location.txt')
if isfile(location_file):
    DATA_DIR = open(location_file, 'r').read()
else:
    DATA_DIR = join(dirname(__file__), '../data/cmu')

def aquire_cmumocap():
    global DATA_DIR
    if not isdir(DATA_DIR):
        makedirs(DATA_DIR)
    
    zip_files = [
        'allasfamc.zip'
    ]

    for zip_name in zip_files:
        url = 'http://mocap.cs.cmu.edu/' + zip_name
        zip_file = join(DATA_DIR, zip_name)
        if not isfile(zip_file):
            print('[data aquisition] - cmu - download', url)
            r = requests.get(url)
            open(zip_file, 'wb').write(r.content)
            print('[data aquisition] - cmu - unzip', url)

            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall(DATA_DIR)

CMU_DIR = join(DATA_DIR, 'all_asfamc/subjects')
