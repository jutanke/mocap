import requests
import zipfile
from os.path import join, isdir, isfile
from os import makedirs
import mocap.settings as settings

DATA_DIR = join(settings.get_data_path(), 'cmu')


def acquire_cmumocap():
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
