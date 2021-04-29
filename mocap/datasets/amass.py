from mocap.datasets.dataset import DataSet, Limb
from os.path import isdir, join, isfile, abspath, dirname
from os import listdir
from os import makedirs
import numpy as np
from subprocess import PIPE, run
from tqdm import tqdm


class AMASS(DataSet):
    def __init__(self, amass_path: str):
        """
        """
        assert isdir(amass_path)

        # step 1: unpack the folders
        folders = [f for f in listdir(amass_path) if isdir(join(amass_path, f))]
        if len(folders) == 0:
            # unzip all files
            zipfiles = [f for f in listdir(amass_path) if f.endswith(".tar.bz2")]
            for f in tqdm(zipfiles):
                fname = join(amass_path, f)
                # print(fname)
                # command = f"tar -xf {fname}"
                command = ["tar", "-xf", fname, "-C", amass_path]
                result = run(command, stdout=PIPE, stderr=PIPE, universal_newlines=True)
                print(result.returncode, result.stdout, result.stderr)

                # subprocess.call(["tar", "-x", fname], shell=True)

        print(folders)
