import sys
from os.path import isdir, join, isfile, abspath, dirname
from os import listdir
from os import makedirs
from human_body_prior.body_model.body_model import BodyModel

assert len(sys.argv) == 2

amass_path = sys.argv[1]
assert isdir(amass_path), amass_path

# ============================
# step 1: unpack the folders
# ============================
folders = [f for f in listdir(amass_path) if isdir(join(amass_path, f))]
if len(folders) == 0:
    print("UNZIP")
    # unzip all files
    zipfiles = [f for f in listdir(amass_path) if f.endswith(".tar.bz2")]
    for f in tqdm(zipfiles):
        fname = join(amass_path, f)
        command = ["tar", "-xf", fname, "-C", amass_path]
        result = run(command, stdout=PIPE, stderr=PIPE, universal_newlines=True)
        print(result.returncode, result.stdout, result.stderr)
