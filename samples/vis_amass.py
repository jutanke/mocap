import sys

sys.path.insert(0, "../")
from mocap.datasets.amass import AMASS

path = "/Users/work/Dev/amass/data/all_data"
ds = AMASS(path)
