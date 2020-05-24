from os.path import isdir, isfile, join, abspath, dirname
import os

path = '../mocap/data/cmu_eval'

assert isdir(path)

for dirpath, dirnames, filenames in os.walk(path):
    allfiles = [f for f in filenames if f.endswith(".txt")]
    if len(allfiles) > 0:
        TXT = "(join(site_package, '" + dirpath[9:] + "'), ["
        is_first = True
        for filename in allfiles:
            if not is_first:
                TXT += ', '
            fkey = join(dirpath, filename)[3:]
            TXT += "'" + fkey + "'"
            is_first = False
        TXT += ']),'
        
        print(TXT)