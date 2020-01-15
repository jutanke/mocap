# mocap

Helper library to handle mocap data. At the moment, the CMU Mocap dataset as well as the Mocap data from the Human3.6M dataset are used.

## Install
This library requires some external tools, such as:

* __matplotlib__: for visualization. ```conda install matplotlib```
* __numba__: to speed-up performance. ```conda install numba```
* __transforms3d__: For handling translations between rotational data. ```pip install transforms3d```
* __tqdm__: for visualization. ```pip install tqdm```
* __spacepy__: Some datasets require to read the CDF file format from NASA. Install as follows (taken from [stackoverflow](https://stackoverflow.com/questions/37232008/how-read-common-data-format-cdf-in-python)).
```
wget -r -l1 -np -nd -nc http://cdaweb.gsfc.nasa.gov/pub/software/cdf/dist/latest-release/linux/ -A cdf*-dist-all.tar.gz
tar xf cdf*-dist-all.tar.gz -C ./
cd cdf*dist
apt install build-essential gfortran libncurses5-dev
make OS=linux ENV=gnu CURSES=yes FORTRAN=no UCOPTIONS=-O2 SHARED=yes -j4 all
make install #no sudo
```

Add to _.bashrc_:
```
export CDF_BASE=$HOME/Libraries/cdf/cdf36_3-dist
export CDF_INC=$CDF_BASE/include
export CDF_LIB=$CDF_BASE/lib
export CDF_BIN=$CDF_BASE/bin
export LD_LIBRARY_PATH=$CDF_BASE/lib:$LD_LIBRARY_PATH
```

Then install spacepy:
```
pip install git+https://github.com/spacepy/spacepy.git
```

## Usage

In case of Human3.6M, follow the steps [below](https://github.com/jutanke/mocap#human36m) first and make sure that you have downloaded the dataset from the official website. In case of CMU, the data will be automatically downloaded.

__basic usage__
```python
# ~~~~~~~~~~~~~~~~~~~~~~~
# using Human3.6M
# ~~~~~~~~~~~~~~~~~~~~~~~
import mocap.datasets.h36m as H36M

all_actors = H36M.ACTORS  # ['S1', 'S5', ..., 'S11']  total number: 7
all_actions = H36M.ACTIONS  # ['walking', ..., 'sittingdown']  total number: 15

ds = H36M.H36M(actors=all_actors)  # 32-joint 3D joint positions, in [m]
seq = ds[0]  # get the first sequence, {n_frames x 96}
print('number of sequences:', len(ds))

for seq in ds:  # loop over entire dataset
    print(seq.shape)  # {n_frames x 96}

# -- with activities --
# For our research we hand-labeled 11 activities
ds = H36M.H36M_withActivities(actors=['S1'])  # We provide 11 framewise activity labels
seq, labels = ds[0]  # get the first sequence, {n_frames x 96}, {n_frames x 11}

for seq, labels in ds:  # loop over entire dataset
    print(seq.shape)  # {n_frames x 96}
    print(labels.shape)  # {n_frames x 11}

# -- fixed skeleton --
# Initially, each skeleton has different dimensions due to the actors being of different
# height and size. However, we also provide processed data where the skeletons of all 
# actors are processed such that they only utilize the skeleton of actor "S1".
ds = H36M.H36M_FixedSkeleton(actor=all_actors)
ds = H36M.H36M_FixedSkeleton_withActivities(actors=all_actors)

# Simplify skeleton:
ds = H36M.H36M_Simplified(ds)
# ds can be used like any other dataset above, it just simplifies the skeleton to 17 joints

# ~~~~~~~~~~~~~~~~~~~~~~~
# using CMU mocap data
# ~~~~~~~~~~~~~~~~~~~~~~~
import mocap.datasets.cmu as CMU

all_subjects = CMU.ALL_SUBJECTS
# different subjects have different actions:
action_for_subject_01 = CMU.GET_ACTIONS('01')

ds = CMU.CMU(['01'])
```

__Advanced iterations__:
```python
import mocap.datasets.h36m as H36M

# include the framerate for each sequence
ds = H36M.H36M(actors=['S1'], iterate_with_framerate=True)
for seq, framerate in ds:
    print(seq.shape)  # {n_frames x 96}
    print('framerate in Hz:', framerate)

# include the unique sequence key for each sequence
ds = H36M.H36M(actors=['S1'], iterate_with_keys=True)
for seq, key in ds:
    print(seq.shape)  # {n_frames x 96}
    print('key:', key)  # h36m: (actor, action, sid) || cmu: (subject, action)

# include both key and framerate per sequence:
ds = H36M.H36M(actors=['S1'],
               iterate_with_keys=True,
               iterate_with_framerate=True)
for seq, framerate, key in ds:
    print(seq.shape)  # {n_frames x 96}
    print('framerate in Hz:', framerate)
    print('key:', key)  # h36m: (actor, action, sid) || cmu: (subject, action)

# this also works with activity labels!
```


## Data
### Human 3.6M

Default skeleton with _32_ joints:
<img width="198" alt="Screenshot 2019-12-27 at 19 58 32" src="https://user-images.githubusercontent.com/831215/71535250-565f6100-28e3-11ea-8a0b-232e9dce3fa4.png">

#### Simplified

Simplified skeleton with _17_ joints:
<img width="188" alt="Screenshot 2019-12-28 at 11 13 00" src="https://user-images.githubusercontent.com/831215/71544883-19d34a00-2963-11ea-8c21-03ea411ac17c.png">

#### Acitivity labels

We provide framewise activity labels for the entire Human3.6M dataset.
The following _11_ human-labeled acitivites are used:
![labels](https://user-images.githubusercontent.com/831215/72436240-44653580-37a0-11ea-85ee-def425e75f3c.png)


### CMU Mocap

Default skeleton with _31_ joints:
<img width="164" alt="Screenshot 2020-01-10 at 15 23 35" src="https://user-images.githubusercontent.com/831215/72164251-40f73600-33bd-11ea-9653-b5a12adf5720.png">

## Download data

### Human3.6M

For Human3.6M, we cannot directly provide a download link due to their distribution policy.
Instead, you first have to download their dataset from the official [website](http://vision.imar.ro/human3.6m/description.php).
Then, call the following script as follows to extract the data:
```bash
$ cd mocap/dataaquisition/scripts
$ python get_h36m_skeleton.py /path/to/h36m/folder/human3.6m
```

### CMU

The CMU mocap dataset is automatically downloaded once make use of it. This may take some time!
