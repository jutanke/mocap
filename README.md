# mocap
Helper library to handle mocap data. At the moment, the CMU Mocap dataset as well as the Mocap data from the Human3.6M dataset are used.

## Install
First you will need to install the general-purpose data library [pak](https://github.com/jutanke/pak):
```
pip install git+https://github.com/jutanke/pak.git
```
and the 3D geometry library [Transforms3D](https://matthew-brett.github.io/transforms3d/):
```
pip install transforms3d
```


A dependency for the Human3.6M dataset is spacepy which can be installed as [follows](https://stackoverflow.com/questions/37232008/how-read-common-data-formatcdf-in-python):
```
wget -r -l1 -np -nd -nc http://cdaweb.gsfc.nasa.gov/pub/software/cdf/dist/latest-release/linux/ -A cdf*-dist-all.tar.gz
tar xf cdf*-dist-all.tar.gz -C ./
cd cdf*dist
apt install build-essential gfortran libncurses5-dev
make OS=linux ENV=gnu CURSES=yes FORTRAN=no UCOPTIONS=-O2 SHARED=yes -j4 all
make install #no sudo
```
Add do .bashrc:
```
export CDF_BASE=$HOME/Libraries/cdf/cdf36_3-dist
export CDF_INC=$CDF_BASE/include
export CDF_LIB=$CDF_BASE/lib
export CDF_BIN=$CDF_BASE/bin
export LD_LIBRARY_PATH=$CDF_BASE/lib:$LD_LIBRARY_PATH
```
And then simply install:
```
pip install git+https://github.com/spacepy/spacepy.git
```

Then the library can be installed simply by:
```
pip install git+https://github.com/jutanke/mocap.git
```
or locally by
```
python setup.py install
```


## Data

### CMU MoCap 

![human_motion_forecasting_cmu](https://user-images.githubusercontent.com/831215/53401966-1f61ab80-39b1-11e9-927e-f4c8de046e50.png)

### Human 3.6m

![human36m_definition](https://user-images.githubusercontent.com/831215/53430714-5bb3fc80-39ef-11e9-83c3-6735db878411.png)

### Simplified

![mocap_simplified](https://user-images.githubusercontent.com/831215/56189771-109a8b00-6029-11e9-9aed-d5b7278ed644.png)
