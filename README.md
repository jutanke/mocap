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
```python
from mocap.data.cmu import CMUHandler

root = '/data/dir/for/cmu_mocap/'
subjects = ['94']
cmu = CMUHandler(root, subjects)
```
![human_motion_forecasting_cmu](https://user-images.githubusercontent.com/831215/53401966-1f61ab80-39b1-11e9-927e-f4c8de046e50.png)

### Human 3.6m
```python
from mocap.data.h36m import Human36mHandler

root = '/data/dir/for/h36m/'
actors = ['S1']
h36m = Human36mHandler(root, actors)
```
![human36m_definition](https://user-images.githubusercontent.com/831215/53430714-5bb3fc80-39ef-11e9-83c3-6735db878411.png)

### Simplified
```python
from mocap.data.cmu import CMUHandler
from mocap.data.h36m import Human36mHandler
from mocap.data.simplified import Simplified

root = '/data/dir/for/h36m/'
actors = ['S1']
h36m = Human36mHandler(root, actors)

root = '/data/dir/for/cmu_mocap/'
subjects = ['94']
cmu = CMUHandler(root, subjects)

cmu_simple = Simplified(cmu)
h36m_simple = Simplified(h36m)
```
![mocap_simplified](https://user-images.githubusercontent.com/831215/56189771-109a8b00-6029-11e9-9aed-d5b7278ed644.png)

### Relative representation
The default data representation is global joint coordinates. However, we also offer a relative representation where a person is centered at the origin (based on a root joint) and faces forward (based on the hip joints). The displacement (x, y, z) is being recorded between the root joints of temporally adjacent joints and the rotation around the z-axis is also remembered.

```python
from mocap.data.cmu import CMUHandler
from mocap.data.simplified import Simplified
from mocap.processing.relative import Transformer

root = '/data/dir/for/cmu_mocap/'
subjects = ['94']
cmu = CMUHandler(root, subjects)
handler = Simplified(cmu)

transformer = Transformer(j_root=handler.j_root,
                          j_left=handler.j_left,
                          j_right=handler.j_right)

seq = handler[0]
seq = transformer.global2relative(seq)  # transform to relative repr.
seq = transformer.relative2global(seq)  # transform back to global rep.
```

### Visualization
A simple visualization pipeline is provided. It allows to either output to the screen or to file, enable/disable certain markers in the scene etc.

```python
from mocap.visualization.sequence import SequenceVisualizer

viz = SequenceVisualizer(data_root=Settings['video_export'],
                         name='gt_cmu', vmax=2, vmin=-2,
                         subsampling=5,
                         with_pauses=False,
                         mark_origin=True,
                         to_file=False)

viz.plot(seq1)  # show a sequence

viz.plot(seq1, plot_jid=True)  # show sequence with joint ids plotted

viz.plot(seq1, noaxis=True)  # draw on a white canvas instead of a grid raster

viz.plot(seq1, seq2)  # draw two consecutive sequences

viz.plot(seq1, seq2=seq2, parallel=True)  # show two sequences in parallel
```
