# mocap

Helper library to handle mocap data. At the moment, the CMU Mocap dataset as well as the Mocap data from the Human3.6M dataset are used.

## Data
### Human 3.6M

Default skeleton with _32_ joints:
<img width="198" alt="Screenshot 2019-12-27 at 19 58 32" src="https://user-images.githubusercontent.com/831215/71535250-565f6100-28e3-11ea-8a0b-232e9dce3fa4.png">

#### Simplified

Simplified skeleton with _17_ joints:
<img width="188" alt="Screenshot 2019-12-28 at 11 13 00" src="https://user-images.githubusercontent.com/831215/71544883-19d34a00-2963-11ea-8c21-03ea411ac17c.png">

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
