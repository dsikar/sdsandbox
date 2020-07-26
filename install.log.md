# Install log

Installation procedure of various platforms

## Intel cloud

Note: this is **only** for training, as the game engine is not expected to run on the Intel platform.  

1. Clone repository  
2. Ran each line in requirements.txt separately e.g.
```
$ python -m pip install numpy
$ python -m pip install pillow
$ python -m pip install h5py
$ python -m pip install gym
$ python -m pip install -e git+https://github.com/tawnkramer/gym-donkeycar.git#egg=gym_donkeycar
$ python -m pip install tensorflow
```
Tensorflow install failed with **MemoryError** message. tensorflow-gpu also fails with same error.  
Also tried:
```
$ conda install -c intel tensorflow
```
which also failed. Finally tried:
```
$ qsub -I
$ pip -m install tensorflow
```
which hung at the end but appears to have installed tensorflow correctly as it prints a version with script e.g.
```
import tensorflow as tf
print(tf.__version__)
```
All being well we should be able to train models.



