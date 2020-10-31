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

## Camber

1. Cloned repository
2. Ran each line in requirements separately
```
python3 -m pip install --user numpy
python3 -m pip install --user pillow
python3 -m pip install --user h5py
python3 -m pip install --user gym
python3 -m pip install --user -e git+https://github.com/tawnkramer/gym-donkeycar.git#egg=gym_donkeycar

```
Ran into error
```
ImportError: No module named 'gym_donkeycar'
```
Two ways to fix this, from src directory, run
```
python3 -m pip install --user -e git+https://github.com/tawnkramer/gym-donkeycar.git#egg=gym_donkeycar
```
Alternatively, add ~/git/sdsandbox to $PYTHONPATH - details to be added - in .bashrc.  
We still get error on camber
```
  File "/home/enterprise.internal.city.ac.uk/aczd097/git/sdsandbox/src/src/gym-donkeycar/gym_donkeycar/envs/donkey_sim.py", line 161
    logger.warning(f'unknown message type {msg_type}')
                                                    ^
SyntaxError: invalid syntax

```
