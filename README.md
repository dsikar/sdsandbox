# SdSandbox + Rain

This is the accompanying code for "Evaluation of Self-Driving Cars Using CNNs In The Rain". Unity game engine is used as a simulation environment, a modified SDSandbox is the Unity wrapper, a modified Automold is used to add rain to images. Data augmetation was done with a modified version of [this repo](https://github.com/naokishibuya/car-behavioral-cloning)


Some modfications were made to the original source code such that:

1. Trained model can actually self-drive - changes to network geometry and image pre-processing

2. Added rain, using the [Automold](https://github.com/UjjwalSaxena/Automold--Road-Augmentation-Library) library

3. Data Augmentation, crops were adjusted such that road geometry of interest are taken from SDSandbox Unity camera image. Original crop correspons to Udacity MOOC Unity wrapper, with a different aspect ratio.

## Running the code

Clone this repo:

```
$ git clone https://github.com/dsikar/sdsandbox
```
Download trained models:
```
$ wget https://bit.ly/39jfp8y
```
Start Unity and choose option "Generated Track"

![SDSandbox](https://user-images.githubusercontent.com/232522/135420165-a135c508-a836-450b-ac82-cd24673e3f9b.png)

Choose option "NN Control Over Network"

![SDSandboxAutoRecNN](https://user-images.githubusercontent.com/232522/135420531-9d7d5bdd-c0c8-471a-a04d-8358d65c5fa3.png)

In a terminal, navigate to the src directory. Start tcpflow to log the run:

```
$ $ sudo tcpflow -i lo -c port 9091 > /tmp/20201207192948_nvidia2_light_rain_mult_1_tcpflow.log
```





# Original SDSandbox README ()

Self Driving Car Sandbox


[![IMAGE ALT TEXT](https://img.youtube.com/vi/e0AFMilaeMI/0.jpg)](https://www.youtube.com/watch?v=e0AFMilaeMI "self driving car sim")


## Summary

Use Unity 3d game engine to simulate car physics in a 3d world.
Generate image steering pairs to train a neural network. Uses NVidia PilotNet NN topology.
Then validate the steering control by sending images to your neural network and feed steering back into the simulator to drive.

## Some videos to help you get started

### Training your first network
[![IMAGE ALT TEXT](https://img.youtube.com/vi/oe7fYuYw8GY/0.jpg)](https://www.youtube.com/watch?v=oe7fYuYw8GY "Getting Started w sdsandbox")

### World complexity
[![IMAGE ALT TEXT](https://img.youtube.com/vi/FhAKaH3ysow/0.jpg)](https://www.youtube.com/watch?v=FhAKaH3ysow "Making a more interesting world.")

### Creating a robust training set

[![IMAGE ALT TEXT](https://img.youtube.com/vi/_h8l7qoT4zQ/0.jpg)](https://www.youtube.com/watch?v=_h8l7qoT4zQ "Creating a robust sdc.")

## Setup

You need to have [Unity](https://unity3d.com/get-unity/download) installed, and all python modules listed in the Requirements section below.

Linix Unity install [here](https://forum.unity3d.com/threads/unity-on-linux-release-notes-and-known-issues.350256/). Check last post in this thread.

You need python 3.4 or higher, 64 bit. You can create a virtual env if you like:
```bash
virtualenv -p python3 env
source env/bin/activate
```

And then you can install the dependancies. This installs a specific version of keras only because it will allow you to load the pre-trained model with fewer problems. If not an issue for you, you can install the latest keras.
```bash
pip install -r requirements.txt
```

This will install [Donkey Gym](https://github.com/tawnkramer/donkey_gym) and [Donkey Car](https://github.com/tawnkramer/donkey) packages from source.

Note: Tensorflow >= 1.10.1 is required

If you have an cuda supported GPU - probably NVidia
```bash
pip install tensorflow-gpu
```

Or without a supported gpu
```bash
pip install tensorflow
```


## Demo

1) Load the Unity project sdsandbox/sdsim in Unity. Double click on Assets/Scenes/road_generator to open that scene.  

2) Hit the start button to launch. Then the "Use NN Steering". When you hit this button, the car will disappear. This is normal. You will see one car per client that connects.

3) Start the prediction server with the pre-trained model.

```bash
cd sdsandbox/src
python predict_client.py --model=../outputs/highway.h5
```
 If you get a crash loading this model, you will not be able to run the demo. But you can still generate your own model. This is a problem between tensorflow/keras versions.

 Note* You can start multiple clients at the same time and you will see them spawn as they connect.

 


#To create your own data and train

## Generate training data

1) Load the Unity project sdsandbox/sdsim in Unity.  

2) Create a dir sdsandbox/sdsim/log.  

3) Hit the start arrow in Unity to launch project.  

4) Hit button "Generate Training Data" to generate image and steering training data. See sdsim/log for output files.  

5) Stop Unity sim by clicking run arrow again.  

6) Run this python script to prepare raw data for training:  

```bash
cd sdsandbox/src
python prepare_data.py
```

7) Repeat 4, 5, 6 until you have lots of training data.



## Train Neural network

```bash
python train.py --model=../outputs/mymodel.h5
```

Let this run. It may take a few hours if running on CPU. Usually far less on a GPU.



## Run car with NN

1) Start Unity project sdsim  


2) Push button "Use NN Steering"


3) Start the prediction client. This listens for images and returns a steering result.  

```bash
python predict_client.py --model=../outputs/mymodel.h5
```



## Requirements
* [python 3.5+ 64 bit](https://www.python.org/)*
* [tensorflow-1.10+](https://github.com/tensorflow/tensorflow)
* [h5py](http://www.h5py.org/)  
* [pillow](https://python-pillow.org/)  
* [pygame](https://pypi.python.org/pypi/Pygame)**  
* [Unity 2018.+](https://unity3d.com/get-unity/download)  


**Note: pygame only needed if using mon_and_predict_server.py which gives a live camera feed during inferencing.


## Credits

Tawn Kramer  
