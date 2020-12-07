import math

training_patience = 6
training_default_epochs = 100
training_default_aug_mult = 1
training_default_aug_percent = 0.0
learning_rate = 0.00001 # 0.00001

# default model name
# model_name = 'nvidia1'
#nvidia 1 - use this size for both
#nvidia 1 and 2, just change nvidia size before
# presenting to neural network
#image_width = 160
#image_height = 120
#nvidia 2

# size augmentation process is expecting, i.e. what came from camera
# AlexNet 224x224, Udacity 320x160, Unity 160x120, etc

# IMAGE DIMS INDEXES
# expected original image size
IMG_WIDTH_IDX = 0
IMG_HEIGHT_IDX = 1
IMG_DEPTH_IDX = 2
# size to be presented to network
IMG_WIDTH_NET_IDX = 3
IMG_HEIGHT_NET_IDX = 4
# to crop road from image
IMG_TOP_CROP_IDX = 5
IMG_BOTTOM_CROP_IDX = 6

# What these lists mean:
# Expected width, height and depth of acquired image,
# width and height of image expected by network
# top crop and bottom crop to remove car and sky from image
# ALEXNET
ALEXNET = 'alexnet'
alexnet_img_dims = [224,224,3,224,224,60,-25]
# NVIDIA1
NVIDIA1 = 'nvidia1' # a.k.a. TawnNet
nvidia1_img_dims = [160,120,3,160,120,60,-25]
# NVIDIA2
NVIDIA2 = 'nvidia2' # a.k.a. NaokiNet
nvidia2_img_dims = [320,160,3,200,66,83,-35]
# NVIDIA_BASELINE
NVIDIA_BASELINE = 'nvidia_baseline' # a.k.a. NaokiNet
nvidia_baseline_img_dims = [320,160,3,200,66,81,-35]

# Alexnet
image_width_alexnet = 224
image_height_alexnet = 224
# nvidia
image_width = 160
image_height = 120
#nvidia2 (Udacity NaokiNet)
#image_width = 160
#image_height = 120

# size network is expecting
image_width_net = 160
image_height_net = 120
# same for all
image_depth = 3

row = image_height_net
col = image_width_net
ch = image_depth

# training for steering and throttle:
num_outputs = 2
# steering alone:
# num_outputs = 1

throttle_out_scale = 1.0
# alexnet
batch_size = 64

# Using class members to avoid passing same parameters through various functions
# The original NVIDIA paper mentions augmentation but no cropping i.e. road only
# augmentation
aug = False
# pre-process image: crop, resize and rgb2yuv
preproc = False
# image normalization constant, Unity model maximum steer
norm_const = 25

# rain type and slant
rt = ''
st = 0

# video recording
VIDEO_WIDTH, VIDEO_HEIGHT = 800, 600
IMAGE_STILL_WIDTH, IMAGE_STILL_HEIGHT = 800, 600
record = False

def setdims(modelname):
    """
    Set image dimensions for training and predicting
    Inputs
        modelname: string, network name
    Outputs
        none
    Example
    setdims('alexnet') # set width and height to 224
    """
    if(modelname=='alexnet'):
        conf.row = image_width_alexnet
        self.col = image_height_alexnet
