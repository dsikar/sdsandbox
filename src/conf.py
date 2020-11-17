import math

training_patience = 60

training_batch_size = 128

training_default_epochs = 100

training_default_aug_mult = 1

training_default_aug_percent = 0.0

# default model name
model_name = 'nvidia1'
#nvidia 1 - use this size for both
#nvidia 1 and 2, just change nvidia size before
# presenting to neural network
#image_width = 160
#image_height = 120
#nvidia 2

# size augmentation process is expecting, i.e. what came from camera
# AlexNet 224x224, Udacity 320x160, Unity 160x120, etc

image_width = 320
image_height = 160
image_depth = 3

# size network is expecting
image_height_net = 66
image_width_net = 200

row = image_height_net
col = image_width_net
ch = image_depth

# training for steering and throttle:
num_outputs = 2
# steering alone:
# num_outputs = 1

throttle_out_scale = 1.0
# alexnet
batch_size = 128 # nvidia1 = 64

# Using class members to avoid passing same parameters through various functions
# The original NVIDIA paper mentions augmentation but no cropping i.e. road only
# augmentation
aug = False
# pre-process image: crop, resize and rgb2yuv
preproc = False


