import math

training_patience = 6

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

# nvidia baseline
image_width = 200
image_height = 66
image_depth = 3

row = image_height
col = image_width
ch = image_depth

#when we wish to try training for steering and throttle:
#num_outputs = 2

#when steering alone:
num_outputs = 1

throttle_out_scale = 1.0
# alexnet
batch_size = 128 # nvidia1 = 64

# Using class members to avoid passing same parameters through various functions
# The original NVIDIA paper mentions augmentation but no cropping i.e. road only
# augmentation
aug = False
# pre-process image: crop, resize and rgb2yuv
preproc = False


