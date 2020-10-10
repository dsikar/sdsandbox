import math

training_patience = 6

training_batch_size = 128

training_default_epochs = 100

training_default_aug_mult = 1

training_default_aug_percent = 0.0

image_width = 160
image_height = 120
image_depth = 3

# Udacity
ud_image_width = 640
ud_image_height = 480

# datasource='unity' # steering angle .json files attributes "user/angle","user/throttle", corresponding to each image
datasource='udacity' # steering

row = image_height
col = image_width
ch = image_depth

#when we wish to try training for steering and throttle:
# num_outputs = 2

#when steering alone:
num_outputs = 1 # for udacity data, we only have steering

throttle_out_scale = 1.0

