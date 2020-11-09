import math

training_patience = 6

training_batch_size = 128

training_default_epochs = 100

training_default_aug_mult = 1

training_default_aug_percent = 0.0
#nvidia 1
#image_width = 160
#image_height = 120
#nvidia 2
image_width = 200
image_height = 66
image_depth = 3

row = image_height
col = image_width
ch = image_depth

#when we wish to try training for steering and throttle:
num_outputs = 2

#when steering alone:
#num_outputs = 1

throttle_out_scale = 1.0


