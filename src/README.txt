** Notes on data **

The original dataset for these scripts is comprised of Unity 3D engine outputs, consisting of a set of images and corresponding .json files.
The .json file contains attributes including the corresponding image name, steering angle and throtle values, as specified in meta.json, as generated with every output.

Script "prepare_data.py" processes .jpg and .json files, ready for "train.py".

In the case of Udacity data, there is one single .csv files with all steering values for corresponding images, the id being the image name excluding the extension ".jpg".

The scripts are then slightly modified to deal with this case.

1. A flag is created in conf.py to identify udacity data.
2. helper_functions.py script is added, containing a getdict() function which creates a dictionary from the udacity .csv file. Keys (filenames) can then be searched to obtain the steering angle.
3. prepare_data.py script is not run for udacity data. The dataset must be unpacked to a known path e.g. dataset/udacity such that the image files will be in directoy e.g. dataset/udacity/Ch2_001/center
4. helper_function.py will contain a reference to the .csv file e.g. 
5. Images are resized from 640 x 480 (udacity standard) to 160 x 120 (unity standard).

filepath = '../dataset/udacity/Ch2_001/final_example.csv'

5. When running train.py, the datapath must be provided e.g.

python train.py --model=../outputs/udacity1.h5 --inputs=../dataset/udacity/Ch2_001/center/*.jpg