# Predict steering angles for a dataset for which a ground truth exists
# dsikar@gmail.com

from __future__ import print_function

import argparse
import fnmatch
import json
import os
import pickle
import random
from datetime import datetime
from time import strftime
import numpy as np
from PIL import Image
from tensorflow import keras
import tensorflow as tf
import conf
import models
from helper_functions import hf_mkdir
from augmentation import augment, preprocess
import cv2
from train import load_json, get_files
from augmentation import preprocess

from tensorflow.python.keras.models import load_model



# 1. get a list of files to predict (sequential)
# 2. read the steering angle (json file)
# 3. generate a prediction
# 4. Store results in list
# 5. Generate a "goodness of steer" value (average steering error)
# 6. Generate graph

inputs = "../dataset/unity/genRoad/.."

def predict_drive(datapath, modelpath):

    print("loading model", modelpath)
    model = load_model(modelpath)
    # In this mode, looks like we have to compile it
    # NB this is a bit tricky, as we might need to use optimizer and loss function used to train model?
    model.compile("sgd", "mse")

    files = get_files(inputs)
    outputs = []
    for fullpath in files:
        frame_number = os.path.basename(fullpath).split("_")[0]
        json_filename = os.path.join(os.path.dirname(fullpath), "record_" + frame_number + ".json")
        data = load_json(json_filename)
        # ground truth
        steering = float(data["user/angle"])
        # prediction
        image = cv2.imread(fullpath)
        image = preprocess(image)
        image = image.reshape((1,) + image.shape)
        mod_pred = model.predict(image)
        # append to list
        outputs = ([steering, outputs[0][0]])

# dataset ../dataset/unity/jungle1/
# model ../trained_models/nvidia2/20201124032017_nvidia2.h5
    # calculate gos (average steering error)
    # plot graph unormalized angles. + average steering error
    # save graph.
    # done

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='prediction server')
    parser.add_argument('--datapath', type=str, default='/home/simbox/git/msc-data/unity/genTrackOneLap', help='model filename')
    parser.add_argument('--modelpath', type=str, default='127.0.0.1', help='server sim host')

    args = parser.parse_args()

    predict_drive(args.model, address, args.constant_throttle, num_cars=args.num_cars, rand_seed=args.rand_seed)
    # max value for slant is 20
    # Example
    # python3 predict_client.py --model=../trained_models/sanity/20201120171015_sanity.h5 --rain=light --slant=0





