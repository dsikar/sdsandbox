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
#from augmentation import augment, preprocess
import Augment_cls
import cv2
from train import load_json, get_files
# from augmentation import preprocess
from utils.steerlib import gos, plotSteeringAngles
from pathlib import Path

from tensorflow.python.keras.models import load_model



# 1. get a list of files to predict (sequential)
# 2. read the steering angle (json file)
# 3. generate a prediction
# 4. Store results in list
# 5. Generate a "goodness of steer" value (average steering error)
# 6. Generate graph

def predict_drive(datapath, modelpath, nc, modelname='nvidia2'):
    """
    Generate predictions from a model for a dataset. Note, this previously relied
    on image dimentions being set in conf.py, when using augmentation. Now we 
    switched to using Augment_cls. To make it backwards compatible, we introduce
    a model name variable, with a default of nvidia2, and use that geometry.
    Inputs
        datapath: string, path to data
        modelpath: string, path to trained model
        nc: steering angle normalization constant
    """
    ag = Augment_cls.Augment_cls(modelname)
    print("loading model", modelpath)
    model = load_model(modelpath)

    # In this mode, looks like we have to compile it
    # NB this is a bit tricky, do need to use optimizer and loss function used to train model?
    model.compile("sgd", "mse")

    files = get_files(datapath, True)
    outputs = []
    for fullpath in files:
        frame_number = os.path.basename(fullpath).split("_")[0]
        json_filename = os.path.join(os.path.dirname(fullpath), "record_" + frame_number + ".json")
        data = load_json(json_filename)
        # ground truth
        steering = float(data["user/angle"]) # normalized - divided by nc by simulator
        # prediction
        image = cv2.imread(fullpath)
        # The image will be 1. resized to expected pre-processing size and 2.resized to expected
        # size to be presented to network. This is network architecture and dataset dependant and
        # currently managed in conf.py
        image = ag.preprocess(image)
        image = image.reshape((1,) + image.shape)
        mod_pred = model.predict(image)
        # append prediction and ground truth to list
        outputs.append([mod_pred[0][1], steering])
    # get goodness of steer
    sarr = np.asarray(outputs)
    p = sarr[:, 0]
    g = sarr[:, 1]
    gs = gos(p,g,nc)
    print(gs)
    # def plotSteeringAngles(p, g=None, n=1, save=False, track= "Track Name", mname="model name", title='title'):
    gss = "{:.2f}".format(gs)
    modelpath = modelpath.split('/')
    datapath = datapath.split('/')
    plotSteeringAngles(p, g, nc, True, datapath[-2], modelpath[-1], 'Gs ' + gss)


# dataset ../dataset/unity/jungle1/
# model ../trained_models/nvidia2/20201124032017_nvidia2.h5
    # calculate gos (average steering error)
    # plot graph unormalized angles. + average steering error
    # save graph.
    # done

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='prediction server')
    parser.add_argument('--datapath', type=str, default='/home/simbox/git/msc-data/unity/genTrackOneLap_3/*.jpg', help='model filename')
    parser.add_argument('--modelpath', type=str, default='/home/simbox/git/sdsandbox/trained_models/sanity/20201120171015_sanity.h5', help='Model')
    parser.add_argument('--nc', type=int, default=1, help='Steering Angle Normalization Constant')
    # time allowing, set image sizes based on model name. For now, these have to be managed in conf.py
    #parser.add_argument('--model', type=str, default='nvidia1', help='model name')

    # set dimensions

    args = parser.parse_args()

    predict_drive(args.datapath, args.modelpath, args.nc)
    # max value for slant is 20
    # Example
    # python3 predict_client.py --model=../trained_models/sanity/20201120171015_sanity.h5 --rain=light --slant=0





