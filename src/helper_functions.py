"""
Additonal functions to help with different file formats
"""
import csv

import os
import sys
import argparse
import time
import json
import base64
import datetime
from io import BytesIO

import tensorflow as tf
from tensorflow.python import keras
from tensorflow.python.keras.models import load_model
from PIL import Image
from PIL import Image
import numpy as np

def predict_sa(model_path, image_path):
    """
    Predict steering angle for a single image.
    # Arguments
        model_path: string, file path to model
        image_path: string, file path to image
    # Outputs
        prediction (steering angle)
    # Usage
        mypred = predict_sa('../outputs/intel_model_qsub_v2.h5', '../dataset/log/logs_Fri_Jul_10_09_16_18_2020/10000_cam-image_array_.jpg')
        print(mypred)
    """
    model = load_model(model_path)
    # we may need to compile model
    # model.compile("sgd", "mse")
    image = Image.open(image_path)
    image = np.array(image, dtype=np.float32)
    # Conform to training shape (dim=4)
    images = []
    images.append(image)
    pred_img = np.array(images)
    # print(pred_img.shape) # (1, 120, 160, 3)
    # print(type(pred_img)) # <class 'numpy.ndarray'>
    return model.predict(pred_img)



def getdict(filepath):
    """
    Create a dictionary from file.
    The expected input format is:

    frame_id,steering_angle
    1479425441182877835,-0.373665106110275
    1479425441232704425,-0.0653962884098291
    (...)

    Where the first column will be the key as string datatype
    and the second column will be the key converted to float.

    # Arguments
        filepath: string, path to file

    # Returns
        dictionary

    # Usage
        filepath = '../dataset/udacity/Ch2_001/final_example.csv'
        mydict = getdict(filepath)
        mydict.get('1479425441182877835')
        >> -0.373665106110275
    """
    file = open(filepath, 'r')
    reader = csv.reader(file)
    # skip first line headers
    next(reader, None)
    mydict = {rows[0]: float(rows[1]) for rows in reader}
    return mydict

# make dict available
filepath = '../dataset/udacity/Ch2_001/final_example.csv'
mydict = getdict(filepath)

if __name__ == "__main__":
    mypred = predict_sa('../outputs/intel_model_qsub_v2.h5', '../dataset/log/logs_Fri_Jul_10_09_16_18_2020/10400_cam-image_array_.jpg')
    print(mypred[0][0])

    """
    Notes on random single tests 
    predict_sa('../outputs/intel_model_qsub_v2.h5', '../dataset/log/logs_Fri_Jul_10_09_16_18_2020/10000_cam-image_array_.jpg')
    0.105303116
    cat ../dataset/log/logs_Fri_Jul_10_09_16_18_2020/record_10000.json  
    "user/angle":0.09035701304674149,

    predict_sa('../outputs/intel_model_qsub_v2.h5', '../dataset/log/logs_Fri_Jul_10_09_16_18_2020/10100_cam-image_array_.jpg')
    -0.12858404
    cat ../dataset/log/logs_Fri_Jul_10_09_16_18_2020/record_10100.json
    "user/angle":-0.13314887881278993

    predict_sa('../outputs/intel_model_qsub_v2.h5', '../dataset/log/logs_Fri_Jul_10_09_16_18_2020/10400_cam-image_array_.jpg')
    0.081949875
    cat ../dataset/log/logs_Fri_Jul_10_09_16_18_2020/record_10400.json
    "user/angle":0.07557757198810578, 
    """
