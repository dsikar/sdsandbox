# Helper library to create videos and plots

import fnmatch
import json
import os
import numpy as np
import matplotlib.pyplot as plt
import statistics
import seaborn as sns
import pickle
from PIL import Image
# prediction
import tensorflow as tf
from tensorflow.python import keras
from tensorflow.python.keras.models import load_model
# rain
from predict_client import add_rain
# Augmentation library
import Augmentation

def load_json(filepath):
    """
    Load a json file
    Inputs
        filepath: string, path to file
    Outputs
        data: dictionary, json key, value pairs
    Example
    path = "~/git/msc-data/unity/roboRacingLeague/log/logs_Sat_Nov_14_12_36_16_2020/record_11640.json"
    js = load_json(path)
    """
    with open(filepath, "rt") as fp:
        data = json.load(fp)
    return data

def GetSteeringFromtcpflow(filename):
    """
    Get a tcpflow log and extract steering values obtained from network communication between.
    Note, we only plot the predicted steering angle jsondict['steering']
    and the value of jsondict['steering_angle'] is ignored. Assumed to be the steering angle
    calculated by PID given the current course.
    sim and prediction engine (predict_client.py)
    Inputs
        filename: string, name of tcpflow log
    Returns
        sa: list of arrays, steering angle predicton and actual value tuple.
    Example


    """
    # open file
    sa = []
    # initialize prediction
    pred = ''
    f = open(filename, "r")
    file = f.read()
    try:
        # readline = f.read()
        lines = file.splitlines()
        for line in lines:
            # print(line)
            start = line.find('{')
            if (start == -1):
                continue
            jsonstr = line[start:]
            # print(jsonstr)
            jsondict = json.loads(jsonstr)
            if "steering" in jsondict:
                # predicted
                pred = jsondict['steering']
                # jsondict['steering_angle']
                # sa.append([float(pred), act])
                sa.append([float(pred), float(pred)])  # append twice to keep code from breaking
            # if "steering_angle" in jsondict:
            # actual
            #   act = jsondict['steering_angle']
            # save pair, only keep last pred in case two were send as it does happen i.e.:
            # 127.000.000.001.59460-127.000.000.001.09091: {"msg_type": "control", "steering": "-0.071960375", "throttle": "0.08249988406896591", "brake": "0.0"}
            # 127.000.000.001.59460-127.000.000.001.09091: {"msg_type": "control", "steering": "-0.079734944", "throttle": "0.08631626516580582", "brake": "0.0"}
            # 127.000.000.001.09091-127.000.000.001.59460: {"msg_type":"telemetry","steering_angle":-0.07196037,(...)
            #   if(len(pred) > 0):
            #      sa.append([float(pred), act])
            #      pred = '' # need to save this image
            # deal with image later, sort out plot first
            # imgString = jsondict["image"]
            # image = Image.open(BytesIO(base64.b64decode(imgString)))
            # img_arr = np.asarray(image, dtype=np.float32)
    except Exception as e:
        print("Exception raise: " + str(e))
    # file should be automatically closed but will close for good measure
    f.close()
    return sa

def GetJSONSteeringAngles(filemask):
    """
    Get steering angles stored as 'user/angle' attributes in .json files
    Inputs:
        filemask: string, path and mask
    Outputs
        svals: list, steering values
    """
    filemask = os.path.expanduser(filemask)
    path, mask = os.path.split(filemask)

    matches = []
    for root, dirnames, filenames in os.walk(path):
        for filename in fnmatch.filter(filenames, mask):
            matches.append(os.path.join(root, filename))
    # sort by create date
    matches = sorted(matches, key=os.path.getmtime)
    # steering values
    svals = []
    for fullpath in matches:
            frame_number = os.path.basename(fullpath).split("_")[0]
            json_filename = os.path.join(os.path.dirname(fullpath), "record_" + frame_number + ".json")
            jobj = load_json(json_filename)
            svals.append(jobj['user/angle'])
    return svals

def jsonSteeringBins(filemask, pname="output", save=True, nc=25):
    """
    Plot a steering values' histogram
    Inputs
        filemask: string, where to search for images, and corresponding .json files
        pname: string, output plot name
        save: boolean, save plot to disk
        nc: int, normalization constant, used in the simulator to put angles in range
        -1, 1. Default is 25.
    Outputs
        svals: list containing non-normalized steering angles
    Example:
    # svals = jsonSteeringBins('~/git/msc-data/unity/genRoad/*.jpg', 'genRoad')
    """
    svals = GetJSONSteeringAngles(filemask)
    values = len(svals)
    svalscp = [element * nc for element in svals]
    mean = ("%.2f" % statistics.mean(svals))
    std = ("%.2f" % statistics.stdev(svals))
    plt.title=(pname)
    # NB Plotted as normalized histogram
    sns.distplot(svalscp, bins=nc*2, kde=False, norm_hist=True,
    axlabel= pname + ' steer. degs. norm. hist. ' + str(values) + ' values, mean = ' + mean + ' std = ' + std)
    #if(save):
    #    sns.save("output.png")
    plt.savefig(pname + '.png')

    # return for downstream processing if required
    return svals

def plot_hist(path):
    """
    Create loss/accuracy plot for training and save to disk
    Inputs
        path: file to history pickle file
    Outputs
        none
    Example:
    path = "/home/simbox/git/sdsandbox/trained_models/nvidia1/20201107144927_nvidia1.history"
    plot_hist(path)
    or
    $ python steerlib.py
    $ python
    $ >>> import steerlib
    $ >>> path = "/home/simbox/git/sdsandbox/trained_models/nvidia1/20201107144927_nvidia1.history"
    $ >>> plot_hist(path)
    """
    history = pickle.load(open(path, "rb"))
    # type(history)
    # <class 'dict'>
    # history.keys()
    # dict_keys(['loss', 'acc', 'val_loss', 'val_acc'])
    do_plot = True
    try:
        if do_plot:
            fig = plt.figure()  # when loading dictionary keys, we omit .history (see train.py)
            plot_name = path.split('/')[-1]
            sp = plot_name \
                 + ' - ' + '(l,vl,a,va)' + '{0:.3f}'.format(history['loss'][-1]) \
                 + ',' + '{0:.3f}'.format(history['val_loss'][-1]) \
                 + ',' + '{0:.3f}'.format(history['acc'][-1]) \
                 + ',' + '{0:.3f}'.format(history['val_acc'][-1])

            fig.suptitle(sp, fontsize=9)
            ax = fig.add_subplot(111)
            ax.plot(history['loss'], 'r-', label='Training Loss', )
            ax.plot(history['val_loss'], 'm-', label='Validation Loss')
            ax2 = ax.twinx()
            ax2.plot(history['acc'], '-', label='Training Accuracy')
            ax2.plot(history['val_acc'], '-', label='Validation Accuracy')
            ax.legend(loc=2) # https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.legend.html
            ax.grid()
            ax.set_xlabel("Epoch")
            ax.set_ylabel(r"Loss")
            ax2.set_ylabel(r"Accurary")
            ax.set_ylim(0, 0.2)
            ax2.set_ylim(0.5, 1)
            ax2.legend(loc=1)
            aimg = plot_name.split('.')[0]
            aimg = str(aimg)  + '_accuracy.png'
            plt.savefig(aimg)
    except Exception as e:
        print("Failed to save accuracy/loss graph: " + str(e))

def gos(p, g, n):
    """
    Calculate the goodness-of-steer between a prediction and a ground truth array.
    Inputs
        p: array of floats, steering angle prediction
        g: array of floats, steering angle ground truth.
        n: float, normalization constant
    Output
        gos: float, average of absolute difference between ground truth and prediction arrays
    """
    # todo add type assertion
    assert len(p) == len(g), "Arrays must be of equal length"
    return sum(abs(p - g)) / len(p) * n
    # print("Goodness of steer: {:.2f}".format(steer))
    
def plotSteeringAngles(p, g=None, n=1, save=False, track= "Track Name", mname="model name", title='title'):
    """
    Plot predicted and (TODO) optionally ground truth steering angles
    Inputs
        p, g: prediction and ground truth float arrays
        n: float, steering normalization constant
        save: boolean, save plot flag
        track, mname, title: string, track (data), trained model and title strings for plot
    Outputs
        plt: pyplot, plot
    Example
    # set argument variables (see data_predict.py)
    plotSteeringAngles(p, g, nc, True, datapath[-2], modelpath[-1], 'Gs ' + gss)
    """

    plt.rcParams["figure.figsize"] = (18,3)

    plt.plot(p*n, label="predicted")
    try:
        if (g is not None):
            plt.plot(g*n, label="ground truth")
    except Exception as e:
        print("problems plotting: " + str(e))

    plt.ylabel('Steering angle')
    plt.xlabel('Frame number')
    # Set a title of the current axes.
    # plt.title('tcpflow log predicted steering angles: track ' + track + ' model ' + mname)
    plt.title(title + ' Steering angles: track ' + track + ', model ' + mname)
    # show a legend on the plot
    plt.legend()
    # Display a figure.
    # horizontal grid only
    plt.grid(axis='y')
    # set limit
    plt.xlim([-5,len(p)+5])
    plt.gca().invert_yaxis()
    # plt.show()
    if(save==True):
        plt.savefig('sa_' + track + '_' + mname + '.png')
    # if need be
    return plt

def plotMultipleSteeringAngles(p, n=25, save=False, track="Track Name", mname="model name", title='title', w=18, h=3):
    """
    Plot multiple predicted and (TODO) optionally ground truth steering angles
    Inputs
        p: list of tuples, prediction and labels
        n: float, steering normalization constant
        save: boolean, save plot flag
        track, mname, title: string, track (data), trained model and title strings for plot
        w: integer, plot width
        h: integer, plot height
    Outputs
        plt: pyplot, plot
    Example
    # get some steering angles
    sa = GetSteeringFromtcpflow('../trained_models/nvidia1/tcpflow/20201207091932_nvidia1_no_rain_tcpflow.log')
    sarr = np.asarray(sa)
    pa = sarr[:,0]
    p.append([pa, 'no rain'])
    plotSteeringAngles(p, g, nc, True, datapath[-2], modelpath[-1], 'Gs ' + gss)
    """
    import matplotlib.pyplot as plt # local copy
    plt.rcParams["figure.figsize"] = (w,h)

    for i in range (0, len(p)):
        plt.plot(p[i][0]*n, label=p[i][1])
    #try:
      #  if (g is not None):
     #       plt.plot(g*n, label="ground truth")
    #except Exception as e:
    #    print("problems plotting: " + str(e))

    plt.ylabel('Steering angle')
    plt.xlabel('Frame number')
    # Set a title of the current axes.
    # plt.title('tcpflow log predicted steering angles: track ' + track + ' model ' + mname)
    plt.title(title + ' Steering angles: track ' + track + ', model ' + mname)
    # show a legend on the plot
    plt.legend()
    # Display a figure.
    # horizontal grid only
    plt.grid(axis='y')
    # set limit
    plt.xlim([-5,len(p[0][0])+5])
    plt.gca().invert_yaxis()
    # plt.show()
    if(save==True):
        plt.savefig('sa_' + track + '_' + mname + '.png')
    # if need be
    return plt

def getSteeringFromtcpflow(filename):
    """
    Get a tcpflow log and extract steering values obtained from network communication between sim and predict_client.py.
    Note, we only plot the predicted steering angle jsondict['steering']
    and the value of jsondict['steering_angle'] is ignored. Assumed to be the steering angle
    calculated by PID given the current course.
    sim and prediction engine (predict_client.py)
    Inputs
        filename: string, name of tcpflow log
    Returns
        sa: list of arrays, steering angle predicton and actual value tuple.
    Example


    """
    # open file
    sa = []
    # initialize prediction
    pred = ''
    f = open(filename, "r")
    file = f.read()
    try:
        # readline = f.read()
        lines = file.splitlines()
        for line in lines:
            # print(line)
            start = line.find('{')
            if (start == -1):
                continue
            jsonstr = line[start:]
            # print(jsonstr)
            jsondict = json.loads(jsonstr)
            if "steering" in jsondict:
                # predicted
                pred = jsondict['steering']
                # jsondict['steering_angle']
                # sa.append([float(pred), act])
                sa.append([float(pred), float(pred)])  # append twice to keep code from breaking
            # if "steering_angle" in jsondict:
            # actual
            #   act = jsondict['steering_angle']
            # save pair, only keep last pred in case two were send as it does happen i.e.:
            # 127.000.000.001.59460-127.000.000.001.09091: {"msg_type": "control", "steering": "-0.071960375", "throttle": "0.08249988406896591", "brake": "0.0"}
            # 127.000.000.001.59460-127.000.000.001.09091: {"msg_type": "control", "steering": "-0.079734944", "throttle": "0.08631626516580582", "brake": "0.0"}
            # 127.000.000.001.09091-127.000.000.001.59460: {"msg_type":"telemetry","steering_angle":-0.07196037,(...)
            #   if(len(pred) > 0):
            #      sa.append([float(pred), act])
            #      pred = '' # need to save this image
            # deal with image later, sort out plot first
            # imgString = jsondict["image"]
            # image = Image.open(BytesIO(base64.b64decode(imgString)))
            # img_arr = np.asarray(image, dtype=np.float32)
    except Exception as e:
        print("Exception raise: " + str(e))
    # file should be automatically closed but will close for good measure
    f.close()
    return sa

"""

# get image

# get steering

# add / don't add rain

outputs = self.model.predict(image_array)

Append groundtruth, predicted to list.
"""
def PrintLatexRowModelGOS(filemask, modelpath, modelname, rt='', st=0):
    """
    Generate a "goodness of fit value" for a model on a given track
    Inputs
        filemask: string, path and mask
        modelpath: string, path to keras model
        modelname: string, canonical model name e.g. nvidia1, nvidia2, nvidia_baseline
        rt: string, rain type e.g. drizzle/light, heavy torrential
        st: integer, -+20 degree rain slant
    Outputs
        svals: list, ground truth steering values and predictions
    """
    # load augmentation library for correct model geometry
    ag = Augmentation.Augmentation(modelname)

    # load model
    print("loading model", modelpath)
    model = load_model(modelpath)

    # In this mode, looks like we have to compile it
    model.compile("sgd", "mse")
    filemask = os.path.expanduser(filemask)
    path, mask = os.path.split(filemask)

    matches = []
    for root, dirnames, filenames in os.walk(path):
        for filename in fnmatch.filter(filenames, mask):
            matches.append(os.path.join(root, filename))

    # steering values
    svals = []
    for fullpath in matches:
        frame_number = os.path.basename(fullpath).split("_")[0]
        json_filename = os.path.join(os.path.dirname(fullpath), "record_" + frame_number + ".json")
        jobj = load_json(json_filename)
        # steering ground truth
        steer_gt = jobj['user/angle']
        # open the image
        img_arr = Image.open(fullpath)
        # Convert PIL Image to numpy array
        img_arr = np.array(img_arr, dtype=np.float32)
        # add rain if need be
        if rt != '':
            img_arr = add_rain(img_arr, rt, st)
        # apply same preprocessing
        # same preprocessing as for training
        img_arr = ag.preprocess(img_arr)

        # put in correct format
        img_arr = img_arr.reshape((1,) + img_arr.shape)
        # generate prediction
        outputs = model.predict(img_arr)
        # store predictions
        steer_pred = outputs[0][0]
        # store ground truth and prediction
        svals.append([steer_pred, steer_gt])
    # get goodness of fit
    sarr = np.asarray(svals)
    p = sarr[:, 0]
    g = sarr[:, 1]
    nc = 25 # unity maximum steering angle / normalization constant - should hold in conf.py and managed with Augmentation
    mygos = gos(p, g, nc)
    # format to human readable/friendlier 2 decimal places
    gos_str = "{:.2f}".format(round(mygos, 2))
    # strip path from modelpath
    modelfile = modelpath.split('/')
    modelfile = modelfile[-1]
    # print latex formated data
    # header
    hd_str = 'Filename & Model & Rain Type & Slant & gos \\\\ \hline'
    # log file
    print('Log: ', path, '\\\\ \hline')
    print(hd_str)
    # results
    res_str = f'{modelfile} & {modelname} & {rt} & {st} & {gos_str} \\\\ \hline'
    print(res_str)

def GetPredictedSteeringAngles(filemask, model, modelname, rt='', st=0):
    """
    Generate a "goodness of fit value" for a model on a given track
    Inputs
        filemask: string, path and mask
        modelpath: string, path to keras model
        modelname: string, canonical model name e.g. nvidia1, nvidia2, nvidia_baseline
        rt: string, rain type e.g. drizzle/light, heavy torrential
        st: integer, -+20 degree rain slant
    Outputs
        svals: list, ground truth steering values and predictions
    """
    # load augmentation library for correct model geometry
    ag = Augmentation.Augmentation(modelname)

    # load model
    # print("loading model", modelpath)
    # assume model is loaded and compiled
    # model = load_model(modelpath)

    # In this mode, looks like we have to compile it
    # model.compile("sgd", "mse")
    filemask = os.path.expanduser(filemask)
    path, mask = os.path.split(filemask)

    matches = []
    for root, dirnames, filenames in os.walk(path):
        for filename in fnmatch.filter(filenames, mask):
            matches.append(os.path.join(root, filename))

    # sort by create date
    matches = sorted(matches, key=os.path.getmtime)
    # steering values
    svals = []
    for fullpath in matches:
        frame_number = os.path.basename(fullpath).split("_")[0]
        json_filename = os.path.join(os.path.dirname(fullpath), "record_" + frame_number + ".json")
        jobj = load_json(json_filename)
        # steering ground truth
        steer_gt = jobj['user/angle']
        # open the image
        img_arr = Image.open(fullpath)
        # Convert PIL Image to numpy array
        img_arr = np.array(img_arr, dtype=np.float32)
        # add rain if need be
        if rt != '':
            img_arr = add_rain(img_arr, rt, st)
        # apply same preprocessing
        # same preprocessing as for training
        img_arr = ag.preprocess(img_arr)

        # put in correct format
        img_arr = img_arr.reshape((1,) + img_arr.shape)
        # generate prediction
        outputs = model.predict(img_arr)
        # store predictions
        steer_pred = outputs[0][0]
        # store ground truth and prediction
        svals.append([steer_pred, steer_gt])
    return svals

def printGOSRows():
    """
    Print GOS rows for results report.
    We are comparing mainly the two best models for nvidia1 and nvidia2, would have been nice to also test
    the driveable nvidia_baseline, and sanity models
    """
    # models
    # nvidia2 - ../../trained_models/nvidia2/20201207192948_nvidia2.h5
    # nvidia1 - ../../trained_models/nvidia1/20201207091932_nvidia1.h5
    # 20201120124421\_nvidia\_baseline.h5
    # log logs_Wed_Nov_25_23_39_22_2020
    ############################################
    # 1. nvidia2 Generated track
    ############################################
    # log = '../../dataset/unity/genTrack/genTrackOneLap/logs_Wed_Nov_25_23_39_22_2020/*.jpg'
    #modelpath = '../../trained_models/nvidia2/20201207192948_nvidia2.h5'
    #modelname = 'nvidia2'
    #rt = 'torrential'
    #st = 20
    #PrintLatexRowModelGOS(log, modelpath, modelname, rt, st)
    # Log:  ../../dataset/unity/genTrack/genTrackOneLap/logs_Wed_Nov_25_23_39_22_2020 \\ \hline
    # Filename & Model & Rain Type & Slant & gos \\ \hline
    # 20201207192948_nvidia2.h5 & nvidia2 &  & 0 & 1.68 \\ \hline
    # Log:  ../../dataset/unity/genTrack/genTrackOneLap/logs_Wed_Nov_25_23_39_22_2020 \\ \hline
    # Filename & Model & Rain Type & Slant & gos \\ \hline
    # 20201207192948_nvidia2.h5 & nvidia2 & light & 0 & 2.12 \\ \hline
    # Log:  ../../dataset/unity/genTrack/genTrackOneLap/logs_Wed_Nov_25_23_39_22_2020 \\ \hline
    # Filename & Model & Rain Type & Slant & gos \\ \hline
    # 20201207192948_nvidia2.h5 & nvidia2 & heavy & 10 & 2.17 \\ \hline
    # Log:  ../../dataset/unity/genTrack/genTrackOneLap/logs_Wed_Nov_25_23_39_22_2020 \\ \hline
    # Filename & Model & Rain Type & Slant & gos \\ \hline
    # 20201207192948_nvidia2.h5 & nvidia2 & torrential & 20 & 2.30 \\ \hline
    #####################################################
    # 2. nvidia1 Generated track
    #####################################################
    # modelpath = '../../trained_models/nvidia1/20201207091932_nvidia1.h5'
    #modelname = 'nvidia1'
    #rt = 'torrential'
    #st = 20
    #PrintLatexRowModelGOS(log, modelpath, modelname, rt, st)
    # Log:  ../../dataset/unity/genTrack/genTrackOneLap/logs_Wed_Nov_25_23_39_22_2020 \\ \hline
    # Filename & Model & Rain Type & Slant & gos \\ \hline
    # 20201207091932_nvidia1.h5 & nvidia1 &  & 0 & 1.82 \\ \hline
    # Log:  ../../dataset/unity/genTrack/genTrackOneLap/logs_Wed_Nov_25_23_39_22_2020 \\ \hline
    # Filename & Model & Rain Type & Slant & gos \\ \hline
    # 20201207091932_nvidia1.h5 & nvidia1 & light & 0 & 2.11 \\ \hline
    # Log:  ../../dataset/unity/genTrack/genTrackOneLap/logs_Wed_Nov_25_23_39_22_2020 \\ \hline
    # Filename & Model & Rain Type & Slant & gos \\ \hline
    # 20201207091932_nvidia1.h5 & nvidia1 & heavy & 10 & 2.13 \\ \hline
    # Log:  ../../dataset/unity/genTrack/genTrackOneLap/logs_Wed_Nov_25_23_39_22_2020 \\ \hline
    # Filename & Model & Rain Type & Slant & gos \\ \hline
    # 20201207091932_nvidia1.h5 & nvidia1 & torrential & 20 & 2.28 \\ \hline
    #####################################################
    # 3. nvidia_baseline Generated track
    #####################################################
    #modelpath = '../../trained_models/nvidia_baseline/20201207201157_nvidia_baseline.h5'
    #modelname = 'nvidia2_baseline'
    #rt = 'torrential'
    #st = 20
    #PrintLatexRowModelGOS(log, modelpath, modelname, rt, st)
    # Log:  ../../dataset/unity/genTrack/genTrackOneLap/logs_Wed_Nov_25_23_39_22_2020 \\ \hline
    # Filename & Model & Rain Type & Slant & gos \\ \hline
    # 20201207201157_nvidia_baseline.h5 & nvidia2_baseline &  & 0 & 2.32 \\ \hline
    # Log:  ../../dataset/unity/genTrack/genTrackOneLap/logs_Wed_Nov_25_23_39_22_2020 \\ \hline
    # Filename & Model & Rain Type & Slant & gos \\ \hline
    # 20201207201157_nvidia_baseline.h5 & nvidia2_baseline & light & 0 & 3.12 \\ \hline
    # Log:  ../../dataset/unity/genTrack/genTrackOneLap/logs_Wed_Nov_25_23_39_22_2020 \\ \hline
    # Filename & Model & Rain Type & Slant & gos \\ \hline
    # 20201207201157_nvidia_baseline.h5 & nvidia2_baseline & heavy & 10 & 3.17 \\ \hline
    # Log:  ../../dataset/unity/genTrack/genTrackOneLap/logs_Wed_Nov_25_23_39_22_2020 \\ \hline
    # Filename & Model & Rain Type & Slant & gos \\ \hline
    # 20201207201157_nvidia_baseline.h5 & nvidia2_baseline & torrential & 20 & 3.39 \\ \hline
    #####################################################
    # 4. sanity Generated track 20201120171015\_ sanity.h5
    #####################################################
    #modelpath = '../../trained_models/sanity/20201120171015_sanity.h5'
    #modelname = 'nvidia1'
    #rt = 'torrential'
    #st = 20
    #PrintLatexRowModelGOS(log, modelpath, modelname, rt, st)
    # Log:  ../../dataset/unity/genTrack/genTrackOneLap/logs_Wed_Nov_25_23_39_22_2020 \\ \hline
    # Filename & Model & Rain Type & Slant & gos \\ \hline
    # 20201120171015_sanity.h5 & nvidia1 &  & 0 & 5.03 \\ \hline
    # Log:  ../../dataset/unity/genTrack/genTrackOneLap/logs_Wed_Nov_25_23_39_22_2020 \\ \hline
    # Filename & Model & Rain Type & Slant & gos \\ \hline
    # 20201120171015_sanity.h5 & nvidia1 & light & 0 & 3.11 \\ \hline
    # Log:  ../../dataset/unity/genTrack/genTrackOneLap/logs_Wed_Nov_25_23_39_22_2020 \\ \hline
    # Filename & Model & Rain Type & Slant & gos \\ \hline
    # 20201120171015_sanity.h5 & nvidia1 & heavy & 10 & 3.07 \\ \hline
    # Log:  ../../dataset/unity/genTrack/genTrackOneLap/logs_Wed_Nov_25_23_39_22_2020 \\ \hline
    # Filename & Model & Rain Type & Slant & gos \\ \hline
    # 20201120171015_sanity.h5 & nvidia1 & torrential & 20 & 3.00 \\ \hline

    ############################################
    # 5. nvidia2 Generated Road
    ############################################
    log = '../../dataset/unity/genRoad/logs_Fri_Jul_10_09_16_18_2020/*.jpg'
    #modelpath = '../../trained_models/nvidia2/20201207192948_nvidia2.h5'
    #modelname = 'nvidia2'
    #rt = 'torrential'
    #st = 20
    #PrintLatexRowModelGOS(log, modelpath, modelname, rt, st)
    # Log:  ../../dataset/unity/genRoad/logs_Fri_Jul_10_09_16_18_2020 \\ \hline
    # Filename & Model & Rain Type & Slant & gos \\ \hline
    # 20201207192948_nvidia2.h5 & nvidia2 &  & 0 & 2.99 \\ \hline # drove 16 minutes on this road https://youtu.be/z9nILq9dQfI
    # Log:  ../../dataset/unity/genRoad/logs_Fri_Jul_10_09_16_18_2020 \\ \hline
    # Filename & Model & Rain Type & Slant & gos \\ \hline
    # 20201207192948_nvidia2.h5 & nvidia2 & light & 0 & 3.20 \\ \hline
    # Log:  ../../dataset/unity/genRoad/logs_Fri_Jul_10_09_16_18_2020 \\ \hline
    # Filename & Model & Rain Type & Slant & gos \\ \hline
    # 20201207192948_nvidia2.h5 & nvidia2 & heavy & 10 & 3.22 \\ \hline
    # Log:  ../../dataset/unity/genRoad/logs_Fri_Jul_10_09_16_18_2020 \\ \hline
    # Filename & Model & Rain Type & Slant & gos \\ \hline
    # 20201207192948_nvidia2.h5 & nvidia2 & torrential & 20 & 3.27 \\ \hline
    #####################################################
    # 6. nvidia1 Generated Road
    #####################################################
    #modelpath = '../../trained_models/nvidia1/20201207091932_nvidia1.h5'
    #modelname = 'nvidia1'
    # rt = 'torrential'
    # st = 20
    # PrintLatexRowModelGOS(log, modelpath, modelname, rt, st)
    # Log:  ../../dataset/unity/genRoad/logs_Fri_Jul_10_09_16_18_2020 \\ \hline
    # Filename & Model & Rain Type & Slant & gos \\ \hline
    # 20201207091932_nvidia1.h5 & nvidia1 &  & 0 & 3.87 \\ \hline
    # Log:  ../../dataset/unity/genRoad/logs_Fri_Jul_10_09_16_18_2020 \\ \hline
    # Filename & Model & Rain Type & Slant & gos \\ \hline
    # 20201207091932_nvidia1.h5 & nvidia1 & light & 0 & 3.75 \\ \hline
    # Log:  ../../dataset/unity/genRoad/logs_Fri_Jul_10_09_16_18_2020 \\ \hline
    # Filename & Model & Rain Type & Slant & gos \\ \hline
    # 20201207091932_nvidia1.h5 & nvidia1 & heavy & 10 & 3.70 \\ \hline
    # Log:  ../../dataset/unity/genRoad/logs_Fri_Jul_10_09_16_18_2020 \\ \hline
    # Filename & Model & Rain Type & Slant & gos \\ \hline
    # 20201207091932_nvidia1.h5 & nvidia1 & torrential & 20 & 3.57 \\ \hline

    #####################################################
    # 7. nvidia_baseline Generated Road
    #####################################################
    #modelpath = '../../trained_models/nvidia_baseline/20201207201157_nvidia_baseline.h5'
    #modelname = 'nvidia2_baseline'
    #rt = 'torrential'
    #st = 20
    #PrintLatexRowModelGOS(log, modelpath, modelname, rt, st)
    # Log:  ../../dataset/unity/genRoad/logs_Fri_Jul_10_09_16_18_2020 \\ \hline
    # Filename & Model & Rain Type & Slant & gos \\ \hline
    # 20201207201157_nvidia_baseline.h5 & nvidia2_baseline &  & 0 & 5.51 \\ \hline
    # Log:  ../../dataset/unity/genRoad/logs_Fri_Jul_10_09_16_18_2020 \\ \hline
    # Filename & Model & Rain Type & Slant & gos \\ \hline
    # 20201207201157_nvidia_baseline.h5 & nvidia2_baseline & light & 0 & 4.97 \\ \hline
    # Log:  ../../dataset/unity/genRoad/logs_Fri_Jul_10_09_16_18_2020 \\ \hline
    # Filename & Model & Rain Type & Slant & gos \\ \hline
    # 20201207201157_nvidia_baseline.h5 & nvidia2_baseline & heavy & 10 & 4.98 \\ \hline
    # Log:  ../../dataset/unity/genRoad/logs_Fri_Jul_10_09_16_18_2020 \\ \hline
    # Filename & Model & Rain Type & Slant & gos \\ \hline
    # 20201207201157_nvidia_baseline.h5 & nvidia2_baseline & torrential & 20 & 5.05 \\ \hline
    #####################################################
    # 8. sanity Generated track 20201120171015\_ sanity.h5
    #####################################################
    modelpath = '../../trained_models/sanity/20201120171015_sanity.h5'
    modelname = 'nvidia1'
    rt = 'torrential'
    st = 20
    PrintLatexRowModelGOS(log, modelpath, modelname, rt, st)
    # Log:  ../../dataset/unity/genRoad/logs_Fri_Jul_10_09_16_18_2020 \\ \hline
    # Filename & Model & Rain Type & Slant & gos \\ \hline
    # 20201120171015_sanity.h5 & nvidia1 &  & 0 & 3.85 \\ \hline
    # Log:  ../../dataset/unity/genRoad/logs_Fri_Jul_10_09_16_18_2020 \\ \hline
    # Filename & Model & Rain Type & Slant & gos \\ \hline
    # 20201120171015_sanity.h5 & nvidia1 & light & 0 & 3.06 \\ \hline
    # Log:  ../../dataset/unity/genRoad/logs_Fri_Jul_10_09_16_18_2020 \\ \hline
    # Filename & Model & Rain Type & Slant & gos \\ \hline
    # 20201120171015_sanity.h5 & nvidia1 & heavy & 10 & 3.05 \\ \hline
    # Log:  ../../dataset/unity/genRoad/logs_Fri_Jul_10_09_16_18_2020 \\ \hline
    # Filename & Model & Rain Type & Slant & gos \\ \hline
    # 20201120171015_sanity.h5 & nvidia1 & torrential & 20 & 3.02 \\ \hline

def printMultiPlots(model1_nvidia2, model2_nvidia1, model3_nvidia_baseline, model4_sanity):
    """
    Print multiple plots to reference first 16 results in Goodness of steer tables
    Inputs
     model1_nvidia2, model2_nvidia1, model3_nvidia_baseline, model4_sanity: the required keras models
    """
    # init plot list
    plot_list = []
    # define log
    log = '../../dataset/unity/genTrack/genTrackOneLap/logs_Wed_Nov_25_23_39_22_2020/*.jpg'
    # Get ground truth values
    gt = GetJSONSteeringAngles(log)
    gt = np.asarray(gt)
    # get predictions
    ############################################
    # 1. nvidia2 Generated track
    ############################################
    # 1.1 dry
    """
    print('Predicting nvidia2 dry...')
    sa = GetPredictedSteeringAngles(log, model1_nvidia2, 'nvidia2', rt='', st=0)
    sarr = np.asarray(sa)
    p = sarr[:,0]
    g = sarr[:, 1]
    plot_list.append([g, 'ground truth'])
    plot_list.append([p, 'prediction - no rain'])
    # 1.2 light rain
    print('Predicting nvidia2 light rain...')
    sa = GetPredictedSteeringAngles(log, model1_nvidia2, 'nvidia2', rt='light', st=0)
    sarr = np.asarray(sa);
    p = sarr[:,0]
    plot_list.append([p, 'prediction - light rain'])
    # 1.3 heavy rain, slant = +-10
    print('Predicting nvidia2 heavy rain...')
    sa = GetPredictedSteeringAngles(log, model1_nvidia2, 'nvidia2', rt='heavy', st=10)
    p = sarr[:,0]
    plot_list.append([p, 'prediction - heavy rain +-10'])
    # 1.4 torrential rain, slant = +-20
    print('Predicting nvidia2 torrential rain...')
    sa = GetPredictedSteeringAngles(log, model1_nvidia2, 'nvidia2', rt='torrential', st=20)
    sarr = np.asarray(sa);
    p = sarr[:,0]
    plot_list.append([p, 'prediction - torrential rain +-20'])
    print('Plotting...')
    plotMultipleSteeringAngles(plot_list, 25, True, "Generated_Track", "20201207192948_nvidia2.h5", title='SDSandbox log genTrackOneLap/logs_Wed_Nov_25_23_39_22_2020', w=18, h=4)
    """
    ############################################
    # 2. nvidia1 Generated track
    ############################################
    # 2.1 dry
    print('Predicting nvidia1 dry...')
    sa = GetPredictedSteeringAngles(log, model2_nvidia1, 'nvidia1', rt='', st=0)
    sarr = np.asarray(sa);
    p = sarr[:,0]
    g = sarr[:, 1]
    plot_list.append([g, 'ground truth'])
    plot_list.append([p, 'prediction - no rain'])
    # 2.2 light rain
    print('Predicting nvidia1 light rain...')
    sa = GetPredictedSteeringAngles(log, model2_nvidia1, 'nvidia1', rt='light', st=0)
    sarr = np.asarray(sa);
    p = sarr[:,0]
    plot_list.append([p, 'prediction - light rain'])
    # 2.3 heavy rain, slant = +-10
    print('Predicting nvidia1 heavy rain...')
    sa = GetPredictedSteeringAngles(log, model2_nvidia1, 'nvidia1', rt='heavy', st=10)
    sarr = np.asarray(sa);
    p = sarr[:,0]
    plot_list.append([p, 'prediction - heavy rain +-10'])
    # 2.4 torrential rain, slant = +-20
    print('Predicting nvidia1 torrential rain...')
    sa = GetPredictedSteeringAngles(log, model2_nvidia1, 'nvidia1', rt='torrential', st=20)
    sarr = np.asarray(sa);
    p = sarr[:,0]
    plot_list.append([p, 'prediction - torrential rain +-20'])
    print('Plotting...')
    plotMultipleSteeringAngles(plot_list, 25, True, "Generated_Track", "20201207091932_nvidia1.h5", title='SDSandbox log genTrackOneLap/logs_Wed_Nov_25_23_39_22_2020', w=18, h=4)

    ############################################
    # 3. nvidia_baseline Generated track
    ############################################
    # 3.1 dry
    """
    print('Predicting model3_nvidia_baseline dry...')
    sa = GetPredictedSteeringAngles(log, model3_nvidia_baseline, 'nvidia1', rt='', st=0)
    sarr = np.asarray(sa);
    p = sarr[:,0]
    g = sarr[:, 1]
    plot_list.append([g, 'ground truth'])
    plot_list.append([p, 'prediction - no rain'])
    # 3.2 light rain
    print('Predicting model3_nvidia_baseline light rain...')
    sa = GetPredictedSteeringAngles(log, model3_nvidia_baseline, 'nvidia1', rt='light', st=0)
    sarr = np.asarray(sa);
    p = sarr[:,0]
    plot_list.append([p, 'prediction - light rain'])
    # 3.3 heavy rain, slant = +-10
    print('Predicting model3_nvidia_baseline heavy rain...')
    sa = GetPredictedSteeringAngles(log, model3_nvidia_baseline, 'nvidia1', rt='heavy', st=10)
    sarr = np.asarray(sa);
    p = sarr[:,0]
    plot_list.append([p, 'prediction - heavy rain +-10'])
    # 3.4 torrential rain, slant = +-20
    print('Predicting model3_nvidia_baseline torrential rain...')
    sa = GetPredictedSteeringAngles(log, model3_nvidia_baseline, 'nvidia1', rt='torrential', st=20)
    sarr = np.asarray(sa);
    p = sarr[:,0]
    plot_list.append([p, 'prediction - torrential rain +-20'])
    print('Plotting...')
    plotMultipleSteeringAngles(plot_list, 25, True, "Generated_Track", "20201207201157_nvidia_baseline.h5", title='SDSandbox log genTrackOneLap/logs_Wed_Nov_25_23_39_22_2020', w=18, h=3)
    """

    """ 
    ############################################
    # 4. sanity Generated track
    ############################################
    # 4.1 dry
    print('Predicting model4_sanity dry...')
    sa = GetPredictedSteeringAngles(log, model4_sanity, 'nvidia1', rt='', st=0)
    sarr = np.asarray(sa);
    p = sarr[:,0]
    g = sarr[:, 1]
    plot_list.append([g, 'ground truth'])
    plot_list.append([p, 'prediction - no rain'])
    # 4.2 light rain
    print('Predicting model4_sanity light rain...')
    sa = GetPredictedSteeringAngles(log, model4_sanity, 'nvidia1', rt='light', st=0)
    sarr = np.asarray(sa);
    p = sarr[:,0]
    plot_list.append([p, 'prediction - light rain'])
    # 4.3 heavy rain, slant = +-10
    print('Predicting model4_sanity heavy rain...')
    sa = GetPredictedSteeringAngles(log, model4_sanity, 'nvidia1', rt='heavy', st=10)
    sarr = np.asarray(sa);
    p = sarr[:,0]
    plot_list.append([p, 'prediction - heavy rain +-10'])
    # 4.4 torrential rain, slant = +-20
    print('Predicting model4_sanity torrential rain...')
    sa = GetPredictedSteeringAngles(log, model4_sanity, 'nvidia1', rt='torrential', st=20)
    sarr = np.asarray(sa);
    p = sarr[:,0]
    plot_list.append([p, 'prediction - torrential rain +-20'])
    print('Plotting...')
    plotMultipleSteeringAngles(plot_list, 25, True, "Generated_Track", "20201120171015_sanity.h5", title='SDSandbox log genTrackOneLap/logs_Wed_Nov_25_23_39_22_2020', w=18, h=3)
    """
if __name__ == "__main__":
    # plot_hist("/home/simbox/git/sdsandbox/trained_models/nvidia1/20201107144927_nvidia1.history")
#if __name__ == "__main__":
#    parser = argparse.ArgumentParser(description='Plot Steering Utils')
#    parser.add_argument('--inputs', type=str, help='file path')

    # args = parser.parse_args()
    #svals = jsonSteeringBins('~/git/msc-data/unity/genRoad/*.jpg', 'genRoad')

    # PrintLatexRowModelGOS('../../dataset/unity/genTrack/genTrackOneLap/logs_Wed_Nov_25_23_39_22_2020/*.jpg', '../../trained_models/nvidia2/20201207192948_nvidia2.h5', 'nvidia2')
    # printGOSRows()
    # load models
    model1_nvidia2 = ''
    model2_nvidia1 = ''
    model3_nvidia_baseline = ''
    model4_sanity = ''
    # modelpath = '../../trained_models/sanity/20201120171015_sanity.h5'
    # modelpath = '../../trained_models/nvidia_baseline/20201207201157_nvidia_baseline.h5'
    #modelpath = '../../trained_models/nvidia1/20201207091932_nvidia1.h5'
    # modelpath = '../../trained_models/nvidia2/20201207192948_nvidia2.h5'
    # assume model is loaded and compiled
    # nvidia2
    modelpath = '../../trained_models/nvidia2/20201207192948_nvidia2.h5'
    model1_nvidia2 = load_model(modelpath)
    model1_nvidia2.compile("sgd", "mse")
    # nvidia1
    modelpath = '../../trained_models/nvidia1/20201207091932_nvidia1.h5'
    model2_nvidia1 = load_model(modelpath)
    model2_nvidia1.compile("sgd", "mse")
    # nvidia_baseline
    modelpath = '../../trained_models/nvidia_baseline/20201207201157_nvidia_baseline.h5'
    model3_nvidia_baseline = load_model(modelpath)
    model3_nvidia_baseline.compile("sgd", "mse")
    # sanity
    modelpath = '../../trained_models/sanity/20201120171015_sanity.h5'
    model4_sanity = load_model(modelpath)
    model4_sanity.compile("sgd", "mse")

    printMultiPlots(model1_nvidia2, model2_nvidia1, model3_nvidia_baseline, model4_sanity)

    #path = 'record_11640.json'
    #js = load_json(path)
    #print(js)
    # plotSteeringAngles(p, None, 25, True, "Generated Track", "20201120171015_sanity.h5", 'tcpflow log predicted')
    # plots1()
