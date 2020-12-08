# Helper library to create videos and plots

import fnmatch
import json
import os
import numpy as np
import argparse
import matplotlib.pyplot as plt
import statistics
import seaborn as sns
import pickle

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

def plotMultipleSteeringAngles(p, n=1, save=False, track= "Track Name", mname="model name", title='title', w, h):
    """
    Plot predicted and (TODO) optionally ground truth steering angles
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

    plt.rcParams["figure.figsize"] = (18,3)

    plt.plot(p*n, label="predicted")
    try:
        if (g is not None):
            plt.plot(g*n, label="ground truth")
    except Exception as e:
        print("problems plotting: " + str(e))

    plt.ylabel('Steering angle')
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

#sa = GetSteeringFromtcpflow('../../dataset/unity/genRoad/tcpflow/20201120184912_sanity.log')
#sarr = np.asarray(sa)
#p = sarr[:,0]
#g = sarr[:,1]



if __name__ == "__main__":
    # plot_hist("/home/simbox/git/sdsandbox/trained_models/nvidia1/20201107144927_nvidia1.history")
#if __name__ == "__main__":
#    parser = argparse.ArgumentParser(description='Plot Steering Utils')
#    parser.add_argument('--inputs', type=str, help='file path')

    #args = parser.parse_args()
    #svals = jsonSteeringBins('~/git/msc-data/unity/genRoad/*.jpg', 'genRoad')
    p = []
    sa = GetSteeringFromtcpflow('../trained_models/nvidia1/tcpflow/20201207091932_nvidia1_no_rain_tcpflow.log')
    sarr = np.asarray(sa)
    pa = sarr[:,0]
    p.append([pa, 'no rain'])

    path = 'record_11640.json'
    js = load_json(path)
    print(js)
    # plotSteeringAngles(p, None, 25, True, "Generated Track", "20201120171015_sanity.h5", 'tcpflow log predicted')
