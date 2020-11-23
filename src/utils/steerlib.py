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

if __name__ == "__main__":
    # plot_hist("/home/simbox/git/sdsandbox/trained_models/nvidia1/20201107144927_nvidia1.history")
#if __name__ == "__main__":
#    parser = argparse.ArgumentParser(description='Plot Steering Utils')
#    parser.add_argument('--inputs', type=str, help='file path')

    #args = parser.parse_args()
    #svals = jsonSteeringBins('~/git/msc-data/unity/genRoad/*.jpg', 'genRoad')
    path = 'record_11640.json'
    js = load_json(path)
    print(js)
