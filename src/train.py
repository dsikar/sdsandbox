'''
Train
Train your nerual network
Author: Tawn Kramer
'''
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

'''
matplotlib can be a pain to setup. So handle the case where it is absent. When present,
use it to generate a plot of training results.
'''
try:
    import matplotlib
    # Force matplotlib to not use any Xwindows backend.
    matplotlib.use('Agg')

    import matplotlib.pyplot as plt
    do_plot = True
except:
    do_plot = False


def shuffle(samples):
    '''
    randomly mix a list and return a new list
    '''
    ret_arr = []
    len_samples = len(samples)
    while len_samples > 0:
        iSample = random.randrange(0, len_samples)
        ret_arr.append(samples[iSample])
        del samples[iSample]
        len_samples -= 1
    return ret_arr

def load_json(filename):
    with open(filename, "rt") as fp:
        data = json.load(fp)
    return data

def generator(samples, batch_size=64,):
    '''
    Rather than keep all data in memory, we will make a function that keeps
    it's state and returns just the latest batch required via the yield command.
    
    As we load images, we can optionally augment them in some manner that doesn't
    change their underlying meaning or features. This is a combination of
    brightness, contrast, sharpness, and color PIL image filters applied with random
    settings. Optionally a shadow image may be overlayed with some random rotation and
    opacity.
    We flip each image horizontally and supply it as a another sample with the steering
    negated.
    '''
    num_samples = len(samples)
    
    while 1: # Loop forever so the generator never terminates
        samples = shuffle(samples)
        #divide batch_size in half, because we double each output by flipping image.
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            
            images = []
            controls = []
            for fullpath in batch_samples:
                try:
                
                    frame_number = os.path.basename(fullpath).split("_")[0]
                    json_filename = os.path.join(os.path.dirname(fullpath), "record_" + frame_number + ".json")
                    data = load_json(json_filename)
                    steering = float(data["user/angle"])
                    throttle = float(data["user/throttle"])
                
                    try:
                        image = Image.open(fullpath)
                    except:
                        print('failed to open', fullpath)
                        continue

                    #PIL Image as a numpy array
                    image = np.array(image, dtype=np.float32)

                    images.append(image)
                    
                    if conf.num_outputs == 2:
                        controls.append([steering, throttle])
                    elif conf.num_outputs == 1:
                        controls.append([steering])
                    else:
                        print("expected 1 or 2 outputs")

                except Exception as e:
                    print(e)
                    print("we threw an exception on:", fullpath)
                    yield [], []


            # final np array to submit to training
            X_train = np.array(images)
            y_train = np.array(controls)
            yield X_train, y_train


def get_files(filemask):
    '''
    use a filemask and search a path recursively for matches
    '''
    #matches = glob.glob(os.path.expanduser(filemask))
    #return matches
    filemask = os.path.expanduser(filemask)
    path, mask = os.path.split(filemask)
    
    matches = []
    for root, dirnames, filenames in os.walk(path):
        for filename in fnmatch.filter(filenames, mask):
            matches.append(os.path.join(root, filename))
    return matches


def train_test_split(lines, test_perc):
    '''
    split a list into two parts, percentage of test used to seperate
    '''
    train = []
    test = []

    for line in lines:
        if random.uniform(0.0, 1.0) < test_perc:
            test.append(line)
        else:
            train.append(line)

    return train, test

def make_generators(inputs, limit=None, batch_size=64):
    '''
    load the job spec from the csv and create some generator for training
    '''
    
    #get the image/steering pairs from the csv files
    lines = get_files(inputs)
    print("found %d files" % len(lines))

    if limit is not None:
        lines = lines[:limit]
        print("limiting to %d files" % len(lines))
    
    train_samples, validation_samples = train_test_split(lines, test_perc=0.2)

    print("num train/val", len(train_samples), len(validation_samples))
    
    # compile and train the model using the generator function
    train_generator = generator(train_samples, batch_size=batch_size)
    validation_generator = generator(validation_samples, batch_size=batch_size)
    
    n_train = len(train_samples)
    n_val = len(validation_samples)
    
    return train_generator, validation_generator, n_train, n_val


def go(model_name, outdir, epochs=50, inputs='./log/*.jpg', limit=None):

    print('working on model', model_name)

    hf_mkdir(outdir)
    outdir += '/' + model_name
    hf_mkdir(outdir)

    # https://docs.python.org/3/library/datetime.html#strftime-strptime-behavior
    dt = strftime("%Y%m%d%H%M%S")
    fp = outdir + '/' + dt + '_' + model_name;
    model_name = fp + '.h5'

    '''
    modify config.json to select the model to train.
    '''
    # model = models.get_nvidia_model_naoki(conf.num_outputs)
    model = models.get_nvidia_model(conf.num_outputs)

    callbacks = [
        # running with naoki's model
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=conf.training_patience, verbose=0),
        keras.callbacks.ModelCheckpoint(model_name, monitor='val_loss', save_best_only=True, verbose=0),
        # keras.callbacks.ModelCheckpoint(('model-{epoch:03d}' +'_' + model_name), monitor='val_loss', save_best_only=True, verbose=0),
    ]
    
    batch_size = conf.training_batch_size


    #Train on session images
    train_generator, validation_generator, n_train, n_val = make_generators(inputs, limit=limit, batch_size=batch_size)

    if n_train == 0:
        print('no training data found')
        return

    steps_per_epoch = n_train // batch_size
    validation_steps = n_val // batch_size

    print("steps_per_epoch", steps_per_epoch, "validation_steps", validation_steps)
    s1 = strftime("%Y%m%d%H%M%S")
    history = model.fit_generator(train_generator, 
        steps_per_epoch = steps_per_epoch,
        validation_data = validation_generator,
        validation_steps = validation_steps,
        epochs=epochs,
        verbose=1,
        callbacks=callbacks)
    s2 = strftime("%Y%m%d%H%M%S")
    FMT = "%Y%m%d%H%M%S"
    tdelta = datetime.strptime(s2, FMT) - datetime.strptime(s1, FMT)
    tdelta = "Total training time: " + str(tdelta)

    # save info
    log = fp + '.log'
    logfile = open(log, 'w')
    logfile.write("Model name: " + model_name + "\r\n")
    logfile.write(tdelta)
    logfile.write('\r\n');
    logfile.write("Training loss: " + '{0:.3f}'.format(history.history['loss'][-1]) + "\r\n")
    logfile.write("Validation loss: " + '{0:.3f}'.format(history.history['val_loss'][-1]) + "\r\n")
    logfile.write("Training accuracy: " + '{0:.3f}'.format(history.history['acc'][-1]) + "\r\n")
    logfile.write("Validation accuracy: " + '{0:.3f}'.format(history.history['val_acc'][-1]) + "\r\n")
    logfile.close()


    # save history
    histfile = fp + '.history'
    with open(histfile, 'wb') as file_pi:
        pickle.dump(history.history, file_pi)    
    try:
        if do_plot:
            fig = plt.figure()
            sp = '(l,vl,a,va)' + '{0:.3f}'.format(history.history['loss'][-1]) \
                 + ',' + '{0:.3f}'.format(history.history['val_loss'][-1]) \
                 + ',' + '{0:.3f}'.format(history.history['acc'][-1]) \
                 + ',' + '{0:.3f}'.format(history.history['val_acc'][-1]) \
                 + ' - ' + model_name.split('\\')[-1]
            fig.suptitle(sp, fontsize=8)
            ax = fig.add_subplot(111)
            #ax.plot(time, Swdown, '-', label='Swdown')
            ax.plot(history.history['loss'], '-', label='Training Loss')
            ax.plot(history.history['val_loss'], '-', label='Validation Loss')
            ax2 = ax.twinx()
            ax2.plot(history.history['acc'], '-', label='Training Accuracy')
            ax2.plot(history.history['val_acc'], '-', label='Validation Accuracy')
            ax.legend(loc=2) # https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.legend.html
            ax.grid()
            ax.set_xlabel("Epoch")
            ax.set_ylabel(r"Loss")
            ax2.set_ylabel(r"Accurary")
            ax.set_ylim(0, 0.2)
            ax2.set_ylim(0.5, 1)
            ax2.legend(loc=1)
            aimg = fp + '_accuracy.png'
            plt.savefig(aimg)
    except:
        print("problems with loss graph")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='train script')
    parser.add_argument('--model', type=str, help='model name')
    parser.add_argument('--outdir', type=str, help='output directory')
    parser.add_argument('--epochs', type=int, default=conf.training_default_epochs, help='number of epochs')
    parser.add_argument('--inputs', default='../dataset/unity/log_sample/*.jpg', help='input mask to gather images')
    parser.add_argument('--limit', type=int, default=None, help='max number of images to train with')
    args = parser.parse_args()
    #print(tf.__version__) 2.2.0
    go(args.model, args.outdir, epochs=args.epochs, limit=args.limit, inputs=args.inputs)

#python train.py ..\outputs\mymodel_aug_90_x4_e200 --epochs=200
