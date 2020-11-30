import argparse
import fnmatch
import json
import os
from io import BytesIO
from PIL import Image
import base64
import numpy as np
import matplotlib.pyplot as plt
from augmentation import preprocess
import cv2
import conf
# import debug.RecordVideo as RecordVideo

def MakeVideo(filename, model, preproc=False):
    """
    Make video from tcpflow logged images.
    video.avi is written to disk
    Inputs
        filename: string, name of tcpflow log
        model: name of model to stamp onto video
        preproc: boolean, show preprocessed image next to original
    Output
        none
    """
    # video name
    video_name = model + '.avi'
    VIDEO_WIDTH, VIDEO_HEIGHT = 800, 600
    IMAGE_WIDTH, IMAGE_HEIGHT = 800, 600
    if(preproc == True): # wide angle
        VIDEO_WIDTH = IMAGE_WIDTH*2
    video = cv2.VideoWriter(video_name, 0, 11, (VIDEO_WIDTH, VIDEO_HEIGHT)) # assumed 11fps
    # font
    font = cv2.FONT_HERSHEY_SIMPLEX

    # normalization constant
    # open file
    sa = []
    # initialize prediction
    pred = ''
    f = open(filename, "r")
    file = f.read()
    try:
        #readline = f.read()
        lines = file.splitlines()
        for line in lines:
            #print(line)
            start = line.find('{')
            if(start == -1):
                continue
            jsonstr = line[start:]
            #print(jsonstr)
            jsondict = json.loads(jsonstr)
            if "steering" in jsondict:
                # predicted
                pred = jsondict['steering']
            if "steering_angle" in jsondict:
                # actual
                act = jsondict['steering_angle']
                # save pair, only keep last pred in case two were send as it does happen i.e.:
                # 127.000.000.001.59460-127.000.000.001.09091: {"msg_type": "control", "steering": "-0.071960375", "throttle": "0.08249988406896591", "brake": "0.0"}
                # 127.000.000.001.59460-127.000.000.001.09091: {"msg_type": "control", "steering": "-0.079734944", "throttle": "0.08631626516580582", "brake": "0.0"}
                # 127.000.000.001.09091-127.000.000.001.59460: {"msg_type":"telemetry","steering_angle":-0.07196037,(...)
                if(len(pred) > 0):
                    # save steering angles
                    sa.append([float(pred), act])
                    pred = '' # need to save this image
                    # process image
                    imgString = jsondict["image"]
                    # decode string
                    image = Image.open(BytesIO(base64.b64decode(imgString)))
                    # try to convert to jpg
                    image = np.array(image)

                    # save
                    #image.save('frame.jpg')
                    # reopen with user-friendlier cv2
                    #image = cv2.imread('frame.jpg') # 120x160x3

                    image_copy = image
                    # resize so we can write some info onto image
                    image = cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT), cv2.INTER_AREA)
                    # add Info to frame
                    cv2.putText(image, model, (50, 50), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
                    # Predicted steering angle
                    pst = sa[len(sa)-1][0]
                    pst *= conf.norm_const
                    simst = "Predicted steering angle: {:.2f}".format(pst)
                    cv2.putText(image, simst, (50, 115), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
                    # create a preprocessed copy to compare what simulator generates to what network "sees"
                    if (preproc == True):  # wide angle
                        image2 = preprocess(image_copy)
                        image2 = cv2.resize(image2, (IMAGE_WIDTH, IMAGE_HEIGHT), cv2.INTER_AREA)
                        cv2.putText(image2, 'Network Image', (50, 50), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
                    # concatenate
                    if (preproc == True):  # wide angle
                        cimgs = np.concatenate((image, image2), axis=1)
                        image = cimgs
                    # model name
                    # model
                    video.write(image);
                    pred = ''
    except Exception as e:
        print("Exception raise: " + str(e))
    # file should be automatically closed but will close for good measure
    f.close()
    cv2.destroyAllWindows()
    video.release()


    return "DummyName.mp4"
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Make Video script')
    parser.add_argument('--filename', type=str, help='tcpflow log')
    parser.add_argument('--model', type=str, help='model name for video label')
    args = parser.parse_args()
    MakeVideo(args.filename, args.model, True)
    # example
    # python MakeVideo.py --filename=/tmp/tcpflow.log --model=20201120184912_sanity.h5
