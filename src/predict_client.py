'''
Predict Server
Create a server to accept image inputs and run them against a trained neural network.
This then sends the steering output back to the client.
Author: Tawn Kramer
'''
from __future__ import print_function
import os
import sys
import argparse
import time
import json
import base64
import datetime
from io import BytesIO
import signal
import tensorflow as tf
from tensorflow.python import keras
from tensorflow.python.keras.models import load_model
from PIL import Image
import numpy as np
from gym_donkeycar.core.fps import FPSTimer
from gym_donkeycar.core.message import IMesgHandler
from gym_donkeycar.core.sim_client import SimClient
# same preprocess as for training
from augmentation import augment, preprocess
import conf
import models
from helper_functions import parse_bool
import utils.RecordVideo as RecordVideo



if tf.__version__ == '1.13.1':
    from tensorflow import ConfigProto, Session

    # Override keras session to work around a bug in TF 1.13.1
    # Remove after we upgrade to TF 1.14 / TF 2.x.
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = Session(config=config)
    keras.backend.set_session(session)

# need to import file TODO
import Automold as am
import Helpers as hp
import numpy as np
# helper function for prediction

def add_rain(image_arr, rt=None, st=0):
    """
    Add rain to image
    Inputs:
        image_arr: numpy array containing image
        rt: string, rain type "heavy" or "torrential"
        st: range to draw a random slant from
    Output
        image_arr: numpy array containing image with rain
    """
    # print("Adding rain...")
    if(st != 0):
        # draw a random number for slant
        st = np.random.randint(-1 * st, st)

    if(rt!='light'): # heavy or torrential
        image_arr = am.add_rain_single(image_arr, rain_type=rt, slant=st)
    else:
         # no slant
        image_arr = am.add_rain_single(image_arr)

    return image_arr

class DonkeySimMsgHandler(IMesgHandler):

    STEERING = 0
    THROTTLE = 1

    def __init__(self, model, constant_throttle, image_cb=None, rand_seed=0):
        self.model = model
        self.constant_throttle = constant_throttle
        self.client = None
        self.timer = FPSTimer()
        self.img_arr = None
        self.image_cb = image_cb
        self.steering_angle = 0.
        self.throttle = 0.
        self.rand_seed = rand_seed
        self.fns = {'telemetry' : self.on_telemetry,\
                    'car_loaded' : self.on_car_created,\
                    'on_disconnect' : self.on_disconnect,
                    'aborted' : self.on_aborted}
        # images to record
        self.img_orig = None
        self.img_add_rain = None
        self.img_processed = None
        self.frame_count = 0
    def on_connect(self, client):
        self.client = client
        self.timer.reset()

    def on_aborted(self, msg):
        self.stop()

    def on_disconnect(self):
        pass

    def on_recv_message(self, message):
        self.timer.on_frame()
        if not 'msg_type' in message:
            print('expected msg_type field')
            print("message:", message)
            return

        msg_type = message['msg_type']
        if msg_type in self.fns:
            self.fns[msg_type](message)
        else:
            print('unknown message type', msg_type)

    def on_car_created(self, data):
        if self.rand_seed != 0:
            self.send_regen_road(0, self.rand_seed, 1.0)

    def on_telemetry(self, data):
        imgString = data["image"]
        image = Image.open(BytesIO(base64.b64decode(imgString)))
        img_arr = np.asarray(image, dtype=np.float32)
        self.frame_count += 1
        self.img_orig = img_arr

        # same preprocessing as for training
        img_arr = preprocess(img_arr)
        self.img_processed = img_arr

        #if(conf.record == True):
        #    text = (['Network Image', 'No Rain'])
        #    rv.add_image(img_arr, text)

        # check for rain
        if(conf.rt != ''):
            img_arr = add_rain(img_arr, conf.rt, conf.st)

        # if we are testing the network with rain
        self.img_arr = img_arr.reshape((1,) + img_arr.shape)

        if self.image_cb is not None:
            self.image_cb(img_arr, self.steering_angle )

    def update(self):
        if self.img_arr is not None:
            self.predict(self.img_arr)
            self.img_arr = None

    def predict(self, image_array):
        outputs = self.model.predict(image_array)
        # check if we are recording
        if (conf.record == True):

            # Add first image, with name of network and frame number
            text = (['20201120171015_sanity.h5', 'Intensity Multiplie: 4', 'Frame:  ' + str(self.frame_count)])
            rv.add_image(self.img_orig, text)

            # Add second image, preprocessed with rain or without
            text = (['Network image', 'No rain'])
            rv.add_image(self.img_processed, text)

            # add third image with prediction
            steering = outputs[0][0]
            steering *= conf.norm_const
            st_str = "{:.2f}".format(steering)
            st_str = "Predicted steering angle: " + st_str
            # st_str = "Predicted steering angle: 20"
            rtype = 'Type: ' + conf.rt
            s = 'Slant: -+' + str(conf.st)
            text = (['Network image added rain', rtype, s, st_str])
            rv.add_image(image_array[0], text)
            rv.add_frame()
        self.parse_outputs(outputs)


    def parse_outputs(self, outputs):
        res = []

        # Expects the model with final Dense(2) with steering and throttle
        for i in range(outputs.shape[1]):
            res.append(outputs[0][i])

        self.on_parsed_outputs(res)
        
    def on_parsed_outputs(self, outputs):
        self.outputs = outputs
        self.steering_angle = 0.0
        self.throttle = 0.2

        if len(outputs) > 0:        
            self.steering_angle = outputs[self.STEERING]

        if self.constant_throttle != 0.0:
            self.throttle = self.constant_throttle
        elif len(outputs) > 1:
            self.throttle = outputs[self.THROTTLE] * conf.throttle_out_scale

        self.send_control(self.steering_angle, self.throttle)

    def send_control(self, steer, throttle):
        # print("send st:", steer, "th:", throttle)
        msg = { 'msg_type' : 'control', 'steering': steer.__str__(), 'throttle':throttle.__str__(), 'brake': '0.0' }
        self.client.queue_message(msg)

    def send_regen_road(self, road_style=0, rand_seed=0, turn_increment=0.0):
        '''
        Regenerate the road, where available. For now only in level 0.
        In level 0 there are currently 5 road styles. This changes the texture on the road
        and also the road width.
        The rand_seed can be used to get some determinism in road generation.
        The turn_increment defaults to 1.0 internally. Provide a non zero positive float
        to affect the curviness of the road. Smaller numbers will provide more shallow curves.
        '''
        msg = { 'msg_type' : 'regen_road',
            'road_style': road_style.__str__(),
            'rand_seed': rand_seed.__str__(),
            'turn_increment': turn_increment.__str__() }
        
        self.client.queue_message(msg)

    def stop(self):
        self.client.stop()

    def __del__(self):
        self.stop()



def clients_connected(arr):
    for client in arr:
        if not client.is_connected():
            return False
    return True


def go(filename, address, constant_throttle=0, num_cars=1, image_cb=None, rand_seed=None):

    print("loading model", filename)
    model = load_model(filename)

    # In this mode, looks like we have to compile it
    model.compile("sgd", "mse")

    clients = []

    for _ in range(0, num_cars):
        # setup the clients
        handler = DonkeySimMsgHandler(model, constant_throttle, image_cb=image_cb, rand_seed=rand_seed)
        client = SimClient(address, handler)
        clients.append(client)

    while clients_connected(clients):
        try:
            time.sleep(0.02)
            for client in clients:
                client.msg_handler.update()
        except KeyboardInterrupt:
            # unless some hits Ctrl+C and then we get this interrupt
            print('stopping')
            break

def stop_exec(signum, frame):
    # restore the original signal handler as otherwise evil things will happen
    # in raw_input when CTRL+C is pressed, and our signal handler is not re-entrant
    signal.signal(signal.SIGINT, original_sigint)

    try:
        # changed raw_input to input
        if input("\nFinish recording video? (y/n)> ").lower().startswith('y'):
            print("*** CTRL+C to stop ***")
            rv.save_video()
            sys.exit(1)

    except KeyboardInterrupt:
        print("Ok ok, quitting")
        sys.exit(1)

    # restore the exit gracefully handler here
    signal.signal(signal.SIGINT, stop_exec)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='prediction server')
    parser.add_argument('--model', type=str, help='model filename')
    parser.add_argument('--host', type=str, default='127.0.0.1', help='server sim host')
    parser.add_argument('--port', type=int, default=9091, help='bind to port')
    parser.add_argument('--num_cars', type=int, default=1, help='how many cars to spawn')
    parser.add_argument('--constant_throttle', type=float, default=0.0, help='apply constant throttle')
    parser.add_argument('--rand_seed', type=int, default=0, help='set road generation random seed')
    parser.add_argument('--rain', type=str, default='', help='type of rain [light|heavy|torrential')
    parser.add_argument('--slant', type=int, default=0, help='Rain slant deviation')
    parser.add_argument('--record', type=parse_bool, default="False", help='Record video of raw and processed images')
    parser.add_argument('--img_cnt', type=int, default=3, help='Number of side by side images to record')


    args = parser.parse_args()
    address = (args.host, args.port)

    conf.rt = args.rain
    conf.st = args.slant
    conf.record = args.record

    if(conf.record == True):
        print("*** When finished, press CTRL+C and y to finish recording, the CTRL+C to quit ***")
        original_sigint = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, stop_exec)
        rv = RecordVideo.RecordVideo(args.model, "video", args.img_cnt)

    go(args.model, address, args.constant_throttle, num_cars=args.num_cars, rand_seed=args.rand_seed)
    # max value for slant is 20
    # Example
    # python3 predict_client.py --model=../trained_models/sanity/20201120171015_sanity.h5 --rain=light --slant=0