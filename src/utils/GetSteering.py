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

def GetSteering(filename):
    """
    Get a tcpflow log and extract steering values received from and sent to sim
    Inputs
        filename: string, name of tcpflow log
    """
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
                    sa.append([float(pred), act])
                    pred = '' # need to save this image
                # deal with image later, sort out plot first
                    imgString = jsondict["image"]
                    image = Image.open(BytesIO(base64.b64decode(imgString)))
                    img_arr = np.asarray(image, dtype=np.float32)
                    img_arr_proc = preprocess(img_arr)
                    stitch = stitchImages(img_arr, img_arr_proc, 160, 120)
                    plt.imshow(stitch)
    except Exception as e:
        print("Exception raise: " + str(e))
    # file should be automatically closed but will close for good measure
    f.close()


def stitchImages(a, b, w, h):
    """
    Stitch two images together side by side
    Inputs
        a, b: floating point image arrays
        w, h: integer width and height dimensions
    Output
        c: floating point stitched image array
    """
    # https://stackoverflow.com/questions/30227466/combine-several-images-horizontally-with-python
    total_width = w * 2
    max_height = h

    a = Image.fromarray(a.astype('uint8'), 'RGB')
    b = Image.fromarray(b.astype('uint8'), 'RGB')

    new_im = Image.new('RGB', (total_width, max_height))
    new_im.paste(a, (0,0))
    new_im.paste(b, (w,0))

    return new_im # new_im



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='train script')
    parser.add_argument('--filename', type=str, help='tcpflow log')
    args = parser.parse_args()
    GetSteering(args.filename)