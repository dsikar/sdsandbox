"""
Go though Unity3D generated files, get all steering angles and plot a histogram
$ python3 makebins.py --inputs=../../dataset/unity/log/*.jpg
"""

import os
import argparse
import fnmatch
import json
import seaborn as sns
import os


def load_json(filename):
    with open(filename, "rt") as fp:
        data = json.load(fp)
    return data

def cleanjson(filemask):

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

    print("Files deleted:", dc)
def parse_bool(b):
    return b == "True"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='JSON missing file handler/cleaner')
    parser.add_argument('--inputs', default='../dataset/unity/jungle1/log/*.jpg', help='input mask to gather images')
    args = parser.parse_args()

    cleanjson(args.inputs)