"""
Go though Unity3D generated files, delete .jpg if a corresponding .json file does not exist
$ python3 jsonclean.py --inputs=../../dataset/unity/log/*.jpg --delete=[False|True]
"""

import os
import argparse
import fnmatch
import json


def load_json(filename):
    with open(filename, "rt") as fp:
        data = json.load(fp)
    return data

def cleanjson(filemask, delete):
    # filemask = '~/git/sdsandbox/dataset/unity/log/*.jpg'
    filemask = os.path.expanduser(filemask)
    path, mask = os.path.split(filemask)

    matches = []
    for root, dirnames, filenames in os.walk(path):
        for filename in fnmatch.filter(filenames, mask):
            matches.append(os.path.join(root, filename))

    # deleted file count
    dc = 0
    for fullpath in matches:
            frame_number = os.path.basename(fullpath).split("_")[0]
            json_filename = os.path.join(os.path.dirname(fullpath), "record_" + frame_number + ".json")
            try:
                load_json(json_filename)
            except:
                print('No matching .json file for: ', fullpath)
                # No matching .json file for:  ../../dataset/unity/log/logs_Mon_Jul_13_09_03_21_2020/35095_cam-image_array_.jpg
                if(delete):
                    print("File deleted.")
                    os.remove(fullpath)
                    dc += 1
                continue

    print("Files deleted:", dc)
def parse_bool(b):
    return b == "True"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='JSON missing file handler/cleaner')
    parser.add_argument('--inputs', default='../dataset/unity/jungle1/log/*.jpg', help='input mask to gather images')
    parser.add_argument('--delete', type=parse_bool, default=False, help='image deletion flag')
    args = parser.parse_args()

    cleanjson(args.inputs, args.delete)