import argparse
import fnmatch
import json
import os

def GetSteering(filename):
    """
    Get a tcpflow log and extract steering values received from and sent to sim
    Inputs
        filename: string, name of tcpflow log
    """
    # open file
    sa = []
    with open(filename) as f:
        readline = f.read()
        start = readline.find('{')
        jsonstr = readline[start:]
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
            sa.append([pred, act])
    # file should be automatically closed but will close for good measure
    f.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='train script')
    parser.add_argument('--filename', type=str, help='tcpflow log')
    args = parser.parse_args()
    GetSteering(args.filename)