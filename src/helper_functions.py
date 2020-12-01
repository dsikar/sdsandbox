"""
Boilerplate code
"""
import os

def hf_mkdir(fp):
    """
    Make directory
    Arguments
        fp: string, directory path
    Returns
        void
    """
    try:
        print("Creating directory:", fp)
        os.mkdir(fp)
    except:
        print("Directory exists:", fp)
        pass

def parse_bool(b):
    """
    Interpret string as boolean data type. "True" will evaluate to true,
    anything else will evaluate to false.
    Input
        b: string
    Output
        b: boolean
    """
    return b == "True"