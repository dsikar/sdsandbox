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