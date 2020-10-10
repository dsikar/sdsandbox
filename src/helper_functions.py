"""
Additonal functions to help with different file formats
"""
import csv

def getdict(filepath):
    """
    Create a dictionary from file.
    The expected input format is:

    frame_id,steering_angle
    1479425441182877835,-0.373665106110275
    1479425441232704425,-0.0653962884098291
    (...)

    Where the first column will be the key as string datatype
    and the second column will be the key converted to float.

    # Arguments
        filepath: string, path to file

    # Returns
        dictionary

    # Usage
    filepath = '../dataset/udacity/Ch2_001/final_example.csv'
    mydict = getdict(filepath)
    mydict.get('1479425441182877835')
    >> -0.373665106110275
    """
    file = open(filepath, 'r')
    reader = csv.reader(file)
    # skip first line headers
    next(reader, None)
    mydict = {rows[0]: float(rows[1]) for rows in reader}
    return mydict

# make dict available
filepath = '../dataset/udacity/Ch2_001/final_example.csv'
mydict = getdict(filepath)