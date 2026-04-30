"""
Python 3.11
"""

import numpy as np

def ObtainImages(pathImages):
    with open(pathImages, 'rb') as file: # rb: Read bytes.
        _ = int.from_bytes(file.read(4), "big") # Magic number: Identification used for MNIST dataset. "big" is used to specify that the bytes should be interpreted as big-endian unsigned integer.
        numImages = int.from_bytes(file.read(4), "big") # In the bytes from 4 to 7 -> number of images.
        numRows = int.from_bytes(file.read(4), "big") # 28 pixels in each row.
        numColumns = int.from_bytes(file.read(4), "big") # 28 pixels in each column.
        images = file.read() # Read the rest of the bytes, which are the pixel values for all the images.
        images = np.frombuffer(images, dtype=np.uint8).reshape((numImages, numRows, numColumns)) # frumbuffer converts the bytes to a 1-dimensional. Then reshape it into a 3D array with shape (numImages, numRows, numColumns)
    return images

def ObtainLabels(pathLabels):
    with open(pathLabels, 'rb') as file:
        labels = file.read()[8:] # From 8 byte starts the label data.
        labels = np.frombuffer(labels, dtype=np.uint8)
    return labels