import os
import numpy as np
import pandas as pd
import pdb
from keras.models import Sequential
from keras.layers import Input, Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from argparse import ArgumentParser

from vgg import VGG

def write_txt(name, arr):
    with open(name, 'w') as f:
        for i in arr:
            f.write(str(i) + '\n')

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-r', dest='data_raw')
    parser.add_argument('-p', dest='data_processed')
    parser.add_argument('-m', dest='model')

    args = parser.parse_args()

    frames = np.load(args.data_raw)
    processed = np.load(args.data_processed)

    INPUT_SHAPE = frames[0].shape

    model = VGG(INPUT_SHAPE,
                1,
                1,
                args.model,
                True)

    predictions = []
    for i in range(len(frames)):
        pred = model.predict([processed[i].reshape(1, processed[i].shape[0], processed[i].shape[1], processed[i].shape[2]), frames[i].reshape(1, frames[i].shape[0], frames[i].shape[1], frames[i].shape[2])])
        predictions.append(pred[0][0])
    write_txt('deliverable.txt', predictions)
