import os
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Input, Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from argparse import ArgumentParser

from vgg import VGG

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

    print(len(frames))
    print(len(processed))
    predictions = model.predict([frames, processed])
    for i in predictions:
        print(i)
