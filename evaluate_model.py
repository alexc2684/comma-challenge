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
    parser.add_argument('-t', dest='testset')
    parser.add_argument('-m', dest='model')

    args = parser.parse_args()

    frames = np.load(args.testset)
    INPUT_SHAPE = frames[0].shape
    model = VGG(INPUT_SHAPE,
                1,
                1,
                args.model,
                False)

    predictions = model.predict(frames)
    for i in predictions:
        print(i)
