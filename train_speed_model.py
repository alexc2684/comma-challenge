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
    parser.add_argument('-e', dest='epochs')
    parser.add_argument('-b', dest='batch_size')
    parser.add_argument('-f', dest='model_name')
    parser.add_argument('-d', dest='dual')
    args = parser.parse_args()

    train_X = np.load('cached/train_X.npy')
    train_y = np.load('cached/train_y.npy')
    test_X = np.load('cached/test_X.npy')
    test_y = np.load('cached/test_y.npy')

    INPUT_SHAPE = train_X[0].shape

    if args.dual:
        raw_train = np.load('cached/train_X_raw.npy')
        raw_test = np.load('cached/test_X_raw.npy')
        train_X = [train_X, raw_train]
        test_X = [test_X, raw_test]

    dual = args.dual == "True"
    model = VGG(INPUT_SHAPE,
                args.epochs,
                args.batch_size,
                args.model_name,
                dual)

    print("Training model")
    model.train(train_X, train_y, test_X, test_y)
    print("Finished training")
