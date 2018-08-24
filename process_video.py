import cv2
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def read_data(name):
    data = []
    with open(name, 'r') as f:
        for line in f.readlines():
            index = line.find('\n')
            if index != -1:
                line = line[:index]
            val = float(line)
            data.append(val)
    return data

def get_frames(filename):
    vidcap = cv2.VideoCapture(filename)
    success = True
    data = []
    while success:
        success,image = vidcap.read()
        if success:
            data.append(image)
    return data

def subtract_frames(arr):
    data = []
    for i in range(1, len(arr)):
        data.append(arr[i] - arr[i-1])
    return np.array(data)

def preprocess(frame, factor):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (img.shape[1]//factor, img.shape[0]//factor))
    img = img.reshape((img.shape[0], img.shape[1], 1))
    return img

if __name__ == "__main__":
    print('Processing data')
    # speed_vals = read_data('data/train.txt')
    # frames = get_frames('data/train.mp4')
    # # sub_frames = subtract_frames(frames)
    # # sub_frames = [preprocess(img, 4) for img in sub_frames]
    # frames = [preprocess(img, 4) for img in frames]
    #
    #
    # # data = list(zip(frames[:len(frames)-1], sub_frames))
    # data = frames
    # train_X, test_X, train_y, test_y = train_test_split(data[:len(speed_vals)-1], speed_vals[:len(speed_vals)-1], test_size=.2, random_state=6)
    # np.save('cached/train_X_raw.npy', train_X)
    # np.save('cached/train_y_raw.npy', train_y)
    # np.save('cached/test_X_raw.npy', test_X)
    # np.save('cached/test_y_raw.npy', test_y)

    frames = get_frames('data/test.mp4')
    # sub_frames = subtract_frames(frames)
    frames = [preprocess(img, 4) for img in frames]

    np.save('cached/eval_raw.npy', frames)
