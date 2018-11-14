# coding: utf-8

import numpy as np

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD, Adam

import random
import os
import time

import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

def make_model(num_actions, input_shape, hidden_size=40):
    model = Sequential()
    model.add(Dense(hidden_size, activation='relu', input_dim=input_shape))
    model.add(Dense(hidden_size*2, activation='relu'))
    model.add(Dense(hidden_size, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_actions, activation='softmax'))

    loss = "mean_squared_error"
    optimizer = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss=loss, optimizer=optimizer)
    model.summary()
    return model
