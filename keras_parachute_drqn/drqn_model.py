# coding: utf-8

import numpy as np

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, GRU, TimeDistributed
from keras.optimizers import SGD, Adam

import random
import os
import time

import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

def make_model(num_actions, input_shape, time_steps=32):
    model = Sequential()
    #model.add(TimeDistributed(Dense(time_steps*2), input_shape=((time_steps, input_shape))))
    model.add(GRU(time_steps, return_sequences=True, input_shape=((time_steps, input_shape))))
    model.add(GRU(time_steps, return_sequences=True))
    model.add(GRU(time_steps))
    model.add(Dense(num_actions, activation='softmax'))

    loss = "mean_squared_error"
    optimizer = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss=loss, optimizer=optimizer)
    model.summary()
    return model
