# coding: utf-8

from multiply import *
import numpy as np

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.models import model_from_json
from keras.optimizers import SGD, Adam
from keras.callbacks import TensorBoard
from collections import deque

import random
import json
import os
import io
import sys
import time

import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

def make_model(num_actions, input_shape, hidden_size):
    model = Sequential()
    model.add(Dense(hidden_size, activation='relu', input_dim=input_shape))
    model.add(Dense(hidden_size, activation='relu'))
    model.add(Dense(num_actions, activation='softmax'))

    loss = "categorical_crossentropy"
    optimizer = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss=loss, optimizer=optimizer)
    model.summary()
    return model

def train():
    max_val = 5
    num_vars = 2
    g = game(max_val, num_vars)
    gamma = 0.99
    epsilon = 1
    epsilon_degrade = 0.00001
    epsilon_min = 0.01
    num_actions = g.max_answer
    batch_size = 25
    replay_memory_capacity = 10000
    hidden_size = g.max_answer
    input_shape = g.state.shape[0]
    print("State shape", input_shape)
    print("Action shape", num_actions)
    model = make_model(num_actions, input_shape, hidden_size)
    accuracy = [0.0]


    total_steps = 0
    total_reward = 0
    episode_length = 100
    num_episodes = 10000
    correct_freq = 0.0
    num_epochs = 1
    epsilon = 1
    replay_memory = deque()
    for episode in range(num_episodes):
        action = None
        prediction = None
        done = False
        prediction_accuracy = 0.0
        num_predictions = 0
        correct_predictions = 0
        ep_acc = 0.0
        corrects = 0
        action_type = ""
        action = None
        pred_printable = ""
        state_printable = ""
        inputs_printable = ""
        reward_printable = ""
        for step in range(episode_length):
            state = g.reset()
            total_steps += 1
            state_printable = "[" + " ".join(map(str, state)) + "]"
            inputs_printable = "[" + " ".join(map(str, g.variables)) + "]"
            if random.random() > epsilon:
                action_type = "pred"
                inp_state = state.reshape((1, input_shape))
                pred = model.predict(inp_state)
                pred_printable = " ".join(["%.2f"%x for x in pred[0]])
                action = np.argmax(pred[0])
                action = action + 1
                num_predictions += 1
            else:
                action_type = "rand"
                if random.random() > correct_freq:
                    action = g.sample_random_action()[0]
                else:
                    action = g.correct_val
            if epsilon >= epsilon_min:
                epsilon = epsilon - epsilon_degrade
            reward = g.step(action)
            reward_printable = str(reward)
            total_reward += reward
            corrects += max(0, reward)
            if action_type == "pred":
                correct_predictions += max(0, reward)
                prediction_accuracy = float(float(correct_predictions)/float(num_predictions))*100.0
            ep_acc = float(float(corrects)/float(step+1)) * 100.0
            replay_memory.append([[state], action-1, reward])
            if len(replay_memory) > replay_memory_capacity:
                replay_memory.popleft()

            logs = None
            if len(replay_memory) > batch_size:
                minibatch = random.sample(replay_memory, batch_size)
                state_t, action_t, reward_t = zip(*minibatch)
                state_t = np.concatenate(state_t)
                targets = model.predict(state_t)
                targets[range(batch_size), action_t] = reward_t
                loss = model.train_on_batch(state_t, targets)
        info = ""
        info += "\n   Max val: " + str(max_val)
        info += "\n   Num vars: " + str(num_vars)
        info += "\n   Classes: " + str(num_actions)
        info += "\n"
        #info += "\n   State: [" + state_printable + "]"
        #info += "\n   Predictions: " + pred_printable
        info += "\n"
        info += "\n   Inputs: [" + inputs_printable + "]"
        info += "\n   [" + action_type + "] " + str(action) 
        info += "\n   [corr] " + str(g.correct_val)
        info += "\n   [reward] " + reward_printable
        info += "\n   "
        info += "\n   Iter: " + "%06d"%total_steps + " Episode: " + "%04d"%episode 
        info += "\n   Epsilon : " + "%.3f"%epsilon 
        info += "\n   Episode accuracy: " + "%.2f"%ep_acc 
        info += "\n   Prediction accuracy: " + "%.2f"%prediction_accuracy
        info += "\n   Prediction this episode: " + str(num_predictions)
        info += "\n"
        sys.stderr.write("\x1b[2J\x1b[H")
        print(info)
        accuracy.append(prediction_accuracy)
        with open("accuracy.json", "w") as f:
            json.dump(accuracy, f, indent=4)


if len(sys.argv) > 1:
    seed = sys.argv[1]
    print("Random seed: " + str(seed))
    random.seed(seed)
print("Starting training.")
train()
