# coding: utf-8

import numpy as np

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from collections import deque

import random
import json
import os
import io
import sys
import time

import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

class ReplayBuffer(object):
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done, goal):
        self.buffer.append((state, action, reward, next_state, done, goal))

    def sample(self, batch_size):
        state, action, reward, next_state, done, goal = zip(*random.sample(self.buffer, batch_size))
        return np.stack(state), action, reward, np.stack(next_state), done, np.stack(goal)

    def __len__(self):
        return len(self.buffer)

class Env(object):
    def __init__(self, num_bits):
        self.num_bits = num_bits

    def reset(self):
        self.done      = False
        self.num_steps = 0
        self.state     = np.random.randint(2, size=self.num_bits)
        self.target    = np.random.randint(2, size=self.num_bits)
        return self.state, self.target

    def step(self, action):
        if self.done:
            raise RESET

        self.state[action] = 1 - self.state[action]

        if self.num_steps > self.num_bits + 1:
            self.done = True
        self.num_steps += 1

        if np.sum(self.state == self.target) == self.num_bits:
            self.done = True
            return np.copy(self.state), 0, self.done, {}
        else:
            return np.copy(self.state), -1, self.done, {}

def make_model(num_actions, input_shape, hidden_size):
    model = Sequential()
    model.add(Dense(hidden_size, activation='relu', input_dim=input_shape))
    model.add(Dense(hidden_size, activation='relu'))
    model.add(Dense(num_actions, activation='linear'))

    loss = "mse"
    optimizer = Adam()
    model.compile(loss=loss, optimizer=optimizer)
    model.summary()
    return model

def update_weights(source, target):
    target.set_weights(source.get_weights()) 

def get_action(model, state, goal, epsilon=0.1):
    if random.random() < epsilon:
        return random.randrange(env.num_bits)
    inp = np.expand_dims(np.concatenate((state,goal)), axis=0)
    q_value = model.predict(inp)
    return np.argmax(q_value)

def gather(a, dim, index):
    expanded_index = [index if dim==i else np.arange(a.shape[i]).reshape([-1 if i==j else 1 for j in range(a.ndim)]) for i in range(a.ndim)]
    return a[expanded_index]

def compute_td_error(batch_size):
    if batch_size > len(replay_buffer):
        return None

    states, actions, rewards, next_states, dones, goals = replay_buffer.sample(batch_size)
    mask = np.array([[1 - x] for x in dones])
    actions = np.array([[x] for x in actions])
    rewards = np.array([[x] for x in rewards])

    inps1 = np.array([np.concatenate((states[i],goals[i])) for i in range(len(states))])
    q_values = model.predict(inps1)
    q_value = gather(q_values, 1, actions)

    inps2 = np.array([np.concatenate((next_states[i],goals[i])) for i in range(len(next_states))])
    next_q_values = target_model.predict(inps2)
    target_actions = np.array([[np.argmax(x)] for x in next_q_values])
    next_q_value = gather(next_q_values, 1, target_actions)

    expected_q_values = rewards + 0.99 * next_q_value * mask

    for n in range(0, batch_size):
        q_values[n][actions[n]] = expected_q_values[n]

    loss2 = np.mean(np.square((q_value - expected_q_values)))
    loss = model.train_on_batch(inps1, q_values)
    return loss, loss2




random.seed(1)

num_bits = 11
env = Env(num_bits)
model = make_model(num_bits, num_bits*2, 256)
target_model = make_model(num_bits, num_bits*2, 256)
update_weights(model, target_model)

batch_size = 5
new_goals = 5
max_frames = 200000
replay_buffer = ReplayBuffer(10000)

episodes = 0
iterations = 0
all_rewards = []
losses = []
losses2 = []

print("Starting training")

while iterations < max_frames:
    episodes += 1
    state, goal = env.reset()
    done = False
    episode = []
    total_reward = 0
    while not done:
        action = get_action(model, state, goal)
        next_state, reward, done, _ = env.step(action)
        episode.append((state, action, reward, done, next_state, goal))
        replay_buffer.push(state, action, reward, next_state, done, goal)
        state = next_state
        total_reward += reward
        iterations += 1
        if iterations % 10000 == 0:
            print("Iterations: " + str(iterations) + " Episodes: " + str(episodes))
            update_weights(model, target_model)

    all_rewards.append(total_reward)

    for state, action, reward, done, next_state, goal in episode:
        for t in np.random.choice(num_bits, new_goals):
            try:
                episode[t]
            except:
                continue
            new_goal = episode[t][-2]
            if np.sum(next_state == new_goal) == num_bits:
                reward = 0
            else:
                reward = -1
            replay_buffer.push(state, action, reward, next_state, done, new_goal)

    loss, loss2 = compute_td_error(batch_size)
    if loss is not None:
        losses.append(loss)
        losses2.append(loss2)
