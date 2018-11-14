# coding: utf-8

from parachute import *
from dqn_model import *
import numpy as np

from collections import deque

import random
import json
import os
import io
import sys
import time

def print_game_state(state):
    printable = ""
    printable += "\n"
    last_row = len(state) -1
    for index, row in enumerate(state):
        if index != last_row:
            printable += "     |" + "|".join([" " if x==0 else "*" for x in row]) + "|"
        else:
            printable += "     |" + " ".join([" " if x==0 else "_" for x in row]) + "|"
        printable += "    " + " ".join(map(str, row)) + "\n"
    return printable

def train():
    g = game()

    epsilon_start = 1.0
    epsilon = epsilon_start
    epsilon_degrade = 0.00001
    epsilon_min = 0.0
    gamma = 0.99
    batch_size = 16
    replay_memory_capacity = 10000
    num_episodes = 10000000
    deterministic_move = 0.00
    num_actions = g.num_actions
    input_shape = g.state.shape[0]
    print("State shape", input_shape)
    print("Action shape", num_actions)
    model = make_model(num_actions, input_shape)

    total_steps = 0
    iterations = 0
    highscore = 0
    last_pred = ""
    loss = 0.0
    replay_memory = deque()
    replay_labels = deque()
    score_memory = deque()
    turns_memory = deque()
    moves = ["left", "stay", "right"]
    move_dist = {}
    move_history = {}
    for m in moves:
        move_dist[m] = 0
        move_history[m] = []
    score_average_history = []
    turns_average_history = []
    score_history = 50
    turns_history = 50
    score_average = 0.0
    turns_average = 0.0

    for episode in range(num_episodes):
        action = None
        done = False
        score = 0
        action_type = ""
        g.reset()
        total_steps = 0
        while done == False:
            total_steps += 1
            state = g.state
            if random.random() > epsilon:
                action_type = "[    |    |pred]"
                pred = model.predict(state.reshape((1, input_shape)))
                action = np.argmax(pred)
                last_pred = "[" + " ".join(["%.3f"%x for x in pred[0]]) + "] -> " + moves[action]
                move_dist[moves[action]] += 1
                replay_labels.append(1)
                for m in moves:
                    move_history[m].append(move_dist[m])
            else:
                if random.random() < deterministic_move:
                    action_type = "[    |proc|    ]"
                    action = g.move_ai()[0]
                else:
                    action_type = "[rand|    |    ]"
                    action = g.sample_random_action()[0]
                replay_labels.append(0)
            if len(replay_labels) > replay_memory_capacity:
                replay_labels.popleft()
            if epsilon >= epsilon_min:
                epsilon *= (1.0 - epsilon_degrade)
            next_state, reward, done, step_info = g.step(action)
            if done == False:
                score += reward
            replay_memory.append([[state], action, reward, done, [next_state]])
            state = next_state
            if len(replay_memory) > replay_memory_capacity:
                replay_memory.popleft()
            if len(replay_memory) > batch_size:
                minibatch = random.sample(replay_memory, batch_size)
                st_t, a_t, r_t, d, st_t1 = zip(*minibatch)
                inputs = np.concatenate(st_t)
                batch_st_t1 = np.concatenate(st_t1)
                targets = model.predict(inputs)
                Q_sa = model.predict(batch_st_t1)
                for index in range(0, batch_size):
                    if d[index] == True:
                        targets[index][a_t[index]] = r_t[index]
                    else:
                        targets[index][a_t[index]] = r_t[index] + gamma*np.max(Q_sa[index])
                loss = model.train_on_batch(inputs, targets)
            pred_replays = 100.0 - (float(sum(replay_labels))/float(replay_memory_capacity) * 100.0)
            info = print_game_state(g.game_space.T.tolist())
            info += "\n   Game " + str(episode) + " step: " + str(total_steps) 
            info += "\n   Score: " + str(score) 
            info += "\n   Action type: " + action_type
            info += "\n   Total iterations: " + str(iterations)
            info += "\n   Epsilon: " + "%.3f"%epsilon
            info += "\n   Items in replay memory: " + str(len(replay_memory))
            info += "\n   Exploration in replay memory: " + "%.2f"%pred_replays + "%"
            info += "\n   High score: " + str(highscore)
            info += "\n   Average score per episode: " + "%.3f"%score_average
            info += "\n   Average turns per episode: " + "%.3f"%turns_average
            info += "\n\n   Predictions"
            info += "\n   ==========="
            info += "\n   Loss: " + "%.3f"%loss
            info += "\n   " + last_pred
            info += "\n   [" + " ".join([m + ":" + str(move_dist[m]) for m in sorted(moves)]) + "]"

            sys.stderr.write("\x1b[2J\x1b[H")
            print(info)

            iterations += 1
        new_highscore = False
        score_memory.append(score)
        if len(score_memory) > score_history:
            score_memory.popleft()
        score_average = float(sum(score_memory))/float(score_history)
        score_average_history.append(score_average)
        turns_memory.append(total_steps)
        if len(turns_memory) > turns_history:
            turns_memory.popleft()
        turns_average = float(sum(turns_memory))/float(turns_history)
        turns_average_history.append(turns_average)
        if score > 0:
            if score > highscore:
                highscore = score
                new_highscore = True
                replay = g.replay
                with open("best_replay.json", "w") as f:
                    json.dump(replay, f)
        if episode % 25 == 0 or new_highscore == True:
            with open("score_history.json", "w") as f:
                json.dump(score_average_history, f, indent=4)
            with open("turns_history.json", "w") as f:
                json.dump(turns_average_history, f, indent=4)
            with open("move_history.json", "w") as f:
                json.dump(move_history, f, indent=4)


if len(sys.argv) > 1:
    seed = sys.argv[1]
    print("Random seed: " + str(seed))
    random.seed(seed)
print("Starting training.")
train()
