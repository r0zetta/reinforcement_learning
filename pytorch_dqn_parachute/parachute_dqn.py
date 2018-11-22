# coding: utf-8

from parachute import *
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from collections import deque

import random
import json
import os
import io
import sys
import time

class DQN(nn.Module):
    def __init__(self, state_len, num_actions, hidden_size=40):
        super(DQN, self).__init__()
        self.lin1 = nn.Linear(state_len, hidden_size)
        self.lin2 = nn.Linear(hidden_size, hidden_size*2)
        self.lin3 = nn.Linear(hidden_size*2, hidden_size)
        self.drop = nn.Dropout(0.5)
        self.out = nn.Linear(hidden_size, num_actions)
 
    def forward(self, x):
        x1 = F.relu(self.lin1(x))
        x2 = F.relu(self.lin2(x1))
        x3 = F.relu(self.lin3(x2))
        x4 = self.drop(x3)
        x5 = self.out(x4)
        x6 = F.softmax(x5)
        return x6

def print_game_state(state):
    printable = ""
    printable += "\n"
    last_row = len(state) -1
    for index, row in enumerate(state):
        if index != last_row:
            printable += "     |" + "|".join([" " if x==0 else "*" for x in row]) + "|"
        else:
            printable += "     |" + " ".join([" " if x==0 else "_" for x in row]) + "|"
        printable += "      "  + " ".join([str(int(x)) for x in row]) + "\n"
    return printable

def train():
    score_memory = deque()
    turns_memory = deque()
    outcomes = ["PASS", "FAIL"]
    moves = ["left", "stay", "right"]
    letters = ["l", "s", "r"]
    move_dist = {}
    move_history = {}
    last_actions = deque()
    for m in moves:
        move_dist[m] = 0
        move_history[m] = []
    score_average_history = []
    turns_average_history = []
    score_history = 50
    turns_history = 50
    score_average = 0.0
    score_average_snapshot = 0.0
    turns_average = 0.0

    game_dim = 5
    g = game(game_dim)
    gamma = 0.99
    num_episodes = 10000000
    epsilon_start = 1.0
    epsilon = epsilon_start
    epsilon_degrade = 0.0001
    epsilon_min = 0.0
    total_steps = 0
    iterations = 0
    highscore = 0
    last_pred = ""
    deterministic_move = 0.0
    BATCH_SIZE = 64
    replay_buffer = deque()
    replay_memory_capacity = 10000
    observe_steps = 9000
    num_actions = g.num_actions
    input_shape = g.state.shape[0]
    hidden_size = 40
    print("State shape", input_shape)
    print("Action shape", num_actions)
    model = DQN(input_shape, num_actions, hidden_size)
    optimizer = optim.SGD(model.parameters(), weight_decay=1e-6, momentum=0.9, nesterov=True, lr=0.01)
    print(model)

    replay_memory = deque()
    print("Populating replay buffer...")
    for episode in range(num_episodes):
        episode_replay_buffer = []
        batch_num = 0
        action = None
        done = False
        score = 0
        action_type = ""
        g.reset()
        state = torch.FloatTensor(g.state).unsqueeze(0)
        total_steps = 0
        while done == False:
            total_steps += 1
            state = torch.FloatTensor(g.state).unsqueeze(0)
            if random.random() > epsilon:
                action_type = "[    |    |pred]"
                with torch.no_grad():
                    pred = model(state.unsqueeze(0))
                last_pred = "[" + " ".join(["%.3f"%x for x in pred[0][0]]) + "]"
                action = np.argmax(pred)
                last_pred += " -> " + moves[action]
                move_dist[moves[action]] += 1
                for l in moves:
                    move_history[l].append(move_dist[l])
                last_actions.append(action)
                if len(last_actions) > BATCH_SIZE:
                    last_actions.popleft()
            else:
                if random.random() < deterministic_move:
                    action_type = "[    |proc|    ]"
                    action = g.move_ai()[0]
                else:
                    action_type = "[rand|    |    ]"
                    action = g.sample_random_action()[0]
            next_state, reward, done, info = g.step(action)
            next_state = torch.FloatTensor(next_state).unsqueeze(0)
            if done == False:
                score += reward
            replay_buffer.append([state, action, reward, done, next_state])
            if len(replay_buffer) > replay_memory_capacity:
                replay_buffer.popleft()
            state = next_state
            reward_printable = ""
            actions_printable = ""
            loss = 0.0
            if iterations > observe_steps:
                if len(replay_buffer) > BATCH_SIZE:
                    if epsilon >= epsilon_min:
                        epsilon -= epsilon_degrade
                    minibatch = random.sample(replay_buffer, BATCH_SIZE)
                    state_t, action_t, reward_t, done_t, state_t1 = zip(*minibatch)
                    actions_printable = "".join([letters[l] for l in action_t])
                    reward_printable = " ".join(map(str, reward_t))
                    state_t = torch.cat(state_t)
                    state_t1 = torch.cat(state_t1)
                    action_batch = torch.LongTensor(action_t).unsqueeze(1)
                    reward_batch = torch.FloatTensor(reward_t).unsqueeze(1)
                    done_mask = [1-int(x) for x in done_t]
                    targets = model(state_t).gather(1, action_batch)
                    Q_sa = model(state_t1).gather(1, action_batch)
                    new_targets = torch.Tensor([reward_batch[n] + done_mask[n] * (gamma * torch.max(Q_sa[n])) for n, m in enumerate(targets)])
                    loss = (targets - new_targets).pow(2).mean()
                    optimizer.zero_grad()
                    loss.backward()
                    for param in model.parameters():
                        param.grad.data.clamp_(-1, 1)
                    optimizer.step()

            last_actions_printable = "".join([letters[l] for l in last_actions])
            info = print_game_state(g.game_space.T.tolist())
            info += "\n   Episode: " + str(episode) + " step: " + "%04d"%total_steps + " Score: " + "%.3f"%score
            info += "\n   Action type: " + action_type
            info += "\n   Epsilon: " + "%.3f"%epsilon + " Total iterations: " + str(iterations)
            info += "\n   Steps in replay memory: " + str(len(replay_buffer))
            #info += "\n   r_t: [" + reward_printable + "]"
            info += "\n   a_t: [" + actions_printable + "]"
            info += "\n   High score: " + "%.3f"%highscore
            info += "\n   Average score per episode: " + "%.3f"%score_average
            info += "\n   Previous average: " + "%.3f"%score_average_snapshot
            info += "\n   Average turns per episode: " + "%.3f"%turns_average
            info += "\n\n   Predictions"
            info += "\n   ==========="
            info += "\n   Loss: " + "%.3f"%loss
            info += "\n   " + last_pred
            info += "\n   [" + last_actions_printable + "]"
            info += "\n   [" + " ".join([m + ":" + str(move_dist[m]) for m in sorted(moves)]) + "]"
            if iterations > observe_steps:
                os.system('clear')
                print(info)
            iterations += 1

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
        if score > highscore:
            highscore = score
            replay = g.replay
            with open("best_replay.json", "w") as f:
                json.dump(replay, f)
        if episode % 25 == 0:
            with open("score_history.json", "w") as f:
                json.dump(score_average_history, f, indent=4)
            with open("turns_history.json", "w") as f:
                json.dump(turns_average_history, f, indent=4)
            with open("move_history.json", "w") as f:
                json.dump(move_history, f, indent=4)
# Adaptive exploration: if score has trended down, add a small amount to epsilon
            if score_average < score_average_snapshot:
                if epsilon < 0.50:
                    epsilon = 0.50
            score_average_snapshot = score_average



if len(sys.argv) > 1:
    seed = sys.argv[1]
    print("Random seed: " + str(seed))
    random.seed(seed)
print("Starting training.")
train()
