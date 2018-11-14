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

class DRQN(nn.Module):
    def __init__(self, state_len, rnn_size, num_layers, num_actions):
        super(DRQN, self).__init__()
        self.gru = nn.GRU(state_len, rnn_size, num_layers)
        self.out = nn.Linear(rnn_size, num_actions)
 
    def forward(self, x):
        x1, hidden  = self.gru(x)
        x_flat = x1.reshape(x1.size()[1:])
        x2 = self.out(x_flat)
        x3 = F.softmax(x2)
        return x3

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
    reward_weight = 0.9
    highscore = 0
    last_pred = ""
    loss = 0.0
    deterministic_move = 0.0
    batch_repeat = 1
    BATCH_SIZE = 64
    num_layers = 3
    replay_memory_capacity = 100
    min_replays = 50
    num_actions = g.num_actions
    input_shape = g.state.shape[0]
    print("State shape", input_shape)
    print("Action shape", num_actions)
    model = DRQN(input_shape, BATCH_SIZE, num_layers, num_actions)
    optimizer = optim.SGD(model.parameters(), weight_decay=1e-6, momentum=0.9, nesterov=True, lr=0.01)
    print(model)

    enriched_replays = deque()
    replay_hashes = deque()
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
        previous_states = []
        for n in range(0, BATCH_SIZE-1):
            previous_states.append(state)
        total_steps = 0
        while done == False:
            total_steps += 1
            state = torch.FloatTensor(g.state).unsqueeze(0)
            previous_states.append(state)
            use_predict = True
# Sometimes use random sampling later in an episode to encourage exploration after epsilon is at zero
            if random.random() < epsilon and total_steps > int(turns_average):
                use_predict = False
            if len(enriched_replays) < min_replays:
                use_predict = False
            if use_predict == True:
                action_type = "[    |    |pred]"
                with torch.no_grad():
                    pred = model(state.unsqueeze(0))
                last_pred = "[" + " ".join(["%.3f"%x for x in pred[0]]) + "]"
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
            episode_replay_buffer.append([state, action, reward, done, next_state])
            state = next_state
            reward_printable = ""
            actions_printable = ""
            loss = 0
            last_train_state = None
            last_train_outcome = ""
            buffer_selected = 0
            buffer_type = ""
            if len(enriched_replays) >= min_replays:
                if epsilon >= epsilon_min:
                    epsilon -= epsilon_degrade
                for r in range(batch_repeat):
                    minibatch = []
                    if len(episode_replay_buffer) > BATCH_SIZE:
                        buffer_type = "EPISODE"
                        minibatch = episode_replay_buffer[-BATCH_SIZE:]
                    else:
                        buffer_type = "REPLAY "
                        buffer_selected = random.randint(0, len(enriched_replays)-1)
                        minibatch = enriched_replays[buffer_selected]
                    if len(minibatch) == BATCH_SIZE:
                        st_t, a_t, r_t, d, st_t1 = zip(*minibatch)
                        outcome = r_t[-1]
                        last_train_outcome = "%.3f"%outcome
                        last_train_state = st_t[-1]
                        actions_printable = "".join([letters[l] for l in a_t])
                        reward_printable = " ".join(map(str, r_t))
                        inputs = torch.cat(st_t).unsqueeze(0)
                        batch_st_t1 = torch.cat(st_t1).unsqueeze(0)
                        targets = model(inputs)[-1]
                        next_targets = model(batch_st_t1)[-1]
                        Q_sa = torch.max(next_targets)
                        t1 = targets[a_t[-1]]
                        t2 = 0.0
                        if d[-1:] == True:
                            t2 = r_t[-1]
                        else:
                            t2 = r_t[-1] + gamma * Q_sa
                        loss = (t1 - t2).pow(2).mean()
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
            info += "\n   Batches in replay memory: " + str(len(enriched_replays))
            #info += "\n   r_t: [" + reward_printable + "]"
            info += "\n   a_t: [" + actions_printable + "]"
            if last_train_state is not None:
                info += "\n\n     " + buffer_type + ": " + last_train_outcome
                info += print_game_state(np.array(last_train_state[0]).reshape((game_dim, game_dim+1)).T.tolist())
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
            if len(enriched_replays) >= min_replays:
                if last_train_state is not None:
                    #sys.stderr.write("\x1b[2J\x1b[H")
                    os.system('clear')
                    print(info)
            else:
                sys.stdout.write("\r")
                sys.stdout.flush()
                sys.stdout.write("Iterations: " + str(iterations) + " Replay buffer: " + str(len(enriched_replays)))
                sys.stdout.flush()
            iterations += 1

        _, _, rewards, _, _ = zip(*episode_replay_buffer[-300:])
        nzri = [r for r,x in enumerate(rewards) if x>0.9 or x<-0.9]
        for n in nzri:
            if n > BATCH_SIZE:
                replay = episode_replay_buffer[1+n-BATCH_SIZE:n+1]
                h = hash(str(replay))
                if h not in replay_hashes:
                    replay_hashes.append(h)
                    if len(replay_hashes) > replay_memory_capacity:
                        replay_hashes.popleft()
                    enriched_replays.append(replay)
                    if len(enriched_replays) > replay_memory_capacity:
                        enriched_replays.popleft()

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
