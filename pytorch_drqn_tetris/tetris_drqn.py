# coding: utf-8

from tetris import *
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
 
    def forward(self, state, goal):
        x = torch.cat((state, goal), dim=2)
        x1, hidden  = self.gru(x)
        x_flat = x1.reshape(x1.size()[1:])
        x2 = self.out(x_flat)
        x3 = F.softmax(x2)
        return x3

def copy_weights(source, target):
    target.load_state_dict(source.state_dict())

def train():
    g = game()
    goal_state = torch.FloatTensor(np.ravel(g.goal_state)).unsqueeze(0)
    state = torch.FloatTensor(np.ravel(g.state)).unsqueeze(0)
    inp_example = torch.cat((goal_state.unsqueeze(0), state.unsqueeze(0)), dim=2)
    print(inp_example.size())

    score_memory = deque()
    turns_memory = deque()
    moves = g.moves
    letters = g.letters
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

    epsilon_start = 1.0
    epsilon_min = 0.0
    use_her = False
    use_curiosity = False
    if use_her == True or use_curiosity == True:
        epsilon_start = 0.1
        epsilon_min = 0.1
    epsilon = epsilon_start
    gamma = 0.99
    num_episodes = 10000000
    epsilon_degrade = 0.00001
    total_steps = 0
    iterations = 0
    highscore = 0
    last_pred = ""
    loss = 0.0
    deterministic_move = 0
    batch_repeat = 1
    printable_length = 30
    replay_memory_capacity = 10000
    min_replays = 1
    rnn_size = 100
    num_layers = 3
    num_actions = g.num_actions
    state = np.ravel(g.state)
    input_shape = state.shape[0] * 2
    print("State shape", input_shape)
    print("Action shape", num_actions)
    model = DRQN(input_shape, rnn_size, num_layers, num_actions)
    reference = DRQN(input_shape, rnn_size, num_layers, num_actions)
    copy_weights(model, reference)
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
        curiosity_score = 0.0
        action_type = ""
        g.reset()
        previous_states = []
        state = torch.FloatTensor(np.ravel(g.state)).unsqueeze(0)
        total_steps = 0
        last_actions = []
        while done == False:
            total_steps += 1
            state = torch.FloatTensor(np.ravel(g.state)).unsqueeze(0)
            previous_states.append(state)
            use_predict = True
            curiosity_reward = 0.0
            if random.random() < epsilon and total_steps > turns_average:
                use_predict = False
            if len(enriched_replays) < min_replays:
                use_predict = False
            if use_predict == True:
                action_type = "[    |    |pred]"
                with torch.no_grad():
                    pred = model(state.unsqueeze(0), goal_state.unsqueeze(0))
                    if use_curiosity == True:
                        curiosity_pred = reference(state.unsqueeze(0), goal_state.unsqueeze(0))
                        curiosity_reward = torch.abs((curiosity_pred - pred).pow(2).mean())
                last_pred = "[" + " ".join(["%.3f"%x for x in pred[0]]) + "]"
                action = np.argmax(pred)
                last_pred += " -> " + moves[action]
                move_dist[moves[action]] += 1
                for l in moves:
                    move_history[l].append(move_dist[l])
                last_actions.append(action)
            else:
                if random.random() < deterministic_move:
                    action_type = "[    |proc|    ]"
                    action = g.move_ai()
                else:
                    action_type = "[rand|    |    ]"
                    action = g.sample_random_action()
            next_state, reward, done, step_info = g.step(action)
            if done == False:
                score += reward
            next_state = torch.FloatTensor(np.ravel(next_state)).unsqueeze(0)
            if use_predict == True and use_curiosity == True:
                reward = curiosity_reward
                curiosity_score += float(curiosity_reward)
            episode_replay_buffer.append([state, action, reward, done, next_state, goal_state])
            state = next_state
            reward_printable = ""
            actions_printable = ""
            loss = 0
            last_train_outcome = ""
            buffer_type = ""
            if len(enriched_replays) >= min_replays:
                if epsilon >= epsilon_min:
                    epsilon -= epsilon_degrade
                for r in range(batch_repeat):
                    minibatch = []
                    if random.random() > 0.5:
                        buffer_type = "EPISODE"
                        minibatch = episode_replay_buffer
                    else:
                        buffer_type = "REPLAY "
                        minibatch = random.choice(enriched_replays)
                    st_t, a_t, r_t, d, st_t1, go = zip(*minibatch)
                    outcome = r_t[-1]
                    last_train_outcome = "%.3f"%outcome
                    actions_printable = " ".join([letters[l] for l in a_t[-printable_length:]])
                    reward_printable = " ".join(["%.2f"%x for x in r_t[-printable_length:]])
                    goals = torch.cat(go).unsqueeze(0)
                    inputs = torch.cat(st_t).unsqueeze(0)
                    batch_st_t1 = torch.cat(st_t1).unsqueeze(0)
                    targets = model(inputs, goals)[-1]
                    next_targets = model(batch_st_t1, goals)[-1]
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

            last_actions_printable = " ".join([letters[l] for l in last_actions[-printable_length:]])
            info = g.print_state()
            #info += step_info
            info += "\n"
            info += "\n   Episode: " + str(episode) + " step: " + "%04d"%total_steps + " Score: " + str(score)
            info += "\n   Curiosity score: " + "%.3f"%curiosity_score
            info += "\n   Action type: " + action_type
            info += "\n   Epsilon: " + "%.3f"%epsilon + " Total iterations: " + str(iterations)
            info += "\n   Batches in replay memory: " + str(len(enriched_replays))
            #info += "\n   r_t: [" + reward_printable + "]"
            info += "\n   a_t: [" + actions_printable + "]"
            info += "\n   High score: " + str(highscore)
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
                #sys.stderr.write("\x1b[2J\x1b[H")
                os.system('clear')
                print(info)
            else:
                sys.stdout.write("\r")
                sys.stdout.flush()
                sys.stdout.write("Iterations: " + str(iterations) + " Replay buffer: " + str(len(enriched_replays)))
                sys.stdout.flush()
            iterations += 1

        enriched_replays.append(episode_replay_buffer)
        if len(enriched_replays) > replay_memory_capacity:
            enriched_replays.popleft()
        if use_her == True:
            for n in range(5):
                her_seq = []
                for r in episode_replay_buffer:
                    state, action, reward, done, next_state, goal = r
                    randep = random.choice(episode_replay_buffer)
                    new_goal = randep[-2]
                    new_reward = -1
                    if np.all(np.array_equal(next_state.numpy(), new_goal.numpy())):
                        new_reward = 0
                    her_seq.append([state, action, new_reward, done, next_state, new_goal])
                enriched_replays.append(her_seq)
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
        if episode % score_history == 0:
            if episode > 0:
                if score_average < score_average_snapshot:
                    if epsilon < 0.10:
                        epsilon = 0.10
            score_average_snapshot = score_average



if len(sys.argv) > 1:
    seed = sys.argv[1]
    print("Random seed: " + str(seed))
    random.seed(seed)
print("Starting training.")
train()
