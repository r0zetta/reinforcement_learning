# coding: utf-8

from multiply import *
from collections import deque
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import sys

max_val = 5
num_vars = 2
gamma = 0.99
epsilon = 1
epsilon_degrade = 0.00001
epsilon_min = 0.0
batch_size = 25
replay_memory_capacity = 10000
num_episodes = 10000
episode_length = 100
accuracy = [0.0]

replay_memory = deque()
iterations = 0

class DQN(nn.Module):
    def __init__(self, num_actions, input_size, hidden_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

g = game(max_val, num_vars)
hidden_size = g.max_answer
num_actions = g.max_answer
input_size = g.state.shape[0]
model = DQN(num_actions, input_size, hidden_size)
optimizer = optim.SGD(model.parameters(), weight_decay=1e-6, momentum=0.9, nesterov=True, lr=0.01)
print(model)

state = torch.tensor(g.state, dtype=torch.float).unsqueeze(0)
print(state)
with torch.no_grad():
    pred = model(state)
    action = model(state).max(1)[1].view(1, 1)
    print(pred)
    print(action)

print("")
print("Starting training...")
print("")
for episode in range(num_episodes):
    action = None
    prediction = None
    done = False
    correct_predictions = 0
    num_predictions = 0
    corrects = 0
    predict_accuracy = 0.0
    episode_accuracy = 0.0
    last_loss = 0.0
    for step in range(episode_length):
        state = g.reset()
        state_printable = "[" + " ".join(map(str, state)) + "]"
        inputs_printable = "[" + " ".join(map(str, g.variables)) + "]"
        state = torch.tensor(state, dtype=torch.float).unsqueeze(0)
        iterations += 1
        action = None
        action_type = ""
        if random.random() > epsilon:
            action_type = "pred"
            num_predictions += 1
            with torch.no_grad():
                action = int(model(state).max(1)[1].view(1, 1)) + 1
        else:
            action_type = "rand"
            action = g.sample_random_action()[0]
        if epsilon > epsilon_min:
            epsilon -= epsilon_degrade
        reward = g.step(action)
        reward_printable = str(reward)
        corrects += max(0, reward)
        if action_type == "pred":
            correct_predictions += max(0, reward)
            predict_accuracy = float(float(correct_predictions)/float(num_predictions))*100.0
        episode_accuracy = float(float(corrects)/float(step+1)) * 100.0
        replay_memory.append([state, action-1, reward])
        if len(replay_memory) > replay_memory_capacity:
            replay_memory.popleft()
        if len(replay_memory) > batch_size:
            minibatch = random.sample(replay_memory, batch_size)
            state_t, action_t, reward_t = zip(*minibatch)
            state_t = torch.cat(state_t)
            action_batch = torch.LongTensor(action_t).unsqueeze(1)
            reward_batch = torch.FloatTensor(reward_t).unsqueeze(1)
            state_action_values = model(state_t).gather(1, action_batch)
            loss = (state_action_values - reward_batch).pow(2).mean()
            last_loss = loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    if episode % 10 == 0:
        info = ""
        #info += "\n   Max val: " + str(max_val)
        #info += "\n   Num vars: " + str(num_vars)
        #info += "\n   Classes: " + str(num_actions)
        #info += "\n"
        info += "\n   Inputs: [" + inputs_printable + "]"
        info += "\n   [" + action_type + "] " + str(action) 
        info += "\n   [corr] " + str(g.correct_val)
        info += "\n   [reward] " + reward_printable
        info += "\n   "
        info += "\n   Iter: " + "%06d"%iterations + " Episode: " + "%04d"%episode 
        info += "\n   Epsilon : " + "%.3f"%epsilon 
        info += "\n   Last loss : " + "%.3f"%last_loss
        info += "\n   Episode accuracy: " + "%.2f"%episode_accuracy
        info += "\n   Prediction accuracy: " + "%.2f"%predict_accuracy
        info += "\n   Prediction this episode: " + str(num_predictions)
        info += "\n"
        #sys.stderr.write("\x1b[2J\x1b[H")
        print(info)
