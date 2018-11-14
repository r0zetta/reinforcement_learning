# coding: utf-8

from qa import *
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from collections import Counter

import random
import json
import os
import io
import sys
import time

class TestGRU(nn.Module):
    def __init__(self, voc_len, sent_len, hidden_size, num_layers, filters, filter_len, num_convs, bidirectional):
        super(TestGRU, self).__init__()
        self.voc_len = voc_len
        self.sent_len = sent_len
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_filters = filters
        self.filter_len = filter_len
        self.num_convs = num_convs
        self.bidirectional = bidirectional
        multiplier = 1
        if self.bidirectional == True:
            multiplier = 2

        self.convs = []
        for n in range(self.num_convs):
            c = nn.Conv1d(sent_len,
                          self.num_filters,
                          n%self.filter_len + 1,
                          padding=0)
            self.convs.append(c)
 
        self.grus = []
        for n in range(self.num_convs):
            g = nn.GRU(self.num_filters * (self.voc_len - n%self.filter_len),
                       self.hidden_size,
                       self.num_layers,
                       bidirectional=self.bidirectional)
            self.grus.append(g)

        self.fcs = []
        for n in range(self.num_convs):
            f =  nn.Linear(multiplier * self.hidden_size,
                           hidden_size)
            self.fcs.append(f)

        self.fc3 = nn.Linear(self.num_convs * self.hidden_size,
                             self.voc_len)

    def forward(self, x):
        x1 = []
        for n in range(self.num_convs):
            l = self.flatten(F.max_pool1d(F.relu(self.convs[n](x)), 3, 1, padding=1))
            x1.append(l)
        g1 = []
        h1 = []
        for n in range(self.num_convs):
            g, h = self.grus[n](x1[n])
            g = g.reshape(g.size()[1:])
            g1.append(g)
            h1.append(h)
        fc = []
        for n in range(self.num_convs):
            f = F.relu(self.fcs[n](g1[n]))
            fc.append(f)
        flat = torch.cat((fc), dim=1)
        out = self.fc3(flat)
        return out

    def flatten(self, v):
        f = self.flat_size(v)
        s = v.size()[0]
        return v.reshape(1, s, f)

    def flat_size(self, t):
        ns = t.size()[1:]
        f = 1
        for s in ns:
            f *= s
        return f


def copy_weights(source, target):
    target.load_state_dict(source.state_dict())


def train():
    episodes = 10000
    c = False
    b = False
    h = 50
    l = 1
    f = 20
    fl = 1
    nc = 5
    w = 0
    lr = 0.0004
    ed = 0.000002
    o = "Adam"
    acc, score, classes, corr_classes = train_with_params(c, h, l, f, fl, nc, b, lr, ed, w, o, episodes)
    params = {"clamp": c,
              "bidirectional": b,
              "hidden": h,
              "layers": l,
              "filters": f,
              "filter_lens": fl,
              "num_convs": nc,
              "learn": lr,
              "epsilon_degrade": ed,
              "wrong": w,
              "opt": o,
              "acc": acc,
              "classes": classes,
              "corr_classes": corr_classes,
              "score": score}
    with open("last_params.json", "w") as f:
        json.dump(params, f, indent=4)

def param_search():
    episodes = 300
    clamp = [False, False]
    bidirectional = [False, False]
    hidden = [10, 40]
    layers = [1,1]
    filters = [20, 40]
    filter_lens = [1,1]
    num_convs = [8,16]
    learn = [0.001, 0.0003]
    eps_d = [0.000009, 0.00001]
    opt = ["SGD", "Adam"]
    wrong = [0, -1]

    best_params = {}
    best_score = 0.0
    for n in range(1000):
        c = True
        b = random.choice(bidirectional)
        h = random.randint(hidden[0], hidden[1])
        l = random.randint(layers[0], layers[1])
        f = random.randint(filters[0], filters[1])
        fl = random.randint(filter_lens[0], filter_lens[1])
        nc = random.randint(num_convs[0], num_convs[1])
        w = random.choice(wrong)
        lr = 0.0004
        ed = 0.00001
        o = random.choice(opt)
        acc, score, classes, corr_classes = train_with_params(c, h, l, f, fl, nc, b, lr, ed, w, o, episodes)
        params = {"clamp": c,
                  "bidirectional": b,
                  "hidden": h,
                  "layers": l,
                  "filters": f,
                  "filter_lens": fl,
                  "num_convs": nc,
                  "learn": lr,
                  "epsilon_degrade": ed,
                  "wrong": w,
                  "opt": o,
                  "acc": acc,
                  "classes": classes,
                  "corr_classes": corr_classes,
                  "score": score}
        filename = "params_" + "%03d"%n + ".json"
        with open(filename, "w") as f:
            json.dump(params, f, indent=4)
        if score > best_score:
            best_score = score
            with open("best_params.json", "w") as f:
                json.dump(params, f, indent=4)


def train_with_params(clamp, hidden, layers, filters, filter_len, num_convs, bidirectional, learn, ed, wrong, opt, episodes):
    clamp_grads = clamp
    print("clamp: " + str(clamp))
    hidden_size = hidden
    print("hidden: " + str(hidden))
    num_layers = layers
    print("layers: " + str(layers))
    filters = filters
    print("filters: " + str(filters))
    filter_len = filter_len
    print("filter_len: " + str(filter_len))
    num_convs = num_convs
    print("num_convs: " + str(num_convs))
    bidirectional = bidirectional
    print("bidirectional: " + str(bidirectional))
    lr = learn
    print("learning_rate: " + str(lr))
    epsilon_degrade = ed
    print("epsilon_degrade: " + str(epsilon_degrade))
    wrong = wrong
    print("wrong answer: " + str(wrong))
    opt = opt
    print("optimizer: " + str(opt))
    num_episodes = episodes
    print("num episodes: " + str(num_episodes))

    num_objects = 3
    num_names = 3
    num_moves = 3
    g = game(num_objects, num_names, num_moves)
    #for s in g.sent:
    #    print(" ".join(s))
    #print(g.answer)

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    state = torch.FloatTensor(g.state).to(device)
    batch_len = state.size()[0]
    sent_len = state.size()[1]
    voc_size = state.size()[2]
    num_actions = g.vocab_size


    epsilon_start = 1.0
    epsilon_min = 0.01
    epsilon = epsilon_start
    print("Sent len", sent_len)
    print("Voc size", voc_size)
    print("Action shape", num_actions)
    print("Test batch length", batch_len)
    model = TestGRU(voc_size, sent_len, hidden_size, num_layers, filters, filter_len, num_convs, bidirectional)
    if torch.cuda.is_available():
        model = model.cuda()
    optimizer = None
    if opt == "SGD":
        optimizer = optim.SGD(model.parameters(), weight_decay=1e-6, momentum=0.9, nesterov=True, lr=learn)
    else:
        optimizer = optim.Adam(model.parameters(), lr=learn, amsgrad=True)
    print(model)

    #print(state.size())
    #verdict = model(state)
    #print(verdict.size())
    #print("")

    iterations = 0
    steps_per_episode = 500
    curriculum_steps = 1000000
    repeats = 1
    classes_hist = []
    corr_classes_hist = []
    pred_acc_hist = []
    ep_acc_hist = []
    loss_hist = []
    reward_hist = []
    losses = []
    rewards = []
    for episode in range(num_episodes):
        predicts = 0
        correct_predicts = 0
        corrects = 0
        guesses = 0
        corr_preds = Counter()
        preds = Counter()
        answers = Counter()
        for step in range(steps_per_episode):
            g.reset()
            state = torch.FloatTensor(g.state).to(device)
            for rep in range(repeats):
                iterations += 1

                action = None
                use_predict = False
                if random.random() > epsilon:
                    use_predict = True
                    predicts += 1
                    with torch.no_grad():
                        pred = model(state)[-1]
                    action = int(np.argmax(pred))
                    preds[action] += 1
                else:
                    if iterations < curriculum_steps:
                        action = g.answer_index
                    else:
                        action = random.randint(0, g.vocab_size-1)

                if epsilon >= epsilon_min:
                    epsilon -= epsilon_degrade

                reward = wrong
                guesses += 1
                answers[g.answer] += 1
                if action == g.answer_index:
                    corrects += 1
                    reward = 1
                if use_predict == True:
                    if reward == 1:
                        correct_predicts += 1
                        corr_preds[g.answer] += 1

                rewards.append(reward)
                targets = model(state)[-1]
                t1 = targets[action].to(device)
                t2 = torch.tensor((reward), dtype=torch.float).to(device)
                diff = t2 - t1
                loss = ((diff * diff) / 2).to(device)
                losses.append(loss.item())
                optimizer.zero_grad()
                loss.backward()
                if clamp_grads == True:
                    for param in model.parameters():
                        if hasattr(param, "grad"):
                            if hasattr(param.grad, "data"):
                                param.grad.data.clamp_(-1, 1)
                optimizer.step()

        predict_acc = 0.0
        episode_acc = 0.0
        mean_loss = torch.mean(torch.tensor(losses))
        mean_rewards = np.mean(rewards)
        if predicts > 0:
            predict_acc = float(correct_predicts)/float(predicts) * 100.0
        if guesses > 0:
            episode_acc = float(corrects)/float(guesses) * 100.0
        pred_acc_hist.append(predict_acc)
        ep_acc_hist.append(episode_acc)
        reward_hist.append(mean_rewards)
        loss_hist.append(float(mean_loss))
        print("Episode: " + str(episode) + " iterations: " + str(iterations) + " epsilon: " + "%.3f"%epsilon + " mean loss: " + "%.3f"%mean_loss + " mean rewards: " + "%.3f"%mean_rewards + " correct: " + str(corrects) + "/" + str(guesses) + " episode accuracy: " + "%.1f"%episode_acc + " predictions: " + str(correct_predicts) + "/" + str(predicts) + " predict accuracy: " + "%.3f"%predict_acc)
        num_answers = len(answers.most_common())
        print("Answers: " + " ".join([w + "(" + str(c) + ")" for w, c in answers.most_common()]))
        print("Preds: " + " ".join([g.voc_inv[n] + "(" + str(c) + ")" for n, c in preds.most_common(num_answers)]))
        classes_hist.append(len(preds))
        print("Correct: " + " ".join([w + "(" + str(c) + ")" for w, c in corr_preds.most_common(num_answers)]))
        corr_classes_hist.append(len(corr_preds))
        print("")
        if episode % 50 == 0:
            with open("predict_accuracy.json", "w") as f:
                json.dump(pred_acc_hist, f, indent=4)
            with open("episode_accuracy.json", "w") as f:
                json.dump(ep_acc_hist, f, indent=4)
            with open("loss_history.json", "w") as f:
                json.dump(loss_hist, f, indent=4)
            with open("reward_history.json", "w") as f:
                json.dump(reward_hist, f, indent=4)
            with open("classes.json", "w") as f:
                json.dump(classes_hist, f, indent=4)
            with open("corr_classes.json", "w") as f:
                json.dump(corr_classes_hist, f, indent=4)
    mean_acc = np.mean(pred_acc_hist)
    mean_classes = np.mean(classes_hist)
    mean_corr_classes = np.mean(corr_classes_hist)
    score = mean_acc * mean_corr_classes
    return mean_acc, score, mean_classes, mean_corr_classes


if len(sys.argv) > 1:
    seed = sys.argv[1]
    print("Random seed: " + str(seed))
    random.seed(seed)
print("Starting training.")
train()
