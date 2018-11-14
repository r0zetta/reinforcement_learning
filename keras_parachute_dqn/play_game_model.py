# coding: utf-8

from parachute import *
from dqn_model import *
import numpy as np

import random
import json
import os
import io
import sys
import time

def print_game_state(state, info):
    sys.stderr.write("\x1b[2J\x1b[H")
    printable = ""
    printable += "\n"
    last_row = len(state) -1
    for index, row in enumerate(state):
        if index != last_row:
            printable += "   |" + "|".join([" " if x==0 else "*" for x in row]) + "|\n"
        else:
            printable += "   |" + " ".join([" " if x==0 else "_" for x in row]) + "|\n"
    print(printable)
    print(info)

g = game()
num_actions = g.num_actions
input_shape = g.state.shape[0]
print("State shape", input_shape)
print("Action shape", num_actions)
model, loaded = make_model(num_actions, input_shape)
done = False
score = 0
info = ""
while done == False:
    space = g.game_space.T.tolist()
    state = g.state
    conc = g.concurrent
    spd = g.speed
    pred = model.predict(state.reshape((1, input_shape)))
    action = [np.argmax(pred)]
    s_t, r, done, inf = g.step(action[0])
    score += r
    info = "Concurrency: " + str(conc) 
    info += "\nSpeed: " + str(spd) 
    info += "\nScore: " + str(score)
    info += "\n[" + "%.3f"%pred[0][0] + " " + "%.3f"%pred[0][1] + " " + "%.3f"%pred[0][2] + "]"
    print_game_state(space, info)
    time.sleep(0.02)
