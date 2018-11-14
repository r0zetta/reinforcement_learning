# -*- coding: utf-8 -*-
import numpy as np
import random
from rescue import *
import time
import sys

delay = 0.3

g = game()
wins = 0
deaths = 0
while 1:
    g.reset()
    done = False
    turns = 0
    while done == False:
        turns += 1
        sys.stderr.write("\x1b[2J\x1b[H")
        a_n, action = g.sample_random_action()
        print("Random action: " + action)
        s_t, r_t, done, info = g.step(action)
        printable = g.print_state()
        print(printable)
        print(info)
        print("Reward: " + str(r_t))
        print("Wins: " + str(wins) + " Deaths: " + str(deaths))
        print("Turns: " + str(turns))
        if done == True:
            if r_t < 0:
                deaths += 1
            if r_t > 0:
                wins += 1
            print("*** GAME OVER ***")
            print("")
        time.sleep(delay)

