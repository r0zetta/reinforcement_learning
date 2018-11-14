# -*- coding: utf-8 -*-
import numpy as np
import random
from rescue import *
from qtable import *
import time
import sys

g = game()
state = g.state
num_states = g.num_states
num_actions = g.num_actions
q = qtable(num_states, num_actions)
epsilon = 1.0
epsilon_degrade = 0.001
epsilon_min = 0.01
gamma = 0.95
learning_rate = 0.9
wins = 0
deaths = 0
iterations = 0
while 1:
    g.reset()
    state = g.get_state()
    done = False
    turns = 0
    while done == False:
        turns += 1
        iterations += 1
        printable = "\n"
        action = ""
        action_type = ""
        a_n = None
        if random.random() > epsilon:
            a_n = q.get_best_action(state)
            action = g.actions[a_n]
            action_type = "Predict "
        else:
            a_n, action = g.sample_random_action()
            action_type = "Random "
        printable += q.print_table_structured(state, a_n)
        if epsilon > epsilon_min:
            epsilon = epsilon * (1.0-epsilon_degrade)
        next_state, reward, done, info = g.step(action)
        cur_qval = q.table[state, a_n]
        best_new_state = np.max(q.table[next_state, :])
        quality = cur_qval + learning_rate * (reward + gamma * best_new_state) - cur_qval
        q.table[state, a_n] = max(0, quality)
        qpad = " "
        if state > 9:
            qpad = ""
        pad = " "
        if reward == -1:
            pad = ""

        printable += "Q(s,a)  = Q(s,a) + lr [R(s,a) + gamma * max Q(s',a') - Q(s,a)]\n"
        printable += "Q(" + str(state) + "," + str(a_n) + ")" + qpad + " = " + "%.2f"%cur_qval + " + " "%.2f"%learning_rate + " [" + str(reward) + pad + "     + " + "%.2f"%gamma + "  *         " + "%.2f"%best_new_state + " -   " + "%.2f"%cur_qval + "] = " + "%.2f"%quality + "\n"
        printable += g.print_state()
        printable += action_type + "action: " + action + "\n"
        printable += "Reward: " + str(reward) + "\n"
        printable += "Wins: " + str(wins) + " Deaths: " + str(deaths) + "\n"
        printable += "Turns: " + str(turns) + "\n"
        printable += "Epsilon: " + "%.4f"%epsilon + "\n"
        sys.stderr.write("\x1b[2J\x1b[H")
        print(printable)
        state = next_state
        if done == True:
            if reward < 0:
                deaths += 1
            if reward > 0:
                wins += 1
        if iterations < 15:
            delay = 2
        elif iterations > 15 and iterations < 30:
            delay = 1
        elif iterations > 30 and iterations < 60:
            delay = 0.5
        elif iterations > 60 and iterations < 100:
            delay = 0.2
        else:
            delay = 0.05
        time.sleep(delay)
