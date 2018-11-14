# -*- coding: utf-8 -*-
import numpy as np
import random

class qtable:
    def __init__(self, num_states, num_actions):
        self.num_states = num_states
        self.num_actions = num_actions
        self.reset()

    def reset(self):
        self.table = np.random.rand(self.num_states, self.num_actions)

    def print_table_flat(self):
        printable = ""
        for s in range(self.num_states):
            printable += "  State: " + str(s) + ":\t" + " ".join(["%.2f"%x for x in self.table[s]]) + "\n"
        return printable

    def print_table_structured(self, state, action):
        sep = " " + "----------------  "*5 + "\n"
        printable = ""
        printable += sep
        for y in range(5):
            for x in range(5):
                index = y*5+x
                printable += "|     "
                if state == index and action == 0:
                    printable += "*" + "%.2f"%self.table[index, 0] + "*"
                else:
                    printable += "[" + "%.2f"%self.table[index, 0] + "]"
                printable += "     |"
            printable += "\n"
            for x in range(5):
                index = y*5+x
                printable += "|"
                if state == index and action == 2:
                    printable += "*" + "%.2f"%self.table[index, 2] + "*"
                else:
                    printable += "[" + "%.2f"%self.table[index, 2] + "]"
                printable += " " + "%02d"%(index) + " "
                if state == index and action == 3:
                    printable += "*" + "%.2f"%self.table[index, 3] + "*"
                else:
                    printable += "[" + "%.2f"%self.table[index, 3] + "]"
                printable += "|"
            printable += "\n"
            for x in range(5):
                index = y*5+x
                printable += "|     "
                if state == index and action == 1:
                    printable += "*" + "%.2f"%self.table[index, 1] + "*"
                else:
                    printable += "[" + "%.2f"%self.table[index, 1] + "]"
                printable += "     |"
            printable += "\n"
            printable += sep
        return printable

    def set_value(self, state, action, value):
        self.table[state, action] = value

    def get_value(self, state, action):
        return self.table[state, action]

    def get_best_action(self, state):
        return np.argmax(self.table[state, :])

