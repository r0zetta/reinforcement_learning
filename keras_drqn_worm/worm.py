import random
import numpy as np
from collections import deque
import time
import sys

class game:
    def __init__(self, dim):
        self.width = dim
        self.height = dim
        self.num_actions = 4
        self.starting_segments = 4
        self.reset()

    def reset(self):
        self.timestep = 1
        #seq = [[2,2], [2,3], [2,4]]
        start_x = random.randint(1, self.width-2)
        start_y = random.randint(1, self.height-2)
        self.segments = []
        #self.segments = seq
        self.segments.append([start_x, start_y])
        self.place_new_food()
        self.replay = []
        self.set_state()
        return self.state

    def place_new_food(self):
        while 1:
            x = random.randint(0, self.width-1)
            y = random.randint(0, self.height-1)
            if [x,y] not in self.segments:
                self.food_pos = [x, y]
                break

    def printable_state(self):
        printable = ""
        s = self.state.T
        for row in s:
            printable += "    " + " ".join(map(str, row)) + "\n"
        return printable

    def pretty_state(self):
        printable = ""
        s = self.state.T
        printable += "     " + "--"*(self.width-1) + "\n"
        for row in s:
            printable += "    |" + " ".join([" " if x==0 else "@" if x==2 else "*" for x in row]) + "|\n"
        printable += "     " + "--"*(self.width-1) + "\n"
        return printable

    def set_state(self):
        self.state = np.zeros((self.width, self.height), dtype = int)
        self.state[self.food_pos[0]][self.food_pos[1]] = 2
        for s in self.segments:
            self.state[s[0]][s[1]] = 1
        self.replay.append(self.state.tolist())

    def move_player(self, move):
        info = ""
        info += "    Number of segments: " + str(len(self.segments)) + "\n"
        for s in self.segments:
            info += "    [" + str(s[0]) + "," + str(s[1]) + "] "
        info += "\n"
        action = move
        current_x = self.segments[0][0]
        current_y = self.segments[0][1]
        new_x = current_x
        new_y = current_y
# Check if we hit the wall
        if action == 0: # Up
            info += "    Move up\n"
            if current_y == 0:
                info += "    Hit wall\n"
                return -1, info
            else:
                new_y = new_y - 1
        elif action == 1: # Down
            info += "    Move down\n"
            if current_y == self.height-1:
                info += "    Hit wall\n"
                return -1, info
            else:
                new_y = new_y + 1
        elif action == 2: # Left
            info += "    Move left\n"
            if current_x == 0:
                info += "    Hit wall\n"
                return -1, info
            else:
                new_x = new_x - 1
        elif action == 3: # Right
            info += "    Move right\n"
            if current_x == self.width-1:
                info += "    Hit wall\n"
                return -1, info
            else:
                new_x = new_x + 1
# Check if we hit our tail
        for s in self.segments:
            if new_x == s[0] and new_y == s[1]:
                info += "    Hit tail\n"
                return -1, info
# Check if we ate food
        info += "    Moving from [" + str(current_x) + " " + str(current_y) + "] to [" + str(new_x) + " " + str(new_y) + "]\n"
        ate_food = False
        if new_x == self.food_pos[0] and new_y == self.food_pos[1]:
            ate_food = True
# Move all segments
        new_segments = []
        new_segments.append([new_x, new_y])
        previous_segments = []
        if ate_food == True:
            previous_segments = self.segments
        else:
            if len(self.segments) > 0:
                previous_segments = self.segments[:-1]
        if len(previous_segments) > 0:
            new_segments += previous_segments
        self.segments = new_segments
        if ate_food == True:
            self.place_new_food()
            return 1*len(self.segments), info
        return 0.01, info

    def move_ai(self):
        xpos = self.segments[0][0]
        ypos = self.segments[0][1]
        safe_moves = []
        if xpos != 0:
            if [xpos-1, ypos] not in self.segments:
                safe_moves.append(2)
        if xpos != self.width-1:
            if [xpos+1, ypos] not in self.segments:
                safe_moves.append(3)
        if ypos != 0:
            if [xpos, ypos-1] not in self.segments:
                safe_moves.append(0)
        if ypos != self.height-1:
            if [xpos, ypos+1] not in self.segments:
                safe_moves.append(1)
        desired_moves = []
        if xpos < self.food_pos[0]:
            desired_moves.append(3)
        if xpos > self.food_pos[0]:
            desired_moves.append(2)
        if ypos < self.food_pos[1]:
            desired_moves.append(1)
        if ypos > self.food_pos[1]:
            desired_moves.append(0)
        possible_moves = []
        for n in desired_moves:
            if n in safe_moves:
                possible_moves.append(n)
        if len(possible_moves) > 0:
            return random.choice(possible_moves)
        if len(safe_moves) > 0:
            return random.choice(safe_moves)
        return 0

    def step(self, action):
        done = False
        reward, info = self.move_player(action)
        if reward > 0:
            self.timestep += 1
            self.set_state()
        else:
            done = True
        return self.state, reward, done, info

    def sample_random_action(self):
        guess = random.randint(0, self.num_actions-1)
        return [guess]
