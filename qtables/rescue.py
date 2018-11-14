# -*- coding: utf-8 -*-
import numpy as np
import random

class game:
    def __init__(self):
        self.x_dim = 5
        self.y_dim = 5
        self.actions = ["up", "down", "left", "right"]
        self.reset()

    def get_state(self):
        return self.state

    def set_state(self):
        self.state = self.player_pos[0] + self.game_state.shape[0] * self.player_pos[1]

    def reset(self):
        self.game_state = np.zeros((self.x_dim, self.y_dim), dtype=int)
        self.guards = [[1, 1], [3, 1], [1, 3], [3, 3]]
        for pos in self.guards:
            self.game_state[pos[0]][pos[1]] = 2
        self.goal = [2, 4]
        self.game_state[self.goal[0]][self.goal[1]] = 3
        self.player_pos = [0, 0]
        self.set_state()
        self.num_states = self.game_state.shape[0] * self.game_state.shape[1]
        self.num_actions = len(self.actions)
        self.game_state[self.player_pos[0]][self.player_pos[1]] = 1

    def print_state(self):
        printable = ""
        st = self.game_state.T
        for x in range(st.shape[0]):
            printable += "  " + " ".join(map(str, st[x])) + "\n"
        return printable

    def step(self, action):
        current_pos = self.player_pos
        reward = 0
        new_pos = []
        valid_move = "valid"
        if action == "up":
            if current_pos[1] > 0:
                new_pos = [current_pos[0], current_pos[1]-1]
            else:
                new_pos = [current_pos[0], current_pos[1]]
                valid_move = "invalid"
        elif action == "down":
            if current_pos[1] < self.game_state.shape[1]-1:
                new_pos = [current_pos[0], current_pos[1]+1]
            else:
                new_pos = [current_pos[0], current_pos[1]]
                valid_move = "invalid"
        elif action == "left":
            if current_pos[0] > 0:
                new_pos = [current_pos[0]-1, current_pos[1]]
            else:
                new_pos = [current_pos[0], current_pos[1]]
                valid_move = "invalid"
        elif action == "right":
            if current_pos[0] < self.game_state.shape[0]-1:
                new_pos = [current_pos[0]+1, current_pos[1]]
            else:
                new_pos = [current_pos[0], current_pos[1]]
                valid_move = "invalid"
        for g in self.guards:
            if new_pos[0] == g[0] and new_pos[1] == g[1]:
                reward -= 1
                return self.game_state, reward, True, "Hit guard"
        if new_pos[0] == self.goal[0] and new_pos[1] == self.goal[1]:
            reward += 1
            return self.game_state, reward, True, "Rescued princess"
        self.game_state[current_pos[0], current_pos[1]] = 0
        self.game_state[new_pos[0], new_pos[1]] = 1
        self.player_pos[0] = new_pos[0]
        self.player_pos[1] = new_pos[1]
        self.set_state()
        return self.state, reward, False, "Move was " + valid_move

    def sample_random_action(self):
        action = random.randint(0,3)
        return action, self.actions[action]

