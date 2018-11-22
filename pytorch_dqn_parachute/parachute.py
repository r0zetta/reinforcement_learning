import random
import numpy as np
from collections import deque

class game:
    def __init__(self, dim):
        self.num_lanes = dim
        self.num_vertical = dim
        self.start_speed = 4 + (self.num_lanes * 2)
        self.start_concurrent = 1
        self.max_concurrent = self.num_vertical
        self.max_speed = self.num_lanes
        self.concurrent_increment = 300
        self.speed_increment = 500
        self.num_previous_states = 1
        self.num_actions = 3
        self.reset()

    def reset(self):
        self.speed = self.start_speed
        self.concurrent = self.start_concurrent
        self.replay = []
        self.timestep = 1
        self.game_space = np.zeros((self.num_lanes, self.num_vertical+1), dtype = int)
        self.player_position = 0
        self.player_lane = self.num_vertical
        self.game_space[self.player_position][self.player_lane] = 1
        self.previous_states = deque()
        for n in range(self.num_previous_states):
            self.previous_states.append(np.ravel(self.game_space))
        self.set_state()
        return self.state

    def set_state(self):
        self.previous_states.append(np.ravel(self.game_space))
        if len(self.previous_states) > self.num_previous_states:
            self.previous_states.popleft()
        self.replay.append(self.game_space.T.tolist())
        self.state = np.ravel(np.array(self.previous_states))

    def get_objects(self):
        objects = []
        for lane in range(self.num_lanes):
            for vert in range(self.num_vertical):
                if self.game_space[lane][vert] == 1:
                    objects.append([lane, vert])
        return objects

    def get_lowest_object(self):
        objects = self.get_objects()
        lowest_lane = None
        lowest_vert = 0
        for obj in objects:
            if obj[1] > lowest_vert:
                lowest_vert = obj[1]
                lowest_lane = obj[0]
        return lowest_lane

    def move_ai(self):
        lowest_lane = self.get_lowest_object()
        action = None
        if self.player_position == lowest_lane:
            action = 1
        elif self.player_position < lowest_lane:
            action = 2
        else:
            action = 0
        return [action]

    def move_objects(self):
        if self.timestep % self.speed != 0:
            return False, False
        objects = self.get_objects()
        captured = False
        for o in objects:
            lane = o[0]
            vert = o[1]
            if vert < self.num_vertical-1:
                self.game_space[lane][vert] = 0
                self.game_space[lane][vert+1] = 1
            else:
                if lane == self.player_position:
                    captured = True
                    self.game_space[lane][vert] = 0
                else:
                    return True, False
        return False, captured

    def add_new_object(self):
        objects = self.get_objects()
        if len(objects) >= self.concurrent:
            return
        for lane in range(self.num_lanes):
            if self.game_space[lane][0] == 1:
                return
        choice = random.randint(0, self.num_lanes - 1)
        self.game_space[choice][0] = 1

    def increase_difficulty(self):
        if self.concurrent <= self.max_concurrent:
            if self.timestep % self.concurrent_increment == 0:
                self.concurrent += 1
        if self.speed > self.max_speed:
            if self.timestep % self.speed_increment == 0:
                self.speed -= 1

    def move_player(self, action):
        move_reward = 0.001
        if action == 0:
            if self.player_position > 0:
                self.game_space[self.player_position][self.player_lane] = 0
                self.player_position -= 1
                self.game_space[self.player_position][self.player_lane] = 1
            else:
                move_reward = -0.04
        elif action == 2:
            if self.player_position < self.num_lanes-1:
                self.game_space[self.player_position][self.player_lane] = 0
                self.player_position += 1
                self.game_space[self.player_position][self.player_lane] = 1
            else:
                move_reward = -0.04
        lowest_lane = self.get_lowest_object()
        if self.player_position == lowest_lane and move_reward > 0:
            move_reward = 0.02
        return move_reward

    def step(self, action):
        done = False
        info = ""
        reward = self.move_player(action)
        reward = 0
        hit_floor, captured = self.move_objects()
        if hit_floor == True:
            done = True
            reward -= 1
        elif captured == True:
            reward += 1
        if done == False:
            self.increase_difficulty()
            self.add_new_object()
            self.timestep += 1
        self.set_state()
        return self.state, reward, done, info

    def sample_random_action(self):
        guess = random.randint(0,2)
        return [guess]
