import random
import numpy as np

class game:
    def __init__(self, max_val, num_vars):
        self.max_val = max_val
        self.num_variables = num_vars
        self.variables = np.zeros(self.num_variables, dtype=int)
        self.max_answer = self.max_val ** self.num_variables
        self.reset()

    def reset(self):
        for x in range(self.num_variables):
            self.variables[x] = random.randint(1, self.max_val)
        self.correct_val = np.prod(self.variables)
        self.set_state()
        return self.state

    def set_state(self):
        self.state = np.zeros(self.max_val*self.num_variables, dtype=int)
        for i, n in enumerate(self.variables):
            self.state[i*self.max_val+n-1] = 1

    def step(self, action):
        reward = 1
        if action != self.correct_val:
            reward = 0
        return reward

    def sample_random_action(self):
        guess = random.randint(1, self.max_answer)
        return [guess]
