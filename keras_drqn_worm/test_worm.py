from worm import *
import sys
import time

delay = 0.03

g = game(30)

while 1:
    g.reset()
    done = False
    reward = 0.0
    while done == False:
        sys.stderr.write("\x1b[2J\x1b[H")
        print(g.pretty_state())
        #action = g.sample_random_action()[0]
        action = g.move_ai()
        _, r_t, done, info = g.step(action)
        reward += r_t
        #print(info)
        #print("Reward: " + str(reward))
        time.sleep(delay)
