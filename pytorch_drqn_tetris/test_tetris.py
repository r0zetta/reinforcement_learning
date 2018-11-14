from tetris import *
import time
import sys

delay = 0.1
g = game()
while 1:
    g.reset()
    done = False
    while done == False:
        action = g.move_ai()
        _, reward, done, info = g.step(action)
        sys.stderr.write("\x1b[2J\x1b[H")
        print(g.print_state())
        print(info)
        time.sleep(delay)
