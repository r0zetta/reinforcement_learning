from parachute import *
import json
import time
import sys

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
done = False
score = 0
while done == False:
    state = g.game_space.T.tolist()
    conc = g.concurrent
    spd = g.speed
    info = "Concurrency: " + str(conc) + "\nSpeed: " + str(spd) + "\nScore: " + str(score)
    print_game_state(state, info)
    action = g.move_ai()
    s_t, r, done, info = g.step(action[0])
    score += r
    time.sleep(0.01)
