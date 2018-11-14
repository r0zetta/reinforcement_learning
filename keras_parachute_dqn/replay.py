import json
import time
import sys

replay = None
with open("best_replay.json", "r") as f:
    replay = json.load(f)
for state in replay:
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
    time.sleep(0.03)
