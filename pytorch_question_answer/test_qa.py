from qa import *
import time
import sys

delay = 0.1
g = game(3, 3, 3)

for s in g.sent:
    print(" ".join(s))

print(g.answer)
