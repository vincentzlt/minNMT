import pandas as pd
import sys
import os
import json

char2subchar = json.load(open(sys.argv[1]))
delim = sys.argv[2]
for l in sys.stdin:
    l = list(map(lambda c: char2subchar.get(c, c), l.strip()))
    print(delim.join(l))
