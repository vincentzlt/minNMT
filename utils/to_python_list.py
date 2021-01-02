import sys

for l in sys.stdin:
    print('['+', '.join(l.strip().split())+']')