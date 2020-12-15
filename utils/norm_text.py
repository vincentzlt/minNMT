import sys
import unicodedata as ud

for l in sys.stdin:
    l = ud.normalize('NFKC', l)
    sys.stdout.write(l)