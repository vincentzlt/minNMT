import pandas as pd
import sys
import os

gran = sys.argv[1]
assert gran in ['ideo', 'ideo_finest', 'stroke']
cur_dir = os.path.dirname(os.path.realpath(__file__))
ids = pd.read_csv(f'{cur_dir}/ids.tsv', sep='\t', index_col='char')
char2subchar = ids[gran].to_dict()
for l in sys.stdin:
    l = list(map(lambda c: char2subchar.get(c, c), l.strip()))
    print(' '.join(l))
