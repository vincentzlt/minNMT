import pandas as pd
import sys

df = pd.read_csv('/pvc/minNMT/utils/kanji_mapping_table.txt',
                 skiprows=17,
                 sep='\t',
                 names=['kanji', 'ts', 'cs'])[['kanji', 'cs']]
cs2kanji = dict([[c, k] for k, cs in df.dropna().values
                 for c in cs.split(',')])
for l in sys.stdin:
    print(''.join([cs2kanji.get(c, c) for c in l.strip()]))
