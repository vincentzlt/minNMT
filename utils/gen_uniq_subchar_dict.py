import pandas as pd
import sys
import json

gran = sys.argv[1]
assert gran in ['ideo', 'ideo_finest', 'stroke']
df = pd.read_csv('/pvc/minNMT/utils/ids.tsv', sep='\t', header=0, index_col=0)
subchar_dict = df[gran].to_dict()
uniq_dict = {}
for k, d in subchar_dict.items():
    if not d in uniq_dict:
        uniq_dict[d] = k
    else:
        tag_num = 1
        while f'{d}_{tag_num}' in uniq_dict:
            tag_num += 1
        uniq_dict[f'{d}_{tag_num}'] = k
uniq_dict = {v: k for k, v in uniq_dict.items()}
print(json.dumps(uniq_dict, ensure_ascii=False, indent=4))