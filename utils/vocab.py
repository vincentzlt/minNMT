import sys
import collections as cl

c = cl.Counter(w for l in sys.stdin for w in l.strip().split())
vocab = ['<PAD>', '<UNK>', '<BOS>', '<EOS>'
         ] + [v for v, freq in c.most_common()]
for i, v in enumerate(vocab):
    print(f'{str(i)}\t{v}')