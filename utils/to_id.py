import sys

vocab_file = sys.argv[1]
vocab = [l.strip().split() for l in open(vocab_file)]
str2id = {v: i for i, v in vocab}
for l in sys.stdin:
    l = [str(str2id.get(w, '1')) for w in l.strip().split()]
    print(' '.join(['2'] + l + ['3']))
