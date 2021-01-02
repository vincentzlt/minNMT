import sys

specials = ['<PAD>', '<UNK>', '<BOS>', '<EOS>']

if len(sys.argv) > 1:
    import collections as cl
    vocab_size = int(sys.argv[1])
    vocab = cl.Counter(w for l in sys.stdin for w in l.strip().split())
    vocab = specials + [v for v, freq in vocab.most_common()]
    vocab = vocab[:vocab_size]

else:
    vocab = specials + list(
        set(w for l in sys.stdin for w in l.strip().split()))

for i, v in enumerate(vocab):
    print(f'{str(i)}\t{v}')