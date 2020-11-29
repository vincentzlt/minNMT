import pandas as pd
import sys
import math


def roundup(x):
    return int(math.ceil(x / 100.0)) * 100


def main():
    train, val, test, out_fname = sys.argv[1:5]
    train = pd.read_pickle(train)
    val = pd.read_pickle(val)
    test = pd.read_pickle(test)

    src, trg = val.columns

    src_maxlen = pd.concat([val[src], test[src]]).map(len).max()
    trg_maxlen = pd.concat([val[trg], test[trg]]).map(len).max()

    train = train[(train[src].map(len) < roundup(src_maxlen))
                  & (train[trg].map(len) < roundup(trg_maxlen))]

    train.to_pickle(out_fname)


if __name__ == "__main__":
    main()