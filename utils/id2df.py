import sys
import pandas as pd
from tqdm import tqdm


def read_int(fname):
    return [[int(i) for i in l.strip().split()]
            for l in tqdm(open(fname), desc=f'read {fname}')]


def main():
    src, src_fname, trg, trg_fname, out_fname = sys.argv[1:6]

    df = pd.DataFrame({src: read_int(src_fname), trg: read_int(trg_fname)})

    df.to_pickle(out_fname)


if __name__ == "__main__":
    main()