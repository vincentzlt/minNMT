import pytorch_lightning as pl
import torch
import pandas as pd
import youtokentome as yttm
import sacremoses as sm
import multiprocessing as mp
import collections as cl
import json
import functools as ft
from tqdm import tqdm
tqdm.pandas()
import numpy as np
import os
import time


class Vocab:
    def __init__(self, lang='en', tok=None, bpe=None):
        self.lang = lang
        self.specials = ['<pad>', '<unk>', '<bos>', '<eos>']
        self.freqs = cl.Counter()
        if tok == 'moses':
            self.tok = ft.partial(sm.MosesTokenizer(lang).tokenize,
                                  return_str=True)
            self.detok = ft.partial(sm.MosesDetokenizer(lang).detokenize,
                                    return_str=False)
            self.is_tok = True
        else:
            self.is_tok = False

        if bpe:
            bpe = yttm.BPE(bpe)
            self.bpe_encode = ft.partial(
                bpe.encode, output_type=yttm.youtokentome.OutputType.SUBWORD)
            self.bpe_decode = lambda l: ''.join(l).strip('▁').split('▁')
            self.is_bpe = True
        else:
            self.is_bpe = False

    def __len__(self):
        return len(self.vocab)

    @staticmethod
    def _read_text_job(lines):
        c = cl.Counter(w for l in lines for w in l.strip().split())
        return c

    @staticmethod
    def iter_file(fname):
        lines = []
        for i, l in enumerate(open(fname)):
            lines.append(l)
            if i > 0 and i % 100000 == 0:
                yield lines
                lines = []

    def read_text(self, fname):
        p = mp.Pool(mp.cpu_count())
        for c in p.imap_unordered(self._read_text_job, self.iter_file(fname)):
            self.freqs += c
        p.close()
        p.join()

    def finalize(self, fname=None):
        self.vocab = dict([(s, 0) for s in self.specials] \
                               + list(self.freqs.items()))
        self.i2s = {i: s for i, s in enumerate(self.vocab.keys())}
        self.s2i = {s: i for i, s in enumerate(self.vocab.keys())}

        if fname:
            self.write_vocab(fname)

    def write_vocab(self, fname):
        json.dump(self.vocab, open(fname, 'wt'), ensure_ascii=False, indent=4)

    def read_vocab(self, fname):
        self.vocab = json.load(open(fname))
        self.i2s = {i: s for i, s in enumerate(self.vocab.keys())}
        self.s2i = {s: i for i, s in enumerate(self.vocab.keys())}

    def ids2str(self, ids):
        strs = [self.i2s.get(i, '<unk>') for i in ids]
        if self.is_bpe:
            strs = self.bpe_decode(strs)
        if self.is_tok:
            strs = self.detok(strs)
        return ' '.join(strs)

    def str2ids(self, str):
        if self.is_tok:
            str = self.tok(str)
        if self.is_bpe:
            str = ' '.join(self.bpe_encode(str))
        return [self.s2i.get(s, 1) for s in str.split()]

    def pad_ids(self, ids, to, bos=True, eos=True):
        if bos:
            ids = [2] + ids
            to += 1
        if eos:
            ids = ids + [3]
            to += 1
        return ids + [0] * (to - len(ids))

    def unpad_ids(self, ids):
        new_ids = []
        if ids[0] == 2:
            ids = ids[1:]
        for i in ids:
            if i == 3:
                break
            else:
                new_ids.append(i)
        return new_ids

    def pad_strs(self, ids, to, bos=True, eos=True):
        if bos:
            ids = ['<bos>'] + ids
            to += 1
        if eos:
            ids = ids + ['<eos>']
            to += 1
        return ids + ['<pad>'] * (to - len(ids))

    def unpad_strs(self, strs):
        new_strs = []
        if strs[0] == '<bos>':
            strs = strs[1:]
        for s in strs:
            if s == '<eos>':
                break
            else:
                new_strs.append(s)
        return new_strs

    def sents2batch(self, sents):
        idss = [self.str2ids(str) for str in sents]
        batch = self.idss2batch(idss)
        return batch

    def batch2sents(self, batch):
        idss = self.batch2idss(batch)
        sents = [self.ids2str(ids) for ids in idss]
        return sents

    def idss2batch(self, idss):
        max_len = max(len(ids) for ids in idss)
        batch = [self.pad_ids(ids, max_len) for ids in idss]
        batch = torch.tensor(batch).T
        return batch

    def batch2idss(self, batch):
        batch = batch.T.tolist()
        idss = [self.unpad_ids(ids) for ids in batch]
        return idss


class Dataset(pl.LightningDataModule):
    def __init__(self,
                 src_vocab,
                 trg_vocab,
                 train_path,
                 val_path,
                 test_path,
                 batch_size=4096):
        super().__init__()

        self.src_vocab = src_vocab
        self.trg_vocab = trg_vocab
        self.train_path = train_path
        self.val_path = val_path
        self.test_path = test_path
        self.batch_size = batch_size

    def read_df(self, src, trg):
        src_strs = open(src).read().strip().split('\n')
        trg_strs = open(trg).read().strip().split('\n')
        df = pd.DataFrame({
            self.src_vocab.lang: src_strs,
            self.trg_vocab.lang: trg_strs
        })
        return df

    def str2ids(self, df):
        tqdm.pandas(leave=True, desc='str2ids')
        slang = self.src_vocab.lang
        tlang = self.trg_vocab.lang
        df[slang] = df[slang].str.split()
        df[tlang] = df[tlang].str.split()
        df[slang] = df[slang].progress_map(
            lambda l: [self.src_vocab.s2i.get(s, 1) for s in l])
        df[tlang] = df[tlang].progress_map(
            lambda l: [self.trg_vocab.s2i.get(s, 1) for s in l])
        return df

    def proc_df(self, df):
        slang = self.src_vocab.lang
        tlang = self.trg_vocab.lang

        tqdm.pandas(leave=True, desc='calc len')
        df[f'{slang}_len'] = df[slang].progress_map(len)
        df[f'{tlang}_len'] = df[tlang].progress_map(len)
        df['ratio'] = df[f'{slang}_len'] / df[f'{tlang}_len']
        df['ratio'] = df['ratio'].progress_map(lambda x: x if x > 1 else 1 / x)

        df = df[df['ratio'] < 3]
        return df[[slang, tlang]]

    def build_datasets(self, src, trg, out):
        df = self.read_df(src, trg)
        df = self.str2ids(df)
        df = self.proc_df(df)
        df.to_pickle(out)

    def setup(self, stage=None):
        self.train_df = pd.read_pickle(self.train_path)
        self.val_df = pd.read_pickle(self.val_path)
        self.test_df = pd.read_pickle(self.test_path)

    def batch_ids(self, data, size):
        slen = [len(d[self.src_vocab.lang]) + 2 for d in data]
        tlen = [len(d[self.trg_vocab.lang]) + 2 for d in data]

        batch_ids = []
        batch_id = []
        batch_slen = []
        batch_tlen = []
        mlen = 0
        for i, (sl, tl) in enumerate(zip(slen, tlen)):
            assert sl <= size and tl <= size
            batch_slen.append(sl)
            batch_tlen.append(tl)
            batch_id.append(i)
            if len(batch_id) * max(batch_slen) > size or len(batch_id) * max(
                    batch_tlen) > size:
                batch_ids.append(batch_id[:-1])
                batch_slen = batch_slen[-1:]
                batch_tlen = batch_tlen[-1:]
                batch_id = batch_id[-1:]
        return batch_ids

    def collate_fn(self, batch):
        slang = self.src_vocab.lang
        tlang = self.trg_vocab.lang
        batch = {
            slang: [d[slang] for d in batch],
            tlang: [d[tlang] for d in batch]
        }
        batch[slang] = self.src_vocab.idss2batch(batch[slang])
        batch[tlang] = self.trg_vocab.idss2batch(batch[tlang])
        return batch

    def train_dataloader(self):
        self.train_df = self.train_df.sample(frac=1)
        data = self.train_df.to_dict('records')
        batch_ids = self.batch_ids(data, self.batch_size)
        dataloader = torch.utils.data.DataLoader(
            data,
            batch_sampler=batch_ids,
            num_workers=8,
            collate_fn=self.collate_fn,
        )
        return dataloader

    def val_dataloader(self):
        data = self.val_df.to_dict('records')
        batch_ids = self.batch_ids(data, self.batch_size // 4)
        dataloader = torch.utils.data.DataLoader(
            data,
            batch_sampler=batch_ids,
            num_workers=8,
            collate_fn=self.collate_fn,
        )
        return dataloader

    def test_dataloader(self):
        data = self.test_df.to_dict('records')
        batch_ids = self.batch_ids(data, self.batch_size // 4)
        dataloader = torch.utils.data.DataLoader(
            data,
            batch_sampler=batch_ids,
            # num_workers=8,
            collate_fn=self.collate_fn,
        )
        return dataloader

    def src_ids2strs(self, ids):
        ids = [id[1:] if id[0] == 2 else id for id in ids]
        ids = [id[:len(id) if 3 not in id else id.index(3)] for id in ids]
        strs = [self.sbpe_model.decode(id) for id in ids]
        strs = [self.sdetok.detokenize(s) for s in strs]
        return strs

    def trg_ids2strs(self, ids):
        ids = [id[1:] if id[0] == 2 else id for id in ids]
        ids = [id[:len(id) if 3 not in id else id.index(3)] for id in ids]
        strs = [self.tbpe_model.decode(id) for id in ids]
        strs = [self.tdetok.detokenize(s) for s in strs]
        return strs


if __name__ == "__main__":
    en_vocab = Vocab('en', tok='moses', bpe='data/bpe.37k.share')
    de_vocab = Vocab('de', tok='moses', bpe='data/bpe.37k.share')

    # en_vocab.read_text('data/train.en')
    # de_vocab.read_text('data/train.de')

    # en_vocab.finalize('data/share.vocab.en')
    # de_vocab.finalize('data/share.vocab.de')

    en_vocab.read_vocab('data/share.vocab.en')
    de_vocab.read_vocab('data/share.vocab.de')

    dataset = Dataset(
        en_vocab, de_vocab,
        '/storage07/user_data/zhanglongtu01/minNMT/data/test-full/newstest2014-ende.pkl.gz',
        '/storage07/user_data/zhanglongtu01/minNMT/data/test-full/newstest2014-ende.pkl.gz',
        '/storage07/user_data/zhanglongtu01/minNMT/data/test-full/newstest2014-ende.pkl.gz',
        4096)

    dataset.build_datasets(
        '/storage07/user_data/zhanglongtu01/minNMT/data/test-full/newstest2014-deen-src.en.subword',
        '/storage07/user_data/zhanglongtu01/minNMT/data/test-full/newstest2014-deen-ref.de.subword',
        '/storage07/user_data/zhanglongtu01/minNMT/data/test-full/newstest2014-ende.pkl.gz'
    )
    # dataset.build_datasets('data/newstest2013.en', 'data/newstest2013.de',
    #                        'data/val.pkl.gz')
    # dataset.build_datasets('data/newstest2014.en', 'data/newstest2014.de',
    #                        'data/test.pkl.gz')

    dataset.setup()
    for b in tqdm(dataset.train_dataloader()):
        assert b['en'].numel() <= 4096
        assert b['de'].numel() <= 4096
    for b in tqdm(dataset.val_dataloader()):
        assert b['en'].numel() <= 4096
        assert b['de'].numel() <= 4096
    for b in tqdm(dataset.test_dataloader()):
        assert b['en'].numel() <= 4096
        assert b['de'].numel() <= 4096
