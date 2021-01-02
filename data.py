import torch
import pandas as pd
from tqdm import tqdm
import sys
import re
import jieba
import MeCab
import time
import os


class Tokenizer:
    def __init__(self, lang, steps, vocab=None):
        """
        steps: a list of strings of the tokenizing steps.
        e.g., ['jieba', 'bpe:model_path']
        valid strings include:
        
        jieba
        mecab
        char
        moses:lang
        meta_space:space_char
        bpe:model_path (yttm)

        the substring after : indicates the necessary information that this step needs

        vocab: the vocab file in [index][tab][vocab] format.
        vocab is optional if the last step is bpe. else it must be given to transform
        tokens to ids.
        """

        self.lang = lang
        self.steps = steps
        if 'jieba' in self.steps:
            jieba.setLogLevel(60)
        if 'mecab' in self.steps:
            self.wakati = MeCab.Tagger("-Owakati")
        moses = [s for s in self.steps if s.startswith('moses:')]
        if moses:
            from sacremoses import MosesTokenizer, MosesDetokenizer
            moses_lang = moses[0].split(':')[1]
            # assume only one moses step is used here
            self.moses_tokenizer, self.moses_detokenizer = MosesTokenizer(
                lang=moses_lang), MosesDetokenizer(lang=moses_lang)
        meta_space = [s for s in self.steps if s.startswith('meta_space:')]
        if meta_space:
            self.meta_space = meta_space[0].split(':')[
                1]  # assume only one meta_space step is used here
        bpe = [s for s in self.steps if s.startswith('bpe:')]
        if bpe:
            import youtokentome as yttm
            model_path = bpe[0].split(':')[1]
            # assume only one meta_space step is used here
            self.bpe = yttm.BPE(model_path)
            self.vocab_size = self.bpe.vocab_size()

        if not self.steps[-1].startswith('bpe:'):
            assert vocab is not None
            self.steps.append('to_id')
        if vocab is not None:
            self.vocab_i2w = {}
            self.vocab_w2i = {}
            for l in open(vocab):
                i, w = l.strip('\n').split('\t')
                i = int(i)
                self.vocab_i2w[i] = w
                self.vocab_w2i[w] = i
            self.vocab_size = len(self.vocab_i2w)

        self.TOK_FUNC = {
            'jieba': self.jieba_tok,
            'mecab': self.mecab_tok,
            'char': self.char_tok,
            'moses': self.moses_tok,
            'meta_space': self.meta_space_tok,
            'bpe': self.bpe_tok,
            'to_id': self.str2ids,
        }
        self.DETOK_FUNC = {
            'jieba': self.jieba_detok,
            'mecab': self.mecab_detok,
            'char': self.char_detok,
            'moses': self.moses_detok,
            'meta_space': self.meta_space_detok,
            'bpe': self.bpe_detok,
            'to_id': self.ids2str,
        }

    def char_tok(self, s):
        return ' '.join(s)

    def char_detok(self, s):
        return ''.join(s.split())

    def jieba_tok(self, s):
        return ' '.join(jieba.lcut(s))

    def jieba_detok(self, s):
        return ''.join(s.split())

    def mecab_tok(self, s):
        return self.wakati.parse(s)

    def mecab_detok(self, s):
        return ''.join(s.split())

    def moses_tok(self, s):
        return self.moses_tokenizer.tokenize(s, return_str=True)

    def moses_detok(self, s):
        return self.moses_detokenizer.detokenize(s.split())

    def meta_space_tok(self, s):
        return s.replace(' ', self.meta_space)

    def meta_space_detok(self, s):
        return s.replace(self.meta_space, ' ')

    def bpe_tok(self, s):
        return self.bpe.encode(s)

    def bpe_detok(self, ids):
        return self.bpe.decode(ids)[0]

    def str2ids(self, s):
        return [self.vocab_w2i.get(w, 1) for w in s.split()]

    def ids2str(self, ids):
        return ' '.join([self.vocab_i2w.get(i, '<UNK>') for i in ids])

    def encode(self, s):
        for step in self.steps:
            step = step.split(':')[0]
            s = self.TOK_FUNC[step](s)
        ids = s
        return ids

    def decode(self, ids):
        s = ids
        for step in reversed(self.steps):
            step = step.split(':')[0]
            s = self.DETOK_FUNC[step](s)
        return s

    def batch_encode(self, strs):
        return [self.encode(s) for s in strs]

    def batch_decode(self, idss):
        return [self.decode(ids) for ids in idss]


class Dataset:
    def __init__(self, src_tok, trg_tok, data_dir):
        self.src_tok = src_tok
        self.trg_tok = trg_tok
        self.data_dir = data_dir

    def setup(self):
        s = self.src_tok.lang
        t = self.trg_tok.lang
        d = self.data_dir

        if os.path.exists(f'{d}/train.pkl'):
            self.train_df = pd.read_pickle(f'{d}/train.pkl')
        else:
            self.train_df = pd.DataFrame({
                s:
                open(f'{d}/train.{s}.id').read().strip().split('\n'),
                t:
                open(f'{d}/train.{t}.id').read().strip().split('\n')
            }).applymap(lambda s: list(map(int, s.split())))
            self.train_df.to_pickle(f'{d}/train.pkl')

        if os.path.exists(f'{d}/val.pkl'):
            self.val_df = pd.read_pickle(f'{d}/val.pkl')
        else:
            self.val_df = pd.DataFrame({
                s:
                open(f'{d}/val.{s}.id').read().strip().split('\n'),
                t:
                open(f'{d}/val.{t}.id').read().strip().split('\n')
            }).applymap(lambda s: list(map(int, s.split())))
            self.val_df.to_pickle(f'{d}/val.pkl')

        if os.path.exists(f'{d}/test.pkl'):
            self.test_df = pd.read_pickle(f'{d}/test.pkl')
        else:
            self.test_df = pd.DataFrame({
                s:
                open(f'{d}/test.{s}.id').read().strip().split('\n'),
                t:
                open(f'{d}/test.{t}.id').read().strip().split('\n')
            }).applymap(lambda s: list(map(int, s.split())))
            self.test_df.to_pickle(f'{d}/test.pkl')

    @staticmethod
    def batch_idxs(dataset, batch_size):
        src, trg = list(dataset[0].keys())
        src_lens = [len(e[src]) + 2 for e in dataset]
        trg_lens = [len(e[trg]) + 2 for e in dataset]
        batch_idxs = []
        batch_idx = []
        src_max = trg_max = 0
        for i, (sl, tl) in enumerate(zip(src_lens, trg_lens)):
            if sl > batch_size or tl > batch_size:
                continue
            src_max = max(src_max, sl)
            trg_max = max(trg_max, tl)
            batch_idx.append(i)
            if len(batch_idx) * src_max > batch_size or len(
                    batch_idx) * trg_max > batch_size:
                batch_idxs.append(batch_idx[:-1])
                batch_idx = batch_idx[-1:]
                src_max = sl
                trg_max = tl
        batch_idxs.append(batch_idx)
        return batch_idxs

    @staticmethod
    def pad_idss(idss):
        """bos, eos, pad"""
        idss = [[2] + ids + [3] for ids in idss]
        mlen = max(len(ids) for ids in idss)
        idss = [ids + [0] * (mlen - len(ids)) for ids in idss]
        return idss

    def collate_fn(self, examples):
        src, trg = list(examples[0].keys())
        src_idss = [e[src] for e in examples]
        trg_idss = [e[trg] for e in examples]
        return {
            src: torch.tensor(self.pad_idss(src_idss)).T,
            trg: torch.tensor(self.pad_idss(trg_idss)).T
        }

    def train_dataloader(self, batch_size=2000):
        dataset = self.train_df.sample(frac=1).to_dict('records')
        batch_idxs = self.batch_idxs(dataset, batch_size)
        return torch.utils.data.DataLoader(dataset,
                                           batch_sampler=batch_idxs,
                                           collate_fn=self.collate_fn,
                                           pin_memory=True)

    def val_dataloader(self, batch_size=2000):
        dataset = self.val_df.to_dict('records')
        batch_idxs = self.batch_idxs(dataset, batch_size)
        return torch.utils.data.DataLoader(dataset,
                                           batch_sampler=batch_idxs,
                                           collate_fn=self.collate_fn)

    def test_dataloader(self, batch_size=2000):
        dataset = self.test_df.to_dict('records')
        batch_idxs = self.batch_idxs(dataset, batch_size)
        return torch.utils.data.DataLoader(dataset,
                                           batch_sampler=batch_idxs,
                                           collate_fn=self.collate_fn)


if __name__ == "__main__":
    print('test bpe tok ...')
    bpe_tok = src_tok = Tokenizer(
        'en', ['bpe:/pvc/minNMT/data/wmt14.en-de/bpe.37000/bpe.37000.share'],
        '/pvc/minNMT/data/wmt14.en-de/bpe.37000/bpe.37000.share.vocab')
    trg_tok = Tokenizer(
        'de', ['bpe:/pvc/minNMT/data/wmt14.en-de/bpe.37000/bpe.37000.share'],
        '/pvc/minNMT/data/wmt14.en-de/bpe.37000/bpe.37000.share.vocab')

    for en, en_id in zip(
            open('/pvc/minNMT/data/wmt14.en-de/raw/test.en'),
            open('/pvc/minNMT/data/wmt14.en-de/bpe.37000/test.en.id')):
        en = en.strip()
        en_id = list(map(int, en_id.strip().split()))
        assert bpe_tok.encode(en) == en_id
        assert bpe_tok.decode(en_id).split() == en.split()
    print('done.')
    print('test jieba tok ...')
    jieba_tok = Tokenizer(
        'zh', ['jieba'],
        '/pvc/minNMT/data/aspec_jc/char.word/train.share.vocab')
    for zh, zh_id in zip(
            open('/pvc/minNMT/data/aspec_jc/char.raw/test.zh'),
            open('/pvc/minNMT/data/aspec_jc/char.word/test.zh.id')):
        zh = zh.strip()
        zh_id = list(map(int, zh_id.strip().split()))
        assert jieba_tok.encode(zh) == zh_id
        # assert jieba_tok.decode(zh_id) == zh.replace(' ', '')
    print('done.')

    print('setup dataset ...')
    t = time.time()
    dataset = Dataset(src_tok, trg_tok,
                      '/pvc/minNMT/data/wmt14.en-de/bpe.37000')
    dataset.setup()
    print(f'done in {time.time()-t} sec.')
    for b in tqdm(dataset.train_dataloader(2000)):
        assert b[src_tok.lang].cuda().numel() <= 2000
        assert b[trg_tok.lang].cuda().numel() <= 2000
