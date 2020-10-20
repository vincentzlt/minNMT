import pytorch_lightning as pl
import torch
import pandas as pd
import collections as cl
import functools as ft


class Dataset(pl.LightningDataModule):
    def __init__(self,
                 src_lang,
                 trg_lang,
                 src_train,
                 trg_train,
                 src_val,
                 trg_val,
                 src_test,
                 trg_test,
                 bpe_file=None,
                 batch_size=4096,
                 num_workers=8):
        super().__init__()

        self.src_lang = src_lang
        self.trg_lang = trg_lang
        self.src_train = src_train
        self.trg_train = trg_train
        self.src_val = src_val
        self.trg_val = trg_val
        self.src_test = src_test
        self.trg_test = trg_test
        self.bpe_file = bpe_file

        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare_data(self) -> None:
        pass

    def read_df(self, src, trg):
        df = pd.DataFrame({
            self.src_lang: open(src).read().strip().split('\n'),
            self.trg_lang: open(trg).read().strip().split('\n')
        })
        df = df.applymap(lambda s: [int(id) for id in s.split()])
        return df

    def setup(self, stage=None):
        self.train_df = self.read_df(self.src_train, self.trg_train)
        self.val_df = self.read_df(self.src_val, self.trg_val)
        self.test_df = self.read_df(self.src_test, self.trg_test)

    def batch_ids(self, data, size):
        slen = [len(d[self.src_lang]) for d in data]
        tlen = [len(d[self.trg_lang]) for d in data]

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
        batch_ids.append(batch_id)
        return batch_ids

    def pad(self, batch):
        maxlen = max(len(b) for b in batch)
        batch = [b + [0] * (maxlen - len(b)) for b in batch]
        return torch.tensor(batch).T

    def unpad(self, batch):
        batch = batch.T.tolist()
        batch = [l[1:] if l[0] == 2 else l for l in batch]
        batch = [l[:l.index(3)] if 3 in l else l for l in batch]
        return batch

    def collate_fn(self, batch):
        slang = self.src_lang
        tlang = self.trg_lang
        batch = {
            slang: [d[slang] for d in batch],
            tlang: [d[tlang] for d in batch]
        }
        batch[slang] = self.pad(batch[slang])
        batch[tlang] = self.pad(batch[tlang])
        return batch

    def train_dataloader(self):
        self.train_df = self.train_df.sample(frac=1)
        data = self.train_df.to_dict('records')
        batch_ids = self.batch_ids(data, self.batch_size)
        dataloader = torch.utils.data.DataLoader(
            data,
            batch_sampler=batch_ids,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )
        return dataloader

    def val_dataloader(self):
        data = self.val_df.to_dict('records')
        batch_ids = self.batch_ids(data, self.batch_size // 4)
        dataloader = torch.utils.data.DataLoader(
            data,
            batch_sampler=batch_ids,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )
        return dataloader

    def test_dataloader(self):
        data = self.test_df.to_dict('records')
        batch_ids = self.batch_ids(data, self.batch_size)
        dataloader = torch.utils.data.DataLoader(
            data,
            batch_sampler=batch_ids,
            # num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )
        return dataloader
