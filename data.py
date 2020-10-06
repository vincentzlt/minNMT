import pytorch_lightning as pl
import torch
import pandas as pd
import youtokentome as yttm
import sacremoses as sm


class Dataset(pl.LightningDataModule):
    def __init__(self,
                 train_path='data/wmt14.en-de/train.pkl.zip',
                 val_path='data/wmt14.en-de/val.pkl.zip',
                 test_path='data/wmt14.en-de/test.pkl.zip',
                 slang='de',
                 tlang='en',
                 is_moses=True,
                 sbpe='data/wmt14.en-de/bpe.32k.de',
                 tbpe='data/wmt14.en-de/bpe.32k.en',
                 batch_size=4096):
        super().__init__()

        self.train_path = train_path
        self.val_path = val_path
        self.test_path = test_path

        self.slang = slang
        self.tlang = tlang
        self.batch_size = batch_size
        self.is_moses = is_moses
        self.sbpe = sbpe
        self.tbpe = tbpe

    def setup(self, stage=None):
        pl._logger.info('setup dataset ...')
        self.train_df = pd.read_pickle(self.train_path).sample(frac=1)
        self.val_df = pd.read_pickle(self.val_path).sample(frac=1)
        self.test_df = pd.read_pickle(self.test_path).sample(frac=1)

        self.sbpe_model = yttm.BPE(self.sbpe)
        self.tbpe_model = yttm.BPE(self.tbpe)
        self.sdetok = sm.MosesDetokenizer(self.slang)
        self.tdetok = sm.MosesDetokenizer(self.tlang)

    def batch_ids(self, data, size):
        slen = [len(d[self.slang]) + 2 for d in data]
        tlen = [len(d[self.tlang]) + 2 for d in data]

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

    def pad(self, ll):
        ll = [[2] + l + [3] for l in ll]
        mlen = max(len(l) for l in ll)
        ll = [l + [0] * (mlen - len(l)) for l in ll]
        return torch.tensor(ll).T

    def collate_fn(self, batch):
        if isinstance(batch, dict):
            batch = [batch]

        keys = list(batch[0].keys())
        batch = {k: [d[k] for d in batch] for k in keys}
        batch[self.slang] = self.pad(batch[self.slang])
        batch[self.tlang] = self.pad(batch[self.tlang])
        return batch

    def train_dataloader(self):
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
            num_workers=8,
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
    dataset = Dataset()
    dataset.setup()
    for b in dataset.train_dataloader():
        pass
