import pytorch_lightning as pl
import torch
import pandas as pd
from tqdm import tqdm
from utils.id2df import read_int


class Dataset(pl.LightningDataModule):
    def __init__(
        self,
        src_lang,
        trg_lang,
        train_pkl=None,  # path to df
        val_pkl=None,  # path to df
        test_src=None,  # path to txt
        test_trg=None,  # path to txt
        batch_size=2500,
        num_workers=4):
        super().__init__()

        self.src_lang = src_lang
        self.trg_lang = trg_lang
        self.train_pkl = train_pkl
        self.val_pkl = val_pkl
        self.test_src = test_src
        self.test_trg = test_trg

        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare_data(self) -> None:
        pass

    def setup(self, stage=None):
        if stage == 'fit' and self.train_pkl is not None and self.val_pkl is not None:
            self.train_df = pd.read_pickle(self.train_pkl)
            self.val_df = pd.read_pickle(self.val_pkl)
        if stage == 'test' and self.test_src is not None and self.test_trg is not None:
            self.test_df = pd.DataFrame({
                self.src_lang: read_int(self.test_src),
                self.trg_lang: read_int(self.test_trg)
            })

    def batch_ids(self, data, size):
        slen = [len(d[self.src_lang]) for d in data]
        tlen = [len(d[self.trg_lang]) for d in data]

        batch_ids = []
        batch_id = []
        batch_slen = []
        batch_tlen = []
        for i, (sl, tl) in enumerate(zip(slen, tlen)):
            assert sl <= size and tl <= size
            batch_slen.append(sl)
            batch_tlen.append(tl)
            batch_id.append(i)
            src_size = len(batch_id) * max(batch_slen)
            trg_size = len(batch_id) * max(batch_tlen)
            if src_size > size or trg_size > size:
                batch_ids.append(batch_id[:-1])
                batch_slen = batch_slen[-1:]
                batch_tlen = batch_tlen[-1:]
                batch_id = batch_id[-1:]
        batch_ids.append(batch_id)
        return batch_ids

    @staticmethod
    def pad(batch):
        maxlen = max(len(b) for b in batch)
        batch = [b + [0] * (maxlen - len(b)) for b in batch]
        return torch.tensor(batch).T

    @staticmethod
    def unpad(batch):
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
        data = self.train_df.sample(frac=1).to_dict('records')
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
        batch_ids = self.batch_ids(data, self.batch_size)
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
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )
        return dataloader


if __name__ == "__main__":
    dataset = Dataset('en', 'de', 'data/wmt14.en-de/train.pkl',
                      'data/wmt14.en-de/val.pkl',
                      'data/wmt14.en-de/test.en.id',
                      'data/wmt14.en-de/test.de.id', 20000, 2)
    print('setup dataset ...')
    dataset.setup('fit')
    print('setup dataloaders ...')
    for b in tqdm(dataset.train_dataloader()):
        assert b['en'].numel() <= 20000
        assert b['de'].numel() <= 20000
    for b in tqdm(dataset.val_dataloader()):
        assert b['en'].numel() <= 20000
        assert b['de'].numel() <= 20000
    dataset.setup('test')
    for b in tqdm(dataset.test_dataloader()):
        assert b['en'].numel() <= 20000
        assert b['de'].numel() <= 20000
    print()