import torch
import os
import json
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import logging
from tqdm import tqdm
from data import Dataset, Tokenizer
from gnmt import GNMT


class Trainer:
    def __init__(self,
                 max_epochs=10,
                 max_steps=1000,
                 batch_size=2000,
                 seed=12345):

        self.max_epochs = max_epochs
        self.max_steps = max_steps
        self.batch_size = batch_size

        self.seed = seed
        self.num_batches = 0
        self.num_steps = 0

        self.should_stop = False

    def init_logger(self):
        # create logger
        logger = logging.getLogger('train')
        logger.setLevel(logging.DEBUG)

        # create console handler and set level to debug
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        fh = logging.FileHandler(os.path.join(self.save_dir, 'train.log'))
        fh.setLevel(logging.DEBUG)

        # create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S')

        # add formatter to ch
        ch.setFormatter(formatter)
        fh.setFormatter(formatter)

        # add ch to logger
        logger.addHandler(ch)
        logger.addHandler(fh)

        return logger

    @property
    def num_epochs(self):
        if hasattr(self, 'train_tqdm'):
            return self.train_tqdm.n
        else:
            return 0

    @property
    def num_batches_this_epoch(self):
        if hasattr(self, 'epoch_tqdm'):
            return self.epoch_tqdm.n
        else:
            return 0

    def fit(self, model, dataset):
        # init loggers
        self.tensorboard = SummaryWriter()
        self.save_dir = self.tensorboard.log_dir
        self.logger = self.init_logger()

        self.logger.info(f'seed torch and numpy: {self.seed}')
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        torch.set_deterministic(True)

        self.logger.info(f'setup dataset ...')
        dataset.setup()
        self.logger.info('done.')

        self.logger.info(f'start training ...')
        self.train_tqdm = tqdm(range(self.max_epochs),
                               desc='epoch',
                               dynamic_ncols=True,
                               postfix=0)
        for i in self.train_tqdm:
            self.num_steps_this_epoch = 0
            self.train_epoch(model, dataset)
            if self.should_stop:
                break

        self.train_tqdm.close()
        self.tensorboard.close()

    def train_epoch(self, model, dataset):
        self.epoch_tqdm = tqdm(dataset.train_dataloader(self.batch_size),
                               desc='batch',
                               dynamic_ncols=True,
                               postfix=1)
        for b in self.epoch_tqdm:
            self.num_batches += 1

            def closure():
                model.zero_grad()
                loss, acc = model.train_step(b)
                loss.backward()
                return loss

            model.optim.step(closure)
            self.num_steps += 1

        self.epoch_tqdm.close()


if __name__ == "__main__":
    print('prepare bpe tok ...')
    bpe_tok = src_tok = Tokenizer(
        'en',
        ['bpe:/pvc/minNMT/data/wmt14.en-de/bpe.37000.h100000/bpe.37000.share'],
        '/pvc/minNMT/data/wmt14.en-de/bpe.37000.h100000/bpe.37000.share.vocab')
    trg_tok = Tokenizer(
        'de',
        ['bpe:/pvc/minNMT/data/wmt14.en-de/bpe.37000.h100000/bpe.37000.share'],
        '/pvc/minNMT/data/wmt14.en-de/bpe.37000.h100000/bpe.37000.share.vocab')

    print("prepare model ...")
    model = GNMT(src_tok, trg_tok, 512, 0.1).cuda()
    print('setup dataset ...')
    dataset = Dataset(src_tok, trg_tok,
                      '/pvc/minNMT/data/wmt14.en-de/bpe.37000.h100000')
    trainer = Trainer()
    trainer.fit(model, dataset)