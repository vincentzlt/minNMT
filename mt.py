import pytorch_lightning as pl
import sacrebleu
import torch
from torch import nn
import torch.nn.functional as F
import argparse as ap
import os

import data
import model
import search
import sacrebleu as sb
ACC = pl.metrics.Accuracy()


class MT(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()

        h = self.hparams = hparams
        if h.model == 'my_transformer':
            self.model = model.MyTransformer(h.dim, h.vocab_size, h.dropout)
        elif h.model == 'pytorch_transformer':
            self.model = model.PytorchTransformer(h.dim, h.vocab_size,
                                                  h.dropout)
        else:
            raise ValueError('model not defined.')

        self.search = search.Search(self.model, h.lenpen)
        self.lr = 0

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ap.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--model',
                            type=str,
                            default='my_transformer',
                            choices=['my_transformer', 'pytorch_transformer'])
        parser.add_argument('--vocab_size', type=int, default=37000)
        parser.add_argument('--dim', type=int, default=512)
        parser.add_argument('--dropout', type=float, default=0.1)
        parser.add_argument('--warmup', type=int, default=4000)
        parser.add_argument(
            '--loss',
            type=str,
            default='label_smooth',
            choices=['ce', 'label_smooth', 'label_smoothed_nll_loss'])
        parser.add_argument('--lenpen', type=float, default=0.6)
        parser.add_argument('--beam_size', type=int, default=5)
        parser.add_argument('--ckpt_save_interval', type=int, default=1500)
        return parser

    def forward(self, x, y):
        return self.model(x, y)

    def training_step(self, batch, batch_idx):
        h = self.hparams
        x = batch[h.src_lang]
        y = batch[h.trg_lang][:-1]
        y_true = batch[h.trg_lang][1:]

        y = self(x, y)
        loss = self.loss(y, y_true)
        self.log('train/loss', loss, on_step=True, prog_bar=False, logger=True)
        self.log('train/lr', self.lr, on_step=True, prog_bar=True, logger=True)

        if batch_idx % (2 * h.accumulate_grad_batches) == 0:
            mask = y_true.ne(0)
            y_hyp = y.max(-1)[1]

            pred = y_hyp.masked_select(mask)
            truth = y_true.masked_select(mask)
            acc = ACC(pred, truth)
            self.log('train/acc', acc.item(), prog_bar=True, logger=True)

        return loss

    def ce_loss(self, output, target):
        output = output.reshape(-1, output.size(-1))
        target = target.reshape(-1)
        loss = F.cross_entropy(output, target, ignore_index=0)
        return loss

    def label_smooth_loss(self, output, target):
        output = output.reshape(-1, output.size(-1)).log_softmax(-1)
        target = target.reshape(-1)
        label = torch.empty_like(output).fill_(0.1 / (output.size(-1) - 1))
        label = label.scatter_(1, target.view(-1, 1), 0.9)
        loss = F.kl_div(output, label, reduction='none')
        mask = target.ne(0).unsqueeze(-1)
        loss = loss.masked_select(mask).sum() / (mask.sum())
        return loss

    def label_smoothed_nll_loss(self, output, target):
        lprobs = output.log_softmax(-1)
        epsilon = 0.1
        ignore_index = 0
        reduce = True
        #calc loss
        target = target.unsqueeze(-1)
        nll_loss = -lprobs.gather(dim=-1, index=target)
        smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
        #handle ignored index
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.)
        smooth_loss.masked_fill_(pad_mask, 0.)
        #handle reduce
        nll_loss = nll_loss.mean()
        smooth_loss = smooth_loss.mean()

        eps_i = epsilon / lprobs.size(-1)
        loss = (1. - epsilon) * nll_loss + eps_i * smooth_loss
        return loss

    def loss(self, output, target):
        if self.hparams.loss == 'ce':
            loss = self.ce_loss(output, target)
        if self.hparams.loss == 'label_smooth':
            loss = self.label_smooth_loss(output, target)
        if self.hparams.loss == 'label_smoothed_nll_loss':
            loss = self.label_smoothed_nll_loss(output, target)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(),
            betas=(0.9, 0.98),
            eps=1e-9,
        )

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx,
                       lambda_closure, using_native_amp, using_lbfgs):
        h = self.hparams
        dim = h.dim
        step = self.trainer.global_step + 1
        lr = dim**(-0.5) * min(step**(-0.5), step * h.warmup**(-1.5))
        for pg in optimizer.param_groups:
            pg['lr'] = lr
        self.lr = lr

        # update params
        optimizer.step()
        optimizer.zero_grad()

        if self.trainer.global_step % h.ckpt_save_interval == 0:
            ckpt_fname = f'{self.trainer.logger.log_dir}/checkpoints/step.{self.trainer.global_step}.ckpt'
            self.trainer.save_checkpoint(ckpt_fname)

    def validation_step(self, batch, batch_idx):
        h = self.hparams
        x = batch[h.src_lang]
        y = batch[h.trg_lang][:-1]
        y_true = batch[h.trg_lang][1:]

        y = self(x, y)
        loss = self.loss(y, y_true)

        mask = y_true.ne(0)
        y_hyp = y.max(-1)[1]

        pred = y_hyp.masked_select(mask)
        truth = y_true.masked_select(mask)
        acc = ACC(pred, truth)

        # y_hyp = self.search.topk(x, int(x.size(0) * 2), 4)
        # unpad = self.trainer.datamodule.unpad
        return {'loss': loss, 'acc': acc}

    def validation_epoch_end(self, outputs):
        loss = torch.stack([o['loss'] for o in outputs]).mean()
        acc = torch.stack([o['acc'] for o in outputs]).mean()
        self.log('val/loss', loss.item(), prog_bar=True)
        self.log('val/acc', acc.item(), prog_bar=True)

    def save_idss(self, idss, fname):
        d = f'{self.trainer.logger.log_dir}/hyps'
        if not os.path.exists(d):
            os.makedirs(d, exist_ok=True)
        with open(f'{d}/{fname}', 'wt') as f:
            f.write('\n'.join(
                [' '.join([str(id) for id in ids]) for ids in idss]))

    def test_step(self, batch, batch_idx):
        x = batch[self.hparams.src_lang]
        y_hyp = self.search.topk(
            x,
            int(x.size(0) * 2),
            self.hparams.beam_size,
        )
        unpad = self.trainer.datamodule.unpad
        idss = unpad(y_hyp)
        for ids in idss:
            print(' '.join([str(i) for i in ids]))


if __name__ == "__main__":
    parser = ap.ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser = MT.add_model_specific_args(parser)
    parser.add_argument('--src_lang')
    parser.add_argument('--trg_lang')
    parser.add_argument('--src_train')
    parser.add_argument('--trg_train')
    parser.add_argument('--src_val')
    parser.add_argument('--trg_val')
    parser.add_argument('--src_test')
    parser.add_argument('--trg_test')
    parser.add_argument('--batch_size', type=int, default=4096)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--seed', type=int, default=12345)
    hparams = parser.parse_args()

    pl.seed_everything(hparams.seed)

    dataset = data.Dataset(src_lang=hparams.src_lang,
                           trg_lang=hparams.trg_lang,
                           src_train=hparams.src_train,
                           trg_train=hparams.trg_train,
                           src_val=hparams.src_val,
                           trg_val=hparams.trg_val,
                           src_test=hparams.src_test,
                           trg_test=hparams.trg_test,
                           batch_size=hparams.batch_size,
                           num_workers=hparams.num_workers)
    mt = MT(hparams)
    pl._logger.info(mt)
    trainer = pl.Trainer.from_argparse_args(
        hparams,
        log_every_n_steps=1,
        replace_sampler_ddp=False,
    )
    trainer.fit(mt, dataset)
