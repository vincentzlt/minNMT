import pytorch_lightning as pl
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
        elif h.model == 'my_transformer':
            self.model = model.PytorchTransformer(h.dim, h.vocab_size,
                                                  h.dropout)
        else:
            raise ValueError('model not defined.')

        self.search = search.Search(self.model)
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
        parser.add_argument('--lr_scale', type=float, default=1)
        parser.add_argument('--warmup', type=int, default=4000)
        parser.add_argument(
            '--loss',
            type=str,
            default='label_smooth',
            choices=['ce', 'label_smooth', 'label_smoothed_nll_loss'])
        return parser

    def forward(self, x, y):
        return self.model(x, y)

    def training_step(self, batch, batch_idx):
        h = self.hparams
        x = batch[h.slang]
        y = batch[h.tlang][:-1]
        y_true = batch[h.tlang][1:]

        y = self(x, y)
        loss = self.loss(y, y_true)
        self.log('train/loss', loss)
        self.log('train/lr', self.lr, prog_bar=True)

        if batch_idx % (100 * h.accumulate_grad_batches) == 0:
            mask = y_true.ne(0)
            y_hyp = y.max(-1)[1]
            acc = ACC(y_hyp.masked_select(mask), y_true.masked_select(mask))
            self.log('train/acc', acc, prog_bar=True)

            if batch_idx % (1000 * h.accumulate_grad_batches) == 0:
                src_vocab = self.trainer.datamodule.src_vocab
                src_str = src_vocab.ids2str(
                    src_vocab.unpad_ids(x.T.tolist()[0]))
                trg_vocab = self.trainer.datamodule.trg_vocab
                trg_str = trg_vocab.ids2str(
                    trg_vocab.unpad_ids(y_true.T.tolist()[0]))
                hyp_str = trg_vocab.ids2str(
                    trg_vocab.unpad_ids(y_hyp.T.tolist()[0]))
                self.print(f'src: {src_str}')
                self.print(f'trg: {trg_str}')
                self.print(f'hyp: {hyp_str}')
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
        lr = lr * h.lr_scale
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
        x = batch[h.slang]
        y = batch[h.tlang][:-1]
        y_true = batch[h.tlang][1:]

        y = self(x, y)
        loss = self.loss(y, y_true)

        y_hyp = self.search.greedy(x)
        src_batch2sents = self.trainer.datamodule.src_vocab.batch2sents
        trg_batch2sents = self.trainer.datamodule.trg_vocab.batch2sents
        src_strs = src_batch2sents(x)
        trg_strs = trg_batch2sents(y_true)
        hyp_strs = trg_batch2sents(y_hyp)
        return {
            'loss': loss,
            'src_strs': src_strs,
            'trg_strs': trg_strs,
            'hyp_strs': hyp_strs,
        }

    def validation_epoch_end(self, outputs):
        loss = torch.stack([o['loss'] for o in outputs]).mean()
        self.log('val/loss', loss.item(), prog_bar=True)

        src_str = [s for o in outputs for s in o['src_strs']]
        trg_str = [s for o in outputs for s in o['trg_strs']]
        hyp_str = [s for o in outputs for s in o['hyp_strs']]
        self.save_str(src_str, 'src')
        self.save_str(trg_str, 'trg')
        self.save_str(hyp_str, f'hyp.{self.trainer.global_step}')
        bleu = sb.corpus_bleu(hyp_str, [trg_str]).score
        self.log('val/bleu', bleu, prog_bar=True)

    def save_str(self, strs, fname):
        d = f'{self.trainer.logger.log_dir}/translations'
        if not os.path.exists(d):
            os.makedirs(d, exist_ok=True)
        with open(f'{d}/{fname}', 'wt') as f:
            f.write('\n'.join(strs))


if __name__ == "__main__":
    parser = ap.ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser = MT.add_model_specific_args(parser)
    parser.add_argument('--train_path', type=str, default='data/train.pkl.gz')
    parser.add_argument('--val_path', type=str, default='data/val.pkl.gz')
    parser.add_argument('--test_path', type=str, default='data/test.pkl.gz')
    parser.add_argument('--src_vocab_path',
                        type=str,
                        default='data/share.vocab.en')
    parser.add_argument('--trg_vocab_path',
                        type=str,
                        default='data/share.vocab.de')
    parser.add_argument('--slang', type=str, default='en')
    parser.add_argument('--tlang', type=str, default='de')
    parser.add_argument('--is_moses', action='store_true', default=True)
    parser.add_argument('--bpe', type=str, default='data/bpe.37k.share')
    parser.add_argument('--batch_size', type=int, default=4096)
    parser.add_argument('--seed', type=int, default=12345)
    parser.add_argument('--ckpt_save_interval', type=int, default=1500)
    hparams = parser.parse_args()

    pl.seed_everything(hparams.seed)
    src_vocab = data.Vocab(hparams.slang, 'moses', hparams.bpe)
    src_vocab.read_vocab(hparams.src_vocab_path)
    trg_vocab = data.Vocab(hparams.tlang, 'moses', hparams.bpe)
    trg_vocab.read_vocab(hparams.trg_vocab_path)

    dataset = data.Dataset(src_vocab,
                           trg_vocab,
                           train_path=hparams.train_path,
                           val_path=hparams.val_path,
                           test_path=hparams.test_path,
                           batch_size=hparams.batch_size)
    mt = MT(hparams)
    pl._logger.info(mt)
    trainer = pl.Trainer.from_argparse_args(
        hparams,
        row_log_interval=1,
        replace_sampler_ddp=False,
    )
    trainer.fit(mt, dataset)
