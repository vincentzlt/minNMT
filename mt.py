import pytorch_lightning as pl
from pytorch_lightning import trainer
import torch
from torch import nn
import torch.nn.functional as F
import argparse as ap
import os

from data import Dataset
from tf import Transformer, Embedding
import sacrebleu as sb
ACC = pl.metrics.Accuracy()


class MT(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()

        h = self.hparams = hparams
        self.src_emb = Embedding(h.src_vocab_size, h.dim, dropout=h.dropout)
        if h.share_vocab:
            self.trg_emb = self.src_emb
        else:
            self.trg_emb = Embedding(h.trg_vocab_size,
                                     h.dim,
                                     dropout=h.dropout)
        self.tf = nn.Transformer(dropout=h.dropout)
        self.proj = nn.Linear(h.dim, h.trg_vocab_size)
        self.proj.weight = self.trg_emb.emb.weight
        self.lr = 0

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ap.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--src_vocab_size', type=int, default=32000)
        parser.add_argument('--trg_vocab_size', type=int, default=32000)
        parser.add_argument('--share_vocab',
                            action='store_true',
                            default=False)
        parser.add_argument('--dim', type=int, default=512)
        parser.add_argument('--dropout', type=float, default=0.1)
        parser.add_argument('--lr_scale', type=float, default=1)
        parser.add_argument('--warmup', type=int, default=4000)
        parser.add_argument(
            '--loss',
            type=str,
            default='ce',
            choices=['ce', 'label_smooth', 'label_smoothed_nll_loss'])
        return parser

    def mask(self, x):
        mask = self.tf.generate_square_subsequent_mask(x.size(0)).type_as(x)
        return mask

    def forward(self, X, Y):
        X = self.src_emb(X)
        Y = self.trg_emb(Y)
        Y = self.tf(X, Y, tgt_mask=self.mask(Y))
        Y = self.proj(Y)
        return Y

    def forward_step(self, X):
        BOS = X[:1]
        Y = BOS[:0]
        finished = torch.zeros_like(BOS).view(-1).bool()

        X = self.src_emb(X)
        X = self.tf.encoder(X)
        for i in range(int(X.size(0) * 1.1)):
            Y = torch.cat([BOS, Y])
            Y = self.trg_emb(Y)
            Y = self.tf.decoder(Y, X, tgt_mask=self.mask(Y))
            Y = self.proj(Y)
            Y = Y.max(-1)[1]
            finished = finished + Y[-1:].view(-1).eq(3)
            if finished.all():
                break
        return Y

    def training_step(self, batch, batch_idx):
        h = self.hparams
        X = batch[h.slang]
        Y = batch[h.tlang][:-1]
        Y_true = batch[h.tlang][1:]

        Y = self(X, Y)
        Y_hyp = Y.max(-1)[1]
        loss = self.loss(Y, Y_true)
        result = pl.TrainResult(loss)
        result.log('train/loss', loss)
        mask = Y_true.ne(0)
        result.log('train/lr', self.lr, prog_bar=True)
        if batch_idx % 100 == 0:
            acc = ACC(Y_hyp.masked_select(mask), Y_true.masked_select(mask))
            result.log('train/acc', acc, prog_bar=True)

        if batch_idx % 1000 == 0:
            src_ids2strs = self.trainer.datamodule.src_ids2strs
            src_str = src_ids2strs(X.T.tolist()[:1])
            trg_ids2strs = self.trainer.datamodule.trg_ids2strs
            trg_str = trg_ids2strs(Y_true.T.tolist()[:1])
            hyp_str = trg_ids2strs(Y_hyp.T.tolist()[:1])
            pl._logger.info(f'src: {src_str[0]}')
            pl._logger.info(f'trg: {trg_str[0]}')
            pl._logger.info(f'hyp: {hyp_str[0]}')
        return result

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
        if target.dim() == lprobs.dim() - 1:
            target = target.unsqueeze(-1)
        nll_loss = -lprobs.gather(dim=-1, index=target)
        smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
        if ignore_index is not None:
            pad_mask = target.eq(ignore_index)
            nll_loss.masked_fill_(pad_mask, 0.)
            smooth_loss.masked_fill_(pad_mask, 0.)
        else:
            nll_loss = nll_loss.squeeze(-1)
            smooth_loss = smooth_loss.squeeze(-1)
        if reduce:
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
            betas=(0.9, 0.999),
            eps=1e-8,
            # weight_decay=1e-4,
        )

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx,
                       lambda_closure, using_native_amp):
        h = self.hparams
        dim = h.dim
        step = self.trainer.global_step + 1
        self.lr = dim**(-0.5) * min(step**(-0.5), step * h.warmup**(-1.5))
        self.lr = self.lr * h.lr_scale
        for pg in optimizer.param_groups:
            pg['lr'] = self.lr

        # update params
        optimizer.step()
        optimizer.zero_grad()

    def validation_step(self, batch, batch_idx):
        h = self.hparams
        X = batch[h.slang]
        Y = batch[h.tlang][:-1]
        Y_true = batch[h.tlang][1:]

        Y = self(X, Y)
        loss = self.loss(Y, Y_true)
        result = pl.EvalResult()
        result.loss = loss

        Y_hyp = self.forward_step(X)
        src_ids2strs = self.trainer.datamodule.src_ids2strs
        result.src_str = src_ids2strs(X.T.tolist())
        trg_ids2strs = self.trainer.datamodule.trg_ids2strs
        result.trg_str = trg_ids2strs(Y_true.T.tolist())
        result.hyp_str = trg_ids2strs(Y_hyp.T.tolist())
        return result

    def validation_epoch_end(self, outputs):
        loss = outputs.loss.mean()
        result = pl.EvalResult(checkpoint_on=loss)
        result.log('val/loss', loss, prog_bar=True)

        src_str = [s for src in outputs.src_str for s in src]
        trg_str = [s for trg in outputs.trg_str for s in trg]
        hyp_str = [s for hyp in outputs.hyp_str for s in hyp]
        self.save_str(src_str, 'src')
        self.save_str(trg_str, 'trg')
        self.save_str(hyp_str, f'hyp.{self.trainer.global_step}')
        bleu = torch.tensor(sb.corpus_bleu(hyp_str, [trg_str]).score)
        result.log('val/bleu', bleu, prog_bar=True)
        return result

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
    parser.add_argument('--train_path',
                        type=str,
                        default='data/wmt14.en-de/train.pkl.zip')
    parser.add_argument('--val_path',
                        type=str,
                        default='data/wmt14.en-de/val.pkl.zip')
    parser.add_argument('--test_path',
                        type=str,
                        default='data/wmt14.en-de/test.pkl.zip')
    parser.add_argument('--slang', type=str, default='de')
    parser.add_argument('--tlang', type=str, default='en')
    parser.add_argument('--is_moses', action='store_true', default=True)
    parser.add_argument('--sbpe',
                        type=str,
                        default='data/wmt14.en-de/bpe.32k.de')
    parser.add_argument('--tbpe',
                        type=str,
                        default='data/wmt14.en-de/bpe.32k.en')
    parser.add_argument('--batch_size', type=int, default=4096)
    hparams = parser.parse_args()

    dataset = Dataset(train_path=hparams.train_path,
                      val_path=hparams.val_path,
                      test_path=hparams.test_path,
                      slang=hparams.slang,
                      tlang=hparams.tlang,
                      is_moses=hparams.is_moses,
                      sbpe=hparams.sbpe,
                      tbpe=hparams.tbpe,
                      batch_size=hparams.batch_size)
    mt = MT(hparams)
    pl._logger.info(mt)
    trainer = pl.Trainer.from_argparse_args(
        hparams,
        row_log_interval=1,
        replace_sampler_ddp=False,
    )
    trainer.fit(mt, dataset)
