import numpy as np
import pytorch_lightning as pl
from pytorch_lightning import metrics
import model
import data
import torch
import youtokentome as yttm
import search
import sacrebleu
import os
import argparse as ap
import subprocess as sp
import nni


class MT(pl.LightningModule):
    def __init__(self,
                 src_lang: str = 'en',
                 trg_lang: str = 'de',
                 vocab_size: int = 37000,
                 d_model: int = 512,
                 nhead: int = 8,
                 num_encoder_layers: int = 6,
                 num_decoder_layers: int = 8,
                 dim_feedforward: int = 2048,
                 dropout: float = 0.1,
                 activation: str = 'relu',
                 warmup: int = 4000,
                 bpe_file: str = '../data/wmt14.en-de/share.bpe.37000',
                 lenpen: float = 0.6,
                 beam_size: int = 4,
                 ckpt_steps: int = 1500):
        super().__init__()

        self.src_lang = src_lang
        self.trg_lang = trg_lang
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.nhead = nhead
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.activation = activation
        self.warmup = warmup
        self.bpe_file = bpe_file
        self.lenpen = lenpen
        self.beam_size = beam_size
        self.ckpt_steps = ckpt_steps

        self.model = model.Transformer(
            vocab_size=self.vocab_size,
            d_model=self.d_model,
            nhead=self.nhead,
            num_encoder_layers=self.num_encoder_layers,
            num_decoder_layers=self.num_decoder_layers,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            activation=self.activation,
        )
        self.train_acc = pl.metrics.Accuracy()
        self.val_acc = pl.metrics.Accuracy()
        self.search = search.Search(self.model, lenpen=self.lenpen)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ap.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--src_lang', type=str, default='en')
        parser.add_argument('--trg_lang', type=str, default='de')
        parser.add_argument('--vocab_size', type=int, default=37000)
        parser.add_argument('--d_model', type=int, default=512)
        parser.add_argument('--nhead', type=int, default=8)
        parser.add_argument('--num_encoder_layers', type=int, default=6)
        parser.add_argument('--num_decoder_layers', type=int, default=8)
        parser.add_argument('--dim_feedforward', type=int, default=2048)
        parser.add_argument('--dropout', type=float, default=0.1)
        parser.add_argument('--activation', type=str, default='relu')
        parser.add_argument('--warmup', type=int, default=4000)
        parser.add_argument('--bpe_file',
                            type=str,
                            default='../data/wmt14.en-de/share.bpe.37000')
        parser.add_argument('--lenpen', type=float, default=0.6)
        parser.add_argument('--beam_size', type=int, default=4)
        return parser

    def forward(self):
        pass

    def training_step(self, batch, batch_idx):
        x = batch[self.src_lang]
        y = batch[self.trg_lang][:-1]
        y_true = batch[self.trg_lang][1:]

        y = self.model(x, y)
        loss = self.label_smoothed_nll_loss(y, y_true)
        mask = y_true.ne(0)
        self.train_acc.update(y[mask], y_true[mask])
        return loss

    def validation_step(self, batch, batch_idx):
        # calc loss
        x = batch[self.src_lang]
        y = batch[self.trg_lang][:-1]
        y_true = batch[self.trg_lang][1:]

        y = self.model(x, y)
        loss = self.label_smoothed_nll_loss(y, y_true)
        mask = y_true.ne(0)
        self.val_acc.update(y[mask], y_true[mask])

        # gen hyps
        x = batch[self.src_lang]
        y_true = batch[self.trg_lang][1:]

        hyp = self.search.topk(x,
                               max_len=int(x.size(0) * 1.1),
                               k=self.beam_size)
        return loss, x, hyp, y_true

    def validation_epoch_end(self, outputs) -> None:
        unpad = self.trainer.datamodule.unpad
        bpe = yttm.BPE(self.bpe_file)

        loss = torch.stack([o[0] for o in outputs]).mean()
        acc = self.val_acc.compute().item()
        self.val_acc.reset()

        x = bpe.decode([l for o in outputs for l in unpad(o[1])])
        hyp = bpe.decode([l for o in outputs for l in unpad(o[2])])
        y_true = bpe.decode([l for o in outputs for l in unpad(o[3])])
        bleu = sacrebleu.corpus_bleu(hyp, [y_true]).score
        self.log('val/loss', loss, True)
        self.log('val/acc', acc, True)
        self.log('val/bleu', bleu, True)
        nni.report_intermediate_result({'bleu': bleu})

        # save files
        self.save_file(f'val.{self.src_lang}', x)
        self.save_file(f'val.{self.trg_lang}', y_true)
        self.save_file(f'val.{self.trg_lang}.{self.global_step}.hyp', hyp)

    def test_step(self, batch, batch_idx):
        x = batch[self.src_lang]
        y_true = batch[self.trg_lang][1:]

        hyp = self.search.topk(x,
                               max_len=int(x.size(0) * 1.1),
                               k=self.beam_size)
        return x, hyp, y_true

    def test_epoch_end(self, outputs) -> None:
        unpad = self.trainer.datamodule.unpad
        bpe = yttm.BPE(self.bpe_file)

        x = bpe.decode([l for o in outputs for l in unpad(o[0])])
        hyp = bpe.decode([l for o in outputs for l in unpad(o[1])])
        y_true = bpe.decode([l for o in outputs for l in unpad(o[2])])
        bleu = sacrebleu.corpus_bleu(hyp, [y_true]).score
        self.log('test/bleu', bleu, True)
        nni.report_final_result({'bleu': bleu})

        # save files
        self.save_file(f'test.{self.src_lang}', x)
        self.save_file(f'test.{self.trg_lang}', y_true)
        self.save_file(f'test.{self.trg_lang}.ckpt_avg.hyp ', hyp)

    def save_file(self, fname: str, l):
        dname = f'{self.logger.log_dir}/hyps'
        if not os.path.exists(dname):
            os.mkdir(dname)
        fname = f'{dname}/{fname}'
        with open(fname, 'w') as f:
            f.write('\n'.join(l))

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(),
            betas=(0.9, 0.98),
            eps=1e-9,
        )

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx,
                       optimizer_closure, on_tpu, using_native_amp,
                       using_lbfgs):
        dim = self.d_model
        step = self.trainer.global_step + 1
        lr = dim**(-0.5) * min(step**(-0.5), step * self.warmup**(-1.5))
        for pg in optimizer.param_groups:
            pg['lr'] = lr
        self.lr = lr

        # log
        train_acc = self.train_acc.compute().item()
        self.train_acc.reset()
        loss = self.trainer.progress_bar_dict['loss']
        metrics = {}
        metrics['train/lr'] = self.lr
        metrics['train/b_acc'] = train_acc
        if loss != 'nan':
            metrics['train/loss'] = float(loss)
        self.logger.log_metrics(metrics)
        self.trainer.progress_bar_metrics.update({
            'train/lr': self.lr,
            'train/b_acc': train_acc
        })

        # update params
        optimizer.step(closure=optimizer_closure)
        optimizer.zero_grad()

        # save
        if self.ckpt_steps != 0 and self.trainer.global_step % self.ckpt_steps == 0:
            ckpt_fname = f'{self.trainer.logger.log_dir}/checkpoints/step.{self.trainer.global_step}.ckpt'
            self.trainer.save_checkpoint(ckpt_fname)

    def label_smoothed_nll_loss(self, output, target):
        # from fairseq
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


if __name__ == "__main__":
    parser = ap.ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    # add PROGRAM level args
    ## add data args
    parser.add_argument('--train_pkl',
                        type=str,
                        default='../data/wmt14.en-de/val.pkl')
    parser.add_argument('--val_pkl',
                        type=str,
                        default='../data/wmt14.en-de/val.pkl')
    parser.add_argument('--src_test',
                        type=str,
                        default='../data/wmt14.en-de/test.en.id')
    parser.add_argument('--trg_test',
                        type=str,
                        default='../data/wmt14.en-de/test.de.id')
    parser.add_argument('--batch_size', type=int, default=2000)
    parser.add_argument('--num_workers', type=int, default=8)

    ## add misc args
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--ckpt_steps', type=int, default=1500)
    parser = MT.add_model_specific_args(parser)
    args = parser.parse_args()
    nni_args = nni.get_next_parameter()
    nni.utils.merge_parameter(args, nni_args)
    pl._logger.info(args)

    pl.seed_everything(args.seed)
    hash = sp.check_output('git rev-parse HEAD'.split())
    pl._logger.info(f'commit #: {hash.decode().strip()}')

    dataset = data.Dataset(args.src_lang, args.trg_lang, args.train_pkl,
                           args.val_pkl, args.src_test, args.trg_test,
                           args.batch_size, args.num_workers)

    mt = MT(src_lang=args.src_lang,
            trg_lang=args.trg_lang,
            vocab_size=args.vocab_size,
            d_model=args.d_model,
            nhead=args.nhead,
            num_encoder_layers=args.num_encoder_layers,
            num_decoder_layers=args.num_decoder_layers,
            dim_feedforward=args.dim_feedforward,
            dropout=args.dropout,
            activation=args.activation,
            warmup=args.warmup,
            bpe_file=args.bpe_file,
            lenpen=args.lenpen,
            beam_size=args.beam_size,
            ckpt_steps=args.ckpt_steps)
    trainer = pl.Trainer.from_argparse_args(args)
    # hack for saving all args
    from pytorch_lightning.core.saving import save_hparams_to_yaml
    os.makedirs(trainer.logger.log_dir)
    save_hparams_to_yaml(f'{trainer.logger.log_dir}/args.yaml', args)

    trainer.fit(mt, datamodule=dataset)

    # average state_dict
    fnames = os.listdir(f'{trainer.logger.log_dir}/checkpoints/')
    fnames = filter(lambda s: s.startswith('step'), fnames)
    fnames = sorted(fnames, key=lambda s: int(s.split('.')[1]))
    fnames = fnames[-5:]
    fnames = [f'{trainer.logger.log_dir}/checkpoints/{f}' for f in fnames]
    state_dicts = [torch.load(f)['state_dict'] for f in fnames]
    state_dict = {}
    for k in state_dicts[0].keys():
        state_dict[k] = torch.stack([s[k] for s in state_dicts]).mean(0)
    mt.load_state_dict(state_dict)
    trainer.test()