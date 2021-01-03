import torch
from torch import nn
from data import Tokenizer, Dataset
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter


class Attn(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.linear_in = nn.Linear(dim, dim, bias=False)
        self.linear_out = nn.Linear(dim * 2, dim, bias=False)
        self.softmax = nn.Softmax(dim=-1)
        self.tanh = nn.Tanh()

    def forward(self, q, c):
        """
        q size: q_len, b, dim
        c size: c_len, b, dim
        """
        q = q.transpose(0, 1)
        c = c.transpose(0, 1)
        q = self.linear_in(q)
        score = q @ (c.transpose(1, 2))
        score = self.softmax(score)
        out = score @ c
        out = torch.cat([out, q], dim=-1)
        out = self.linear_out(out)
        out = self.tanh(out)
        return out.transpose(0, 1), score


class GNMT(nn.Module):
    def __init__(self, src_tok, trg_tok, dim, dropout):
        super().__init__()

        self.dim = dim
        self.src_tok = src_tok
        self.trg_tok = trg_tok
        self.src_embed = nn.Embedding(self.src_tok.vocab_size, dim)
        self.trg_embed = nn.Embedding(self.trg_tok.vocab_size, dim)
        self.proj = nn.Linear(dim, self.trg_tok.vocab_size)
        self.encoder_layers = nn.ModuleList([nn.LSTM(dim, dim)] * 8)
        self.decoder_layers = nn.ModuleList([nn.LSTM(dim, dim)] * 8)
        self.attn = Attn(dim)

        self.optim = torch.optim.Adam(self.parameters())

    def forward(self, strs):
        x = Dataset.pad_idss(self.src_tok.batch_encode(strs))
        y_step = x[:1]
        y = [y_step]
        x = self.src_embed(x)
        x = self.encode(x)
        hidden = None
        for i in range(len(x)):
            y_step, hidden = self.decode(y_step, hidden)
            y_step = self.attn(y_step, x)
            y.append(y_step)
        return torch.cat(y)

    def train_step(self, batch):
        s, t = list(batch.keys())
        x = batch[s].to(self.device)
        y = batch[t].to(self.device)
        y_true = y[1:]
        y = y[:-1]
        x = self.src_embed(x)
        y = self.trg_embed(y)
        x = self.encode(x)
        y, hidden = self.decode(y)
        y, score = self.attn(y, x)
        y = self.proj(y)
        loss = nn.functional.cross_entropy(y.reshape(-1,
                                                     self.trg_tok.vocab_size),
                                           y_true.reshape(-1),
                                           ignore_index=0)
        acc = y.max(-1)[1].eq(y_true)
        mask = y_true.ne(0)
        acc = acc.masked_select(mask).sum() / mask.sum()
        return loss, acc

    def encode(self, x):
        x_fwd, (h, c) = self.encoder_layers[0](x)
        x_bkd, (h, c) = self.encoder_layers[1](x.flip(0))
        x = x_fwd + x_bkd
        for l in self.encoder_layers[2:]:
            res = x
            x, (h, c) = l(x)
            x = x + res
        return x

    def decode(self, y, hidden=None):
        new_hidden = {}
        for i, l in enumerate(self.decoder_layers):
            res = y
            if hidden is None:
                y, (h, c) = l(y)
            else:
                y, (h, c) = l(y, hidden[i])
            new_hidden[i] = (h, c)
            y = y + res
        return y, new_hidden

    @property
    def device(self):
        return next(self.parameters()).device


if __name__ == "__main__":
    print('test bpe tok ...')
    bpe_tok = src_tok = Tokenizer(
        'en',
        ['bpe:/pvc/minNMT/data/wmt14.en-de/bpe.37000.h100000/bpe.37000.share'],
        '/pvc/minNMT/data/wmt14.en-de/bpe.37000.h100000/bpe.37000.share.vocab')
    trg_tok = Tokenizer(
        'de',
        ['bpe:/pvc/minNMT/data/wmt14.en-de/bpe.37000.h100000/bpe.37000.share'],
        '/pvc/minNMT/data/wmt14.en-de/bpe.37000.h100000/bpe.37000.share.vocab')

    print("prepare model ...")
    gnmt = GNMT(src_tok, trg_tok, 512, 0.1).cuda()

    print('setup dataset ...')
    dataset = Dataset(src_tok, trg_tok,
                      '/pvc/minNMT/data/wmt14.en-de/bpe.37000.h100000')
    dataset.setup()
    print(f'done.')

    tensorboard = SummaryWriter()
    train_tqdm = tqdm(dataset.train_dataloader(20000))
    for b in train_tqdm:

        def closure():
            gnmt.zero_grad()
            loss, acc = gnmt.train_step(b)
            loss.backward()
            train_tqdm.set_postfix({'loss': loss.item(), 'acc': acc.item()})
            tensorboard.add_scalar('train/loss', loss.item(), train_tqdm.n)
            tensorboard.add_scalar('train/acc', acc.item(), train_tqdm.n)
            return loss

        gnmt.optim.step(closure)

    tensorboard.close()
    train_tqdm.close()

    strs = [
        'Gutach: Increased safety for pedestrians',
        'They are not even 100 metres apart: On Tuesday, the new B 33 pedestrian lights in Dorfparkplatz in Gutach became operational - within view of the existing Town Hall traffic lights.',
        'Two sets of lights so close to one another: intentional or just a silly error?',
        'Yesterday, Gutacht\'s Mayor gave a clear answer to this question.',
        '"At the time, the Town Hall traffic lights were installed because this was a school route," explained Eckert yesterday.',
        'The Kluser lights protect cyclists, as well as those travelling by bus and the residents of Bergle.',
        'The system, which officially became operational yesterday, is of importance to the Sulzbachweg/Kirchstrasse junction.',
        'We have the museum, two churches, the spa gardens, the bus stop, a doctor\'s practice and a bank, not to mention the traffic from the \'Grub\' residential area.',
        '"At times of high road and pedestrian traffic, an additional set of lights were required to ensure safety," said Eckert.',
        'This was also confirmed by Peter Arnold from the Offenburg District Office.',
    ]
    x = src_tok.batch_encode(strs)
    x = torch.tensor(Dataset.pad_idss(x)).T.cuda()
    y_step = x[:1]
    y = [y_step]
    x = gnmt.src_embed(x)
    x = gnmt.encode(x)
    hidden = None
    for i in range(len(x)):
        y_step = gnmt.trg_embed(y_step)
        y_step, hidden = gnmt.decode(y_step, hidden)
        y_step, score = gnmt.attn(y_step, x)
        y_step = y_step.max(-1)[1]
        y.append(y_step)
    y = torch.cat(y)
    print()