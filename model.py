import torch
from torch import nn


class Embedding(nn.Module):
    def __init__(self, vocab_size, dim=512, padding_idx=0, dropout=0.1):
        super().__init__()

        self.emb = nn.Embedding(vocab_size, dim, padding_idx=padding_idx)
        nn.init.uniform_(self.emb.weight, -0.01, 0.01)

        max_len = 1024
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, dim, 2).float() *
            (-torch.tensor(10000.0).log() / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
        self.dropout = nn.Dropout(dropout)
        self.dim = dim

    def forward(self, x, pos=0):
        x = self.emb(x) * (self.dim**0.5)
        x = x + self.pe[pos:x.size(0) + pos]
        x = self.dropout(x)
        return x


class Pytorch_TF(nn.Module):
    def __init__(self,
                 dim,
                 src_vocab_size,
                 trg_vocab_size,
                 share=True,
                 dropout=0.1) -> None:
        super().__init__()

        self.src_emb = Embedding(src_vocab_size, dim, dropout=0.1)
        self.proj = nn.Linear(dim, trg_vocab_size, bias=False)
        if share:
            assert src_vocab_size == trg_vocab_size
            self.trg_emb = self.src_emb
            self.proj.weight = self.src_emb.emb.weight
        else:
            self.trg_emb = Embedding(trg_vocab_size, dim, dropout=0.1)

        self.nn = nn.Transformer(dim)

    def mask(self, x):
        return self.nn.generate_square_subsequent_mask(x.size(0)).type_as(x)

    def forward(self, x, y):
        x = self.src_emb(x)
        y = self.src_emb(y)

        x = self.nn.encoder(x)
        y = self.nn.decoder(y, x, self.mask(y))

        y = self.proj(y)
        return y

    def forward_step(self, y_step, state=None):
        x, y_prev = state
        Y_prev = y = torch.cat([y_prev, y_step])

        y = self.src_emb(y)
        y = self.nn.decoder(y, x, self.mask(y))
        y_step_prob = self.proj(y[-1:]).softmax(-1)

        state = (x, y_prev)
        return y_step_prob, state

    def init_state(self, x):
        y = x[:1]
        x = self.src_emb(x)
        x = self.nn.encoder(x)
        state = (x, y)
        return state