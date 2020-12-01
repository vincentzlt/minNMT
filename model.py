import torch
from torch import nn
import copy


class Embedding(nn.Module):
    def __init__(self, vocab_size, dim=512, padding_idx=0, dropout=0.1):
        super().__init__()

        self.emb = nn.Embedding(vocab_size, dim, padding_idx=padding_idx)

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


class Transformer(nn.Module):
    def __init__(self,
                 vocab_size: int = 37000,
                 d_model: int = 512,
                 nhead: int = 8,
                 num_encoder_layers: int = 6,
                 num_decoder_layers: int = 6,
                 dim_feedforward: int = 2048,
                 dropout: float = 0.1,
                 activation: str = 'relu') -> None:
        super().__init__()

        self.emb = Embedding(vocab_size=vocab_size,
                             dim=d_model,
                             padding_idx=0,
                             dropout=dropout)
        self.tf = nn.Transformer(d_model=d_model,
                                 nhead=nhead,
                                 num_encoder_layers=num_encoder_layers,
                                 num_decoder_layers=num_decoder_layers,
                                 dim_feedforward=dim_feedforward,
                                 dropout=dropout,
                                 activation=activation)
        self.proj = nn.Linear(in_features=d_model,
                              out_features=vocab_size,
                              bias=True)
        self.proj.weight = self.emb.emb.weight

    def mask(self, x):
        return self.tf.generate_square_subsequent_mask(x.size(0)).type_as(x)

    def forward(self, x, y):
        x = self.emb(x)
        y = self.emb(y)
        y = self.tf(x, y, tgt_mask=self.mask(y))
        y = self.proj(y)
        return y

    def forward_step(self, y_step, state):
        x, y_prev = state
        y_prev = y = torch.cat([y_prev, y_step])

        y = self.emb(y)
        y = self.tf.decoder(y, x, tgt_mask=self.mask(y))
        y_step_prob = self.proj(y[-1:]).softmax(-1)

        state = (x, y_prev)
        return y_step_prob, state

    def init_state(self, x):
        y_bos = x[:1]
        x = self.emb(x)
        x = self.tf.encoder(x)
        state = (x, y_bos)
        return state


if __name__ == "__main__":
    tf = Transformer()

    x = torch.randint(0, 37000, (8, 4))
    y = torch.randint(0, 37000, (9, 4))
    y = tf(x, y)
    assert y.shape == (9, 4, 37000)

    x = torch.randint(0, 37000, (8, 4))
    y_step = x[:0]
    state = tf.init_state(x)
    for i in range(10):
        y_step_prob, state = tf.forward_step(y_step, state)
        y_step = y_step_prob.max(-1)[1]
    assert y_step_prob.shape == (1, 4, 37000)

    print()