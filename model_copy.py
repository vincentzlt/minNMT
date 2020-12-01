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


def attn(q, k, v, mask=None):
    q = q.transpose(0, 1)
    k = k.transpose(0, 1)
    v = v.transpose(0, 1)
    score = (q @ k.transpose(-2, -1)) / (q.size(-1)**0.5)
    if mask is not None:
        score = score.masked_fill(mask, -float('inf'))
    score = score.softmax(-1)
    out = (score @ v)
    out = out.transpose(0, 1)
    return out, score


def triu_mask(x):
    sz = x.size(0)
    mask = torch.triu(torch.ones(sz, sz), 1).type_as(x).bool()
    return mask


class MultiheadAttention(nn.Module):
    def __init__(self, dim=512, num_heads=8):
        super().__init__()

        self.q_in = nn.Linear(dim, dim, bias=False)
        self.k_in = nn.Linear(dim, dim, bias=False)
        self.v_in = nn.Linear(dim, dim, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

    def forward(self, q, k, v, mask=None):
        q_len = q.size(0)
        k_len = k.size(0)
        v_len = v.size(0)
        bsz = q.size(1)

        q = self.q_in(q)
        k = self.k_in(k)
        v = self.v_in(v)

        q = q.reshape(q_len, bsz * self.num_heads, self.head_dim)
        k = k.reshape(k_len, bsz * self.num_heads, self.head_dim)
        v = v.reshape(v_len, bsz * self.num_heads, self.head_dim)

        out, score = attn(q, k, v, mask)

        out = out.reshape(q_len, bsz, self.dim)
        return out


class FFN(nn.Module):
    def __init__(self, dim=512, inner_dim=2048):
        super().__init__()

        self.l1 = nn.Linear(dim, inner_dim)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(inner_dim, dim)

    def forward(self, x):
        x = self.l1(x)
        x = self.relu(x)
        x = self.l2(x)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, dim=512, num_heads=8, dropout=0.1):
        super().__init__()

        self.self_attn = MultiheadAttention(dim, num_heads)
        self.ffn = FFN(dim, dim * 4)

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.dropout(self.self_attn(x, x, x))
        x = self.norm1(x)
        x = x + self.dropout(self.ffn(x))
        x = self.norm2(x)
        return x


class DecoderLayer(nn.Module):
    def __init__(self, dim=512, num_heads=8, dropout=0.1):
        super().__init__()

        self.self_attn = MultiheadAttention(dim, num_heads)
        self.ctx_attn = MultiheadAttention(dim, num_heads)
        self.ffn = FFN(dim, dim * 4)

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, y, x):
        y = y + self.dropout(self.self_attn(y, y, y, triu_mask(y)))
        y = self.norm1(y)
        y = y + self.dropout(self.ctx_attn(y, x, x))
        y = self.norm2(y)
        y = y + self.dropout(self.ffn(y))
        y = self.norm3(y)
        return y


class Encoder(nn.Module):
    def __init__(self, layer, num_layers=6):
        super().__init__()

        self.layers = nn.ModuleList(
            [copy.deepcopy(layer) for i in range(num_layers)])

    def forward(self, x):
        for l in self.layers:
            x = l(x)
        return x


class Decoder(nn.Module):
    def __init__(self, layer, num_layers=6):
        super().__init__()

        self.layers = nn.ModuleList(
            [copy.deepcopy(layer) for i in range(num_layers)])

    def forward(self, y, x):
        for l in self.layers:
            y = l(y, x)
        return y


class Transformer(nn.Module):
    def __init__(self, dim=512, num_layers=6, num_heads=8, dropout=0):
        super().__init__()

        encoder_layer = EncoderLayer(dim, num_heads, dropout)
        encoder = Encoder(encoder_layer, num_layers)
        decoder_layer = DecoderLayer(dim, num_heads, dropout)
        decoder = Decoder(decoder_layer, num_layers)
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x, y):
        x = self.encoder(x)
        y = self.decoder(y, x)
        return y


class MyTransformer(nn.Module):
    def __init__(self, dim, vocab_size, dropout=0.1):
        super().__init__()

        self.emb = Embedding(vocab_size, dim, dropout=dropout)
        self.tf = Transformer(dim, dropout=dropout)
        self.proj = nn.Linear(dim, vocab_size, bias=False)
        self.proj.weight = self.emb.emb.weight

    def forward(self, x, y):
        x = self.emb(x)
        y = self.emb(y)
        y = self.tf(x, y)
        y = self.proj(y)
        return y

    def forward_step(self, y_step, state):
        x, y_prev = state
        y_prev = y = torch.cat([y_prev, y_step])

        y = self.emb(y)
        y = self.tf.decoder(y, x)
        y_step_prob = self.proj(y[-1:]).softmax(-1)

        state = (x, y_prev)
        return y_step_prob, state

    def init_state(self, x):
        y_bos = x[:1]
        x = self.emb(x)
        x = self.tf.encoder(x)
        state = (x, y_bos)
        return state


class PytorchTransformer(nn.Module):
    def __init__(self, dim, vocab_size, dropout=0.1):
        super().__init__()

        self.emb = Embedding(vocab_size, dim, dropout=dropout)
        self.tf = nn.Transformer(dim, dropout=dropout)
        self.proj = nn.Linear(dim, vocab_size, bias=False)
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