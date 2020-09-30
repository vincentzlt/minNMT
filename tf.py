import torch
from torch import dropout, nn
import torch.nn.functional as F
import copy

# def Linear(in_features, out_features, bias=True):
#     m = nn.Linear(in_features, out_features, bias)
#     nn.init.normal_(m.weight,)
#     # if bias:
#     #     nn.init.ones_(m.bias)
#     return m

# def LayerNorm(dim, eps=1e-6):
#     return nn.LayerNorm(dim, eps=eps)


class ScaleNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super(ScaleNorm, self).__init__()
        self.scale = nn.Parameter(torch.tensor(dim**0.5))
        self.eps = eps

    def forward(self, x):
        norm = self.scale / torch.norm(x, dim=-1,
                                       keepdim=True).clamp(min=self.eps)
        return x * norm


class Embedding(nn.Module):
    def __init__(self, vocab_size, dim=512, padding_idx=0, dropout=0.1):
        super().__init__()

        self.emb = nn.Embedding(vocab_size, dim, padding_idx=padding_idx)
        nn.init.uniform_(self.emb.weight, -0.01, 0.01)
        F.normalize(self.emb.weight)

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
        self.dropout = dropout
        self.dim = dim

    def forward(self, x, pos=0):
        x = self.emb(x) * (self.dim**0.5)
        x = x + self.pe[pos:x.size(0) + pos]
        x = F.dropout(x, self.dropout)
        return x


def attn(q, k, v, mask=None):
    q = q.transpose(0, 1)
    k = k.transpose(0, 1)
    v = v.transpose(0, 1)
    score = (q @ k.transpose(-2, -1)) / (q.size(-1)**0.5)
    if mask is not None:
        score = score + mask.unsqueeze(0)
    score = score.softmax(-1)
    out = (score @ v)
    out = out.transpose(0, 1)
    return out, score


def triu_mask(x):
    sz = x.size(0)
    mask = torch.triu(torch.empty(sz, sz).fill_(-float('inf')), 1).type_as(x)
    return mask


class MultiheadAttention(nn.Module):
    def __init__(self, dim=512, num_heads=8):
        super().__init__()

        self.q_in = nn.Linear(dim, dim, bias=False)
        nn.init.normal_(self.q_in.weight, 0, (2 / (5 * dim))**0.5)
        self.k_in = nn.Linear(dim, dim, bias=False)
        nn.init.normal_(self.k_in.weight, 0, (2 / (5 * dim))**0.5)
        self.v_in = nn.Linear(dim, dim, bias=False)
        nn.init.normal_(self.v_in.weight, 0, (2 / (5 * dim))**0.5)
        self.out = nn.Linear(dim, dim, bias=False)
        nn.init.normal_(self.out.weight, 0, (2 / (5 * dim))**0.5)

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
        nn.init.normal_(self.l1.weight, 0, (2 / (5 * dim))**0.5)
        self.l2 = nn.Linear(inner_dim, dim)
        nn.init.normal_(self.l2.weight, 0, (2 / (5 * dim))**0.5)

    def forward(self, x):
        x = self.l1(x)
        x = F.relu(x)
        x = self.l2(x)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, dim=512, num_heads=8, dropout=0.1):
        super().__init__()

        self.self_attn = MultiheadAttention(dim, num_heads)
        self.ffn = FFN(dim, dim * 4)

        self.norm1 = ScaleNorm(dim)
        self.norm2 = ScaleNorm(dim)
        self.dropout = dropout

    def forward(self, x):
        x = self.norm1(x)
        x = x + F.dropout(self.self_attn(x, x, x), self.dropout)
        x = self.norm2(x)
        x = x + F.dropout(self.ffn(x), self.dropout)
        return x


class DecoderLayer(nn.Module):
    def __init__(self, dim=512, num_heads=8, dropout=0.1):
        super().__init__()

        self.self_attn = MultiheadAttention(dim, num_heads)
        self.ctx_attn = MultiheadAttention(dim, num_heads)
        self.ffn = FFN(dim, dim * 4)

        self.norm1 = ScaleNorm(dim)
        self.norm2 = ScaleNorm(dim)
        self.norm3 = ScaleNorm(dim)

        self.dropout = dropout

    def forward(self, y, x):
        y = self.norm1(y)
        y = y + F.dropout(self.self_attn(y, y, y, triu_mask(y)), self.dropout)
        y = self.norm2(y)
        y = y + F.dropout(self.ctx_attn(y, x, x), self.dropout)
        y = self.norm3(y)
        y = y + F.dropout(self.ffn(y), self.dropout)
        return y


class Encoder(nn.Module):
    def __init__(self, layer, num_layers=6):
        super().__init__()

        self.layers = nn.ModuleList(
            [copy.deepcopy(layer) for i in range(num_layers)])
        self.norm = ScaleNorm(layer.self_attn.dim)

    def forward(self, x):
        for l in self.layers:
            x = l(x)
        x = self.norm(x)
        return x


class Decoder(nn.Module):
    def __init__(self, layer, num_layers=6):
        super().__init__()

        self.layers = nn.ModuleList(
            [copy.deepcopy(layer) for i in range(num_layers)])
        self.norm = ScaleNorm(layer.self_attn.dim)

    def forward(self, y, x):
        for l in self.layers:
            y = l(y, x)
        y = self.norm(y)
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
