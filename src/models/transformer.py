'''
salka: gpt.py
adaptation of OpenAI's GPT and GPT-2 architectures

https://openai.com/blog/better-language-models/
'''

import torch
from torch import nn, optim

def gelu(x):
    return x * torch.sigmoid(x * 1.702)

class Transformer(nn.Module):
    def __init__(self, input_dim, attn_dim, num_heads, nn_dim,
                 dropout=0.0, input_norm=False):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        self.ln0 = nn.LayerNorm(input_dim) if input_norm else lambda x: x

        self.attn = nn.MultiheadAttention(attn_dim, num_heads, dropout=dropout)
        self.resample1 = nn.Conv1d(input_dim, attn_dim, 1)
        self.ln1 = nn.LayerNorm(attn_dim) # TODO: make sure this is correct

        self.dense = nn.Linear(attn_dim, nn_dim)
        self.resample2 = nn.Conv1d(attn_dim, nn_dim, 1)
        self.ln2 = nn.LayerNorm(nn_dim)

    def forward(self, x, padding_mask=None, mask=None):
        x = self.ln0(x)
        res = self.drop(self.resample1(x.transpose(1, 2)).transpose(1, 2))
        x = x.transpose(0, 1)
        x, _ = self.attn(x, x, x, key_padding_mask=padding_mask, attn_mask=mask)
        x = x.transpose(0, 1)
        x = self.ln1(x + res)
        res = self.drop(self.resample2(x.transpose(1, 2)).transpose(1, 2))
        x = gelu(self.dense(x))
        x = self.ln2(x + res)

        return x

class GPTModel(nn.Module):
    def __init__(self, attn_dim, num_heads, nn_dim, n_blocks, vocab_size,
                 context_size=512, dropout=0.0, scale_res=False, block_norm=False,
                 tied_weights=True, device='cuda'):
        super().__init__()
        self.device = device
        self.n = n_blocks
        self.scale_res = scale_res

        self.embed = nn.Embedding(vocab_size, nn_dim)
        self.pos_embed = nn.Parameter(torch.empty(context_size, nn_dim).normal_(0.0, 0.02))
        self.drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([Transformer(nn_dim, attn_dim, num_heads, nn_dim, dropout, block_norm)
                                     for _ in range(n_blocks)])

        for block in self.blocks:
            block.apply(self.__gpt_weight_init)

        self.lnf = nn.LayerNorm(nn_dim)

        self.out = nn.Linear(nn_dim, vocab_size)
        if tied_weights:
            self.out.weights = self.embed.weight

    def __gpt_weight_init(self, model):
        if type(model) == nn.Linear:
            nn.init.normal_(model.weight, 0.0, 0.02)
            model.bias.data.fill_(0.01)
        elif type(model) == nn.MultiheadAttention:
            nn.init.normal_(model.in_proj_weight, 0.0, 0.02)
            model.in_proj_bias.data.fill_(0.01)
            nn.init.normal_(model.out_proj.weight, 0.0, 0.02)
            model.out_proj.bias.data.fill_(0.01)

        # scale residual layer weights
        if type(model) == nn.Conv1d and self.scale_res:
            scale = 1. / (self.n ** 0.5)
            model.weight.data *= scale
            model.bias.data *= scale

    def forward(self, x, mask=None, pad_key=None):
        if pad_key is not None:
            pad_key = (x == pad_key).to(self.device)

        if mask is not None:
            seq_len = x.size(1)
            mask = torch.tril(torch.ones(seq_len, seq_len)).to(self.device)

        x = self.drop(self.embed(x) + self.pos_embed)
        for block in self.blocks:
            x = block(x, padding_mask=pad_key, mask=mask)
        x = self.lnf(x)
        x = self.out(x)

        return x

