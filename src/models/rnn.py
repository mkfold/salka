# salka: rnn.py

import torch
from torch import nn

from torch.nn.utils.rnn import pack_padded_sequence

class RNNLanguageModel(nn.Module):
    def __init__(self, x_size, vocab_size, h_size, n_layers,
                 cell_type='gru', dropout=0, residual=False, bidir=False,
                 tied_weights=False):
        super().__init__()
        self.n_layers = n_layers
        self.h_size = h_size
        self.states = 2 if bidir else 1

        if cell_type == 'gru':
            cell = nn.GRU
        elif cell_type == 'lstm':
            cell = nn.LSTM
            self.states *= 2
        else:
            cell = nn.RNN

        self.embedding = nn.Embedding(vocab_size, x_size, padding_idx=0)

        self.layers = nn.ModuleList([cell(x_size, h_size, dropout=dropout,
                                          bidirectional=bidir, batch_first=True)])
        for i in range(n_layers - 1):
            self.layers += [cell(h_size, h_size, dropout=dropout,
                                 bidirectional=bidir, batch_first=True)]

        if tied_weights:
            assert x_size == (h_size * 2 if bidir else h_size)
            self.out = nn.Linear(x_size, vocab_size)
            self.out.weights = self.embedding.weight
        else:
            self.out = nn.Linear(h_size * 2 if bidir else h_size, vocab_size)

        if residual:
            self.residual = nn.ModuleList([nn.Conv1d(x_size, h_size, 1)])
            for i in range(n_layers - 1):
                self.residual += [nn.Conv1d(h_size, h_size, 1)]
            self.residual += [nn.Conv1d(h_size, vocab_size, 1)]
        else:
            self.residual = None

    def forward(self, x, lens):
        # assume that input is N x L x C
        zeros = self.__zeros(len(x))#.size(0))
        x = self.embedding(x)

        pack_x = pack_padded_sequence(x, lens, batch_first=True, enforce_sorted=False)

        for i in range(self.n_layers):
            res = self.residual[i](x.transpose(1, 2)).transpose(1, 2)\
                    if self.residual is not None else 0.
            x, _ = self.layers[i](x)#, zeros[i])
            x += res

        x = self.out(x) #+ (self.residual[i](x.transpose(1, 2)).transpose(1, 2)) \
                        #    if self.residual is not None else 0.

        return x

    def __zeros(self, batch_size):
        return torch.zeros(self.n_layers, self.states, batch_size, self.h_size)
