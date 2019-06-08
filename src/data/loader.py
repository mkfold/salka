from pickle import load
import mmap
import string

import torch
from torch.utils.data import Dataset
from bpe import Encoder
from bpe.encoder import DEFAULT_SOW, DEFAULT_EOW

_BASE_VOCAB = { '<pad>': 0, '<s>': 1, '</s>': 2 }
CHAR_VOCAB = { x for x in string.printable if ord(x) - 29 >= 0 } 

_WINDOW_SIZE = { 'D' : 86400, 'H' : 3600, 'M' : 60 }

class TimeBufferedCSVReader(object):
    def __init__(self, filename, sep=',', resolution='H', multiplier=1, skiprows=1,
                 skipdays=[], **kwargs):
        self.sep = sep
        self.window = multiplier * _WINDOW_SIZE[resolution]
        self.cur_time = 0
        self.cur_day = 0
        self.cur_window = 0
        self.cur_line = 0
        self.skipdays = set(skipdays)
        self.skiprows = skiprows
        self.file_ = open(filename, 'r+b')
        self.mmap_ = mmap.mmap(self.file_.fileno(), 0, prot=mmap.PROT_READ)
        for _ in range(self.skiprows): self.file_.readline()

    def reset(self):
        self.cur_time = 0
        self.file_.seek(0)
        for _ in range(self.skiprows): self.file_.readline(); self.cur_line += 1

    def __iter__(self):
        return self

    # FIXME: cur_window returns -1 for last buffer
    def __next__(self):
        if self.cur_time == -1:
            raise StopIteration

        while self.cur_day in self.skipdays:
            line = str(self.file_.readline(), 'utf-8')
            if line == '':
                self.cur_time = -1
                break
            self.cur_time = int(line.partition(self.sep)[0])
            self.cur_day = self.cur_time // 86400
            self.cur_line += 1

        window_end = self.cur_time + self.window

        data = []
        while self.cur_time < window_end:
            line = str(self.file_.readline(), 'utf-8').strip()
            if line == '':
                self.cur_time = -1
                break
            self.cur_time = int(line.partition(self.sep)[0])
            self.cur_day = self.cur_time // 86400
            data += [(self.cur_line, line)]
            self.cur_line += 1
        self.cur_window = self.cur_time // self.window

        return data

# apply rules for BPE
def _apply_rules(line, enc):
    return enc.transform(line)

class CSVDataset(Dataset):
    def __init__(self, data, vocab=CHAR_VOCAB, maxlen=None, rewrites=None,
                 word_sep=None, sep=',', ignore_cols=[0], meta_cols=[0,1],
                 **kwargs):
        super().__init__()

        if rewrites and not word_sep:
            raise ValueError

        self.data = []
        self.maxlen = 0
        if type(vocab) == Encoder:
            self.vocab = vocab
            self.pad_idx = self.vocab.word_vocab['<pad>']
            self.sos_idx = self.vocab.word_vocab['<s>']
            self.eos_idx = self.vocab.word_vocab['</s>']
            if self.vocab.pct_bpe != 0.:
                self.bound_idx = {self.vocab.bpe_vocab[DEFAULT_SOW],
                                  self.vocab.bpe_vocab[DEFAULT_EOW]}
            else:
                self.bound_idx = set()
            self.vocab_size = self.vocab.vocab_size
            new_sep = f' {sep} ' if self.vocab.pct_bpe != 0. else ' '
            self.transform = lambda x: x.replace(' ', '_').replace(sep, new_sep)
        else:
            self.vocab = {**_BASE_VOCAB,
                          **{ x : i + len(_BASE_VOCAB) for i, x in enumerate(vocab)}}
            self.pad_idx = self.vocab['<pad>']
            self.sos_idx = self.vocab['<s>']
            self.eos_idx = self.vocab['</s>']
            self.bound_idx = set()
            self.vocab_size = len(self.vocab) + 1
            self.transform = lambda x: x

        for line_no, line in data:
            # remove ignored columns
            meta = ''
            if len(ignore_cols) != 0 or len(meta_cols) != 0:
                line = line.split(sep)
                meta = ','.join([line[x] for x in meta_cols])
                line = [x for i, x in enumerate(line) if i not in ignore_cols]
                line = sep.join(line)

            line = self.transform(line)

            # apply rewrite rules
            if type(self.vocab) == Encoder:
                line = next(_apply_rules(['<s> ' + line + ' </s>'], self.vocab))
                line = [x for x in line if x not in self.bound_idx]
            else:
                line = ['<s>'] + list(line) + ['</s>']
                line = [self.vocab[x] if x in self.vocab else self.vocab_size for x in line]

            line_len = len(line)
            self.maxlen = max(self.maxlen, line_len)
            self.data += [(line_no, meta, line)]

        if maxlen is not None: self.maxlen = maxlen + 1
        self.length = len(self.data)

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        seq = self.data[i][2]
        if len(seq) < self.maxlen:
            seq += [self.pad_idx] * (self.maxlen - len(seq))
        elif len(seq) > self.maxlen:
            seq = seq[:self.maxlen]
        seq_tensor = torch.tensor(seq, dtype=torch.long)
        seq_len = torch.tensor(len(seq), dtype=torch.long)

        return self.data[i][0], self.data[i][1], seq_tensor, seq_len


