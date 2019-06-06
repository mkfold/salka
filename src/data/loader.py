import torch
from torch.utils.data import Dataset
from pickle import load
import mmap
import string

_BASE_VOCAB = { '<p>': 0, '<s>': 1, '</s>': 2 }
CHAR_VOCAB = { x for x in string.printable if ord(x) - 29 >= 0 } 

# apply rules for BPE
def _apply_rules(line, rules):
    for rule in rules:
        line = line.replace(' '.join(rule), ''.join(rule))
    return line

_WINDOW_SIZE = { 'D' : 86400, 'H' : 3600 }

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

class CSVDataset(Dataset):
    def __init__(self, data, vocab=CHAR_VOCAB, rewrites=None,
                 word_sep=None, pre_sep=',', ignore_cols=[0], meta_cols=[0,1],
                 **kwargs):
        super().__init__()

        if rewrites and not word_sep:
            raise ValueError

        self.data = []
        self.maxlen = 0
        self.vocab = {**_BASE_VOCAB,
                      **{ x : i + len(_BASE_VOCAB) for i, x in enumerate(vocab)}}

        self.pad_idx = self.vocab['<p>']
        self.sos_idx = self.vocab['<s>']
        self.eos_idx = self.vocab['</s>']

        self.vocab_size = len(self.vocab) + 1
        for line_no, line in data:
            # remove ignored columns
            meta = ''
            if len(ignore_cols) != 0 or len(meta_cols) != 0:
                line = line.split(pre_sep)
                meta = ','.join([line[x] for x in meta_cols])
                line = [x for i, x in enumerate(line) if i not in ignore_cols]
                line = pre_sep.join(line)

            # apply BPE rewrite rules
            if rewrites is not None:
                line = _apply_rules('<s> ' + ' '.join(line) + ' </s>', rewrites)

            # split by word/BPE delimiter or by characters
            if word_sep is not None:
                line = line.split(word_sep)
            else:
                line = ['<s>'] + list(line) + ['</s>']

            line = [self.vocab[x] if x in self.vocab else self.vocab_size for x in line]
            line_len = len(line)
            self.maxlen = max(self.maxlen, line_len)
            self.data += [(line_no, meta, line)]

        self.length = len(self.data)

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        seq = self.data[i][2]
        pad = [self.pad_idx] * (self.maxlen - len(seq))
        seq_tensor = torch.tensor(seq + pad, dtype=torch.long)
        seq_len = torch.tensor(len(seq), dtype=torch.long)

        return self.data[i][0], self.data[i][1], seq_tensor, seq_len


