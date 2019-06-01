import torch
from torch.utils.data import Dataset

class CharDataset(Dataset):
    def __init__(self, filename, labels=None):
        super().__init__()

        self.data = []
        self.maxlen = 0
        with open(filename, 'r') as f:
            for line in f:
                line = [ord(x) - 29 for x in line.strip()]
                line_len = len(line) + 2
                self.maxlen = max(self.maxlen, line_len)
                self.data += [line]

        self.vocab_size = 89

        self.length = len(self.data)

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        seq = self.data[i][:-1]
        pad = [0] * (self.maxlen - len(seq))
        seq_tensor = torch.tensor([1] + seq + [2] + pad, dtype=torch.long)
        seq_len = torch.tensor(len(seq) + 2, dtype=torch.long)

        return seq_tensor, seq_len

