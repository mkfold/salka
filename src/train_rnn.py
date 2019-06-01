import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from models.data import CharDataset
from models.rnn import RNNLanguageModel

from tqdm import tqdm

import click

_OPTS = {'adam' : optim.Adam, 'sgd' : optim.SGD}

# TODO: add evaluation fcn to group
@click.group()
def cli():
    pass

@click.command()
@click.argument('data_file', type=str, required=True)
@click.option('--layers', type=int, default=1)
@click.option('--hidden_dim', type=int, default=32)
@click.option('--dropout', type=float, default=0.4)
@click.option('--rnn_cell', type=str, default='gru')
@click.option('--embedding_dim', type=int, default=16)
@click.option('--tied_weights', type=bool, default=False)
@click.option('--bidir', type=bool, default=False)
@click.option('--residual', type=bool, default=False)
@click.option('--optimizer', type=str, default='adam') # TODO: choice type
@click.option('--lr', type=float, default=0.01)
@click.option('--mb', type=int, default=32)
@click.option('--num_epochs', type=int, default=100)
def train(data_file, layers, hidden_dim, dropout, rnn_cell, embedding_dim,
          tied_weights, bidir, residual, optimizer, lr, mb, num_epochs):
    dataset = CharDataset(data_file)
    data = DataLoader(dataset, shuffle=True, batch_size=mb, num_workers=8)

    model = RNNLanguageModel(embedding_dim, 255, hidden_dim, layers, rnn_cell,
                             dropout=dropout, residual=residual, bidir=bidir,
                             tied_weights=tied_weights)

    opt = _OPTS[optimizer](model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    for e in range(num_epochs):
        model.train()
        avg_loss = 0.
        batches = 0
        train_iter = tqdm(data)
        for b in train_iter:
            opt.zero_grad()
            seqs, lens = b
            x = seqs[:, :-1]
            y = seqs[:, 1:]
            y_mask = (y != 0).float().unsqueeze(2)

            preds = model(x, lens - 1)

            loss = criterion(preds.transpose(1, 2), y)
            loss.backward()

            opt.step()

            avg_loss += loss.item()
            batches += 1
            train_iter.set_description(f'batch loss: {avg_loss / batches:.4f}')

        print(avg_loss / len(data))

cli.add_command(train)

if __name__ == '__main__':
    cli()
