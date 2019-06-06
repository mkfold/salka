import gzip
from os.path import basename

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import click

from utils.config import load_config
from data.loader import CSVDataset, TimeBufferedCSVReader
from models.rnn import RNNLanguageModel

_OPTS = {'adam' : optim.Adam, 'sgd' : optim.SGD}

# TODO: add evaluation fcn to group
@click.group()
def cli():
    pass

@click.command()
@click.argument('data_file', type=str, required=True)
@click.argument('config', type=str, required=True)
@click.option('--layers', type=int, default=1)
@click.option('--hidden_dim', type=int, default=32)
@click.option('--dropout', type=float, default=0.4)
@click.option('--rnn_cell', type=str, default='gru')
@click.option('--embedding_dim', type=int, default=16)
@click.option('--tied_weights', is_flag=True)
@click.option('--bidir', is_flag=True)
@click.option('--residual', is_flag=True)
@click.option('--optimizer', type=str, default='adam') # TODO: choice type
@click.option('--lr', type=float, default=0.01)
@click.option('--mb', type=int, default=256)
@click.option('--num_epochs', type=int, default=100)
@click.option('--cpu', is_flag=True)
def train(data_file, config, layers, hidden_dim, dropout, rnn_cell, embedding_dim,
          tied_weights, bidir, residual, optimizer, lr, mb, num_epochs, cpu):
    config = load_config(config)
    window_batches = TimeBufferedCSVReader(data_file, **config['reader'])

    device = torch.device('cuda' if torch.cuda.is_available() and not cpu else 'cpu')
    model = RNNLanguageModel(embedding_dim, 255, hidden_dim, layers, rnn_cell,
                             dropout=dropout, residual=residual, bidir=bidir,
                             tied_weights=tied_weights).to(device)

    opt = _OPTS[optimizer](model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(reduction='none', ignore_index=0)

    scores = gzip.open(basename(data_file) + '.scores.gz', 'wt')

    train_window = CSVDataset(next(window_batches), **config['dataset'])
    prev_window = window_batches.cur_window
    for eval_window in window_batches:
        eval_window = CSVDataset(eval_window, **config['dataset'])
        train_data = DataLoader(train_window, shuffle=False, batch_size=mb, num_workers=8)
        eval_data = DataLoader(eval_window, shuffle=False, batch_size=mb, num_workers=8)

        cur_window = window_batches.cur_window

        # train on window
        model.train()
        avg_train_loss = 0.
        batches = 0
        train_iter = tqdm(train_data)
        for b in train_iter:
            opt.zero_grad()
            _, _, seqs, lens = b
            x = seqs[:, :-1].to(device)
            y = seqs[:, 1:].to(device)
            y_mask = (y != 0).float().unsqueeze(2).to(device)

            preds = model(x, lens - 1)

            loss = criterion(preds.transpose(1, 2), y)
            loss = loss.sum(dim=1) / y_mask.sum(dim=1).squeeze()

            loss = loss.mean()
            loss.backward()

            opt.step()

            avg_train_loss += loss.cpu().item()
            batches += 1
            train_iter.set_description(f'[TRAIN] window={prev_window} loss={avg_train_loss / batches:.8f}')

        # evaluate on next window
        model.eval()
        avg_eval_loss = 0.
        batches = 0
        eval_iter = tqdm(eval_data)
        for b in eval_iter:
            line_nums, meta, seqs, lens = b
            x = seqs[:, :-1].to(device)
            y = seqs[:, 1:].to(device)
            y_mask = (y != 0).float().unsqueeze(2).to(device)

            preds = model(x, lens - 1)

            loss = criterion(preds.transpose(1, 2), y)
            loss = loss.sum(dim=1) / y_mask.sum(dim=1).squeeze()

            for line, user, score in zip(line_nums, meta, loss):
                scores.write(f'{line},{user},{score}\n')

            loss = loss.mean()

            avg_eval_loss += loss.cpu().item()
            batches += 1
            eval_iter.set_description(f'[EVAL]  window={cur_window} loss={avg_eval_loss / batches:.8f}')

        train_window = eval_window
        prev_window = cur_window

    scores.close()

cli.add_command(train)

if __name__ == '__main__':
    cli()

