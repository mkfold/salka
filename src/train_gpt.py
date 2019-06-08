import gzip
from os.path import basename

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import click
from bpe import Encoder

from utils.config import load_config
from utils.training import opt_map
from data.loader import CSVDataset, TimeBufferedCSVReader
from models.transformer import GPTModel

# TODO: add evaluation fcn to group
@click.group()
def cli():
    pass

@click.command()
@click.argument('data_file', type=str, required=True)
@click.argument('config', type=str, required=True)
@click.option('--blocks', type=int, default=1)
@click.option('--attn_dim', type=int, default=32)
@click.option('--num_heads', type=int, default=1)
@click.option('--nn_dim', type=int, default=32)
@click.option('--scale_residuals', is_flag=True)
@click.option('--block_norm', is_flag=True)
@click.option('--dropout', type=float, default=0.4)
@click.option('--tied_weights', is_flag=True)
@click.option('--optimizer', type=str, default='adam') # TODO: choice type
@click.option('--lr', type=float, default=2.5e-4)
@click.option('--mb', type=int, default=32)
@click.option('--cpu', is_flag=True)
def train(data_file, config, blocks, attn_dim, num_heads, nn_dim, dropout, tied_weights,
          optimizer, lr, mb, scale_residuals, block_norm, cpu):
    config = load_config(config)

    context_size = config['dataset']['maxlen']

    if 'vocab' in config['dataset']:
        vocab = Encoder.load(config['dataset']['vocab'])
        config['dataset']['vocab'] = vocab
        vocab_size = config['dataset']['vocab'].vocab_size
        pad_idx = vocab.word_vocab[vocab.PAD]
    else:
        vocab_size = 255
        pad_idx = 0

    window_batches = TimeBufferedCSVReader(data_file, **config['reader'])

    device = torch.device('cuda' if torch.cuda.is_available() and not cpu else 'cpu')

    model = GPTModel(attn_dim, num_heads, nn_dim, blocks, vocab_size, context_size,
                     dropout=dropout, scale_res=scale_residuals, block_norm=block_norm,
                     tied_weights=tied_weights, device=device).to(device)

    opt = opt_map[optimizer](model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(reduction='none', ignore_index=pad_idx)

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
            _, _, seqs, _ = b
            x = seqs[:, :-1].to(device)
            y = seqs[:, 1:].to(device)
            y_mask = (y != pad_idx).float().unsqueeze(2).to(device)

            preds = model(x, mask=True, pad_key=pad_idx)

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
            line_nums, meta, seqs, _ = b
            x = seqs[:, :-1].to(device)
            y = seqs[:, 1:].to(device)
            y_mask = (y != pad_idx).float().unsqueeze(2).to(device)

            preds = model(x, mask=True, pad_key=pad_idx)

            loss = criterion(preds.transpose(1, 2), y)
            loss = loss.sum(dim=1) / y_mask.sum(dim=1).squeeze()

            for line_no, line_meta, line_score in zip(line_nums, meta, loss):
                scores.write(f'{line_no},{line_meta},{line_score}\n')

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

