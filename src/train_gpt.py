import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from models.data import CharDataset
from models.transformer import GPTModel

from tqdm import tqdm

import click

_OPTS = {'adam' : optim.Adam, 'sgd' : optim.SGD, 'rms' : optim.RMSprop}

# TODO: add evaluation fcn to group
@click.group()
def cli():
    pass

@click.command()
@click.argument('data_file', type=str, required=True)
@click.option('--blocks', type=int, default=1)
@click.option('--attn_dim', type=int, default=32)
@click.option('--num_heads', type=int, default=1)
@click.option('--nn_dim', type=int, default=32)
@click.option('--scale_residuals', is_flag=True)
@click.option('--block_norm', is_flag=True)
#@click.option('--embedding_dim', type=int, default=16)
@click.option('--dropout', type=float, default=0.4)
@click.option('--tied_weights', is_flag=True)
@click.option('--optimizer', type=str, default='adam') # TODO: choice type
@click.option('--lr', type=float, default=2.5e-4)
@click.option('--mb', type=int, default=32)
@click.option('--num_epochs', type=int, default=100)
def train(data_file, blocks, attn_dim, num_heads, nn_dim, dropout, tied_weights,
          optimizer, lr, mb, num_epochs, scale_residuals, block_norm):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = CharDataset(data_file)
    context_size = dataset.maxlen + 1
    data = DataLoader(dataset, shuffle=True, batch_size=mb, num_workers=8)

    model = GPTModel(attn_dim, num_heads, nn_dim, blocks, dataset.vocab_size, context_size,
                     dropout=dropout, scale_res=scale_residuals, block_norm=block_norm,
                     tied_weights=tied_weights, device=device)
    model = model.to(device)

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
            x = seqs[:, :-1].to(device)
            y = seqs[:, 1:].to(device)
            y_mask = (y != 0).float().unsqueeze(2).to(device)

            preds = model(x, mask=True, pad_key=0)

            loss = criterion(preds.transpose(1, 2), y)
            loss.backward()

            opt.step()

            avg_loss += loss.cpu().item()
            batches += 1
            train_iter.set_description(f'batch loss: {avg_loss / batches:.4f}')

        print(avg_loss / len(data))

cli.add_command(train)

if __name__ == '__main__':
    cli()
