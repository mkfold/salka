import click
from os.path import split, join
from bpe import Encoder

_BPE, _WORD = (0, 'bpe'), (1, 'word')

@click.command()
@click.argument('filename', type=str, required=True)
@click.option('--vocab_size', type=int, default=256,
              help='Size of generated vocabulary.')
@click.option('--ngram_max', type=int, default=16,
              help='Maximum length of byte-pair encodings.')
@click.option('--pct_bpe', type=float, default=1.0,
              help='Percentage of vocabulary comprised of byte-pair encodings.')
@click.option('--sep', type=str, default=',',
              help='Delimiter used by input.\
                    Removed if word encodings used, padded with spaces for BPE.')
@click.option('--ignore_cols', type=set, default={0},
              help='Column(s) to ignore when generating vocabulary.')
@click.option('-v', is_flag=True, help='Show encoding progress.')
def make_vocab(filename, vocab_size, ngram_max, pct_bpe, sep, ignore_cols, v):
    '''
    Creates word or byte-pair encoding vocabulary and mappings from a sample of
    text. Because this script will load the entire input text into memory, for
    large corpora it is recommended to use a representative sample of text.

    Vocabulary will be saved in a JSON file with the same base name as the input
    file, suffixed with "_word" or "_bpe" depending on the encoding used.
    '''

    kind = _BPE if pct_bpe else _WORD

    with open(filename, 'r') as f:
        sample = f.readlines()

    new_sep = f' {sep} ' if kind == _BPE else ' '
    sample = ['<s> ' + x.replace(' ', '_').replace(sep, new_sep) + '</s>' \
                  for i, x in enumerate(sample) if i not in ignore_cols]

    enc = Encoder(vocab_size, pct_bpe=pct_bpe, silent=not v, ngram_max=ngram_max,
                  required_tokens={'<s>', '</s>'},
                  PAD='<pad>', UNK='<unk>')
    enc.fit(sample)
    enc.vocab_size = len(enc.word_vocab) + len(enc.bpe_vocab)
    enc.mute()
    dir_, name = split(filename)
    enc.save(join(dir_, name.split('.')[0] + f'_{kind[1]}.json'))

if __name__ == '__main__':
    make_vocab()

