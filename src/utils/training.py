import torch
from torch import nn, optim

opt_map = {'adam' : optim.Adam, 'sgd' : optim.SGD, 'rms' : optim.RMSprop}

