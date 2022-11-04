import time

import numpy as np
import torch
from torch import nn
from torch.backends import cudnn

FORCE_CPU = False

# If the following conditions are satisfied: 1) cudnn is enabled, 2) input data is on the GPU 3) input data has dtype
# torch.float16 4) V100 GPU is used, 5) input data is not in PackedSequence format
# persistent algorithm can be selected to improve performance.

# NOTICE: GPU processing requires pytorch CUDA installed via pip

def get_device():
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        cudnn.benchmark = True
    else:
        device = torch.device('cpu')
    return device


device = get_device()
if FORCE_CPU:
    device = torch.device('cpu')


class SingleOutRNN(nn.Module):
    r"""
    An RNN followed by a single output unit (sigmoid activation)
    """

    def __init__(self, input_size, hidden_size, num_layers, nonlinearity='tanh', bidirectional=False, sigmoid=False,
                 dtype=torch.float):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.n_direction = 2 if bidirectional else 1
        self.recurrent = nn.RNN(input_size, hidden_size, num_layers,
                                batch_first=True,
                                nonlinearity=nonlinearity, bidirectional=bidirectional)
        self.final = nn.Linear(hidden_size * self.n_direction, 1)
        self.sigmoid = sigmoid
        self.dtype = dtype
        # self.apply(init_parameters)

    def forward(self, data, hidden=None):
        if hidden is None:
            batch_size = data.shape[0]
            hidden = torch.zeros((self.num_layers * self.n_direction, batch_size, self.hidden_size),
                                 device=device, dtype=self.dtype)
        output, hidden_next = self.recurrent.forward(data, hidden)
        output = self.final.forward(output)
        if self.sigmoid:
            output = torch.sigmoid(output)
        return output, hidden_next

    def init_normal(self):
        for name, param in self.recurrent.named_parameters():
            if name.startswith("weight"):
                nn.init.normal_(param, std=0.1)
            else:
                param.data.fill_(0)
        for name, param in self.final.named_parameters():
            if name.startswith("weight"):
                nn.init.normal_(param, std=0.1)
            else:
                param.data.fill_(0)
