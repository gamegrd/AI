import torch
import torch.nn as nn
import torch.nn.functional as func


class Mish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * (torch.tanh(func.softplus(x)))
