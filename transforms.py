import torch
from torch import nn
from torchaudio import transforms as T
import numpy as np

class MinMaxNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return (x - x.min()) / (x.max() - x.min())

def random_time_shift(spec, Tshift):
    deltat = int(np.random.uniform(low=0.0, high=Tshift))
    if deltat == 0:
        return spec
    return torch.roll(spec, shifts=deltat, dims=-1)

class TimeShift(nn.Module):
    def __init__(self, Tshift):
        super().__init__()
        self.Tshift = Tshift

    def forward(self, spec):
        return random_time_shift(spec, self.Tshift)