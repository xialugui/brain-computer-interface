import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from braindecode.modules import (
    Ensure4d,
)


class ShallowConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.ensuredims = Ensure4d()
        self.dimshuffle =  Rearrange("batch C T 1 -> batch 1 C T")
        self.temporal_filter = nn.Conv2d(1, 40, (1, 25))
        self.spatial_filter = nn.Conv2d(40, 40, (22, 1))
        self.mean_pooling = nn.AvgPool2d((1, 75), stride=(1, 15))
        self.line_classification = nn.Linear(2760, 4)

    def forward(self, x):
        print("Input shape:", x.shape)
        x = self.ensuredims(x)
        x = self.dimshuffle(x)
        x = self.temporal_filter(x)
        x = self.spatial_filter(x)
        x = torch.square(x)
        x = self.mean_pooling(x)
        x = torch.log1p(x)
        x = torch.flatten(x, 1)
        return self.line_classification(x)
