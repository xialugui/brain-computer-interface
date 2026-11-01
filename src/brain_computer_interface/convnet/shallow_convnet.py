import torch
import torch.nn as nn
import torch.nn.functional as F


class ShallowConvNet(nn.Module):
    def __init__(self):
        self.temporal = nn.Conv2d(1, 40, (25, 1))
        self.spatial_filter = nn.Conv2d(40, 40, (1, 44))
        self.mean_pooling = nn.AvgPool2d((75, 1), stride=(15, 1))
        self.line_classification = nn.Linear(1200, 4)

    def forward(self, x):
        x = self.temporal(x)
        x = self.spatial_filter(x)
        x = torch.square(x)
        x = self.mean_pooling(x)
        x = torch.log1p(x)
        x = torch.flatten(x, 1)
        return self.line_classification(x)
