import torch
from torch import nn
from torch.functional as F

class Residual(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, use_1x1conv: bool = False, stride=1
    ):
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
            stride=stride,
        )
        self.conv2 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
            stride=stride,
        )
        self.conv3 = None
        if use_1x1conv:
            self.conv3 = nn.Conv2d(out_channels,out_channels,kernel_size=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
    def forward(self,X):
        Y =F.relu(self.bn1(self.conv1(X)))
        Y =self.bn2(self.conv2(Y))
        if(self.conv3):
            X = self.conv3(X)
        Y+=X
        return F.relu(Y)
