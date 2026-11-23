from torch import nn


class LeNet(nn.Module):
    def __init__(self):
        self.super.__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(1, 6, (5, 5), padding=2),
            nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, (5, 5)),
            nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(16 * 5 * 5, 120),
            nn.Sigmoid(),
            nn.Linear(120, 84),
            nn.Sigmoid(),
            nn.Linear(84, 10),
        )
