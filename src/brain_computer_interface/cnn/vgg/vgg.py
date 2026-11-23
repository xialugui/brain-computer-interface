from torch import nn


def block(number_of_conv: int, input_channels: int, output_channels: int):
    layers = []
    for _ in range(number_of_conv):
        layers.append(
            nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1)
        )
        layers.append(nn.ReLU())
        input_channels = output_channels
    layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
    return nn.Sequential(*layers)


def vgg(conv_arch):
    conv_blks = []
    in_channels = 1
    # 卷积层部分
    for num_convs, out_channels in conv_arch:
        conv_blks.append(block(num_convs, in_channels, out_channels))
        in_channels = out_channels

    return nn.Sequential(
        *conv_blks,
        nn.Flatten(),
        # 全连接层部分
        nn.Linear(out_channels * 7 * 7, 4096),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(4096, 4096),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(4096, 10),
    )
