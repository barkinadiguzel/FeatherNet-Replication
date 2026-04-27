import torch
import torch.nn as nn

class DWConvStream(nn.Module):
    def __init__(self, channels, kernel_size=3, stride=2):
        super().__init__()

        self.dwconv = nn.Conv2d(
            channels, channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            groups=channels,
            bias=False
        )

    def forward(self, x):
        return self.dwconv(x)
