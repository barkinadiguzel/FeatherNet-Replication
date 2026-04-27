import torch
import torch.nn as nn

class DownsampleB(nn.Module):
    def __init__(self, inp, oup):
        super().__init__()

        self.branch1 = nn.Sequential(
            nn.Conv2d(inp, inp, 3, stride=2, padding=1, groups=inp, bias=False),
            nn.BatchNorm2d(inp)
        )
        self.branch2 = nn.Sequential(
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(inp, oup, 1, bias=False),
            nn.BatchNorm2d(oup)
        )

        self.bn = nn.BatchNorm2d(oup)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x1 = self.branch1(x)
        x2 = self.branch2(x)

        x = x1 + x2

        return self.relu(self.bn(x))
