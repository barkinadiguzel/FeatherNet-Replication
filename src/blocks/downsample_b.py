import torch
import torch.nn as nn

class DownsampleB(nn.Module):
    def __init__(self, inp, oup):
        super().__init__()

        self.branch1 = nn.Sequential(
            nn.Conv2d(inp, oup, 3, stride=2, padding=1,
                      groups=inp, bias=False),
            nn.BatchNorm2d(oup)
        )

        self.branch2 = nn.Sequential(
            nn.AvgPool2d(2, stride=2),
            nn.Conv2d(inp, oup, 1, bias=False),
            nn.BatchNorm2d(oup)
        )

        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.branch1(x) + self.branch2(x))
