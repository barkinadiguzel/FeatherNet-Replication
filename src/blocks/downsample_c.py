import torch
import torch.nn as nn

class DownsampleC(nn.Module):
    def __init__(self, inp, oup):
        super().__init__()

        self.op = nn.Sequential(
            nn.Conv2d(inp, oup, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(oup),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.op(x)
