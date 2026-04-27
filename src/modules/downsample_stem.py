import torch.nn as nn

class Stem(nn.Module):
    def __init__(self, inp=3, oup=32):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(inp, oup, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(oup),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)
