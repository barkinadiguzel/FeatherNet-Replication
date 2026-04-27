import torch.nn as nn
from blocks.inverted_residual import InvertedResidual
from blocks.downsample_b import DownsampleB
from blocks.downsample_c import DownsampleC
from blocks.se_module import SEModule

class FeatherStage(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        layers = []

        for t, c, n, block_type in cfg:
            for i in range(n):
                if block_type == "A":
                    layers.append(InvertedResidual(c, c, 1, t))
                elif block_type == "B":
                    layers.append(DownsampleB(c, c))
                elif block_type == "C":
                    layers.append(DownsampleC(c, c))

                layers.append(SEModule(c))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
