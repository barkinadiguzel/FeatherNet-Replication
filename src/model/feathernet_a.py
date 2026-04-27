import torch.nn as nn
from modules.downsample_stem import Stem
from modules.feather_stage import FeatherStage
from streaming.dwconv_stream import DWConvStream

class FeatherNetA(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.stem = Stem()
        self.stage = FeatherStage(cfg["A"])
        self.stream = DWConvStream(64)
        self.fc = nn.Linear(64 * 4 * 4, 2)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage(x)
        x = self.stream(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)
