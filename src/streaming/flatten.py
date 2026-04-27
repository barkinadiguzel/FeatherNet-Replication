import torch

class FlattenStream:
    def forward(self, x):
        return x.view(x.size(0), -1)
