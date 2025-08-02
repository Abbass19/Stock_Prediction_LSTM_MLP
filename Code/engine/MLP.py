import torch
from torch import nn as nn


class SimpleBaseLine(nn.Module):
    def __init__(self, hidden_size: int, input_feature: int):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(input_feature, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size//2),
            nn.ReLU(),
            nn.Linear(hidden_size//2, 1)
        )

    def forward(self,  x: torch.Tensor):
        return self.model(x)




class PyramidStyle(nn.Module):
    def __init__(self, hidden_size: int, input_feature: int):
        super().__init__()
        self.module= nn.Sequential(
            nn.Linear(input_feature, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size//2),
            nn.ReLU(),
            nn.Linear(hidden_size//2, hidden_size//4),
            nn.ReLU(),
            nn.Linear(hidden_size//4, 1)
        )
    def forward(self, x: torch.Tensor):
        return self.module(x)





class WideShallow(nn.Module):
    def __init__(self, hidden_size: int, input_feature: int):
        super().__init__()
        self.module= nn.Sequential(
            nn.Linear(input_feature, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, x: torch.Tensor):
        return self.module(x)





class RegularizedDeep(nn.Module):
    def __init__(self, hidden_size: int, input_feature: int):
        super().__init__()
        self.module= nn.Sequential(
            nn.Linear(input_feature, hidden_size),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(hidden_size, hidden_size//2),
            nn.ReLU(),
            nn.Linear(hidden_size//2, hidden_size//4),
            nn.ReLU(),
            nn.Linear( hidden_size//4, 1)
        )
    def forward(self, x: torch.Tensor):
        return self.module(x)

