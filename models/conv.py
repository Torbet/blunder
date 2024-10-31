import torch
import torch.nn as nn
from models.dense import Dense1, Dense3, Dense6


class Conv1(nn.Module):
    def __init__(self):
        super(Conv1, self).__init__()
        self.conv1 = nn.Conv3d(6, 64, kernel_size=(5, 3, 3), stride=(2, 1, 1))
        self.conv2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), stride=(2, 1, 1))
        self.dense1 = Dense1(16384)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1, 3, 4)
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        x = self.dense1(x)
        return x


class Conv3(nn.Module):
    def __init__(self):
        super(Conv3, self).__init__()
        self.conv1 = nn.Conv3d(6, 64, kernel_size=(5, 3, 3), stride=(2, 1, 1))
        self.conv2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), stride=(2, 1, 1))
        self.dense3 = Dense3(16384)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1, 3, 4)
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        x = self.dense3(x)
        return x


class Conv6(nn.Module):
    def __init__(self):
        super(Conv6, self).__init__()
        self.conv1 = nn.Conv3d(6, 64, kernel_size=(5, 3, 3), stride=(2, 1, 1))
        self.conv2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), stride=(2, 1, 1))
        self.dense6 = Dense6(16384)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1, 3, 4)
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        x = self.dense6(x)
        return x
