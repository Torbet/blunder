import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv3D(nn.Module):
    def __init__(self):
        super(Conv3D, self).__init__()
        self.conv1 = nn.Conv3d(40, 64, kernel_size=(3, 3, 3))
        self.conv2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3))
        self.pool = nn.MaxPool3d(kernel_size=(2, 2, 2))
        self.l1 = nn.Linear(512, 128)
        self.l2 = nn.Linear(128, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.l1(x))
        x = self.l2(x)
        return F.sigmoid(x)
