import torch
import torch.nn as nn
import torch.nn.functional as F


class Dense1(nn.Module):
    def __init__(self, in_features: int = 15360):
        super(Dense1, self).__init__()
        self.l1 = nn.Linear(in_features, 512)
        self.l2 = nn.Linear(512, 4)

    def forward(self, x: torch.Tensor, *args: list[torch.Tensor]) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        x = F.relu(self.l1(x))
        x = self.l2(x)
        return x


class Dense3(nn.Module):
    def __init__(self, in_features: int = 15360):
        super(Dense3, self).__init__()
        self.l1 = nn.Linear(in_features, 512)
        self.l2 = nn.Linear(512, 512)
        self.l3 = nn.Linear(512, 64)
        self.l4 = nn.Linear(64, 4)

    def forward(self, x: torch.Tensor, *args: list[torch.Tensor]) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = self.l4(x)
        return x


class Dense6(nn.Module):
    def __init__(self, in_features: int = 15360):
        super(Dense6, self).__init__()
        self.l1 = nn.Linear(in_features, 2048)
        self.l2 = nn.Linear(2048, 2048)
        self.l3 = nn.Linear(2048, 512)
        self.l4 = nn.Linear(512, 512)
        self.l5 = nn.Linear(512, 128)
        self.l6 = nn.Linear(128, 128)
        self.l7 = nn.Linear(128, 4)

    def forward(self, x: torch.Tensor, *args: list[torch.Tensor]) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = F.relu(self.l4(x))
        x = F.relu(self.l5(x))
        x = F.relu(self.l6(x))
        x = self.l7(x)
        return x
