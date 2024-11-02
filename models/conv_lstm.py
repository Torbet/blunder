import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvLSTM(nn.Module):
    def __init__(self):
        super(ConvLSTM, self).__init__()
        self.conv1 = nn.Conv2d(6, 64, kernel_size=3)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3)
        self.lstm = nn.LSTM(1024, 256, batch_first=True)
        self.l1 = nn.Linear(256, 2)

    def forward(self, x: torch.Tensor, *args: list[torch.Tensor]) -> torch.Tensor:
        bs, ts, C, H, W = x.size()
        out = []
        for i in range(ts):
            xt = x[:, i]
            xt = F.relu(self.conv1(xt))
            xt = F.relu(self.conv2(xt))
            xt = F.relu(self.conv3(xt))
            out.append(xt.flatten(1))
        x = torch.stack(out, dim=1)
        x, _ = self.lstm(x)
        x = x[:, -1]
        x = self.l1(x)
        x = F.sigmoid(x)
        return x


class ConvLSTMExtra(nn.Module):
    def __init__(self):
        super(ConvLSTMExtra, self).__init__()
        self.conv1 = nn.Conv2d(6, 64, kernel_size=3)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3)
        self.lstm = nn.LSTM(1026, 256, batch_first=True)
        self.l1 = nn.Linear(256, 2)

    def forward(
        self, board: torch.Tensor, evals: torch.Tensor, times: torch.Tensor
    ) -> torch.Tensor:
        bs, ts, C, H, W = board.size()
        out = []
        for i in range(ts):
            xt = board[:, i]
            xt = F.relu(self.conv1(xt))
            xt = F.relu(self.conv2(xt))
            xt = F.relu(self.conv3(xt))
            xt = xt.flatten(1)
            xt = torch.cat((xt, evals[:, i : i + 1], times[:, i : i + 1]), dim=1)
            out.append(xt)

        x = torch.stack(out, dim=1)
        x, _ = self.lstm(x)
        x = x[:, -1]
        x = self.l1(x)
        x = F.sigmoid(x)
        return x
