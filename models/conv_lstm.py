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
        self.l1 = nn.Linear(256, 4)

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
        return x


class ConvLSTMExtra(nn.Module):
    def __init__(self, bidirectional: bool = False):
        super(ConvLSTMExtra, self).__init__()
        self.conv1 = nn.Conv2d(6, 64, kernel_size=3)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3)
        self.lstm = nn.LSTM(1026, 512, batch_first=True, bidirectional=bidirectional)
        lstm_out = 512 * (2 if bidirectional else 1)
        self.l1 = nn.Linear(lstm_out, 4)

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
        return x


class ConvLSTMExtra2(nn.Module):
    def __init__(self, bidirectional: bool = False):
        super(ConvLSTMExtra2, self).__init__()
        self.conv1 = nn.Conv2d(6, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(2, 2)
        self.lstm = nn.LSTM(130, 256, batch_first=True, bidirectional=bidirectional)
        lstm_out = 256 * (2 if bidirectional else 1)
        self.l1 = nn.Linear(lstm_out, 128)
        self.dropout = nn.Dropout(0.5)
        self.l2 = nn.Linear(128, 4)

    def forward(
        self, moves: torch.Tensor, evals: torch.Tensor, times: torch.Tensor
    ) -> torch.Tensor:
        bs, ts, C, H, W = moves.size()
        out = []
        for i in range(ts):
            xt = moves[:, i]
            xt = self.pool(F.relu(self.bn1(self.conv1(xt))))
            xt = self.pool(F.relu(self.bn2(self.conv2(xt))))
            xt = self.pool(F.relu(self.bn3(self.conv3(xt))))
            xt = xt.flatten(1)
            xt = torch.cat((xt, evals[:, i : i + 1], times[:, i : i + 1]), dim=1)
            out.append(xt)

        x = torch.stack(out, dim=1)
        x, _ = self.lstm(x)
        x = x[:, -1]
        x = F.relu(self.l1(x))
        x = self.dropout(x)
        x = self.l2(x)
        return x
