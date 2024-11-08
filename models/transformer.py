import torch
import torch.nn as nn
from torchvision.models.video import swin3d_t


class Swin3D(nn.Module):
    def __init__(self):
        super(Swin3D, self).__init__()
        self.model = swin3d_t()
        conv_layer = self.model.patch_embed.proj
        self.model.patch_embed.proj = nn.Conv3d(
            6,
            conv_layer.out_channels,
            kernel_size=conv_layer.kernel_size,
            stride=conv_layer.stride,
            padding=conv_layer.padding,
        )
        self.model.head = nn.Linear(self.model.head.in_features, 4)

    def forward(self, x: torch.Tensor, *args: list[torch.Tensor]) -> torch.Tensor:
        x = x.permute(0, 2, 1, 3, 4)
        return self.model(x)
