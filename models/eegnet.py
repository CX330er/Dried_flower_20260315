import torch
from torch import nn


class EEGNet(nn.Module):
    """EEGNet (Lawhern et al.) style model."""

    def __init__(
        self,
        n_channels: int = 22,
        n_classes: int = 4,
        input_time: int = 1000,
        f1: int = 8,
        d: int = 2,
    ):
        super().__init__()
        f2 = f1 * d

        self.block1 = nn.Sequential(
            nn.Conv2d(1, f1, kernel_size=(1, 64), padding=(0, 32), bias=False),
            nn.BatchNorm2d(f1),
            nn.Conv2d(f1, f2, kernel_size=(n_channels, 1), groups=f1, bias=False),
            nn.BatchNorm2d(f2),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(0.5),
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(f2, f2, kernel_size=(1, 16), padding=(0, 8), groups=f2, bias=False),
            nn.Conv2d(f2, f2, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(f2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(0.5),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_channels, input_time)
            feat = self.block2(self.block1(dummy))
            flat_dim = feat.flatten(1).shape[1]
        self.classifier = nn.Linear(flat_dim, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)
        x = self.block1(x)
        x = self.block2(x)
        x = torch.flatten(x, start_dim=1)
        return self.classifier(x)
