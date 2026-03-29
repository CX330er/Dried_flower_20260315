import torch
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_t: int = 10):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=(1, kernel_t), bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ELU(),
            nn.MaxPool2d((1, 3)),
            nn.Dropout(0.5),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DeepConvNet(nn.Module):
    """Deep ConvNet variant for MI-EEG classification."""

    def __init__(self, n_channels: int = 22, n_classes: int = 4, input_time: int = 1000):
        super().__init__()
        self.first = nn.Sequential(
            nn.Conv2d(1, 25, kernel_size=(1, 10), bias=False),
            nn.Conv2d(25, 25, kernel_size=(n_channels, 1), groups=25, bias=False),
            nn.BatchNorm2d(25),
            nn.ELU(),
            nn.MaxPool2d((1, 3)),
            nn.Dropout(0.5),
        )
        self.block2 = ConvBlock(25, 50)
        self.block3 = ConvBlock(50, 100)
        self.block4 = ConvBlock(100, 200)

        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_channels, input_time)
            feat = self._forward_features(dummy)
            flat_dim = feat.flatten(1).shape[1]
        self.classifier = nn.Linear(flat_dim, n_classes)

    def _forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.first(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)
        x = self._forward_features(x)
        x = torch.flatten(x, start_dim=1)
        return self.classifier(x)
