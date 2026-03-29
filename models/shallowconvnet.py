import torch
from torch import nn


class ShallowConvNet(nn.Module):
    """Shallow ConvNet variant for MI-EEG classification."""

    def __init__(self, n_channels: int = 22, n_classes: int = 4, input_time: int = 1000):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 40, kernel_size=(1, 25), bias=False),
            nn.Conv2d(40, 40, kernel_size=(n_channels, 1), groups=40, bias=False),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 75), stride=(1, 15)),
            nn.Dropout(0.5),
        )

        feat_time = (input_time - 25 + 1)
        feat_time = (feat_time - 75) // 15 + 1
        self.classifier = nn.Linear(40 * feat_time, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T] -> [B, 1, C, T]
        x = x.unsqueeze(1)
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)
        return self.classifier(x)
