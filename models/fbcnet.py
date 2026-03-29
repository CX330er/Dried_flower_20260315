import torch
from torch import nn


class FBCNet(nn.Module):
    """A compact FBCNet-style model for MI-EEG classification.

    Input: [B, C, T]
    """

    def __init__(
        self,
        n_channels: int = 22,
        n_classes: int = 4,
        input_time: int = 1000,
        n_temporal_filters: int = 16,
        temporal_kernel: int = 51,
        pool_kernel: int = 25,
        pool_stride: int = 10,
        dropout: float = 0.5,
    ):
        super().__init__()

        self.temporal = nn.Sequential(
            nn.Conv2d(1, n_temporal_filters, kernel_size=(1, temporal_kernel), padding=(0, temporal_kernel // 2), bias=False),
            nn.BatchNorm2d(n_temporal_filters),
            nn.ELU(),
        )

        self.spatial = nn.Sequential(
            nn.Conv2d(
                n_temporal_filters,
                n_temporal_filters,
                kernel_size=(n_channels, 1),
                groups=n_temporal_filters,
                bias=False,
            ),
            nn.BatchNorm2d(n_temporal_filters),
            nn.ELU(),
        )

        self.temporal_pool = nn.Sequential(
            nn.AvgPool2d(kernel_size=(1, pool_kernel), stride=(1, pool_stride)),
            nn.Dropout(dropout),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_channels, input_time)
            feat = self._forward_features(dummy)
            flat_dim = feat.flatten(1).shape[1]

        self.classifier = nn.Sequential(
            nn.Linear(flat_dim, 128),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(128, n_classes),
        )

    def _forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.temporal(x)
        x = self.spatial(x)
        x = self.temporal_pool(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)
        x = self._forward_features(x)
        x = torch.flatten(x, start_dim=1)
        return self.classifier(x)
