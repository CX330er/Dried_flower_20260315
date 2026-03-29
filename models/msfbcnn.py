import torch
from torch import nn


class _TemporalBranch(nn.Module):
    def __init__(self, out_channels: int, kernel_size: int):
        super().__init__()
        self.branch = nn.Sequential(
            nn.Conv2d(1, out_channels, kernel_size=(1, kernel_size), padding=(0, kernel_size // 2), bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.branch(x)


class MSFBCNN(nn.Module):
    """Multi-Scale Filter Bank CNN variant for MI-EEG classification.

    Input: [B, C, T]
    """

    def __init__(
        self,
        n_channels: int = 22,
        n_classes: int = 4,
        input_time: int = 1000,
        branch_channels: int = 8,
        kernels: tuple[int, int, int] = (15, 31, 63),
        dropout: float = 0.5,
    ):
        super().__init__()

        self.branches = nn.ModuleList([_TemporalBranch(branch_channels, k) for k in kernels])
        merged_channels = branch_channels * len(kernels)

        self.spatial = nn.Sequential(
            nn.Conv2d(
                merged_channels,
                merged_channels,
                kernel_size=(n_channels, 1),
                groups=merged_channels,
                bias=False,
            ),
            nn.BatchNorm2d(merged_channels),
            nn.ELU(),
        )

        self.fusion = nn.Sequential(
            nn.Conv2d(merged_channels, 64, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropout),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_channels, input_time)
            feat = self._forward_features(dummy)
            flat_dim = feat.flatten(1).shape[1]

        self.classifier = nn.Linear(flat_dim, n_classes)

    def _forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.cat([branch(x) for branch in self.branches], dim=1)
        x = self.spatial(x)
        x = self.fusion(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)
        x = self._forward_features(x)
        x = torch.flatten(x, start_dim=1)
        return self.classifier(x)
