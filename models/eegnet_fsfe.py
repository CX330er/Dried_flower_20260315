import torch
from torch import nn


class FrequencyStableTemporalConv(nn.Module):
    """Temporal convolution with frequency-semantic band-pass kernels.

    The kernel center frequencies are fixed by predefined sub-bands.
    Learnable parameters are per-filter amplitude and bandwidth scale.
    """

    def __init__(
        self,
        bands_hz: tuple[tuple[float, float], ...],
        kernel_size: int = 63,
        sfreq: float = 250.0,
        learn_bandwidth: bool = True,
        center_tune_hz: float = 0.0,
    ):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd (e.g., 31 or 63).")

        self.bands_hz = bands_hz
        self.kernel_size = kernel_size
        self.sfreq = float(sfreq)
        self.center_tune_hz = float(center_tune_hz)

        centers = [(lo + hi) * 0.5 for lo, hi in bands_hz]
        bandwidths = [max(hi - lo, 1e-3) for lo, hi in bands_hz]

        self.center_base = nn.Parameter(torch.tensor(centers, dtype=torch.float32), requires_grad=False)
        self.bandwidth_base = nn.Parameter(torch.tensor(bandwidths, dtype=torch.float32), requires_grad=False)

        self.log_amplitude = nn.Parameter(torch.zeros(len(bands_hz), dtype=torch.float32))

        if learn_bandwidth:
            self.bandwidth_delta = nn.Parameter(torch.zeros(len(bands_hz), dtype=torch.float32))
        else:
            self.register_parameter("bandwidth_delta", None)

        if self.center_tune_hz > 0.0:
            self.center_delta = nn.Parameter(torch.zeros(len(bands_hz), dtype=torch.float32))
        else:
            self.register_parameter("center_delta", None)

        self.register_buffer("window", torch.hamming_window(kernel_size, periodic=False))

    def _effective_center_bandwidth(self) -> tuple[torch.Tensor, torch.Tensor]:
        center = self.center_base
        if self.center_delta is not None:
            center = center + torch.tanh(self.center_delta) * self.center_tune_hz

        bandwidth = self.bandwidth_base
        if self.bandwidth_delta is not None:
            bandwidth = bandwidth * torch.nn.functional.softplus(1.0 + self.bandwidth_delta)

        nyquist = self.sfreq / 2.0
        bandwidth = torch.clamp(bandwidth, min=1.0, max=nyquist - 1.0)
        low = torch.clamp(center - 0.5 * bandwidth, min=0.5)
        high = torch.clamp(center + 0.5 * bandwidth, max=nyquist - 0.5)
        bandwidth = torch.clamp(high - low, min=1.0)
        center = 0.5 * (low + high)
        return center, bandwidth

    def _make_kernel_bank(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        half = self.kernel_size // 2
        t = torch.arange(-half, half + 1, device=device, dtype=dtype) / self.sfreq

        center, bandwidth = self._effective_center_bandwidth()
        center = center.to(device=device, dtype=dtype)
        bandwidth = bandwidth.to(device=device, dtype=dtype)

        low = center - 0.5 * bandwidth
        high = center + 0.5 * bandwidth

        kernels = []
        for lo, hi, amp_log in zip(low, high, self.log_amplitude):
            lo = torch.clamp(lo, min=0.5)
            hi = torch.clamp(hi, max=self.sfreq / 2.0 - 0.5)

            bandpass = (2.0 * hi * torch.sinc(2.0 * hi * t)) - (2.0 * lo * torch.sinc(2.0 * lo * t))
            bandpass = bandpass * self.window.to(device=device, dtype=dtype)
            bandpass = bandpass / (bandpass.norm(p=2) + 1e-6)
            amplitude = torch.exp(amp_log).to(device=device, dtype=dtype)
            kernels.append(amplitude * bandpass)

        kernel_bank = torch.stack(kernels, dim=0).unsqueeze(1).unsqueeze(2)  # [F1, 1, 1, L]
        return kernel_bank

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        kernel_bank = self._make_kernel_bank(device=x.device, dtype=x.dtype)
        return nn.functional.conv2d(x, kernel_bank, bias=None, stride=1, padding=(0, self.kernel_size // 2))


class EEGNetFSFE(nn.Module):
    """EEGNet variant with Frequency-Stable Front-end replacing temporal conv."""

    def __init__(
        self,
        n_channels: int = 22,
        n_classes: int = 4,
        input_time: int = 1000,
        f1: int = 8,
        d: int = 2,
        kernel_size: int = 63,
        sfreq: float = 250.0,
        center_tune_hz: float = 0.0,
    ):
        super().__init__()
        bands = (
            (4.0, 8.0),
            (8.0, 12.0),
            (12.0, 16.0),
            (16.0, 20.0),
            (20.0, 24.0),
            (24.0, 28.0),
            (28.0, 32.0),
            (32.0, 40.0),
        )
        if f1 != len(bands):
            raise ValueError(f"EEGNetFSFE currently expects f1={len(bands)} to match predefined MI bands.")

        f2 = f1 * d

        self.freq_stable_frontend = nn.Sequential(
            FrequencyStableTemporalConv(
                bands_hz=bands,
                kernel_size=kernel_size,
                sfreq=sfreq,
                learn_bandwidth=True,
                center_tune_hz=center_tune_hz,
            ),
            nn.BatchNorm2d(f1),
            nn.ELU(),
        )

        self.block1_rest = nn.Sequential(
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
            feat = self.block2(self.block1_rest(self.freq_stable_frontend(dummy)))
            flat_dim = feat.flatten(1).shape[1]
        self.classifier = nn.Linear(flat_dim, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)
        x = self.freq_stable_frontend(x)
        x = self.block1_rest(x)
        x = self.block2(x)
        x = torch.flatten(x, start_dim=1)
        return self.classifier(x)
