import torch
from torch import nn

class LightweightSharedSpatialPrior(nn.Module):
    """Shared spatial prior module for 22-channel MI EEG.

    Build regional depthwise spatial features for:
    - Left sensorimotor region
    - Right sensorimotor region
    - Midline region
    - Global region

    Then explicitly construct:
    Fsum = FL + FR
    Fdiff = FL - FR
    Fshared = Concat(Fsum, Fdiff, FM, FG)
    and apply 1x1 pointwise fusion to obtain Ffuse.
    """

    # Canonical BCIC-IV-2a 22-channel order used in this project.
    #
    # Note:
    # - BCIC-IV-2a uses a mixed montage containing classic 10-20 labels
    #   (e.g., Fz/Cz/Pz/FC3/C3/CP3...) and intermediate labels often
    #   associated with 10-10 spacing (e.g., FC1/FC2/C1/C2/CP1/CP2).
    # - The left/right/midline regions below are intentionally defined as
    #   physiologically motivated subsets for MI priors, not a partition of
    #   all 22 channels. Remaining channels are still modeled through FG.
    CHANNEL_NAMES = (
        "Fz",
        "FC3",
        "FC1",
        "FCz",
        "FC2",
        "FC4",
        "C5",
        "C3",
        "C1",
        "Cz",
        "C2",
        "C4",
        "C6",
        "CP3",
        "CP1",
        "CPz",
        "CP2",
        "CP4",
        "P1",
        "Pz",
        "P2",
        "POz",
    )

    def __init__(self, in_filters: int, out_filters: int, n_channels: int = 22):
        super().__init__()
        if n_channels != 22:
            raise ValueError(
                "LightweightSharedSpatialPrior currently requires n_channels=22 "
                "to match fixed 10-20 topology region definitions."
            )

        name_to_idx = {name: idx for idx, name in enumerate(self.CHANNEL_NAMES)}
        left_idx = [name_to_idx[ch] for ch in ("FC3", "C3", "CP3", "FC1", "C1", "CP1")]
        right_idx = [name_to_idx[ch] for ch in ("FC4", "C4", "CP4", "FC2", "C2", "CP2")]
        mid_idx = [name_to_idx[ch] for ch in ("FCz", "Cz", "CPz")]
        # FG uses all channels to capture complementary/global information from
        # channels that are outside L/R/M subsets (e.g., Fz, C5/C6, P1/P2/POz...).
        global_idx = list(range(n_channels))

        self.register_buffer("left_idx", torch.tensor(left_idx, dtype=torch.long), persistent=False)
        self.register_buffer("right_idx", torch.tensor(right_idx, dtype=torch.long), persistent=False)
        self.register_buffer("mid_idx", torch.tensor(mid_idx, dtype=torch.long), persistent=False)
        self.register_buffer("global_idx", torch.tensor(global_idx, dtype=torch.long), persistent=False)

        # Region-wise depthwise spatial convolution (kernel height == region channel count).
        self.left_spatial = nn.Conv2d(in_filters, in_filters, kernel_size=(len(left_idx), 1), groups=in_filters, bias=False)
        self.right_spatial = nn.Conv2d(in_filters, in_filters, kernel_size=(len(right_idx), 1), groups=in_filters, bias=False)
        self.mid_spatial = nn.Conv2d(in_filters, in_filters, kernel_size=(len(mid_idx), 1), groups=in_filters, bias=False)
        self.global_spatial = nn.Conv2d(in_filters, in_filters, kernel_size=(len(global_idx), 1), groups=in_filters, bias=False)

        # Pointwise fusion after concatenating Fsum/Fdiff/FM/FG.
        self.pointwise_fuse = nn.Conv2d(4 * in_filters, out_filters, kernel_size=(1, 1), bias=False)

    def _region_depthwise(self, x: torch.Tensor, region_idx: torch.Tensor, conv: nn.Module) -> torch.Tensor:
        x_region = torch.index_select(x, dim=2, index=region_idx)
        return conv(x_region)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fl = self._region_depthwise(x, self.left_idx, self.left_spatial)
        fr = self._region_depthwise(x, self.right_idx, self.right_spatial)
        fm = self._region_depthwise(x, self.mid_idx, self.mid_spatial)
        fg = self._region_depthwise(x, self.global_idx, self.global_spatial)

        f_sum = fl + fr
        f_diff = fl - fr
        f_shared = torch.cat([f_sum, f_diff, fm, fg], dim=1)
        f_fuse = self.pointwise_fuse(f_shared)
        return f_fuse

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
        dropout_rate: float = 0.35,
    ):
        super().__init__()
        if not (0.25 <= dropout_rate <= 0.5):
            raise ValueError("dropout_rate should be in [0.25, 0.5] for the lightweight EEGNet back-end.")

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

        # Explicit stage-2 spatial prior module:
        # FrequencyStableTemporalConv -> LightweightSharedSpatialPrior.
        self.shared_spatial_prior = LightweightSharedSpatialPrior(
            in_filters=f1,
            out_filters=f2,
            n_channels=n_channels,
        )

        self.post_spatial_block = nn.Sequential(
            nn.BatchNorm2d(f2),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropout_rate),
        )

        # Third-layer design: EEGNet back-half style separable conv + avg pooling + dropout + ELU.
        self.third_layer = nn.Sequential(
            nn.Conv2d(f2, f2, kernel_size=(1, 16), padding=(0, 8), groups=f2, bias=False),
            nn.Conv2d(f2, f2, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(f2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropout_rate),
        )
        self.flatten = nn.Flatten(start_dim=1)

        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_channels, input_time)
            feat = self.freq_stable_frontend(dummy)
            feat = self.shared_spatial_prior(feat)
            feat = self.post_spatial_block(feat)
            feat = self.third_layer(feat)
            flat_dim = feat.flatten(1).shape[1]
        self.classifier = nn.Linear(flat_dim, n_classes)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)
        x = self.freq_stable_frontend(x)
        x = self.shared_spatial_prior(x)
        x = self.post_spatial_block(x)
        x = self.third_layer(x)
        return self.flatten(x)

    def forward(self, x: torch.Tensor, return_features: bool = False, return_probs: bool = False):
        features = self.forward_features(x)
        logits = self.classifier(features)
        if return_features and return_probs:
            return logits, features, torch.softmax(logits, dim=1)
        if return_features:
            return logits, features
        if return_probs:
            return torch.softmax(logits, dim=1)
        return logits
