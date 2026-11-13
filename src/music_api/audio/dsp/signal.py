from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

FloatAudio = NDArray[Any]


@dataclass(frozen=True)
class AudioSignal:
    """Decoded PCM as frames × channels, floating point, nominally in [-1, 1]."""

    samples: FloatAudio
    sample_rate: int

    def __post_init__(self) -> None:
        samples = np.asarray(self.samples)
        if samples.ndim == 1:
            samples = samples[:, np.newaxis]
            object.__setattr__(self, "samples", samples)
        if samples.ndim != 2 or samples.shape[0] == 0 or samples.shape[1] == 0:
            raise ValueError("Audio samples must be a non-empty frames-by-channels array.")
        if samples.dtype.kind != "f":
            raise TypeError("Audio samples must use a floating-point representation.")
        if not np.isfinite(samples).all():
            raise ValueError("Audio samples must not contain NaN or infinity.")
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive.")

    @property
    def frame_count(self) -> int:
        return int(self.samples.shape[0])

    @property
    def channel_count(self) -> int:
        return int(self.samples.shape[1])

    @property
    def duration_seconds(self) -> float:
        return self.frame_count / self.sample_rate
