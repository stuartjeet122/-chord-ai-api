"""Reusable digital signal processing primitives for decoded audio."""

from music_api.audio.dsp.analysis import (
    detect_silence,
    generate_waveform,
    inspect_quality,
)
from music_api.audio.dsp.operations import (
    apply_fade,
    apply_filter,
    apply_gain,
    change_channels,
    normalize_peak,
    normalize_rms,
    resample_audio,
    trim_audio,
)
from music_api.audio.dsp.signal import AudioSignal

__all__ = [
    "AudioSignal",
    "apply_fade",
    "apply_filter",
    "apply_gain",
    "change_channels",
    "detect_silence",
    "generate_waveform",
    "inspect_quality",
    "normalize_peak",
    "normalize_rms",
    "resample_audio",
    "trim_audio",
]
