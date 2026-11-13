from __future__ import annotations

import math
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray
from scipy import signal
from scipy.signal import resample_poly

from music_api.audio.dsp.signal import AudioSignal

FilterKind = Literal["highpass", "lowpass", "bandpass", "notch"]


def trim_audio(signal_data: AudioSignal, start_seconds: float, end_seconds: float) -> AudioSignal:
    _finite_nonnegative(start_seconds, "start_seconds")
    _finite_nonnegative(end_seconds, "end_seconds")
    start = round(start_seconds * signal_data.sample_rate)
    end = round(end_seconds * signal_data.sample_rate)
    if start >= end or end > signal_data.frame_count:
        raise ValueError("Trim bounds must select a non-empty range within the source.")
    return AudioSignal(signal_data.samples[start:end].copy(), signal_data.sample_rate)


def apply_gain(
    signal_data: AudioSignal,
    *,
    gain_db: float | None = None,
    linear_gain: float | None = None,
    channel_gains_db: tuple[float, ...] | None = None,
    prevent_clipping: bool = True,
) -> tuple[AudioSignal, dict[str, float | bool]]:
    if sum(value is not None for value in (gain_db, linear_gain, channel_gains_db)) != 1:
        raise ValueError("Specify exactly one gain mode.")
    if channel_gains_db is not None:
        if len(channel_gains_db) != signal_data.channel_count:
            raise ValueError("Provide one channel gain for each audio channel.")
        if any(not math.isfinite(value) for value in channel_gains_db):
            raise ValueError("Channel gains must be finite.")
        gains = np.power(10.0, np.asarray(channel_gains_db, dtype=np.float64) / 20.0)
    elif gain_db is not None:
        _finite(gain_db, "gain_db")
        gains = np.full(signal_data.channel_count, 10.0 ** (gain_db / 20.0))
    elif linear_gain is not None:
        _finite_nonnegative(linear_gain, "linear_gain")
        gains = np.full(signal_data.channel_count, linear_gain)
    else:
        raise ValueError("Channel gain mode was not provided.")
    adjusted = signal_data.samples.astype(np.float64, copy=True) * gains
    requested_peak = float(np.max(np.abs(adjusted)))
    clipped_samples = float(np.count_nonzero(np.abs(adjusted) > 1.0))
    clipped = requested_peak > 1.0
    if prevent_clipping:
        np.clip(adjusted, -1.0, 1.0, out=adjusted)
    return AudioSignal(adjusted, signal_data.sample_rate), {
        "clipping_prevented": clipped and prevent_clipping,
        "clipped_samples": clipped_samples,
        "requested_peak": requested_peak,
    }


def normalize_peak(
    signal_data: AudioSignal,
    target_dbfs: float = -1.0,
    *,
    ceiling_dbfs: float = 0.0,
) -> tuple[AudioSignal, dict[str, float | bool | None]]:
    _validate_dbfs(target_dbfs, "target_dbfs", upper=0)
    _validate_dbfs(ceiling_dbfs, "ceiling_dbfs", upper=0)
    peak = float(np.max(np.abs(signal_data.samples)))
    if peak == 0:
        return AudioSignal(signal_data.samples.copy(), signal_data.sample_rate), {
            "input_peak_dbfs": None,
            "output_peak_dbfs": None,
            "gain_db": 0.0,
            "limited_by_ceiling": False,
        }
    requested_gain = target_dbfs - _dbfs(peak)
    maximum_gain = ceiling_dbfs - _dbfs(peak)
    gain = min(requested_gain, maximum_gain)
    result, _ = apply_gain(signal_data, gain_db=gain)
    output_peak = float(np.max(np.abs(result.samples)))
    return result, {
        "input_peak_dbfs": _dbfs(peak),
        "output_peak_dbfs": _dbfs(output_peak),
        "gain_db": gain,
        "limited_by_ceiling": gain < requested_gain,
    }


def normalize_rms(
    signal_data: AudioSignal,
    target_dbfs: float = -18.0,
    *,
    ceiling_dbfs: float = -1.0,
) -> tuple[AudioSignal, dict[str, float | bool]]:
    _validate_dbfs(target_dbfs, "target_dbfs", upper=0)
    _validate_dbfs(ceiling_dbfs, "ceiling_dbfs", upper=0)
    rms = float(np.sqrt(np.mean(np.square(signal_data.samples, dtype=np.float64))))
    if rms == 0:
        raise ValueError("Cannot RMS-normalize silent audio.")
    requested_gain = target_dbfs - _dbfs(rms)
    peak = float(np.max(np.abs(signal_data.samples)))
    maximum_gain = ceiling_dbfs - _dbfs(peak) if peak else requested_gain
    gain = min(requested_gain, maximum_gain)
    result, _ = apply_gain(signal_data, gain_db=gain)
    output_rms = float(np.sqrt(np.mean(np.square(result.samples, dtype=np.float64))))
    return result, {
        "input_rms_dbfs": _dbfs(rms),
        "output_rms_dbfs": _dbfs(output_rms),
        "gain_db": gain,
        "limited_by_ceiling": gain < requested_gain,
    }


def apply_fade(
    signal_data: AudioSignal,
    *,
    fade_in_seconds: float = 0,
    fade_out_seconds: float = 0,
    curve: Literal["linear", "equal_power"] = "linear",
) -> AudioSignal:
    _finite_nonnegative(fade_in_seconds, "fade_in_seconds")
    _finite_nonnegative(fade_out_seconds, "fade_out_seconds")
    if fade_in_seconds + fade_out_seconds > signal_data.duration_seconds:
        raise ValueError("Fade durations cannot exceed the available audio duration.")
    gain = np.ones(signal_data.frame_count, dtype=np.float64)
    fade_in_frames = round(fade_in_seconds * signal_data.sample_rate)
    fade_out_frames = round(fade_out_seconds * signal_data.sample_rate)
    if curve == "linear":

        def make_curve(value: NDArray[Any]) -> NDArray[Any]:
            return value
    elif curve == "equal_power":

        def make_curve(value: NDArray[Any]) -> NDArray[Any]:
            return np.sin(value * math.pi / 2)
    else:
        raise ValueError("curve must be 'linear' or 'equal_power'.")
    if fade_in_frames:
        gain[:fade_in_frames] *= make_curve(np.linspace(0.0, 1.0, fade_in_frames, endpoint=True))
    if fade_out_frames:
        gain[-fade_out_frames:] *= make_curve(np.linspace(1.0, 0.0, fade_out_frames, endpoint=True))
    return AudioSignal(signal_data.samples * gain[:, np.newaxis], signal_data.sample_rate)


def change_channels(
    signal_data: AudioSignal,
    mode: Literal["mono", "stereo", "left", "right", "swap", "split"],
    *,
    channel_layout: str | None = None,
) -> AudioSignal | tuple[AudioSignal, ...]:
    samples = signal_data.samples
    channels = signal_data.channel_count
    if mode == "mono":
        if channels == 1:
            mono = samples.copy()
        elif channels == 2:
            mono = samples.mean(axis=1, keepdims=True)
        else:
            matrix = _downmix_matrix(channels, channel_layout)
            mono = samples @ matrix
        return AudioSignal(mono, signal_data.sample_rate)
    if mode == "stereo":
        if channels == 1:
            return AudioSignal(np.repeat(samples, 2, axis=1), signal_data.sample_rate)
        if channels == 2:
            return AudioSignal(samples.copy(), signal_data.sample_rate)
        matrix = _downmix_matrix(channels, channel_layout, stereo=True)
        return AudioSignal(samples @ matrix, signal_data.sample_rate)
    if mode == "left":
        if channels < 2:
            raise ValueError("Left-channel extraction requires at least two channels.")
        return AudioSignal(samples[:, :1].copy(), signal_data.sample_rate)
    if mode == "right":
        if channels < 2:
            raise ValueError("Right-channel extraction requires at least two channels.")
        return AudioSignal(samples[:, 1:2].copy(), signal_data.sample_rate)
    if mode == "swap":
        if channels != 2:
            raise ValueError("Channel swapping requires stereo audio.")
        return AudioSignal(samples[:, ::-1].copy(), signal_data.sample_rate)
    if mode == "split":
        return tuple(
            AudioSignal(samples[:, channel : channel + 1].copy(), signal_data.sample_rate)
            for channel in range(channels)
        )
    raise ValueError("Unsupported channel operation.")


def resample_audio(signal_data: AudioSignal, target_sample_rate: int) -> AudioSignal:
    if target_sample_rate <= 0:
        raise ValueError("target_sample_rate must be positive.")
    if target_sample_rate == signal_data.sample_rate:
        return AudioSignal(signal_data.samples.copy(), target_sample_rate)
    divisor = math.gcd(target_sample_rate, signal_data.sample_rate)
    converted = resample_poly(
        signal_data.samples,
        target_sample_rate // divisor,
        signal_data.sample_rate // divisor,
        axis=0,
    )
    return AudioSignal(converted, target_sample_rate)


def apply_filter(
    signal_data: AudioSignal,
    kind: FilterKind,
    *,
    cutoff_hz: float | tuple[float, float],
    order: int = 4,
    zero_phase: bool = False,
) -> AudioSignal:
    if order < 1 or order > 12:
        raise ValueError("Filter order must be between 1 and 12.")
    nyquist = signal_data.sample_rate / 2
    if kind in {"lowpass", "highpass"}:
        if not isinstance(cutoff_hz, (int, float)):
            raise ValueError("A single cutoff frequency is required.")
        _valid_frequency(float(cutoff_hz), nyquist)
        sos = signal.butter(
            order,
            float(cutoff_hz),
            btype=kind,
            fs=signal_data.sample_rate,
            output="sos",
        )
    elif kind == "bandpass":
        if not isinstance(cutoff_hz, tuple) or len(cutoff_hz) != 2:
            raise ValueError("Band-pass filtering requires a lower and upper cutoff.")
        low, high = cutoff_hz
        _valid_frequency(low, nyquist)
        _valid_frequency(high, nyquist)
        if low >= high:
            raise ValueError("Band-pass lower cutoff must be below its upper cutoff.")
        sos = signal.butter(
            order,
            (low, high),
            btype="bandpass",
            fs=signal_data.sample_rate,
            output="sos",
        )
    elif kind == "notch":
        if not isinstance(cutoff_hz, (int, float)):
            raise ValueError("Notch filtering requires one center frequency.")
        _valid_frequency(float(cutoff_hz), nyquist)
        b, a = signal.iirnotch(
            float(cutoff_hz), Q=max(1.0, float(order)), fs=signal_data.sample_rate
        )
        sos = signal.tf2sos(b, a)
    else:
        raise ValueError("Unsupported filter kind.")
    if zero_phase:
        if signal_data.frame_count < 3 * (2 * len(sos) + 1):
            raise ValueError("Audio is too short for zero-phase filtering.")
        filtered = signal.sosfiltfilt(sos, signal_data.samples, axis=0)
    else:
        filtered = signal.sosfilt(sos, signal_data.samples, axis=0)
    return AudioSignal(filtered, signal_data.sample_rate)


def remove_dc_offset(signal_data: AudioSignal) -> AudioSignal:
    return AudioSignal(
        signal_data.samples - signal_data.samples.mean(axis=0, keepdims=True),
        signal_data.sample_rate,
    )


def crossfade(
    first: AudioSignal,
    second: AudioSignal,
    duration_seconds: float,
) -> AudioSignal:
    _compatible_signals(first, second)
    _finite_nonnegative(duration_seconds, "duration_seconds")
    frames = round(duration_seconds * first.sample_rate)
    if frames > min(first.frame_count, second.frame_count):
        raise ValueError("Crossfade cannot exceed either segment duration.")
    if not frames:
        return AudioSignal(np.concatenate((first.samples, second.samples)), first.sample_rate)
    phase = np.linspace(0.0, math.pi / 2, frames, endpoint=True)[:, np.newaxis]
    blend = first.samples[-frames:] * np.cos(phase) + second.samples[:frames] * np.sin(phase)
    return AudioSignal(
        np.concatenate((first.samples[:-frames], blend, second.samples[frames:])),
        first.sample_rate,
    )


def _downmix_matrix(
    channels: int,
    channel_layout: str | None,
    *,
    stereo: bool = False,
) -> NDArray[np.float64]:
    layout = (channel_layout or "").casefold()
    if channels == 6 and layout in {"5.1", "5.1(side)"}:
        # FFmpeg/libswresample's common 5.1 order: FL, FR, FC, LFE, surrounds.
        center = 2**-0.5
        surround = 2**-0.5
        matrix = np.asarray(
            [
                [1.0, 0.0, center, 0.0, surround, 0.0],
                [0.0, 1.0, center, 0.0, 0.0, surround],
            ]
        )
        return matrix if stereo else matrix.mean(axis=0, keepdims=True)
    if channels == 4 and layout in {"quad", "4.0"}:
        matrix = np.asarray([[1.0, 0.0, 0.0, 0.5], [0.0, 1.0, 0.5, 0.0]])
        return matrix if stereo else matrix.mean(axis=0, keepdims=True)
    raise ValueError("Explicit downmix is supported only for stereo, quad/4.0, and 5.1 layouts.")


def _compatible_signals(first: AudioSignal, second: AudioSignal) -> None:
    if first.sample_rate != second.sample_rate or first.channel_count != second.channel_count:
        raise ValueError("Audio sample rates and channel counts must match.")


def _valid_frequency(value: float, nyquist: float) -> None:
    if not math.isfinite(value) or value <= 0 or value >= nyquist:
        raise ValueError("Filter cutoff must be positive and below Nyquist.")


def _validate_dbfs(value: float, name: str, *, upper: float) -> None:
    if not math.isfinite(value) or value < -160 or value > upper:
        raise ValueError(f"{name} must be finite and between -160 and {upper} dBFS.")


def _finite(value: float, name: str) -> None:
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite.")


def _finite_nonnegative(value: float, name: str) -> None:
    _finite(value, name)
    if value < 0:
        raise ValueError(f"{name} must be non-negative.")


def _dbfs(amplitude: float) -> float:
    return -math.inf if amplitude == 0 else 20.0 * math.log10(amplitude)
