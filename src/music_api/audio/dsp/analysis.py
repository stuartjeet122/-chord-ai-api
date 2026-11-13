from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import soundfile
from numpy.typing import NDArray

MAX_WAVEFORM_POINTS = 2_000_000


def generate_waveform(
    path: Path,
    *,
    samples_per_second: int = 100,
    max_points: int = MAX_WAVEFORM_POINTS,
    block_frames: int = 262_144,
) -> dict[str, Any]:
    if samples_per_second <= 0 or max_points <= 0 or block_frames <= 0:
        raise ValueError("Waveform resolution and resource limits must be positive.")
    with soundfile.SoundFile(path) as audio:
        sample_rate = audio.samplerate
        if samples_per_second > sample_rate:
            raise ValueError("samples_per_second cannot exceed the source sample rate.")
        frame_count = len(audio)
        if frame_count == 0:
            raise ValueError("Cannot create a waveform for empty audio.")
        bin_count = math.ceil(frame_count * samples_per_second / sample_rate)
        if bin_count > max_points:
            raise ValueError(f"Requested waveform has {bin_count} points; maximum is {max_points}.")
        channels = audio.channels
        minima = np.full((bin_count, channels), np.inf, dtype=np.float64)
        maxima = np.full((bin_count, channels), -np.inf, dtype=np.float64)
        sum_squares = np.zeros((bin_count, channels), dtype=np.float64)
        counts = np.zeros(bin_count, dtype=np.int64)
        position = 0
        while position < frame_count:
            data = audio.read(
                min(block_frames, frame_count - position),
                dtype="float64",
                always_2d=True,
            )
            local = 0
            while local < len(data):
                absolute_frame = position + local
                bin_index = min(
                    bin_count - 1,
                    absolute_frame * samples_per_second // sample_rate,
                )
                bin_end = min(
                    len(data),
                    math.ceil((bin_index + 1) * sample_rate / samples_per_second) - position,
                )
                segment = data[local:bin_end]
                minima[bin_index] = np.minimum(minima[bin_index], segment.min(axis=0))
                maxima[bin_index] = np.maximum(maxima[bin_index], segment.max(axis=0))
                sum_squares[bin_index] += np.square(segment).sum(axis=0)
                counts[bin_index] += len(segment)
                local = bin_end
            position += len(data)

    rms = np.sqrt(sum_squares / np.maximum(counts[:, np.newaxis], 1))
    timestamps = (np.arange(bin_count, dtype=np.float64) + 0.5) / samples_per_second
    timestamps[-1] = min(
        timestamps[-1],
        frame_count / sample_rate,
    )
    return {
        "sample_rate": sample_rate,
        "channel_count": channels,
        "duration_seconds": frame_count / sample_rate,
        "samples_per_second": samples_per_second,
        "timestamps_seconds": timestamps.tolist(),
        "channels": [
            {
                "minimum": minima[:, channel].tolist(),
                "maximum": maxima[:, channel].tolist(),
                "peak": np.maximum(
                    np.abs(minima[:, channel]),
                    np.abs(maxima[:, channel]),
                ).tolist(),
                "rms": rms[:, channel].tolist(),
            }
            for channel in range(channels)
        ],
    }


def detect_silence(
    path: Path,
    *,
    threshold_dbfs: float = -60.0,
    relative: bool = False,
    window_ms: int = 20,
    min_silence_ms: int = 300,
    min_sound_ms: int = 0,
    block_frames: int = 262_144,
) -> dict[str, Any]:
    if not math.isfinite(threshold_dbfs) or threshold_dbfs > 0 or threshold_dbfs < -160:
        raise ValueError("threshold_dbfs must be finite and between -160 and 0.")
    if window_ms <= 0 or min_silence_ms < 0 or min_sound_ms < 0 or block_frames <= 0:
        raise ValueError("Silence detection durations and block size are invalid.")
    with soundfile.SoundFile(path) as audio:
        sample_rate = audio.samplerate
        frame_count = len(audio)
        window_frames = max(1, round(sample_rate * window_ms / 1000))
        window_levels: list[float] = []
        squares = 0.0
        samples = 0
        buffer = np.empty((0, audio.channels), dtype=np.float64)
        while True:
            block = audio.read(block_frames, dtype="float64", always_2d=True)
            if not len(block):
                break
            buffer = np.concatenate((buffer, block), axis=0)
            usable = len(buffer) // window_frames * window_frames
            if usable:
                windows = buffer[:usable].reshape(-1, window_frames, audio.channels)
                rms = np.sqrt(np.mean(np.square(windows), axis=(1, 2)))
                window_levels.extend(rms.tolist())
                squares += float(np.square(buffer[:usable]).sum())
                samples += usable * audio.channels
            buffer = buffer[usable:]
        if len(buffer):
            rms = float(np.sqrt(np.mean(np.square(buffer))))
            window_levels.append(rms)
            squares += float(np.square(buffer).sum())
            samples += buffer.size
    if not window_levels:
        raise ValueError("Cannot detect silence in empty audio.")

    reference = max(window_levels) if relative else 1.0
    threshold = reference * (10.0 ** (threshold_dbfs / 20.0))
    silent = np.asarray(window_levels) <= threshold
    minimum_windows = max(1, math.ceil(min_silence_ms / window_ms))
    minimum_sound_windows = math.ceil(min_sound_ms / window_ms)
    if minimum_sound_windows:
        silent = _remove_short_runs(silent, minimum_sound_windows, target=False)
    silent = _remove_short_runs(silent, minimum_windows, target=True)
    regions: list[dict[str, float]] = []
    start: int | None = None
    for index, is_silent in enumerate(np.append(silent, False)):
        if is_silent and start is None:
            start = index
        elif not is_silent and start is not None:
            first = start * window_frames
            last = min(index * window_frames, frame_count)
            region_rms = float(np.mean(window_levels[start:index]))
            regions.append(
                {
                    "start_seconds": first / sample_rate,
                    "end_seconds": last / sample_rate,
                    "duration_seconds": (last - first) / sample_rate,
                    "mean_rms": region_rms,
                    "threshold_amplitude": threshold,
                    "confidence": (
                        1.0 if threshold == 0 else min(1.0, max(0.0, 1.0 - region_rms / threshold))
                    ),
                }
            )
            start = None
    duration = frame_count / sample_rate
    total_silence = sum(region["duration_seconds"] for region in regions)
    return {
        "sample_rate": sample_rate,
        "duration_seconds": duration,
        "threshold_dbfs": threshold_dbfs,
        "threshold_amplitude": threshold,
        "relative_threshold": relative,
        "window_ms": window_ms,
        "min_silence_ms": min_silence_ms,
        "regions": regions,
        "silence_percentage": 100.0 * total_silence / duration if duration else 0.0,
    }


def inspect_quality(
    path: Path,
    *,
    silence_threshold_dbfs: float = -60.0,
    block_frames: int = 262_144,
) -> dict[str, Any]:
    if not math.isfinite(silence_threshold_dbfs) or not -160 <= silence_threshold_dbfs <= 0:
        raise ValueError("silence_threshold_dbfs must be finite and between -160 and 0.")
    if block_frames <= 0:
        raise ValueError("block_frames must be positive.")
    with soundfile.SoundFile(path) as audio:
        sample_rate = audio.samplerate
        channels = audio.channels
        frames = len(audio)
        if not frames:
            raise ValueError("Cannot inspect quality of empty audio.")
        peak = np.zeros(channels, dtype=np.float64)
        sum_squares = np.zeros(channels, dtype=np.float64)
        sums = np.zeros(channels, dtype=np.float64)
        clipped = 0
        silent_frames = 0
        frame_total = 0
        window_rms: list[float] = []
        while True:
            block = audio.read(block_frames, dtype="float64", always_2d=True)
            if not len(block):
                break
            absolute = np.abs(block)
            peak = np.maximum(peak, absolute.max(axis=0))
            sum_squares += np.square(block).sum(axis=0)
            sums += block.sum(axis=0)
            clipped += int(np.count_nonzero(absolute >= 0.999))
            frame_rms = np.sqrt(np.mean(np.square(block), axis=1))
            threshold = 10.0 ** (silence_threshold_dbfs / 20.0)
            silent_frames += int(np.count_nonzero(frame_rms <= threshold))
            frame_total += len(block)
            window_rms.extend(
                np.sqrt(
                    np.mean(
                        np.square(block[: len(block) // sample_rate * sample_rate]).reshape(
                            -1, sample_rate, channels
                        ),
                        axis=(1, 2),
                    )
                ).tolist()
                if len(block) >= sample_rate
                else []
            )

    count = frame_total
    rms = np.sqrt(sum_squares / count)
    peak_dbfs = _dbfs(peak)
    rms_dbfs = _dbfs(rms)
    crest_factor_db = peak_dbfs - rms_dbfs
    noise_floor = float(np.percentile(window_rms, 10)) if window_rms else None
    return {
        "sample_rate": sample_rate,
        "channel_count": channels,
        "duration_seconds": frames / sample_rate,
        "peak_amplitude": peak.tolist(),
        "peak_dbfs": peak_dbfs.tolist(),
        "rms_amplitude": rms.tolist(),
        "rms_dbfs": rms_dbfs.tolist(),
        "clipped_sample_percentage": 100.0 * clipped / (count * channels),
        "dc_offset": (sums / count).tolist(),
        "silence_percentage": 100.0 * silent_frames / count,
        "crest_factor_db": crest_factor_db.tolist(),
        "estimated_noise_floor_rms": noise_floor,
        "noise_floor_method": (
            "10th percentile of one-second RMS windows; rough estimate, not a calibrated "
            "noise measure."
            if noise_floor is not None
            else None
        ),
        "integrated_loudness_lufs": None,
        "true_peak_dbfs": None,
        "loudness_measurement_note": (
            "Integrated loudness and true peak are not inferred from sample peak or RMS."
        ),
    }


def _dbfs(amplitude: NDArray[Any]) -> NDArray[Any]:
    floor = np.finfo(np.float64).tiny
    return np.asarray(20.0 * np.log10(np.maximum(amplitude, floor)))


def _remove_short_runs(mask: NDArray[Any], min_length: int, *, target: bool) -> NDArray[Any]:
    if min_length <= 1:
        return mask
    cleaned = mask.copy()
    padded = np.concatenate(([not target], mask, [not target]))
    changes = np.flatnonzero(padded[1:] != padded[:-1])
    for start, stop in zip(changes[::2], changes[1::2], strict=True):
        if padded[start + 1] == target and stop - start < min_length:
            cleaned[start:stop] = not target
    return cleaned
