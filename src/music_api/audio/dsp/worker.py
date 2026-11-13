from __future__ import annotations

import asyncio
import json
import math
import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import soundfile
from numpy.typing import NDArray
from scipy import signal

from music_api.audio.dsp.analysis import detect_silence, generate_waveform, inspect_quality
from music_api.audio.dsp.chunking import AudioChunk, plan_chunks
from music_api.audio.dsp.operations import (
    apply_filter,
    change_channels,
    resample_audio,
)
from music_api.audio.dsp.pipeline import apply_pipeline
from music_api.audio.dsp.signal import AudioSignal
from music_api.audio.toolchain import ProbeResult
from music_api.audio.transforms import run_soundstretch

Progress = Callable[[float], Awaitable[bool]]
MAX_IN_MEMORY_BYTES = 256 * 1024 * 1024
MAX_SEGMENT_OUTPUTS = 1000
_BLOCK_FRAMES = 262_144


@dataclass(frozen=True)
class DspOutput:
    path: Path
    asset_type: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class DspResult:
    outputs: list[DspOutput]
    result: dict[str, Any]
    duration_ms: int


class ProcessingCancelledError(Exception):
    pass


async def process_audio(
    source_path: Path,
    work_dir: Path,
    *,
    source_asset_id: uuid.UUID,
    operation: str,
    options: dict[str, Any],
    probe: ProbeResult,
    progress: Progress,
    soundtouch_path: str = "soundstretch",
    soundtouch_version: str = "unknown",
) -> DspResult:
    started = time.monotonic()
    if operation == "time_pitch":
        return await _time_pitch(
            source_path,
            work_dir,
            options,
            probe,
            progress,
            soundtouch_path,
            soundtouch_version,
            started,
        )

    if operation in {"waveform", "silence", "quality"}:
        if operation == "waveform":
            result = await asyncio.to_thread(generate_waveform, source_path, **options)
        elif operation == "silence":
            result = await asyncio.to_thread(detect_silence, source_path, **options)
        else:
            result = await asyncio.to_thread(inspect_quality, source_path)
        output = work_dir / f"{operation}.json"
        _write_json(output, result)
        return DspResult(
            [DspOutput(output, "waveform" if operation == "waveform" else "analysis_artifact", {})],
            result,
            int((time.monotonic() - started) * 1000),
        )

    if operation == "concatenate":
        source_paths = [Path(path) for path in options.get("_source_paths", [])]
        if len(source_paths) < 2:
            raise ValueError("Concatenation requires at least two prepared audio sources.")
        output = work_dir / "concatenated.wav"
        total_frames, sample_rate, channels = await _concatenate_paths(
            source_paths,
            output,
            float(options.get("crossfade_seconds", 0)),
            progress,
        )
        return _result(
            [
                DspOutput(
                    output,
                    "processed_audio",
                    {"source_asset_ids": options.get("asset_ids", [])},
                )
            ],
            {
                "source_count": len(source_paths),
                "duration_seconds": total_frames / sample_rate,
                "sample_rate": sample_rate,
                "channel_count": channels,
                "crossfade_seconds": options.get("crossfade_seconds", 0),
            },
            started,
        )

    if operation == "pipeline":
        await _ensure_in_memory_limit(source_path)
        data, sample_rate = await asyncio.to_thread(
            soundfile.read,
            source_path,
            dtype="float64",
            always_2d=True,
        )
        signal_data, reports = await asyncio.to_thread(
            apply_pipeline,
            AudioSignal(data, sample_rate),
            options["operations"],
            channel_layout=probe.channel_layout,
        )
        if not await progress(0.9):
            raise ProcessingCancelledError
        output = work_dir / "pipeline.wav"
        await asyncio.to_thread(_write_audio, output, signal_data.samples, signal_data.sample_rate)
        if not await progress(1.0):
            raise ProcessingCancelledError
        return _result(
            [DspOutput(output, "processed_audio", {})],
            {
                "operations": reports,
                "sample_rate": signal_data.sample_rate,
                "channel_count": signal_data.channel_count,
                "duration_seconds": signal_data.duration_seconds,
            },
            started,
        )

    if operation == "trim":
        start = float(options["start_seconds"])
        end = float(options["end_seconds"])
        start_frame = round(start * probe.sample_rate)
        end_frame = round(end * probe.sample_rate)
        if start_frame >= end_frame or end_frame > round(
            probe.duration_seconds * probe.sample_rate
        ):
            raise ValueError("Trim bounds must select a non-empty range within the source.")
        output = work_dir / "trimmed.wav"
        await _write_range(source_path, output, start_frame, end_frame, progress)
        item = DspOutput(
            output,
            "trimmed_audio",
            {
                "start_sample": start_frame,
                "end_sample": end_frame,
                "start_seconds": start_frame / probe.sample_rate,
                "end_seconds": end_frame / probe.sample_rate,
            },
        )
        return _result(
            [item], {"duration_seconds": (end_frame - start_frame) / probe.sample_rate}, started
        )

    if operation in {"split", "chunks"}:
        chunks = _make_chunks(source_asset_id, probe, options, operation, source_path)
        if len(chunks) > MAX_SEGMENT_OUTPUTS:
            raise ValueError(f"Segmentation exceeds the {MAX_SEGMENT_OUTPUTS}-output limit.")
        outputs: list[DspOutput] = []
        for index, chunk in enumerate(chunks):
            output = work_dir / f"segment-{index:05d}.wav"
            await _write_range(
                source_path,
                output,
                chunk.start_sample,
                chunk.end_sample,
                progress,
                progress_start=index / len(chunks),
                progress_span=1 / len(chunks),
            )
            output_metadata = {
                "chunk_id": str(chunk.id),
                "position": index,
                "start_sample": chunk.start_sample,
                "end_sample": chunk.end_sample,
                "start_seconds": chunk.start_seconds,
                "end_seconds": chunk.end_seconds,
                "duration_seconds": chunk.duration_seconds,
                "sample_rate": chunk.sample_rate,
                "processing_config": {
                    **chunk.processing_config,
                    "profile_id": options.get("profile_id"),
                },
            }
            if operation == "chunks":
                output, output_metadata = await _prepare_analysis_chunk(
                    output,
                    work_dir,
                    index,
                    options,
                    probe.channel_layout,
                    output_metadata,
                )
            outputs.append(DspOutput(output, "processed_audio", output_metadata))
        return _result(outputs, {"chunk_count": len(chunks)}, started)

    if operation == "gain":
        gain_db = options.get("gain_db")
        linear_gain = options.get("linear_gain")
        channel_gains = options.get("channel_gains_db")
        if sum(value is not None for value in (gain_db, linear_gain, channel_gains)) != 1:
            raise ValueError("Specify exactly one gain mode.")
        if channel_gains is not None and len(channel_gains) != probe.channels:
            raise ValueError("Provide one channel gain for each audio channel.")
        if channel_gains is not None:
            gains = np.power(10.0, np.asarray(channel_gains, dtype=np.float64) / 20.0)
        elif gain_db is not None:
            gains = np.full(probe.channels, 10.0 ** (float(gain_db) / 20.0))
        elif linear_gain is not None:
            gains = np.full(probe.channels, float(linear_gain))
        else:
            raise ValueError("Gain mode was not provided.")
        output = work_dir / "gain.wav"
        clipped = await _write_transform(
            source_path, output, progress, lambda data, _: data * gains
        )
        return _result(
            [DspOutput(output, "processed_audio", {})],
            {"requested_gain_db": gain_db, "linear_gain": linear_gain, "clipped_samples": clipped},
            started,
        )

    if operation == "normalize":
        method = options["method"]
        if method == "loudness":
            raise ValueError("Loudness normalization must use the FFmpeg EBU R128 service.")
        target = float(options["target_dbfs"])
        ceiling = float(options["ceiling_dbfs"])
        metrics = await asyncio.to_thread(_measure_for_normalization, source_path, method)
        input_level = metrics["peak"] if method == "peak" else metrics["rms"]
        if input_level <= 0:
            if method == "rms":
                raise ValueError("Cannot rms-normalize silent audio.")
            output = work_dir / "normalized.wav"
            await _copy_audio(source_path, output, progress)
            return _result(
                [DspOutput(output, "normalized_audio", {})],
                {
                    "method": method,
                    "input_peak_dbfs": None,
                    "input_rms_dbfs": None,
                    "output_peak_dbfs": None,
                    "output_rms_dbfs": None,
                    "target_dbfs": target,
                    "ceiling_dbfs": ceiling,
                    "gain_db": 0.0,
                    "limited_by_ceiling": False,
                    "clipped_samples": 0,
                },
                started,
            )
        requested_gain = target - 20 * math.log10(input_level)
        peak_gain = ceiling - 20 * math.log10(metrics["peak"])
        gain = min(requested_gain, peak_gain)
        output = work_dir / "normalized.wav"
        clipped = await _write_transform(
            source_path,
            output,
            progress,
            lambda data, _: data * 10.0 ** (gain / 20.0),
        )
        normalization_result = {
            "method": method,
            "input_peak_dbfs": _dbfs(metrics["peak"]),
            "input_rms_dbfs": _dbfs(metrics["rms"]),
            "output_peak_dbfs": _dbfs(min(1.0, metrics["peak"] * 10.0 ** (gain / 20.0))),
            "output_rms_dbfs": _dbfs(metrics["rms"] * 10.0 ** (gain / 20.0)),
            "target_dbfs": target,
            "ceiling_dbfs": ceiling,
            "gain_db": gain,
            "limited_by_ceiling": gain < requested_gain,
            "clipped_samples": clipped,
        }
        return _result(
            [DspOutput(output, "normalized_audio", {})],
            normalization_result,
            started,
        )

    if operation == "fade":
        fade_in = round(float(options.get("fade_in_seconds", 0)) * probe.sample_rate)
        fade_out = round(float(options.get("fade_out_seconds", 0)) * probe.sample_rate)
        frames = round(probe.duration_seconds * probe.sample_rate)
        if fade_in + fade_out > frames:
            raise ValueError("Fade durations cannot exceed the available audio duration.")
        curve = options.get("curve", "linear")

        def fade(data: NDArray[Any], offset: int) -> NDArray[Any]:
            indices = np.arange(offset, offset + len(data), dtype=np.int64)
            gains = np.ones(len(data), dtype=np.float64)
            if fade_in:
                active = indices < fade_in
                phase = indices[active] / max(1, fade_in - 1)
                gains[active] = _fade_curve(phase, curve, rising=True)
            if fade_out:
                active = indices >= frames - fade_out
                phase = (frames - 1 - indices[active]) / max(1, fade_out - 1)
                gains[active] = _fade_curve(phase, curve, rising=True)
            return data * gains[:, np.newaxis]

        output = work_dir / "faded.wav"
        await _write_transform(source_path, output, progress, fade)
        return _result(
            [DspOutput(output, "processed_audio", {})],
            {
                "fade_in_seconds": fade_in / probe.sample_rate,
                "fade_out_seconds": fade_out / probe.sample_rate,
                "curve": curve,
            },
            started,
        )

    if operation == "channels":
        mode = options["mode"]
        output = work_dir / f"channels-{mode}.wav"
        channel_paths = await _write_channels(
            source_path,
            output,
            mode,
            probe.channel_layout,
            progress,
        )
        outputs = [
            DspOutput(path, "processed_audio", {"channel_index": index})
            for index, path in enumerate(channel_paths)
        ]
        return _result(outputs, {"mode": mode, "output_count": len(outputs)}, started)

    if operation == "filter":
        return await _filter_file(source_path, work_dir, options, probe, progress, started)

    if operation == "resample":
        target_rate = int(options["sample_rate"])
        if target_rate == probe.sample_rate:
            output = work_dir / "resampled.wav"
            await _copy_audio(source_path, output, progress)
        else:
            await _ensure_in_memory_limit(source_path)
            data, sample_rate = await asyncio.to_thread(
                soundfile.read,
                source_path,
                dtype="float64",
                always_2d=True,
            )
            converted = resample_audio(AudioSignal(data, sample_rate), target_rate)
            output = work_dir / "resampled.wav"
            await asyncio.to_thread(_write_audio, output, converted.samples, target_rate)
        return _result(
            [DspOutput(output, "processed_audio", {})],
            {"input_sample_rate": probe.sample_rate, "output_sample_rate": target_rate},
            started,
        )

    raise ValueError(f"Unsupported audio DSP operation: {operation}.")


async def _time_pitch(
    source_path: Path,
    work_dir: Path,
    options: dict[str, Any],
    probe: ProbeResult,
    progress: Progress,
    soundtouch_path: str,
    soundtouch_version: str,
    started: float,
) -> DspResult:
    source_info = soundfile.info(source_path)
    if source_info.samplerate != probe.sample_rate or source_info.channels != probe.channels:
        raise ValueError("Decoded audio properties do not match the probed source.")
    total_frames = source_info.frames
    start_seconds = options.get("start_seconds")
    end_seconds = options.get("end_seconds")
    start_frame = 0 if start_seconds is None else round(float(start_seconds) * probe.sample_rate)
    end_frame = (
        total_frames if end_seconds is None else round(float(end_seconds) * probe.sample_rate)
    )
    if start_frame < 0 or end_frame <= start_frame or end_frame > total_frames:
        raise ValueError("The selected time range must be non-empty and within the source.")

    input_path = source_path
    selected_start = float(options.get("_source_start_seconds", 0.0))
    selected_end = float(options.get("_source_end_seconds", probe.duration_seconds))
    if start_frame != 0 or end_frame != total_frames or bool(options.get("_selected_section")):
        input_path = work_dir / "selected-section.wav"
        await _write_transform_section(
            source_path,
            input_path,
            start_frame,
            end_frame,
            int(options.get("edge_fade_ms", 10)),
            progress,
        )

    output_path = work_dir / "transformed.wav"
    tempo_factor = float(options.get("tempo_factor", 1.0))
    semitones = float(options.get("semitones", 0.0))

    async def report_engine_progress(fraction: float) -> bool:
        return await progress(0.05 + min(1.0, max(0.0, fraction)) * 0.85)

    engine_result = await run_soundstretch(
        input_path,
        output_path,
        executable=soundtouch_path,
        semitones=semitones,
        tempo_factor=tempo_factor,
        progress=report_engine_progress,
        timeout_seconds=int(options.get("_timeout_seconds", 21_600)),
        version=soundtouch_version,
    )
    input_quality = await asyncio.to_thread(inspect_quality, input_path)
    non_finite_samples = await asyncio.to_thread(_count_non_finite_samples, output_path)
    if non_finite_samples:
        raise ValueError("SoundTouch output contains NaN or infinite samples.")
    quality = await asyncio.to_thread(inspect_quality, output_path)
    peak = max(float(value) for value in quality["peak_amplitude"])
    normalization_gain_db = 0.0
    peak_ceiling = 10 ** (-1.01 / 20)
    if peak > peak_ceiling:
        normalization_gain_db = 20 * math.log10(peak_ceiling / peak)
        normalized_path = work_dir / "transformed-peak-limited.wav"

        async def report_normalization_progress(fraction: float) -> bool:
            return await progress(0.9 + min(1.0, max(0.0, fraction)) * 0.1)

        await _write_transform(
            output_path,
            normalized_path,
            report_normalization_progress,
            lambda data, _: data * 10 ** (normalization_gain_db / 20),
        )
        output_path = normalized_path
        quality = await asyncio.to_thread(inspect_quality, output_path)
        engine_result["output_subtype"] = soundfile.info(output_path).subtype
    elif not await progress(1.0):
        raise ProcessingCancelledError
    source_duration = (end_frame - start_frame) / probe.sample_rate
    result = {
        **engine_result,
        "source_start_seconds": selected_start + start_frame / probe.sample_rate,
        "source_end_seconds": selected_start + end_frame / probe.sample_rate,
        "source_duration_seconds": source_duration,
        "expected_duration_seconds": source_duration / tempo_factor,
        "input_quality": input_quality,
        "output_quality": quality,
        "non_finite_sample_count": non_finite_samples,
        "peak_adjustment_db": normalization_gain_db,
        "limitations": [
            "SoundTouch is a WSOLA-style time stretcher; transient and phase artifacts "
            "depend on source material and processing ratio."
        ],
    }
    return _result(
        [
            DspOutput(
                output_path,
                "processed_audio",
                {
                    "source_start_seconds": selected_start + start_frame / probe.sample_rate,
                    "source_end_seconds": selected_end,
                    "output_duration_seconds": engine_result["output_duration_seconds"],
                    "tempo_factor": tempo_factor,
                    "semitones": semitones,
                },
            )
        ],
        result,
        started,
    )


async def _write_transform_section(
    source_path: Path,
    output_path: Path,
    start_frame: int,
    end_frame: int,
    fade_ms: int,
    progress: Progress,
) -> None:
    with soundfile.SoundFile(source_path) as source:
        if start_frame < 0 or end_frame <= start_frame or end_frame > len(source):
            raise ValueError("The selected time range must be non-empty and within the source.")
        source.seek(start_frame)
        total_frames = end_frame - start_frame
        fade_frames = min(round(source.samplerate * fade_ms / 1000), total_frames // 2)
        with soundfile.SoundFile(
            output_path,
            mode="w",
            samplerate=source.samplerate,
            channels=source.channels,
            format="WAV",
            subtype="FLOAT",
        ) as target:
            written = 0
            while written < total_frames:
                block = source.read(
                    min(_BLOCK_FRAMES, total_frames - written),
                    dtype="float32",
                    always_2d=True,
                )
                if not len(block):
                    raise ValueError("Source ended before the selected time range.")
                if fade_frames:
                    positions = np.arange(written, written + len(block))
                    gains = np.ones(len(block), dtype=np.float32)
                    fade_in = positions < fade_frames
                    gains[fade_in] *= positions[fade_in] / max(1, fade_frames)
                    fade_out = positions >= total_frames - fade_frames
                    gains[fade_out] *= (total_frames - 1 - positions[fade_out]) / max(
                        1, fade_frames
                    )
                    block *= gains[:, np.newaxis]
                target.write(block)
                written += len(block)
                if not await progress(written / total_frames * 0.05):
                    raise ProcessingCancelledError


def _make_chunks(
    source_asset_id: uuid.UUID,
    probe: ProbeResult,
    options: dict[str, Any],
    operation: str,
    source_path: Path,
) -> list[AudioChunk]:
    mode = options.get("mode", "chunks" if operation == "chunks" else "equal")
    frame_count = round(probe.duration_seconds * probe.sample_rate)
    if mode == "timestamps":
        boundaries = [0.0, *options["boundaries_seconds"], probe.duration_seconds]
        if boundaries != sorted(set(boundaries)) or boundaries[-1] > probe.duration_seconds:
            raise ValueError("Split boundaries must be unique and within source duration.")
        return _chunks_from_boundaries(source_asset_id, probe, boundaries)
    if mode in {"equal", "chunks"}:
        chunk_seconds = options.get("segment_seconds", options.get("chunk_seconds"))
        if chunk_seconds is None:
            raise ValueError(f"{mode} splitting requires a segment duration.")
        return plan_chunks(
            source_asset_id,
            frame_count=frame_count,
            sample_rate=probe.sample_rate,
            chunk_seconds=float(chunk_seconds),
            overlap_seconds=float(options.get("overlap_seconds", 0)),
            min_chunk_seconds=options.get("min_segment_seconds"),
            max_chunk_seconds=options.get("max_segment_seconds"),
        )
    if mode == "silence":
        analysis = detect_silence(
            source_path,
            threshold_dbfs=float(options.get("silence_threshold_dbfs", -45)),
            min_silence_ms=options.get("min_silence_ms", 300),
        )
        boundaries = [0.0]
        boundaries.extend(region["end_seconds"] for region in analysis["regions"])
        boundaries.append(probe.duration_seconds)
        unique = sorted({value for value in boundaries if 0 <= value <= probe.duration_seconds})
        return _chunks_from_boundaries(source_asset_id, probe, unique)
    raise ValueError("Unsupported splitting mode.")


def _chunks_from_boundaries(
    source_asset_id: uuid.UUID,
    probe: ProbeResult,
    boundaries: list[float],
) -> list[AudioChunk]:
    chunks = []
    for start, end in zip(boundaries, boundaries[1:], strict=False):
        start_frame = round(start * probe.sample_rate)
        end_frame = round(end * probe.sample_rate)
        if end_frame <= start_frame:
            continue
        chunks.append(
            AudioChunk(
                id=uuid.uuid4(),
                source_asset_id=source_asset_id,
                start_sample=start_frame,
                end_sample=end_frame,
                sample_rate=probe.sample_rate,
                processing_config={"mode": "timestamps"},
            )
        )
    if not chunks:
        raise ValueError("Split boundaries produced no non-empty segments.")
    return chunks


async def _prepare_analysis_chunk(
    source_path: Path,
    work_dir: Path,
    index: int,
    options: dict[str, Any],
    channel_layout: str | None,
    metadata: dict[str, Any],
) -> tuple[Path, dict[str, Any]]:
    target_rate = options.get("sample_rate")
    target_channels = str(options.get("channels", "preserve"))
    if target_rate is None and target_channels == "preserve":
        return source_path, metadata
    await _ensure_in_memory_limit(source_path)
    samples, sample_rate = await asyncio.to_thread(
        soundfile.read,
        source_path,
        dtype="float64",
        always_2d=True,
    )
    transformed = AudioSignal(samples, sample_rate)
    if target_channels != "preserve":
        converted = change_channels(
            transformed,
            target_channels,  # type: ignore[arg-type]
            channel_layout=channel_layout,
        )
        if not isinstance(converted, AudioSignal):
            raise ValueError("Chunk preparation must produce a single channel layout.")
        transformed = converted
    if target_rate is not None:
        transformed = resample_audio(transformed, int(target_rate))
    prepared_path = work_dir / f"segment-{index:05d}-prepared.wav"
    await asyncio.to_thread(
        _write_audio, prepared_path, transformed.samples, transformed.sample_rate
    )
    metadata["output_sample_rate"] = transformed.sample_rate
    metadata["output_channel_count"] = transformed.channel_count
    return prepared_path, metadata


async def _write_range(
    source_path: Path,
    output_path: Path,
    start_frame: int,
    end_frame: int,
    progress: Progress,
    *,
    progress_start: float = 0.0,
    progress_span: float = 1.0,
) -> None:
    with soundfile.SoundFile(source_path) as source:
        if start_frame < 0 or end_frame <= start_frame or end_frame > len(source):
            raise ValueError("Sample range is outside the source audio.")
        source.seek(start_frame)
        with soundfile.SoundFile(
            output_path,
            mode="w",
            samplerate=source.samplerate,
            channels=source.channels,
            format="WAV",
            subtype="PCM_24",
        ) as target:
            offset = start_frame
            while offset < end_frame:
                data = source.read(
                    min(_BLOCK_FRAMES, end_frame - offset),
                    dtype="float32",
                    always_2d=True,
                )
                if not len(data):
                    raise ValueError("Source ended before the requested sample range.")
                target.write(data)
                offset += len(data)
                if not await progress(
                    progress_start
                    + progress_span * (offset - start_frame) / (end_frame - start_frame)
                ):
                    raise ProcessingCancelledError


async def _concatenate_paths(
    source_paths: list[Path],
    output_path: Path,
    crossfade_seconds: float,
    progress: Progress,
) -> tuple[int, int, int]:
    if not math.isfinite(crossfade_seconds) or crossfade_seconds < 0:
        raise ValueError("crossfade_seconds must be finite and non-negative.")
    sources = [soundfile.SoundFile(path) for path in source_paths]
    try:
        sample_rate = sources[0].samplerate
        channels = sources[0].channels
        if any(
            source.samplerate != sample_rate or source.channels != channels for source in sources
        ):
            raise ValueError("Prepared concatenation inputs must have compatible PCM settings.")
        crossfade_frames = round(crossfade_seconds * sample_rate)
        if any(len(source) <= crossfade_frames for source in sources):
            raise ValueError("Each source must be longer than the requested crossfade.")
        total_source_frames = sum(len(source) for source in sources)
        expected_frames = total_source_frames - crossfade_frames * (len(sources) - 1)
        pending_tail: NDArray[Any] | None = None
        completed_source_frames = 0
        written_frames = 0
        with soundfile.SoundFile(
            output_path,
            mode="w",
            samplerate=sample_rate,
            channels=channels,
            format="WAV",
            subtype="PCM_24",
        ) as target:
            for source_index, source in enumerate(sources):
                offset = 0
                if source_index and crossfade_frames:
                    if pending_tail is None:
                        raise RuntimeError("Crossfade tail was not retained.")
                    head = source.read(crossfade_frames, dtype="float64", always_2d=True)
                    if len(head) != crossfade_frames:
                        raise ValueError("Source is too short for the requested crossfade.")
                    phase = np.linspace(0.0, math.pi / 2, crossfade_frames, endpoint=True)[
                        :, np.newaxis
                    ]
                    target.write(
                        np.clip(
                            pending_tail * np.cos(phase) + head * np.sin(phase),
                            -1.0,
                            1.0,
                        )
                    )
                    written_frames += crossfade_frames
                    offset = crossfade_frames
                    pending_tail = None
                rolling_tail: NDArray[Any] | None = None
                while True:
                    block = source.read(_BLOCK_FRAMES, dtype="float64", always_2d=True)
                    if not len(block):
                        break
                    if crossfade_frames:
                        combined = (
                            block
                            if rolling_tail is None
                            else np.concatenate((rolling_tail, block), axis=0)
                        )
                        if len(combined) > crossfade_frames:
                            prefix = combined[:-crossfade_frames]
                            target.write(np.clip(prefix, -1.0, 1.0))
                            written_frames += len(prefix)
                            rolling_tail = combined[-crossfade_frames:]
                        else:
                            rolling_tail = combined
                    else:
                        target.write(np.clip(block, -1.0, 1.0))
                        written_frames += len(block)
                    offset += len(block)
                    completed_source_frames += len(block)
                    if not await progress(
                        (completed_source_frames + source_index * crossfade_frames)
                        / max(1, total_source_frames)
                    ):
                        raise ProcessingCancelledError
                if crossfade_frames:
                    if rolling_tail is None or len(rolling_tail) != crossfade_frames:
                        raise ValueError("Source is too short to retain the crossfade region.")
                    if source_index + 1 == len(sources):
                        target.write(np.clip(rolling_tail, -1.0, 1.0))
                        written_frames += len(rolling_tail)
                    else:
                        pending_tail = rolling_tail
            if written_frames != expected_frames:
                raise RuntimeError(
                    "Concatenated output frame count did not match its expected duration."
                )
        return expected_frames, sample_rate, channels
    finally:
        for source in sources:
            source.close()


async def _write_transform(
    source_path: Path,
    output_path: Path,
    progress: Progress,
    transform: Callable[[NDArray[Any], int], NDArray[Any]],
) -> int:
    clipped = 0
    with soundfile.SoundFile(source_path) as source:
        with soundfile.SoundFile(
            output_path,
            mode="w",
            samplerate=source.samplerate,
            channels=source.channels,
            format="WAV",
            subtype="PCM_24",
        ) as target:
            offset = 0
            total = len(source)
            while True:
                data = source.read(_BLOCK_FRAMES, dtype="float64", always_2d=True)
                if not len(data):
                    break
                transformed = transform(data, offset)
                clipped += int(np.count_nonzero(np.abs(transformed) > 1.0))
                target.write(np.clip(transformed, -1.0, 1.0))
                offset += len(data)
                if not await progress(offset / max(1, total)):
                    raise ProcessingCancelledError
    return clipped


async def _write_channels(
    source_path: Path,
    output_path: Path,
    mode: str,
    channel_layout: str | None,
    progress: Progress,
) -> list[Path]:
    with soundfile.SoundFile(source_path) as source:
        source_rate = source.samplerate
        source_channels = source.channels
        source_frames = len(source)
        output_channels = (
            source_channels if mode == "split" else 1 if mode in {"mono", "left", "right"} else 2
        )
        paths = (
            [
                output_path.with_name(f"{output_path.stem}-{index + 1}.wav")
                for index in range(source_channels)
            ]
            if mode == "split"
            else [output_path]
        )
        targets = [
            soundfile.SoundFile(
                path,
                mode="w",
                samplerate=source_rate,
                channels=1 if mode == "split" else output_channels,
                format="WAV",
                subtype="PCM_24",
            )
            for path in paths
        ]
        try:
            offset = 0
            while True:
                block = source.read(_BLOCK_FRAMES, dtype="float64", always_2d=True)
                if not len(block):
                    break
                converted = change_channels(
                    AudioSignal(block, source_rate),
                    mode,  # type: ignore[arg-type]
                    channel_layout=channel_layout,
                )
                if isinstance(converted, tuple):
                    for target, channel in zip(targets, converted, strict=True):
                        target.write(np.clip(channel.samples, -1.0, 1.0))
                else:
                    targets[0].write(np.clip(converted.samples, -1.0, 1.0))
                offset += len(block)
                if not await progress(offset / max(1, source_frames)):
                    raise ProcessingCancelledError
        finally:
            for target in targets:
                target.close()
    return paths


async def _filter_file(
    source_path: Path,
    work_dir: Path,
    options: dict[str, Any],
    probe: ProbeResult,
    progress: Progress,
    started: float,
) -> DspResult:
    output = work_dir / "filtered.wav"
    if options.get("zero_phase"):
        await _ensure_in_memory_limit(source_path)
        data, sample_rate = await asyncio.to_thread(
            soundfile.read, source_path, dtype="float64", always_2d=True
        )
        cutoff = options["cutoff_hz"]
        if isinstance(cutoff, list):
            cutoff = tuple(cutoff)
        filtered = apply_filter(
            AudioSignal(data, sample_rate),
            options["kind"],
            cutoff_hz=cutoff,
            order=options.get("order", 4),
            zero_phase=True,
        )
        await asyncio.to_thread(_write_audio, output, filtered.samples, sample_rate)
    else:
        kind = options["kind"]
        cutoff = options["cutoff_hz"]
        if isinstance(cutoff, list):
            cutoff = tuple(cutoff)
        # Design coefficients once; carry filter state between bounded PCM blocks.
        sample_rate = probe.sample_rate
        if kind == "notch":
            b, a = signal.iirnotch(
                float(cutoff), Q=max(1.0, float(options.get("order", 4))), fs=sample_rate
            )
            sos = signal.tf2sos(b, a)
        else:
            sos = signal.butter(
                options.get("order", 4),
                cutoff,
                btype=kind,
                fs=sample_rate,
                output="sos",
            )
        await _write_filter_stream(source_path, output, sos, progress)
    return _result(
        [DspOutput(output, "processed_audio", {})],
        {
            "kind": options["kind"],
            "cutoff_hz": options["cutoff_hz"],
            "order": options.get("order", 4),
            "zero_phase": bool(options.get("zero_phase")),
            "phase_note": (
                "Zero-phase filtering may produce boundary transients."
                if options.get("zero_phase")
                else "Causal filtering introduces phase shift."
            ),
        },
        started,
    )


async def _write_filter_stream(
    source_path: Path,
    output_path: Path,
    sos: NDArray[Any],
    progress: Progress,
) -> None:
    with soundfile.SoundFile(source_path) as source:
        zi: NDArray[Any] | None = None
        with soundfile.SoundFile(
            output_path,
            mode="w",
            samplerate=source.samplerate,
            channels=source.channels,
            format="WAV",
            subtype="PCM_24",
        ) as target:
            offset = 0
            total = len(source)
            while True:
                block = source.read(_BLOCK_FRAMES, dtype="float64", always_2d=True)
                if not len(block):
                    break
                if zi is None:
                    zi = (
                        signal.sosfilt_zi(sos)[:, :, np.newaxis]
                        * block[0][np.newaxis, np.newaxis, :]
                    )
                filtered, zi = signal.sosfilt(sos, block, axis=0, zi=zi)
                target.write(np.clip(filtered, -1.0, 1.0))
                offset += len(block)
                if not await progress(offset / max(1, total)):
                    raise ProcessingCancelledError


def _measure_for_normalization(path: Path, method: str) -> dict[str, float]:
    peak = 0.0
    square_sum = 0.0
    sample_count = 0
    with soundfile.SoundFile(path) as audio:
        while True:
            block = audio.read(_BLOCK_FRAMES, dtype="float64", always_2d=True)
            if not len(block):
                break
            peak = max(peak, float(np.max(np.abs(block))))
            square_sum += float(np.square(block).sum())
            sample_count += block.size
    rms = math.sqrt(square_sum / sample_count) if sample_count else 0.0
    return {"peak": peak, "rms": rms}


def _fade_curve(phase: NDArray[Any], curve: str, *, rising: bool) -> NDArray[Any]:
    bounded = np.clip(phase, 0.0, 1.0)
    if not rising:
        bounded = 1.0 - bounded
    if curve == "linear":
        return bounded
    if curve == "equal_power":
        return np.sin(bounded * math.pi / 2)
    raise ValueError("curve must be 'linear' or 'equal_power'.")


async def _copy_audio(source: Path, destination: Path, progress: Progress) -> None:
    await _write_range(source, destination, 0, soundfile.info(source).frames, progress)


async def _ensure_in_memory_limit(path: Path) -> None:
    info = await asyncio.to_thread(soundfile.info, path)
    estimate = info.frames * info.channels * 8 * 3
    if estimate > MAX_IN_MEMORY_BYTES:
        raise ValueError(
            "This operation requires full-signal filtering/resampling and exceeds the "
            "configured in-memory processing limit."
        )


def _write_audio(path: Path, samples: NDArray[Any], sample_rate: int) -> None:
    soundfile.write(path, np.clip(samples, -1.0, 1.0), sample_rate, subtype="PCM_24")


def _count_non_finite_samples(path: Path) -> int:
    count = 0
    with soundfile.SoundFile(path) as audio:
        while True:
            block = audio.read(_BLOCK_FRAMES, dtype="float64", always_2d=True)
            if not len(block):
                break
            count += int(np.count_nonzero(~np.isfinite(block)))
    return count


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, separators=(",", ":"), allow_nan=False), encoding="utf-8")


def _dbfs(value: float) -> float | None:
    return None if value <= 0 else 20.0 * math.log10(value)


def _result(
    outputs: list[DspOutput],
    result: dict[str, Any],
    started: float,
) -> DspResult:
    return DspResult(outputs, result, int((time.monotonic() - started) * 1000))
