from __future__ import annotations

import asyncio
import math
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

import soundfile

from music_api.audio.dsp.schemas import AudioTransformRequest
from music_api.audio.toolchain import AudioToolError
from music_api.music.theory import transpose_between_keys


def resolve_transform_parameters(request: AudioTransformRequest) -> dict[str, Any]:
    if request.source_key is not None and request.target_key is not None:
        shift = transpose_between_keys(
            ["C"],
            request.source_key,
            request.target_key,
        )["semitones"]
    else:
        shift = request.semitones or 0.0

    if request.source_bpm is not None and request.target_bpm is not None:
        tempo_factor = request.target_bpm / request.source_bpm
    else:
        tempo_factor = request.tempo_factor or 1.0
    if not 0.25 <= tempo_factor <= 4.0:
        raise ValueError("The resulting tempo factor must be between 0.25 and 4.")

    return {
        "semitones": float(shift),
        "source_key": request.source_key,
        "target_key": request.target_key,
        "tempo_factor": float(tempo_factor),
        "source_bpm": request.source_bpm,
        "target_bpm": request.target_bpm,
        "start_seconds": request.start_seconds,
        "end_seconds": request.end_seconds,
        "edge_fade_ms": request.edge_fade_ms,
        "output_format": request.output_format,
        "bitrate_kbps": request.bitrate_kbps,
        "engine": "SoundTouch SoundStretch",
    }


async def run_soundstretch(
    source: Path,
    destination: Path,
    *,
    executable: str,
    semitones: float,
    tempo_factor: float,
    progress: Callable[[float], Awaitable[bool]],
    timeout_seconds: int,
    version: str,
) -> dict[str, Any]:
    if not math.isfinite(semitones) or not -24 <= semitones <= 24:
        raise ValueError("Pitch shift must be between -24 and +24 semitones.")
    if not math.isfinite(tempo_factor) or not 0.25 <= tempo_factor <= 4:
        raise ValueError("Tempo factor must be between 0.25 and 4.")
    info = soundfile.info(source)
    if info.samplerate < 8000 or info.samplerate > 48000:
        raise ValueError("SoundTouch supports sample rates from 8 kHz through 48 kHz.")
    if info.channels not in {1, 2}:
        raise ValueError("SoundTouch supports mono or stereo audio only.")

    tempo_percent = (tempo_factor - 1.0) * 100.0
    arguments = [
        executable,
        str(source),
        str(destination),
        f"-tempo={tempo_percent:+.6f}",
        f"-pitch={semitones:+.6f}",
    ]
    try:
        process = await asyncio.create_subprocess_exec(
            *arguments,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except OSError as exc:
        raise AudioToolError(
            "audio_transform_engine_unavailable",
            "SoundTouch SoundStretch is unavailable; install it and configure SOUNDTOUCH_PATH.",
        ) from exc

    communication = asyncio.create_task(process.communicate())
    expected_frames = max(1, round(info.frames / tempo_factor))
    expected_bytes = expected_frames * info.channels * 4 + 128
    last_progress = -1
    try:
        async with asyncio.timeout(timeout_seconds):
            while not communication.done():
                produced_bytes = destination.stat().st_size if destination.exists() else 0
                fraction = min(0.95, produced_bytes / expected_bytes)
                progress_value = int(fraction * 1000)
                if progress_value != last_progress:
                    if not await progress(fraction):
                        process.terminate()
                        await communication
                        raise AudioToolError("job_cancelled", "Audio transformation was cancelled.")
                    last_progress = progress_value
                await asyncio.sleep(0.25)
            stdout, stderr = await communication
    except TimeoutError as exc:
        await _terminate_process(process)
        await communication
        raise AudioToolError(
            "audio_transform_timeout", "Audio transformation exceeded its time limit."
        ) from exc
    except asyncio.CancelledError:
        await _terminate_process(process)
        await communication
        raise
    except AudioToolError:
        await _terminate_process(process)
        raise
    if process.returncode != 0:
        message = (stderr or stdout).decode("utf-8", errors="replace")[-2000:]
        raise AudioToolError(
            "audio_transform_failed",
            f"SoundTouch could not process the audio: {message.strip()[:200]}",
        )
    if not destination.is_file() or destination.stat().st_size <= 44:
        raise AudioToolError(
            "audio_transform_invalid_output", "SoundTouch produced an empty audio file."
        )
    try:
        output_info = soundfile.info(destination)
    except (RuntimeError, OSError) as exc:
        raise AudioToolError(
            "audio_transform_invalid_output", "SoundTouch produced an unreadable WAV file."
        ) from exc
    if output_info.frames <= 0 or output_info.channels != info.channels:
        raise AudioToolError(
            "audio_transform_invalid_output",
            "SoundTouch output has invalid duration or changed the channel count.",
        )
    if not await progress(1.0):
        raise AudioToolError("job_cancelled", "Audio transformation was cancelled.")

    return {
        "engine": "SoundTouch SoundStretch",
        "engine_version": version,
        "tempo_factor": tempo_factor,
        "tempo_percent": tempo_percent,
        "semitones": semitones,
        "input_frames": info.frames,
        "output_frames": output_info.frames,
        "sample_rate": output_info.samplerate,
        "channel_count": output_info.channels,
        "output_duration_seconds": output_info.frames / output_info.samplerate,
        "output_subtype": output_info.subtype,
    }


async def _terminate_process(process: asyncio.subprocess.Process) -> None:
    if process.returncode is not None:
        return
    process.terminate()
    try:
        await asyncio.wait_for(process.wait(), timeout=5)
    except TimeoutError:
        process.kill()
        await process.wait()
