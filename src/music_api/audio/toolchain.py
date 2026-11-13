from __future__ import annotations

import asyncio
import json
import logging
import math
import re
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from music_api.config import Settings

logger = logging.getLogger(__name__)
ProgressCallback = Callable[[float], Awaitable[bool]]


class AudioToolError(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class ProbeResult:
    container: str
    codec: str
    codec_profile: str | None
    duration_seconds: float
    bitrate: int | None
    sample_rate: int
    channels: int
    channel_layout: str | None
    bit_depth: int | None
    audio_stream_count: int
    file_size_bytes: int
    raw_format_tags: dict[str, str]
    variable_bitrate: bool | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "container": self.container,
            "codec": self.codec,
            "codec_profile": self.codec_profile,
            "duration_seconds": self.duration_seconds,
            "bitrate": self.bitrate,
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "channel_layout": self.channel_layout,
            "bit_depth": self.bit_depth,
            "audio_stream_count": self.audio_stream_count,
            "file_size_bytes": self.file_size_bytes,
            "variable_bitrate": self.variable_bitrate,
        }


class AudioToolchain:
    def __init__(self, settings: Settings) -> None:
        self.ffmpeg_path = settings.ffmpeg_path
        self.ffprobe_path = settings.ffprobe_path
        self.timeout_seconds = settings.audio_conversion_timeout_seconds
        self.max_duration_seconds = settings.max_audio_duration_seconds

    async def versions(self) -> tuple[str, str]:
        ffmpeg = await self._version(self.ffmpeg_path, "ffmpeg")
        ffprobe = await self._version(self.ffprobe_path, "ffprobe")
        return ffmpeg, ffprobe

    async def availability(self) -> dict[str, str]:
        checks: dict[str, str] = {}
        for label, executable in (
            ("ffmpeg", self.ffmpeg_path),
            ("ffprobe", self.ffprobe_path),
        ):
            try:
                await self._version(executable, label)
                checks[label] = "ok"
            except AudioToolError:
                checks[label] = "unavailable"
        return checks

    async def _version(self, executable: str, label: str) -> str:
        try:
            stdout, _ = await self._run(
                [executable, "-version"],
                timeout_seconds=10,
                error_code="audio_tools_unavailable",
                error_message=(
                    f"{label} is unavailable; install it and configure its executable path."
                ),
            )
        except AudioToolError:
            raise
        return stdout.decode("utf-8", errors="replace").splitlines()[0][:128]

    async def inspect(self, source: Path) -> ProbeResult:
        stdout, _ = await self._run(
            [
                self.ffprobe_path,
                "-v",
                "error",
                "-show_entries",
                (
                    "format=format_name,duration,bit_rate,size:format_tags:stream=codec_type,"
                    "codec_name,profile,sample_rate,channels,channel_layout,bits_per_sample,"
                    "bits_per_raw_sample,bit_rate"
                ),
                "-of",
                "json",
                str(source),
            ],
            timeout_seconds=min(self.timeout_seconds, 120),
            error_code="invalid_audio",
            error_message="FFprobe could not parse a valid audio file.",
        )
        try:
            data = json.loads(stdout)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise AudioToolError(
                "invalid_audio", "FFprobe returned invalid inspection data."
            ) from exc

        streams = [
            stream for stream in data.get("streams", []) if stream.get("codec_type") == "audio"
        ]
        if not streams:
            raise AudioToolError("no_audio_stream", "The uploaded file contains no audio stream.")
        stream = streams[0]
        format_info = data.get("format", {})
        try:
            duration = float(stream.get("duration") or format_info.get("duration"))
            sample_rate = int(stream["sample_rate"])
            channels = int(stream["channels"])
            file_size = int(format_info.get("size") or source.stat().st_size)
        except (KeyError, TypeError, ValueError, OSError) as exc:
            raise AudioToolError(
                "invalid_audio_metadata", "Audio technical metadata is incomplete."
            ) from exc
        if not all(math.isfinite(value) for value in (duration,)):
            raise AudioToolError("invalid_audio_metadata", "Audio duration is not finite.")
        if duration <= 0 or duration > self.max_duration_seconds:
            raise AudioToolError(
                "audio_duration_limit",
                f"Audio duration must be between 0 and {self.max_duration_seconds} seconds.",
            )
        if file_size <= 0 or sample_rate <= 0 or channels <= 0:
            raise AudioToolError("invalid_audio_metadata", "Audio technical metadata is invalid.")

        bitrate = _optional_int(stream.get("bit_rate")) or _optional_int(
            format_info.get("bit_rate")
        )
        raw_tags = format_info.get("tags", {})
        return ProbeResult(
            container=str(format_info.get("format_name") or "unknown")[:128],
            codec=str(stream.get("codec_name") or "unknown")[:64],
            codec_profile=str(stream["profile"])[:128] if stream.get("profile") else None,
            duration_seconds=duration,
            bitrate=bitrate,
            sample_rate=sample_rate,
            channels=channels,
            channel_layout=(
                str(stream["channel_layout"])[:64] if stream.get("channel_layout") else None
            ),
            bit_depth=_optional_int(stream.get("bits_per_raw_sample"))
            or _optional_int(stream.get("bits_per_sample")),
            audio_stream_count=len(streams),
            file_size_bytes=file_size,
            raw_format_tags={str(key)[:128]: str(value)[:4096] for key, value in raw_tags.items()},
            variable_bitrate=None,
        )

    async def verify_decodable(
        self,
        source: Path,
        progress: ProgressCallback | None = None,
    ) -> None:
        arguments = [
            self.ffmpeg_path,
            "-hide_banner",
            "-nostdin",
            "-v",
            "error",
            "-xerror",
            "-stats_period",
            "1",
            "-i",
            str(source),
            "-map",
            "0:a:0",
            "-progress",
            "pipe:1",
            "-nostats",
            "-f",
            "null",
            "-",
        ]
        if progress is None:
            await self._run(
                arguments,
                timeout_seconds=self.timeout_seconds,
                error_code="corrupt_audio",
                error_message="FFmpeg could not decode the complete audio stream.",
            )
            return
        await self._run_with_progress(
            arguments,
            timeout_seconds=self.timeout_seconds,
            progress=progress,
            error_code="corrupt_audio",
            error_message="FFmpeg could not decode the complete audio stream.",
        )

    async def convert(
        self,
        source: Path,
        destination: Path,
        options: dict[str, Any],
        source_probe: ProbeResult,
        progress: ProgressCallback,
    ) -> tuple[str, int]:
        ffmpeg_version, _ = await self.versions()
        destination.parent.mkdir(parents=True, exist_ok=True)
        output_format = str(options["output_format"])
        codec = _encoder(output_format, int(options["bit_depth"]), bool(options["float32"]))
        requested_rate = options.get("sample_rate")
        sample_rate = int(requested_rate) if requested_rate else None
        if (
            sample_rate is not None
            and sample_rate > source_probe.sample_rate
            and not bool(options.get("allow_upsampling"))
        ):
            sample_rate = source_probe.sample_rate

        arguments = [
            self.ffmpeg_path,
            "-hide_banner",
            "-nostdin",
            "-y",
            "-stats_period",
            "1",
            "-i",
            str(source),
            "-map",
            "0:a:0",
            "-threads",
            "2",
            "-vn",
            "-sn",
            "-dn",
        ]
        start = options.get("trim_start_seconds")
        end = options.get("trim_end_seconds")
        if start is not None:
            arguments.extend(["-ss", _number(float(start))])
        if end is not None:
            trim_length = float(end) - float(start or 0)
            arguments.extend(["-t", _number(trim_length)])

        channels = str(options.get("channels", "original"))
        if channels != "original":
            arguments.extend(["-ac", "1" if channels == "mono" else "2"])
        if sample_rate is not None:
            arguments.extend(["-ar", str(sample_rate)])
        if bool(options.get("normalize")):
            arguments.extend(["-af", "loudnorm=I=-16:TP=-1.5:LRA=11"])
        arguments.extend(["-c:a", codec])
        if output_format in {"mp3", "aac", "opus"}:
            bitrate = int(options.get("bitrate_kbps") or 192)
            arguments.extend(["-b:a", f"{bitrate}k"])
        if output_format == "flac":
            arguments.extend(["-compression_level", "5"])
        if output_format == "wav":
            arguments.extend(["-f", "wav"])
        arguments.extend(["-progress", "pipe:1", "-nostats", str(destination)])

        started = time.monotonic()
        try:
            process = await asyncio.create_subprocess_exec(
                *arguments,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except OSError as exc:
            raise AudioToolError(
                "audio_tools_unavailable",
                "FFmpeg could not be started; verify FFMPEG_PATH.",
            ) from exc

        assert process.stdout is not None
        assert process.stderr is not None
        stderr_task = asyncio.create_task(_read_limited(process.stderr, 16_384))
        stderr_tail = bytearray()
        try:
            async with asyncio.timeout(self.timeout_seconds):
                async for raw_line in process.stdout:
                    line = raw_line.decode("ascii", errors="ignore").strip()
                    if line.startswith("out_time_ms="):
                        try:
                            elapsed = int(line.partition("=")[2]) / 1_000_000
                            if not await progress(elapsed):
                                raise AudioToolError(
                                    "job_cancelled", "Audio conversion was cancelled."
                                )
                        except ValueError:
                            logger.debug("Ignoring malformed FFmpeg progress line")
                return_code = await process.wait()
                stderr_tail.extend(await stderr_task)
            if return_code != 0:
                logger.warning(
                    "FFmpeg conversion failed: %s", stderr_tail.decode(errors="replace")[-2000:]
                )
                raise AudioToolError(
                    "conversion_failed", "FFmpeg could not convert this audio file."
                )
            if not destination.is_file() or destination.stat().st_size == 0:
                raise AudioToolError(
                    "invalid_conversion_output", "FFmpeg produced an empty output file."
                )
            return ffmpeg_version, int((time.monotonic() - started) * 1000)
        except TimeoutError as exc:
            await _terminate(process)
            stderr_tail.extend(await stderr_task)
            logger.warning(
                "FFmpeg conversion timed out: %s", stderr_tail.decode(errors="replace")[-2000:]
            )
            raise AudioToolError(
                "conversion_timeout", "Audio conversion exceeded its time limit."
            ) from exc
        except asyncio.CancelledError:
            await _terminate(process)
            await stderr_task
            raise
        except AudioToolError:
            if process.returncode is None:
                await _terminate(process)
            if not stderr_task.done():
                await stderr_task
            raise
        except Exception:
            await _terminate(process)
            if not stderr_task.done():
                await stderr_task
            raise

    async def normalize_loudness(
        self,
        source: Path,
        destination: Path,
        *,
        target_lufs: float,
        true_peak_dbfs: float,
        loudness_range_lu: float,
        limit: bool,
        sample_rate: int,
        progress: ProgressCallback,
    ) -> dict[str, Any]:
        if sample_rate <= 0:
            raise AudioToolError("invalid_loudness_options", "sample_rate must be positive.")
        for value, name, minimum, maximum in (
            (target_lufs, "target_lufs", -36.0, -5.0),
            (true_peak_dbfs, "true_peak_dbfs", -9.0, 0.0),
            (loudness_range_lu, "loudness_range_lu", 1.0, 20.0),
        ):
            if not math.isfinite(value) or not minimum <= value <= maximum:
                raise AudioToolError(
                    "invalid_loudness_options", f"{name} is outside supported limits."
                )
        filter_options = (
            f"I={_number(target_lufs)}:TP={_number(true_peak_dbfs)}:"
            f"LRA={_number(loudness_range_lu)}"
        )
        measurement_args = [
            self.ffmpeg_path,
            "-hide_banner",
            "-nostdin",
            "-i",
            str(source),
            "-map",
            "0:a:0",
            "-af",
            f"loudnorm={filter_options}:print_format=json",
            "-f",
            "null",
            "-",
        ]
        _, measurement_output = await self._run(
            measurement_args,
            timeout_seconds=self.timeout_seconds,
            error_code="loudness_measurement_failed",
            error_message="FFmpeg could not measure EBU R128 loudness.",
        )
        measured = _parse_loudnorm_stats(measurement_output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        normalization_filter = (
            f"loudnorm={filter_options}:measured_I={measured['input_i']}:"
            f"measured_TP={measured['input_tp']}:"
            f"measured_LRA={max(1.0, float(measured['input_lra']))}:"
            f"measured_thresh={measured['input_thresh']}:offset={measured['target_offset']}:"
            f"linear={'false' if limit else 'true'}:print_format=json"
        )
        normalize_args = [
            self.ffmpeg_path,
            "-hide_banner",
            "-nostdin",
            "-y",
            "-i",
            str(source),
            "-map",
            "0:a:0",
            "-threads",
            "2",
            "-ar",
            str(sample_rate),
            "-vn",
            "-sn",
            "-dn",
            "-af",
            normalization_filter,
            "-c:a",
            "pcm_f32le",
            "-f",
            "wav",
            "-progress",
            "pipe:1",
            "-nostats",
            str(destination),
        ]
        output = await self._run_with_progress(
            normalize_args,
            timeout_seconds=self.timeout_seconds,
            progress=progress,
            error_code="loudness_normalization_failed",
            error_message="FFmpeg could not normalize integrated loudness.",
        )
        normalized = _parse_loudnorm_stats(output)
        if not destination.is_file() or destination.stat().st_size == 0:
            raise AudioToolError(
                "invalid_normalization_output",
                "FFmpeg produced an empty loudness-normalized file.",
            )
        return {
            "method": "ebu_r128_loudnorm",
            "target_lufs": target_lufs,
            "target_true_peak_dbfs": true_peak_dbfs,
            "target_loudness_range_lu": loudness_range_lu,
            "limiting_requested": limit,
            "integrated_loudness_input_lufs": float(measured["input_i"]),
            "integrated_loudness_output_lufs": float(normalized["output_i"]),
            "loudness_range_input_lu": float(measured["input_lra"]),
            "loudness_range_output_lu": float(normalized["output_lra"]),
            "true_peak_input_dbfs": float(measured["input_tp"]),
            "true_peak_output_dbfs": float(normalized["output_tp"]),
            "target_offset_lu": float(normalized["target_offset"]),
        }

    async def _run(
        self,
        arguments: list[str],
        *,
        timeout_seconds: int,
        error_code: str,
        error_message: str,
    ) -> tuple[bytes, bytes]:
        try:
            process = await asyncio.create_subprocess_exec(
                *arguments,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except OSError as exc:
            raise AudioToolError(
                "audio_tools_unavailable",
                "FFmpeg or FFprobe is unavailable; verify FFMPEG_PATH and FFPROBE_PATH.",
            ) from exc
        try:
            if process.stdout is None or process.stderr is None:
                raise AudioToolError(
                    "audio_tool_failed", "Audio tool output could not be captured."
                )
            stdout_task = asyncio.create_task(_read_bounded(process.stdout, 4 * 1024 * 1024))
            stderr_task = asyncio.create_task(_read_bounded(process.stderr, 16 * 1024))
            try:
                async with asyncio.timeout(timeout_seconds):
                    stdout_result, stderr_result = await asyncio.gather(stdout_task, stderr_task)
                    return_code = await process.wait()
            except TimeoutError:
                await _terminate(process)
                await asyncio.gather(stdout_task, stderr_task, return_exceptions=True)
                raise
        except TimeoutError as exc:
            raise AudioToolError(
                "audio_tool_timeout", "Audio inspection exceeded its time limit."
            ) from exc
        if stdout_result[1]:
            raise AudioToolError(
                "audio_tool_output_too_large", "Audio inspection output exceeded its limit."
            )
        if return_code != 0:
            logger.info("Audio tool exited with %s: %s", return_code, stderr_result[0][-2000:])
            raise AudioToolError(error_code, error_message)
        return stdout_result[0], stderr_result[0]

    async def _run_with_progress(
        self,
        arguments: list[str],
        *,
        timeout_seconds: int,
        progress: ProgressCallback,
        error_code: str,
        error_message: str,
    ) -> bytes:
        try:
            process = await asyncio.create_subprocess_exec(
                *arguments,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except OSError as exc:
            raise AudioToolError(
                "audio_tools_unavailable",
                "FFmpeg or FFprobe is unavailable; verify FFMPEG_PATH and FFPROBE_PATH.",
            ) from exc
        assert process.stdout is not None
        assert process.stderr is not None
        stderr_task = asyncio.create_task(_read_limited(process.stderr, 16_384))
        stderr = b""
        try:
            async with asyncio.timeout(timeout_seconds):
                async for raw_line in process.stdout:
                    line = raw_line.decode("ascii", errors="ignore").strip()
                    if line.startswith("out_time_ms="):
                        try:
                            elapsed = int(line.partition("=")[2]) / 1_000_000
                            if not await progress(elapsed):
                                raise AudioToolError(
                                    "job_cancelled",
                                    "Audio processing was cancelled.",
                                )
                        except ValueError:
                            logger.debug("Ignoring malformed FFmpeg progress line")
                return_code = await process.wait()
                stderr = await stderr_task
        except TimeoutError as exc:
            await _terminate(process)
            await stderr_task
            raise AudioToolError(
                "audio_tool_timeout", "Audio inspection exceeded its time limit."
            ) from exc
        except asyncio.CancelledError:
            await _terminate(process)
            await stderr_task
            raise
        except AudioToolError:
            await _terminate(process)
            await stderr_task
            raise
        except Exception:
            await _terminate(process)
            await stderr_task
            raise
        if return_code != 0:
            logger.info("Audio tool exited with %s: %s", return_code, stderr[-2000:])
            raise AudioToolError(error_code, error_message)
        return stderr


def _optional_int(value: Any) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _parse_loudnorm_stats(output: bytes) -> dict[str, float]:
    text = output.decode("utf-8", errors="replace")
    candidates = re.findall(r"\{[^{}]*\"input_i\"[^{}]*\}", text, flags=re.DOTALL)
    if not candidates:
        raise AudioToolError(
            "loudness_measurement_failed",
            "FFmpeg did not return EBU R128 loudness measurements.",
        )
    try:
        values = json.loads(candidates[-1])
        parsed = {
            key: float(values[key])
            for key in (
                "input_i",
                "input_tp",
                "input_lra",
                "input_thresh",
                "target_offset",
                "output_i",
                "output_tp",
                "output_lra",
            )
        }
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise AudioToolError(
            "loudness_measurement_failed",
            "FFmpeg returned incomplete EBU R128 measurements.",
        ) from exc
    if not all(math.isfinite(value) for value in parsed.values()):
        raise AudioToolError(
            "loudness_measurement_failed",
            "FFmpeg returned non-finite EBU R128 measurements.",
        )
    return parsed


def _encoder(output_format: str, bit_depth: int, float32: bool) -> str:
    if output_format == "wav":
        if float32:
            return "pcm_f32le"
        return {16: "pcm_s16le", 24: "pcm_s24le", 32: "pcm_s32le"}[bit_depth]
    if output_format == "flac":
        return "flac"
    if output_format == "mp3":
        return "libmp3lame"
    if output_format == "aac":
        return "aac"
    if output_format == "opus":
        return "libopus"
    raise AudioToolError("unsupported_output_format", "The requested output format is unsupported.")


def _number(value: float) -> str:
    if not math.isfinite(value):
        raise AudioToolError("invalid_conversion_options", "Trim values must be finite numbers.")
    return format(value, ".6f").rstrip("0").rstrip(".") or "0"


async def _read_limited(stream: asyncio.StreamReader, limit: int) -> bytes:
    output = bytearray()
    while chunk := await stream.read(4096):
        remaining = limit - len(output)
        if remaining > 0:
            output.extend(chunk[:remaining])
    return bytes(output)


async def _read_bounded(stream: asyncio.StreamReader, limit: int) -> tuple[bytes, bool]:
    output = bytearray()
    exceeded = False
    while chunk := await stream.read(8192):
        remaining = limit - len(output)
        if remaining > 0:
            output.extend(chunk[:remaining])
        if len(chunk) > remaining:
            exceeded = True
    return bytes(output), exceeded


async def _terminate(process: asyncio.subprocess.Process) -> None:
    if process.returncode is not None:
        return
    process.terminate()
    try:
        await asyncio.wait_for(process.wait(), timeout=5)
    except TimeoutError:
        process.kill()
        await process.wait()
