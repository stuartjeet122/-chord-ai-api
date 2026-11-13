from __future__ import annotations

from typing import Any

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


def apply_pipeline(
    source: AudioSignal,
    operations: list[dict[str, Any]],
    *,
    channel_layout: str | None = None,
) -> tuple[AudioSignal, list[dict[str, Any]]]:
    """Apply validated operation dictionaries sequentially to a decoded signal."""
    current = source
    reports: list[dict[str, Any]] = []
    for index, operation in enumerate(operations):
        kind = operation.get("type")
        try:
            if kind == "trim":
                current = trim_audio(
                    current,
                    float(operation["start_seconds"]),
                    float(operation["end_seconds"]),
                )
                report: dict[str, Any] = {
                    "start_seconds": operation["start_seconds"],
                    "end_seconds": operation["end_seconds"],
                }
            elif kind == "gain":
                current, report = apply_gain(current, gain_db=float(operation["gain_db"]))
            elif kind == "fade":
                current = apply_fade(
                    current,
                    fade_in_seconds=float(operation.get("fade_in_seconds", 0)),
                    fade_out_seconds=float(operation.get("fade_out_seconds", 0)),
                    curve=operation.get("curve", "linear"),
                )
                report = {"curve": operation.get("curve", "linear")}
            elif kind == "filter":
                cutoff = operation["cutoff_hz"]
                if isinstance(cutoff, list):
                    cutoff = tuple(cutoff)
                current = apply_filter(
                    current,
                    operation["kind"],
                    cutoff_hz=cutoff,
                    order=int(operation.get("order", 4)),
                    zero_phase=bool(operation.get("zero_phase", False)),
                )
                report = {
                    "filter": operation["kind"],
                    "cutoff_hz": operation["cutoff_hz"],
                    "zero_phase": bool(operation.get("zero_phase", False)),
                }
            elif kind == "resample":
                current = resample_audio(current, int(operation["sample_rate"]))
                report = {"sample_rate": current.sample_rate}
            elif kind == "channels":
                converted = change_channels(
                    current,
                    operation["mode"],
                    channel_layout=channel_layout,
                )
                if not isinstance(converted, AudioSignal):
                    raise ValueError(
                        "Channel splitting cannot be used inside a single-output pipeline."
                    )
                current = converted
                report = {"mode": operation["mode"], "channels": current.channel_count}
            elif kind == "normalize":
                method = operation.get("method", "peak")
                if method == "peak":
                    current, report = normalize_peak(
                        current,
                        float(operation.get("target_dbfs", -1)),
                        ceiling_dbfs=float(operation.get("ceiling_dbfs", -1)),
                    )
                elif method == "rms":
                    current, report = normalize_rms(
                        current,
                        float(operation.get("target_dbfs", -18)),
                        ceiling_dbfs=float(operation.get("ceiling_dbfs", -1)),
                    )
                else:
                    raise ValueError("Unsupported normalization method.")
            else:
                raise ValueError(f"Unsupported pipeline operation at position {index}.")
            reports.append({"position": index, "type": kind, "result": report})
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Pipeline operation {index} ({kind}) failed: {exc}") from exc
    return current, reports
