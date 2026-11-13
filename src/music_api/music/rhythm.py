from __future__ import annotations

import math
from typing import Any, Literal

Subdivision = Literal["beat", "eighth", "sixteenth", "triplet"]
BeatUnit = Literal["quarter", "eighth", "dotted-quarter", "half"]

_SUPPORTED_DENOMINATORS = {2, 4, 8, 16}


def _require_bpm(value: float, *, field_name: str = "bpm") -> float:
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{field_name} must be a positive finite number.")
    return float(value)


def validate_bpm(value: float, *, minimum: float = 40.0, maximum: float = 300.0) -> float:
    bpm = _require_bpm(value)
    if bpm < minimum or bpm > maximum:
        raise ValueError(f"bpm must be between {minimum} and {maximum}.")
    return bpm


def parse_time_signature(signature: str | dict[str, Any]) -> dict[str, Any]:
    if isinstance(signature, dict):
        numerator = int(signature.get("numerator", 4))
        denominator = int(signature.get("denominator", 4))
        beats_per_measure = int(signature.get("beats_per_measure", numerator))
        beat_unit = str(signature.get("beat_unit", "quarter"))
    elif isinstance(signature, str):
        text = signature.strip()
        if "/" not in text:
            raise ValueError("Time signatures must use numerator/denominator format.")
        numerator_raw, denominator_raw = text.split("/", 1)
        numerator = int(numerator_raw)
        denominator = int(denominator_raw)
        if denominator in {8, 16} and numerator % 3 == 0:
            beats_per_measure = numerator // 3
            beat_unit = "dotted-quarter"
        else:
            beats_per_measure = numerator
            beat_unit = "quarter"
    else:
        raise ValueError("Time signature must be a string or mapping.")

    if numerator <= 0:
        raise ValueError("Time signature numerator must be positive.")
    if denominator not in _SUPPORTED_DENOMINATORS:
        raise ValueError("Supported denominators are 2, 4, 8, or 16.")
    if beats_per_measure <= 0:
        raise ValueError("beats_per_measure must be positive.")

    if beat_unit not in {"quarter", "eighth", "dotted-quarter", "half"}:
        raise ValueError("Unsupported beat unit. Use quarter, eighth, dotted-quarter, or half.")
    return {
        "numerator": numerator,
        "denominator": denominator,
        "beats_per_measure": beats_per_measure,
        "beat_unit": beat_unit,
    }


def beat_duration_seconds(bpm: float, *, beat_unit: str | None = None) -> float:
    bpm_value = validate_bpm(bpm)
    beat_unit_name = (beat_unit or "quarter").lower()
    factor = {
        "quarter": 1.0,
        "eighth": 0.5,
        "dotted-quarter": 1.5,
        "half": 2.0,
    }.get(beat_unit_name)
    if factor is None:
        raise ValueError("Unsupported beat unit.")
    return 60.0 * factor / bpm_value


def measure_duration_seconds(
    bpm: float,
    *,
    time_signature: str | dict[str, Any] | None = None,
    beats_per_measure: int | None = None,
    beat_unit: str | None = None,
) -> float:
    bpm_value = validate_bpm(bpm)
    if time_signature is not None:
        parsed = parse_time_signature(time_signature)
        beats_per_measure = parsed["beats_per_measure"]
        beat_unit = parsed["beat_unit"]
    if beats_per_measure is None:
        raise ValueError("beats_per_measure is required when a time signature is not supplied.")
    if beats_per_measure <= 0:
        raise ValueError("beats_per_measure must be positive.")
    return beat_duration_seconds(bpm_value, beat_unit=beat_unit) * float(beats_per_measure)


def _subdivision_offsets(subdivision: str, beat_seconds: float, swing_ratio: float) -> list[float]:
    if subdivision == "beat":
        return [0.0]
    if subdivision == "eighth":
        if swing_ratio <= 0:
            return [0.0, 0.5, 1.0]
        first = (0.5 + swing_ratio / 2.0) * beat_seconds
        return [0.0, first, beat_seconds]
    if subdivision == "sixteenth":
        quarter = beat_seconds / 4.0
        if swing_ratio <= 0:
            return [0.0, quarter, quarter * 2, quarter * 3, beat_seconds]
        return [
            0.0,
            quarter * (1.0 + swing_ratio / 2.0),
            quarter * 2.0,
            quarter * 3.0 + quarter * (swing_ratio / 2.0),
            beat_seconds,
        ]
    if subdivision == "triplet":
        segment = beat_seconds / 3.0
        return [0.0, segment, segment * 2, beat_seconds]
    raise ValueError("Unsupported subdivision.")


def _resolve_bpm_at(tempo_points: list[dict[str, Any]], timestamp_seconds: float) -> float:
    if not tempo_points:
        raise ValueError("Tempo points must not be empty.")
    sorted_points = sorted(tempo_points, key=lambda item: float(item["timestamp_seconds"]))
    if timestamp_seconds <= float(sorted_points[0]["timestamp_seconds"]):
        return float(sorted_points[0]["bpm"])
    for previous, current in zip(sorted_points, sorted_points[1:], strict=False):
        previous_time = float(previous["timestamp_seconds"])
        current_time = float(current["timestamp_seconds"])
        if current_time >= timestamp_seconds:
            if current_time == previous_time:
                return float(current["bpm"])
            ratio = (timestamp_seconds - previous_time) / (current_time - previous_time)
            return float(previous["bpm"]) + (float(current["bpm"]) - float(previous["bpm"])) * ratio
    return float(sorted_points[-1]["bpm"])


def generate_tempo_ramp(
    start_bpm: float,
    end_bpm: float,
    *,
    duration_seconds: float,
    steps: int = 8,
) -> dict[str, Any]:
    validate_bpm(start_bpm)
    validate_bpm(end_bpm)
    if duration_seconds <= 0:
        raise ValueError("duration_seconds must be greater than zero.")
    if steps < 2:
        raise ValueError("steps must be at least 2.")
    points = []
    for index in range(steps):
        fraction = index / max(1, steps - 1)
        points.append(
            {
                "timestamp_seconds": round(duration_seconds * fraction, 6),
                "bpm": start_bpm + (end_bpm - start_bpm) * fraction,
            }
        )
    return {
        "start_bpm": start_bpm,
        "end_bpm": end_bpm,
        "duration_seconds": duration_seconds,
        "points": points,
        "average_bpm": (start_bpm + end_bpm) / 2.0,
    }


def build_tempo_plan(
    *,
    start_bpm: float,
    target_bpm: float,
    stage_duration_seconds: float,
    stage_count: int = 1,
    interval_seconds: float | None = None,
) -> list[dict[str, Any]]:
    validate_bpm(start_bpm)
    validate_bpm(target_bpm)
    if stage_count <= 0:
        raise ValueError("stage_count must be positive.")
    if interval_seconds is not None and interval_seconds <= 0:
        raise ValueError("interval_seconds must be positive when supplied.")
    stage_duration = stage_duration_seconds if stage_duration_seconds > 0 else 0.0
    stages: list[dict[str, Any]] = []
    delta = (target_bpm - start_bpm) / max(1, stage_count)
    for index in range(stage_count):
        stage_bpm = start_bpm + delta * index
        next_bpm = start_bpm + delta * (index + 1)
        stages.append(
            {
                "index": index,
                "start_bpm": stage_bpm,
                "target_bpm": next_bpm,
                "duration_seconds": stage_duration,
                "interval_seconds": interval_seconds,
            }
        )
    return stages


def build_schedule_window(
    *,
    bpm: float,
    time_signature: str | dict[str, Any],
    start_seconds: float = 0.0,
    duration_seconds: float,
    subdivisions: str = "eighth",
    swing_ratio: float = 0.0,
    count_in_bars: int = 0,
    accent_pattern: list[int] | None = None,
    max_events: int = 10_000,
) -> dict[str, Any]:
    bpm_value = validate_bpm(bpm)
    parsed = parse_time_signature(time_signature)
    if duration_seconds <= 0:
        raise ValueError("duration_seconds must be positive.")
    if max_events <= 0:
        raise ValueError("max_events must be positive.")
    if swing_ratio < 0 or swing_ratio > 1:
        raise ValueError("swing_ratio must be between 0 and 1.")
    beat_seconds = beat_duration_seconds(bpm_value, beat_unit=parsed["beat_unit"])
    if count_in_bars < 0:
        raise ValueError("count_in_bars cannot be negative.")
    accent_list = accent_pattern or [1] + [0] * max(0, parsed["beats_per_measure"] - 1)
    accent_list = accent_list[: parsed["beats_per_measure"]]
    if len(accent_list) < parsed["beats_per_measure"]:
        accent_list.extend([0] * (parsed["beats_per_measure"] - len(accent_list)))

    events: list[dict[str, Any]] = []
    start_time = max(0.0, start_seconds)
    total_measures = max(
        1,
        math.ceil(
            (duration_seconds + start_time)
            / measure_duration_seconds(bpm_value, time_signature=parsed)
        ),
    )
    measure_duration = measure_duration_seconds(bpm_value, time_signature=parsed)

    for measure_index in range(total_measures):
        measure_start = start_time + measure_index * measure_duration
        for beat_index in range(parsed["beats_per_measure"]):
            event_time = measure_start + beat_index * beat_seconds
            if event_time < start_time:
                continue
            if event_time > start_time + duration_seconds + 1e-9:
                break
            events.append(
                {
                    "timestamp_seconds": round(event_time, 6),
                    "measure_index": measure_index,
                    "beat_index": beat_index,
                    "subdivision_index": 0,
                    "event_type": "downbeat" if beat_index == 0 else "beat",
                    "accent_level": accent_list[beat_index],
                    "bpm": bpm_value,
                    "phase": "main",
                }
            )
            if len(events) >= max_events:
                return {"events": events, "truncated": True, "total_events": len(events)}
            if subdivisions == "beat":
                continue
            offsets = _subdivision_offsets(subdivisions, beat_seconds, swing_ratio)
            for index, offset in enumerate(offsets[1:], start=1):
                sub_time = event_time + offset
                if sub_time < start_time:
                    continue
                if sub_time > start_time + duration_seconds + 1e-9:
                    continue
                events.append(
                    {
                        "timestamp_seconds": round(sub_time, 6),
                        "measure_index": measure_index,
                        "beat_index": beat_index,
                        "subdivision_index": index,
                        "event_type": "subdivision",
                        "accent_level": accent_list[beat_index],
                        "bpm": bpm_value,
                        "phase": "main",
                    }
                )
                if len(events) >= max_events:
                    return {"events": events, "truncated": True, "total_events": len(events)}

    if count_in_bars > 0:
        count_in_events = []
        for bar in range(count_in_bars):
            for beat_index in range(parsed["beats_per_measure"]):
                event_time = bar * measure_duration + beat_index * beat_seconds
                count_in_events.append(
                    {
                        "timestamp_seconds": round(event_time, 6),
                        "measure_index": bar,
                        "beat_index": beat_index,
                        "subdivision_index": 0,
                        "event_type": "count-in",
                        "accent_level": accent_list[beat_index],
                        "bpm": bpm_value,
                        "phase": "count_in",
                    }
                )
        events = count_in_events + events

    return {
        "events": events[:max_events],
        "truncated": len(events) > max_events,
        "total_events": len(events),
    }


def build_variable_tempo_map(
    *,
    base_bpm: float,
    tempo_points: list[dict[str, Any]],
    duration_seconds: float,
) -> dict[str, Any]:
    validate_bpm(base_bpm)
    if not tempo_points:
        raise ValueError("tempo_points must include at least one value.")
    normalized = [
        {
            "timestamp_seconds": float(item["timestamp_seconds"]),
            "bpm": validate_bpm(float(item["bpm"])),
        }
        for item in tempo_points
    ]
    points = sorted(normalized, key=lambda item: item["timestamp_seconds"])
    if points[0]["timestamp_seconds"] < 0:
        raise ValueError("Tempo map timestamps cannot be negative.")
    final_bpm = float(points[-1]["bpm"])
    if duration_seconds <= 0:
        raise ValueError("duration_seconds must be positive.")
    return {
        "base_bpm": base_bpm,
        "final_bpm": final_bpm,
        "duration_seconds": duration_seconds,
        "tempo_points": points,
        "bpm_at_end": _resolve_bpm_at(points, duration_seconds),
    }


def build_count_in_schedule(
    *,
    bpm: float,
    time_signature: str | dict[str, Any],
    bar_count: int,
    subdivision: str = "beat",
    start_offset_seconds: float = 0.0,
) -> list[dict[str, Any]]:
    if bar_count < 0:
        raise ValueError("bar_count cannot be negative.")
    schedule = build_schedule_window(
        bpm=bpm,
        time_signature=time_signature,
        start_seconds=start_offset_seconds,
        duration_seconds=(
            measure_duration_seconds(bpm, time_signature=time_signature) * max(1, bar_count)
        ),
        subdivisions=subdivision,
        count_in_bars=bar_count,
        max_events=10_000,
    )
    return [event for event in schedule["events"] if event.get("phase") == "count_in"]


def time_to_beat(
    *,
    bpm: float,
    time_signature: str | dict[str, Any],
    timestamp_seconds: float,
    tempo_points: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if timestamp_seconds < 0:
        raise ValueError("timestamp_seconds cannot be negative.")
    parsed = parse_time_signature(time_signature)
    beat_seconds = beat_duration_seconds(bpm, beat_unit=parsed["beat_unit"])
    if tempo_points:
        current_bpm = _resolve_bpm_at(tempo_points, timestamp_seconds)
        beat_seconds = 60.0 / current_bpm
    measures = timestamp_seconds / (beat_seconds * parsed["beats_per_measure"])
    measure_index = int(math.floor(measures))
    remainder = timestamp_seconds - measure_index * beat_seconds * parsed["beats_per_measure"]
    beat_index = int(math.floor(remainder / beat_seconds))
    beat_offset = remainder - beat_index * beat_seconds
    return {
        "measure_index": measure_index,
        "beat_index": beat_index,
        "beat_offset_seconds": beat_offset,
        "time_seconds": timestamp_seconds,
        "bpm": bpm if not tempo_points else _resolve_bpm_at(tempo_points, timestamp_seconds),
    }


def beat_to_time(
    *,
    bpm: float,
    time_signature: str | dict[str, Any],
    measure_index: int,
    beat_index: int,
    offset_seconds: float = 0.0,
    tempo_points: list[dict[str, Any]] | None = None,
) -> float:
    parsed = parse_time_signature(time_signature)
    if beat_index < 0 or beat_index >= parsed["beats_per_measure"]:
        raise ValueError("beat_index is outside the current measure.")
    if offset_seconds < 0:
        raise ValueError("offset_seconds cannot be negative.")
    measure_duration = measure_duration_seconds(bpm, time_signature=parsed)
    base = measure_index * measure_duration
    if tempo_points:
        for dominant in tempo_points:
            if float(dominant["timestamp_seconds"]) > base:
                break
        start_bpm = _resolve_bpm_at(tempo_points, base)
        beat_time = 60.0 / start_bpm
        return base + beat_index * beat_time + offset_seconds
    beat_time = 60.0 / bpm
    return base + beat_index * beat_time + offset_seconds
