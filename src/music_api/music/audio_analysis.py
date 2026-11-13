from __future__ import annotations

import json
import re
import subprocess
import threading
from pathlib import Path
from typing import Any, TypedDict

import numpy as np
from numpy.typing import NDArray

from music_api.music.theory import NOTE_NAMES

SAMPLE_RATE = 22_050
HOP_LENGTH = 512
MAX_DURATION_SECONDS = 600
FFMPEG_TIMEOUT_SECONDS = 180
_ANALYSIS_SLOTS = threading.BoundedSemaphore(2)
_LRC_TIMESTAMP = re.compile(r"\[(\d{1,3}):([0-5]\d)(?:\.(\d{1,3}))?\]")
_CHORD_QUALITIES = {
    "": (0, 4, 7),
    "m": (0, 3, 7),
    "dim": (0, 3, 6),
    "aug": (0, 4, 8),
    "sus2": (0, 2, 7),
    "sus4": (0, 5, 7),
}
_MAJOR_PROFILE = (6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88)
_MINOR_PROFILE = (6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17)


class BeatEventData(TypedDict):
    timestamp_seconds: float
    beat_index: int


class ChordEventData(TypedDict):
    start_seconds: float
    end_seconds: float
    chord: str
    confidence: float


class NoteEventData(TypedDict):
    start_seconds: float
    end_seconds: float
    notes: list[str]


class LyricLineData(TypedDict):
    start_seconds: float
    text: str


class AnalysisResult(TypedDict):
    duration_seconds: float
    tempo_bpm: float | None
    time_signature: str | None
    estimated_key: str | None
    beats: list[BeatEventData]
    chords: list[ChordEventData]
    estimated_note_events: list[NoteEventData]


class AnalysisCapacityError(RuntimeError):
    pass


class AudioMetadata(TypedDict):
    duration_seconds: float
    codec: str | None
    sample_rate_hz: int | None
    channels: int | None
    bitrate_kbps: float | None
    title: str | None
    artist: str | None
    album: str | None


def parse_lrc(text: str, duration_seconds: float) -> list[LyricLineData]:
    if len(text) > 100_000:
        raise ValueError("LRC lyrics must be no larger than 100,000 characters.")

    lines: list[LyricLineData] = []
    for raw_line in text.splitlines():
        timestamps = list(_LRC_TIMESTAMP.finditer(raw_line))
        if not timestamps:
            if raw_line.strip() and not (raw_line.strip().startswith("[") and "]" in raw_line):
                raise ValueError("Lyrics must use LRC timestamps, for example [00:12.50]Line.")
            continue
        lyric = _LRC_TIMESTAMP.sub("", raw_line).strip()
        if not lyric:
            continue
        for timestamp in timestamps:
            fraction = timestamp.group(3) or ""
            seconds = float(int(timestamp.group(1)) * 60 + int(timestamp.group(2)))
            seconds += float(f"0.{fraction}") if fraction else 0
            if seconds > duration_seconds:
                raise ValueError("An LRC timestamp is beyond the end of the audio.")
            lines.append({"start_seconds": round(seconds, 3), "text": lyric})
    if not lines:
        raise ValueError("Provide at least one timestamped LRC lyric line.")
    return sorted(lines, key=lambda line: float(line["start_seconds"]))


def _validate_mp3(path: Path) -> None:
    with path.open("rb") as audio_file:
        header = audio_file.read(4096)
    if not header:
        raise ValueError("The uploaded audio file is empty.")
    if not (header.startswith(b"ID3") or any(
        header[index] == 0xFF and header[index + 1] & 0xE0 == 0xE0
        for index in range(len(header) - 1)
    )):
        raise ValueError("The uploaded file does not appear to be a valid MP3.")


def analyze_mp3(path: Path) -> AnalysisResult:
    _validate_mp3(path)
    if not _ANALYSIS_SLOTS.acquire(timeout=30):
        raise AnalysisCapacityError(
            "Audio analysis capacity is currently full; try again shortly."
        )
    try:
        return _analyze_mp3(path)
    finally:
        _ANALYSIS_SLOTS.release()


def probe_mp3(path: Path) -> AudioMetadata:
    _validate_mp3(path)
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "a:0",
                "-show_entries",
                (
                    "format=duration,bit_rate:format_tags=title,artist,album:"
                    "stream=codec_name,sample_rate,channels,bit_rate:"
                    "stream_tags=title,artist,album"
                ),
                "-of",
                "json",
                str(path),
            ],
            check=True,
            capture_output=True,
            timeout=15,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("FFprobe is required to inspect MP3 metadata.") from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("MP3 metadata inspection exceeded the time limit.") from exc
    except subprocess.CalledProcessError as exc:
        raise ValueError("Could not read metadata from the uploaded MP3.") from exc

    try:
        metadata = json.loads(result.stdout)
        stream = metadata["streams"][0]
        format_info = metadata.get("format", {})
        tags = {**format_info.get("tags", {}), **stream.get("tags", {})}
        duration = float(format_info["duration"])
        if not 0 < duration <= MAX_DURATION_SECONDS:
            raise ValueError(f"Audio must be {MAX_DURATION_SECONDS // 60} minutes or shorter.")
        bitrate = stream.get("bit_rate") or format_info.get("bit_rate")
        return {
            "duration_seconds": round(duration, 3),
            "codec": stream.get("codec_name"),
            "sample_rate_hz": int(stream["sample_rate"]) if stream.get("sample_rate") else None,
            "channels": int(stream["channels"]) if stream.get("channels") else None,
            "bitrate_kbps": round(int(bitrate) / 1000, 2) if bitrate else None,
            "title": tags.get("title"),
            "artist": tags.get("artist"),
            "album": tags.get("album"),
        }
    except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError("Could not read metadata from the uploaded MP3.") from exc


def _analyze_mp3(path: Path) -> AnalysisResult:
    import librosa

    wav_path = path.with_suffix(".wav")
    try:
        try:
            subprocess.run(
                [
                    "ffmpeg",
                    "-nostdin",
                    "-v",
                    "error",
                    "-i",
                    str(path),
                    "-map",
                    "0:a:0",
                    "-vn",
                    "-ac",
                    "1",
                    "-ar",
                    str(SAMPLE_RATE),
                    "-t",
                    str(MAX_DURATION_SECONDS + 1),
                    "-y",
                    str(wav_path),
                ],
                check=True,
                capture_output=True,
                timeout=FFMPEG_TIMEOUT_SECONDS,
            )
        except FileNotFoundError as exc:
            raise RuntimeError("FFmpeg is required to analyze audio.") from exc
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError("Audio decoding exceeded the time limit.") from exc
        except subprocess.CalledProcessError as exc:
            detail = exc.stderr.decode("utf-8", errors="replace").strip()
            raise ValueError(f"Could not decode the uploaded MP3: {detail[:300]}") from exc

        audio, sample_rate = librosa.load(wav_path, sr=SAMPLE_RATE, mono=True)
        duration = float(librosa.get_duration(y=audio, sr=sample_rate))
        if duration <= 0:
            raise ValueError("The uploaded audio contains no sound.")
        if duration > MAX_DURATION_SECONDS:
            raise ValueError(f"Audio must be {MAX_DURATION_SECONDS // 60} minutes or shorter.")

        chroma = librosa.feature.chroma_stft(
            y=audio,
            sr=sample_rate,
            n_fft=4096,
            hop_length=HOP_LENGTH,
        )
        onset_envelope = librosa.onset.onset_strength(
            y=audio,
            sr=sample_rate,
            hop_length=HOP_LENGTH,
        )
        tempo, beat_frames = librosa.beat.beat_track(
            onset_envelope=onset_envelope,
            sr=sample_rate,
            hop_length=HOP_LENGTH,
            units="frames",
            trim=False,
        )
        beat_times = librosa.frames_to_time(
            beat_frames,
            sr=sample_rate,
            hop_length=HOP_LENGTH,
        )
        beat_times = [float(time) for time in beat_times if 0 <= time < duration]
        bpm = float(np.asarray(tempo).reshape(-1)[0]) if np.size(tempo) else 0.0
        key = _estimate_key(np.asarray(chroma).mean(axis=1))
        beats: list[BeatEventData] = [
            {"timestamp_seconds": round(time, 3), "beat_index": index + 1}
            for index, time in enumerate(beat_times)
        ]
        chords, estimated_note_events = _extract_events(chroma, beat_times, duration)
        return {
            "duration_seconds": round(duration, 3),
            "tempo_bpm": round(bpm, 2) if bpm > 0 else None,
            "time_signature": None,
            "estimated_key": key,
            "beats": beats,
            "chords": chords,
            "estimated_note_events": estimated_note_events,
        }
    finally:
        wav_path.unlink(missing_ok=True)


def _estimate_key(chroma: NDArray[Any]) -> str | None:
    values = np.asarray(chroma, dtype=float)
    if not values.any():
        return None
    candidates: list[tuple[float, str]] = []
    for root in range(12):
        for mode, profile in (("major", _MAJOR_PROFILE), ("minor", _MINOR_PROFILE)):
            rotated = np.roll(np.asarray(profile, dtype=float), root)
            correlation = float(np.corrcoef(values, rotated)[0, 1])
            if np.isfinite(correlation):
                candidates.append((correlation, f"{NOTE_NAMES[root]} {mode}"))
    return max(candidates)[1] if candidates else None


def _extract_events(
    chroma: NDArray[Any],
    beat_times: list[float],
    duration: float,
) -> tuple[list[ChordEventData], list[NoteEventData]]:
    values = np.asarray(chroma, dtype=float)
    frame_times = np.arange(values.shape[1]) * HOP_LENGTH / SAMPLE_RATE
    boundaries = [0.0, *beat_times, duration]
    boundaries = sorted(set(boundaries))
    chords: list[ChordEventData] = []
    notes: list[NoteEventData] = []

    for start, end in zip(boundaries, boundaries[1:], strict=False):
        frame_mask = (frame_times >= start) & (frame_times < end)
        if not frame_mask.any() or end <= start:
            continue
        profile = values[:, frame_mask].mean(axis=1)
        chord, confidence, intervals = _best_chord(profile)
        note_names: list[str] = []
        if chord is not None:
            root = max(
                (index for index, name in enumerate(NOTE_NAMES) if chord.startswith(name)),
                key=lambda index: len(NOTE_NAMES[index]),
            )
            note_names = [NOTE_NAMES[(root + interval) % 12] for interval in intervals]
        if note_names:
            _append_note_event(notes, start, end, note_names)
        if chord is not None:
            if chords and chords[-1]["chord"] == chord:
                chords[-1]["end_seconds"] = round(end, 3)
                chords[-1]["confidence"] = round(
                    (float(chords[-1]["confidence"]) + confidence) / 2,
                    3,
                )
            else:
                chords.append(
                    {
                        "start_seconds": round(start, 3),
                        "end_seconds": round(end, 3),
                        "chord": chord,
                        "confidence": round(confidence, 3),
                    }
                )
    return chords, notes


def _best_chord(
    profile: NDArray[Any],
) -> tuple[str | None, float, tuple[int, ...]]:
    values = np.asarray(profile, dtype=float)
    norm = float(np.linalg.norm(values))
    if norm <= 1e-10:
        return None, 0.0, ()
    normalized = values / norm
    best_label: str | None = None
    best_score = 0.0
    best_intervals: tuple[int, ...] = ()
    for root in range(12):
        for suffix, intervals in _CHORD_QUALITIES.items():
            template = np.zeros(12)
            template[(root + np.asarray(intervals)) % 12] = 1.0
            score = float(np.dot(normalized, template) / np.linalg.norm(template))
            if score > best_score:
                best_label = f"{NOTE_NAMES[root]}{suffix}"
                best_score = score
                best_intervals = intervals
    if best_score < 0.52:
        return None, best_score, ()
    return best_label, best_score, best_intervals


def _append_note_event(
    events: list[NoteEventData],
    start: float,
    end: float,
    notes: list[str],
) -> None:
    if events and events[-1]["notes"] == notes:
        events[-1]["end_seconds"] = round(end, 3)
    else:
        events.append(
            {
                "start_seconds": round(start, 3),
                "end_seconds": round(end, 3),
                "notes": notes,
            }
        )
