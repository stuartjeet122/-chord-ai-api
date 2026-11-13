from __future__ import annotations

import math
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import soundfile
from numpy.typing import NDArray
from scipy.signal import fftconvolve, find_peaks

from music_api.music.theory import (
    CHORD_INTERVALS,
    MAJOR_PROFILE,
    MINOR_PROFILE,
    NOTE_NAMES,
    KeyCandidate,
    diatonic_degree,
    spell_chord,
)

_ENGINE_VERSION = "dsp-harmonic-0.1.0"
_MIN_MIDI = 28
_MAX_MIDI = 84
_CHUNK_SECONDS = 8
_CHORD_SCORE_FLOOR = 0.48
_MIN_CHORD_MARGIN = 0.018


@dataclass(frozen=True)
class FeatureTrack:
    times: NDArray[np.float64]
    pitch_salience: NDArray[np.float64]
    rms: NDArray[np.float64]
    onset_strength: NDArray[np.float64]
    duration_seconds: float


class AnalysisCancelledError(Exception):
    pass


def analyze_audio_file(
    path: Path,
    cancelled: threading.Event | None = None,
) -> dict[str, Any]:
    tracks = extract_harmonic_features(path, cancelled=cancelled)
    chords = estimate_chords(tracks)
    keys = estimate_keys(tracks)
    _attach_roman_numerals(chords, keys)
    rhythm = estimate_rhythm(tracks)
    return {
        "engine_version": _ENGINE_VERSION,
        "duration_seconds": tracks.duration_seconds,
        "sample_rate": _analysis_sample_rate(path),
        "chords": chords,
        "key": keys,
        "rhythm": rhythm,
        "bass_events": _bass_events(tracks),
        "melody": {
            "status": "unavailable",
            "reason": "No validated note-transcription provider is installed.",
        },
        "sections": {
            "status": "unavailable",
            "reason": "No validated song-structure provider is installed.",
            "items": [],
        },
        "methodology": {
            "chords": "harmonic partial salience and pitch-class template candidates",
            "key": "whole-track and local Krumhansl-Schmuckler profile correlation",
            "rhythm": "spectral-flux onset autocorrelation and tempo-periodic grid",
            "confidence": "numeric confidence is withheld until calibration on annotated data",
        },
    }


def extract_harmonic_features(
    path: Path,
    *,
    cancelled: threading.Event | None = None,
) -> FeatureTrack:
    with soundfile.SoundFile(path) as audio:
        sample_rate = int(audio.samplerate)
        total_frames = len(audio)
        if total_frames <= 0 or sample_rate <= 0:
            raise ValueError("Audio must contain at least one frame.")
        frame_length = max(4096, round(sample_rate * 0.46))
        hop_length = max(1, round(sample_rate * 0.10))
        fft_length = 1 << (frame_length - 1).bit_length()
        window = np.hanning(frame_length).astype(np.float64)
        midi_values = np.arange(_MIN_MIDI, _MAX_MIDI + 1)
        frequencies: NDArray[np.float64] = np.asarray(
            np.fft.rfftfreq(fft_length, 1.0 / sample_rate),
            dtype=np.float64,
        )
        harmonic_bins = _harmonic_bins(midi_values, frequencies, sample_rate)
        buffer: NDArray[np.float64] = np.empty((0, audio.channels), dtype=np.float64)
        buffer_start = 0
        next_start = 0
        absolute_read = 0
        salience_rows: list[NDArray[np.float64]] = []
        rms_values: list[float] = []
        onset_values: list[float] = []
        times: list[float] = []
        previous_spectrum: NDArray[np.float64] | None = None
        while absolute_read < total_frames:
            if cancelled is not None and cancelled.is_set():
                raise AnalysisCancelledError("Music analysis was cancelled.")
            block = audio.read(
                min(sample_rate * _CHUNK_SECONDS, total_frames - absolute_read),
                dtype="float64",
                always_2d=True,
            )
            if not len(block):
                break
            buffer = np.concatenate((buffer, block), axis=0)
            absolute_read += len(block)
            buffer_end = buffer_start + len(buffer)
            while next_start + frame_length <= buffer_end:
                local_start = next_start - buffer_start
                frame = buffer[local_start : local_start + frame_length]
                channel_spectra = np.abs(
                    np.fft.rfft(frame * window[:, np.newaxis], n=fft_length, axis=0)
                )
                spectrum = np.mean(channel_spectra, axis=1)
                salience_rows.append(_sample_harmonics(spectrum, harmonic_bins))
                rms_values.append(float(np.sqrt(np.mean(np.square(frame)))))
                if previous_spectrum is None:
                    onset_values.append(0.0)
                else:
                    bins = min(2048, len(spectrum))
                    current = np.log1p(spectrum[:bins])
                    previous = np.log1p(previous_spectrum[:bins])
                    onset_values.append(float(np.maximum(current - previous, 0).mean()))
                previous_spectrum = spectrum
                times.append((next_start + frame_length / 2) / sample_rate)
                next_start += hop_length
            discard = max(0, next_start - buffer_start)
            if discard:
                buffer = buffer[discard:]
                buffer_start += discard
        if not salience_rows:
            raise ValueError("Audio is shorter than the analysis window.")
    return FeatureTrack(
        times=np.asarray(times, dtype=np.float64),
        pitch_salience=np.asarray(salience_rows, dtype=np.float64),
        rms=np.asarray(rms_values, dtype=np.float64),
        onset_strength=np.asarray(onset_values, dtype=np.float64),
        duration_seconds=total_frames / sample_rate,
    )


def estimate_chords(track: FeatureTrack) -> list[dict[str, Any]]:
    labels: list[dict[str, Any]] = []
    salience = track.pitch_salience
    for row, rms in zip(salience, track.rms, strict=True):
        pitch_classes = np.zeros(12, dtype=np.float64)
        for offset, value in enumerate(row):
            pitch_classes[(_MIN_MIDI + offset) % 12] += value
        ranked = _rank_chords(pitch_classes)
        if rms < 1e-5 or not ranked or ranked[0]["score"] < _CHORD_SCORE_FLOOR:
            labels.append(_unknown_chord())
            continue
        best = ranked[0]
        margin = best["score"] - (ranked[1]["score"] if len(ranked) > 1 else 0.0)
        bass_candidates = row[: min(len(row), 21)]
        bass_index = int(np.argmax(bass_candidates)) if bass_candidates.size else 0
        bass_midi = _MIN_MIDI + bass_index
        bass_name = NOTE_NAMES[bass_midi % 12]
        root = str(best["root"])
        root_pc = NOTE_NAMES.index(root)
        bass_diff = (bass_midi % 12 - root_pc) % 12
        intervals = CHORD_INTERVALS[str(best["quality"])]
        inversion = (
            _inversion_name(bass_diff, intervals)
            if bass_diff != 0 and bass_diff in intervals
            else None
        )
        ambiguous = margin < _MIN_CHORD_MARGIN
        labels.append(
            {
                "symbol": "N" if ambiguous else spell_chord(root_pc, str(best["quality"])),
                "root": None if ambiguous else root,
                "quality": "unknown" if ambiguous else str(best["quality"]),
                "bass_note": bass_name,
                "inversion": inversion if not ambiguous else None,
                "score": float(best["score"]),
                "margin": float(margin),
                "confidence_status": "ambiguous" if ambiguous else "uncalibrated",
                "candidates": ranked[:3],
            }
        )
    return _merge_chord_frames(track.times, labels, track.duration_seconds)


def estimate_keys(track: FeatureTrack) -> dict[str, Any]:
    whole = _key_candidates(track.pitch_salience)
    profile_margin = whole[0].score - whole[1].score if len(whole) > 1 else None
    decisive = bool(
        whole and profile_margin is not None and profile_margin >= 0.08 and whole[0].score >= 0.25
    )
    local_candidates: list[dict[str, Any]] = []
    frames_per_window = max(1, round(20.0 / _frame_period(track.times)))
    for start in range(0, len(track.times), frames_per_window):
        end = min(len(track.times), start + frames_per_window)
        if end - start < max(8, frames_per_window // 2):
            continue
        candidates = _key_candidates(track.pitch_salience[start:end])
        if (
            len(candidates) < 2
            or candidates[0].score - candidates[1].score < 0.08
            or candidates[0].score < 0.25
        ):
            continue
        top = candidates[0]
        local_candidates.append(
            {
                "start_seconds": float(track.times[start]),
                "end_seconds": float(track.times[end - 1]),
                "tonic": top.tonic,
                "mode": top.mode,
                "profile_score": top.score,
                "confidence": None,
            }
        )
    return {
        "selected": (
            {
                "tonic": whole[0].tonic,
                "mode": whole[0].mode,
                "profile_score": whole[0].score,
                "confidence": None,
            }
            if decisive
            else None
        ),
        "best_candidate": (
            {
                "tonic": whole[0].tonic,
                "mode": whole[0].mode,
                "profile_score": whole[0].score,
            }
            if whole
            else None
        ),
        "alternatives": [
            {"tonic": item.tonic, "mode": item.mode, "profile_score": item.score}
            for item in whole[1:4]
        ],
        "changes": _key_changes(local_candidates),
        "confidence_status": "uncalibrated" if decisive else "ambiguous",
        "profile_margin": profile_margin,
    }


def _attach_roman_numerals(chords: list[dict[str, Any]], key: dict[str, Any]) -> None:
    selected = key.get("selected")
    margin = key.get("profile_margin")
    if not isinstance(selected, dict) or not isinstance(margin, float | int) or margin < 0.08:
        return
    expected_qualities = {
        "major": {
            "I": {"major", "major7", "major6", "major9"},
            "ii": {"minor", "minor7", "minor6", "minor9", "minor11"},
            "iii": {"minor", "minor7", "minor6", "minor9", "minor11"},
            "IV": {"major", "major7", "major6", "major9"},
            "V": {"major", "dominant7", "dominant9", "dominant13"},
            "vi": {"minor", "minor7", "minor6", "minor9", "minor11"},
            "vii°": {"diminished", "diminished7", "half_diminished7"},
        },
        "minor": {
            "i": {"minor", "minor7", "minor6", "minor9", "minor11"},
            "ii°": {"diminished", "diminished7", "half_diminished7"},
            "III": {"major", "major7", "major6", "major9"},
            "iv": {"minor", "minor7", "minor6", "minor9", "minor11"},
            "v": {"minor", "minor7", "dominant7"},
            "VI": {"major", "major7", "major6", "major9"},
            "VII": {"major", "major7", "dominant7"},
        },
    }
    mode = str(selected["mode"])
    for chord in chords:
        root = chord.get("root")
        if root is None:
            continue
        degree = diatonic_degree(str(root), str(selected["tonic"]), mode)
        allowed = expected_qualities.get(mode, {}).get(degree or "", set())
        if chord.get("quality") in allowed:
            chord["roman_numeral"] = degree


def estimate_rhythm(track: FeatureTrack) -> dict[str, Any]:
    frame_period = _frame_period(track.times)
    onset = track.onset_strength
    if len(onset) < 10 or float(np.max(onset)) <= 1e-12:
        return {
            "tempo_bpm": None,
            "alternatives_bpm": [],
            "beats": [],
            "downbeats": [],
            "time_signature": None,
            "tempo_map": [],
            "confidence": None,
            "status": "insufficient_onset_evidence",
        }
    centered = onset - float(np.mean(onset))
    autocorrelation = fftconvolve(centered, centered[::-1], mode="full")[len(centered) - 1 :]
    minimum_lag = max(1, round(60.0 / 200.0 / frame_period))
    maximum_lag = min(len(autocorrelation) - 1, round(60.0 / 50.0 / frame_period))
    if maximum_lag <= minimum_lag:
        return {
            "tempo_bpm": None,
            "alternatives_bpm": [],
            "beats": [],
            "downbeats": [],
            "time_signature": None,
            "tempo_map": [],
            "confidence": None,
            "status": "insufficient_duration",
        }
    region = autocorrelation[minimum_lag : maximum_lag + 1]
    peaks, _ = find_peaks(region)
    if not len(peaks):
        peaks = np.asarray([int(np.argmax(region))])
    ranked_lags = sorted(
        (minimum_lag + int(index) for index in peaks),
        key=lambda lag: float(autocorrelation[lag]),
        reverse=True,
    )
    selected_lag = ranked_lags[0]
    if float(autocorrelation[selected_lag]) <= 0:
        return {
            "tempo_bpm": None,
            "alternatives_bpm": [],
            "beats": [],
            "downbeats": [],
            "time_signature": None,
            "tempo_map": [],
            "confidence": None,
            "status": "insufficient_periodic_evidence",
        }
    period = selected_lag * frame_period
    tempo = 60.0 / period
    alternatives = sorted(
        {
            round(value, 2)
            for factor in (0.5, 2.0)
            for value in (tempo * factor,)
            if 50 <= value <= 200 and abs(value - tempo) > 1
        }
    )
    tempo_map = _estimate_tempo_map(track)
    beats = _beats_from_tempo_map(track, tempo_map)
    peak = float(autocorrelation[selected_lag])
    baseline = float(np.mean(autocorrelation[minimum_lag : maximum_lag + 1]))
    return {
        "tempo_bpm": round(tempo, 3),
        "alternatives_bpm": alternatives,
        "beats": beats,
        "downbeats": [],
        "time_signature": None,
        "tempo_map": tempo_map,
        "confidence": None,
        "tempo_periodicity_score": (
            max(0.0, (peak - baseline) / (abs(peak) + 1e-12)) if peak > 0 else None
        ),
        "status": (
            "estimated_variable_tempo_unvalidated"
            if len({item["bpm"] for item in tempo_map}) > 1
            else "estimated_unvalidated"
        ),
    }


def _harmonic_bins(
    midi_values: NDArray[np.int64],
    frequencies: NDArray[np.float64],
    sample_rate: int,
) -> NDArray[np.int64]:
    result = np.full((len(midi_values), 8), -1, dtype=np.int64)
    nyquist = sample_rate / 2.0
    resolution = sample_rate / (2 * (len(frequencies) - 1))
    for note_index, midi in enumerate(midi_values):
        fundamental = 440.0 * 2.0 ** ((int(midi) - 69) / 12.0)
        for harmonic in range(1, result.shape[1] + 1):
            frequency = fundamental * harmonic
            if frequency >= nyquist:
                break
            result[note_index, harmonic - 1] = round(frequency / resolution)
    return result


def _sample_harmonics(
    spectrum: NDArray[np.float64],
    harmonic_bins: NDArray[np.int64],
) -> NDArray[np.float64]:
    salience = np.zeros(harmonic_bins.shape[0], dtype=np.float64)
    for note_index, bins in enumerate(harmonic_bins):
        valid = bins[bins >= 0]
        if not len(valid):
            continue
        energies = [
            float(np.max(spectrum[max(0, index - 1) : min(len(spectrum), index + 2)]))
            for index in valid
        ]
        salience[note_index] = sum(
            value / (harmonic + 1) for harmonic, value in enumerate(energies)
        )
    return salience


def _rank_chords(profile: NDArray[np.float64]) -> list[dict[str, Any]]:
    norm = float(np.linalg.norm(profile))
    if norm <= 1e-12:
        return []
    ranked: list[dict[str, Any]] = []
    for root, root_name in enumerate(NOTE_NAMES):
        for quality, intervals in CHORD_INTERVALS.items():
            template = np.zeros(12, dtype=np.float64)
            template[(root + np.asarray(intervals, dtype=np.int64)) % 12] = 1.0
            cosine = float(np.dot(profile, template) / (norm * np.linalg.norm(template)))
            coverage = float(np.sum(profile * template) / (np.sum(profile) + 1e-12))
            score = 0.75 * cosine + 0.25 * coverage - max(0, len(intervals) - 3) * 0.012
            ranked.append(
                {
                    "root": root_name,
                    "quality": quality,
                    "symbol": spell_chord(root, quality),
                    "score": score,
                }
            )
    ranked.sort(key=lambda item: float(item["score"]), reverse=True)
    # Margin is measured across distinct root/quality interpretations, not enharmonic duplicates.
    return ranked


def _merge_chord_frames(
    times: NDArray[np.float64],
    frames: list[dict[str, Any]],
    duration: float,
) -> list[dict[str, Any]]:
    if not frames:
        return []
    hop = _frame_period(times)
    half = hop / 2
    merged: list[dict[str, Any]] = []
    start = 0
    for index in range(1, len(frames) + 1):
        if index < len(frames) and frames[index]["symbol"] == frames[start]["symbol"]:
            continue
        first = frames[start]
        start_seconds = 0.0 if start == 0 else max(0.0, float(times[start]) - half)
        end_seconds = (
            duration if index == len(frames) else min(duration, float(times[index - 1]) + half)
        )
        merged.append(
            {
                "symbol": first["symbol"],
                "root": first["root"],
                "quality": first["quality"],
                "inversion": first["inversion"],
                "bass_note": first["bass_note"],
                "start_seconds": start_seconds,
                "end_seconds": max(start_seconds, end_seconds),
                "duration_seconds": max(0.0, end_seconds - start_seconds),
                "confidence": None,
                "confidence_status": first["confidence_status"],
                "sources": ["harmonic_dsp_template"],
                "evidence": {
                    "template_score": first.get("score"),
                    "runner_up_margin": first.get("margin"),
                    "top_candidates": first.get("candidates", []),
                },
                "roman_numeral": None,
                "manually_corrected": False,
            }
        )
        start = index
    return merged


def _unknown_chord() -> dict[str, Any]:
    return {
        "symbol": "N",
        "root": None,
        "quality": "unknown",
        "inversion": None,
        "bass_note": None,
        "confidence_status": "unknown",
        "candidates": [],
    }


def _inversion_name(bass_interval: int, intervals: tuple[int, ...]) -> str | None:
    try:
        position = intervals.index(bass_interval)
    except ValueError:
        return None
    return ("root", "first", "second", "third", "fourth", "fifth")[position]


def _key_candidates(pitch_salience: NDArray[np.float64]) -> list[KeyCandidate]:
    if not len(pitch_salience):
        return []
    profile = np.zeros(12, dtype=np.float64)
    for offset, column in enumerate(pitch_salience.T):
        profile[(_MIN_MIDI + offset) % 12] += float(np.sum(column))
    if float(np.linalg.norm(profile)) <= 1e-12:
        return []
    if np.count_nonzero(profile >= float(np.max(profile)) * 0.05) < 3:
        return []
    normalized_profile: NDArray[np.float64] = np.asarray(
        (profile - np.mean(profile)) / (np.std(profile) + 1e-12),
        dtype=np.float64,
    )
    candidates: list[KeyCandidate] = []
    for tonic in range(12):
        for mode, source in (("major", MAJOR_PROFILE), ("minor", MINOR_PROFILE)):
            rotated = np.roll(np.asarray(source, dtype=np.float64), tonic)
            rotated = (rotated - np.mean(rotated)) / (np.std(rotated) + 1e-12)
            score = float(np.dot(normalized_profile, rotated) / 12)
            scale = (0, 2, 4, 5, 7, 9, 11) if mode == "major" else (0, 2, 3, 5, 7, 8, 10)
            candidates.append(
                KeyCandidate(NOTE_NAMES[tonic], mode, score, tuple((tonic + n) % 12 for n in scale))
            )
    return sorted(candidates, key=lambda item: item.score, reverse=True)


def _key_changes(local: list[dict[str, Any]]) -> list[dict[str, Any]]:
    changes: list[dict[str, Any]] = []
    prior: tuple[str, str] | None = None
    for candidate in local:
        current = (str(candidate["tonic"]), str(candidate["mode"]))
        if prior is not None and current != prior:
            changes.append(
                {
                    "start_seconds": candidate["start_seconds"],
                    "tonic": current[0],
                    "mode": current[1],
                    "profile_score": candidate["profile_score"],
                    "confidence": None,
                }
            )
        prior = current
    return changes


def _bass_events(track: FeatureTrack) -> list[dict[str, Any]]:
    upper_index = min(len(track.pitch_salience[0]), 25)
    if upper_index <= 0:
        return []
    midi = np.argmax(track.pitch_salience[:, :upper_index], axis=1) + _MIN_MIDI
    active = track.rms > 1e-5
    events: list[dict[str, Any]] = []
    start = 0
    for index in range(1, len(midi) + 1):
        if index < len(midi) and midi[index] == midi[start] and active[index] == active[start]:
            continue
        if active[start]:
            half = _frame_period(track.times) / 2
            begin = max(0.0, float(track.times[start]) - half)
            end = min(
                track.duration_seconds,
                float(track.times[index - 1]) + half if index > start else begin,
            )
            events.append(
                {
                    "note": NOTE_NAMES[int(midi[start]) % 12],
                    "midi": int(midi[start]),
                    "start_seconds": begin,
                    "end_seconds": end,
                    "confidence": None,
                    "confidence_status": "uncalibrated",
                    "method": "lowest_harmonic_salience",
                }
            )
        start = index
    return events


def _best_beat_phase(
    onset: NDArray[np.float64],
    period: float,
    frame_period: float,
) -> float:
    best_phase = 0.0
    best_score = -math.inf
    frame_count = len(onset)
    for phase in np.arange(0.0, min(period, max(frame_count * frame_period, 0.0)), frame_period):
        indices = np.arange(
            round(phase / frame_period), frame_count, max(1, round(period / frame_period))
        )
        score = float(np.sum(onset[indices]))
        if score > best_score:
            best_phase, best_score = float(phase), score
    return best_phase


def _estimate_tempo_map(track: FeatureTrack) -> list[dict[str, Any]]:
    frame_period = _frame_period(track.times)
    window_frames = max(1, round(20.0 / frame_period))
    minimum_frames = max(1, round(5.0 / frame_period))
    segments: list[dict[str, Any]] = []
    for start in range(0, len(track.onset_strength), window_frames):
        end = min(len(track.onset_strength), start + window_frames)
        if end - start < minimum_frames:
            continue
        bpm = _tempo_from_onsets(track.onset_strength[start:end], frame_period)
        if bpm is None:
            continue
        segments.append(
            {
                "start_seconds": float(track.times[start]),
                "end_seconds": min(track.duration_seconds, float(track.times[end - 1])),
                "bpm": round(bpm, 3),
                "time_signature": None,
                "confidence": None,
                "method": "local_onset_autocorrelation_unvalidated",
            }
        )
    return segments


def _beats_from_tempo_map(
    track: FeatureTrack,
    tempo_map: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    frame_period = _frame_period(track.times)
    beats: list[dict[str, Any]] = []
    for segment in tempo_map:
        start = min(len(track.times) - 1, max(0, round(segment["start_seconds"] / frame_period)))
        end = min(len(track.times), round(segment["end_seconds"] / frame_period) + 1)
        local_onset = track.onset_strength[start:end]
        if not len(local_onset):
            continue
        period = 60.0 / float(segment["bpm"])
        phase = _best_beat_phase(local_onset, period, frame_period)
        segment_start = float(segment["start_seconds"])
        beat_time = segment_start + phase
        segment_end = float(segment["end_seconds"])
        while beat_time <= segment_end:
            local_frame = min(
                len(track.onset_strength) - 1,
                max(0, round(beat_time / frame_period)),
            )
            if not beats or beat_time - float(beats[-1]["time_seconds"]) > frame_period:
                beats.append(
                    {
                        "time_seconds": beat_time,
                        "strength": float(track.onset_strength[local_frame]),
                        "downbeat": None,
                    }
                )
            beat_time += period
    return beats


def _tempo_from_onsets(onset: NDArray[np.float64], frame_period: float) -> float | None:
    if len(onset) < 10 or float(np.max(onset)) <= 1e-12:
        return None
    centered = onset - float(np.mean(onset))
    autocorrelation = fftconvolve(centered, centered[::-1], mode="full")[len(centered) - 1 :]
    minimum_lag = max(1, round(60.0 / 200.0 / frame_period))
    maximum_lag = min(len(autocorrelation) - 1, round(60.0 / 50.0 / frame_period))
    if maximum_lag <= minimum_lag:
        return None
    region = autocorrelation[minimum_lag : maximum_lag + 1]
    return 60.0 / ((minimum_lag + int(np.argmax(region))) * frame_period)


def _frame_period(times: NDArray[np.float64]) -> float:
    return float(np.median(np.diff(times))) if len(times) > 1 else 0.1


def _analysis_sample_rate(path: Path) -> int:
    info = soundfile.info(path)
    return int(info.samplerate)
