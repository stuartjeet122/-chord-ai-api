from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile

from music_api.music.dsp_analysis import (
    FeatureTrack,
    analyze_audio_file,
    estimate_chords,
    estimate_keys,
    estimate_rhythm,
    extract_harmonic_features,
)
from music_api.music.theory import diatonic_degree, pitch_class, spell_chord


def _write_chord_progression(path: Path) -> None:
    sample_rate = 22_050
    seconds_per_chord = 2.0
    notes = ((60, 64, 67), (55, 59, 62))
    samples: list[np.ndarray] = []
    for chord in notes:
        time = np.arange(round(sample_rate * seconds_per_chord)) / sample_rate
        samples.append(
            sum(np.sin(2 * np.pi * 440.0 * 2 ** ((midi - 69) / 12) * time) for midi in chord)
            / len(chord)
        )
    soundfile.write(path, np.concatenate(samples), sample_rate, subtype="PCM_16")


def test_theory_helpers_support_enharmonics_chords_and_degrees() -> None:
    assert pitch_class("Db") == pitch_class("C#") == 1
    assert spell_chord(0, "minor7") == "Cm7"
    assert diatonic_degree("A", "C", "major") == "vi"
    assert diatonic_degree("F#", "C", "major") is None


def test_dsp_chords_keep_timeline_and_supporting_evidence(tmp_path: Path) -> None:
    audio_path = tmp_path / "progression.wav"
    _write_chord_progression(audio_path)

    track = extract_harmonic_features(audio_path)
    events = estimate_chords(track)
    detected = [event for event in events if event["root"] is not None]

    assert detected
    assert detected[0]["root"] == "C"
    assert detected[-1]["root"] == "G"
    assert events[0]["start_seconds"] == 0.0
    assert events[-1]["end_seconds"] == track.duration_seconds
    assert detected[0]["start_seconds"] < 0.25
    assert all(event["confidence"] is None for event in events)
    assert all(event["evidence"]["top_candidates"] for event in detected)


def test_silent_audio_reports_unknown_chords_and_no_invented_tempo(tmp_path: Path) -> None:
    audio_path = tmp_path / "silence.wav"
    soundfile.write(audio_path, np.zeros(22_050), 22_050, subtype="PCM_16")

    result = analyze_audio_file(audio_path)

    assert all(event["symbol"] == "N" for event in result["chords"])
    assert result["rhythm"]["tempo_bpm"] is None
    assert result["rhythm"]["time_signature"] is None
    assert result["key"]["selected"] is None


def test_key_profile_candidates_and_beat_periodicity_are_explicitly_uncalibrated() -> None:
    times = np.arange(100, dtype=np.float64) * 0.1
    salience = np.zeros((100, 57), dtype=np.float64)
    for midi in (60, 64, 67, 72):
        salience[:, midi - 28] = 1.0
    onset = np.zeros(100, dtype=np.float64)
    onset[::5] = 1.0
    track = FeatureTrack(
        times=times,
        pitch_salience=salience,
        rms=np.ones(100, dtype=np.float64),
        onset_strength=onset,
        duration_seconds=10.0,
    )

    key = estimate_keys(track)
    rhythm = estimate_rhythm(track)

    assert key["best_candidate"] is not None
    assert key["confidence_status"] in {"ambiguous", "uncalibrated"}
    assert rhythm["tempo_bpm"] == 120.0
    assert rhythm["confidence"] is None
    assert rhythm["time_signature"] is None
    assert rhythm["beats"]
