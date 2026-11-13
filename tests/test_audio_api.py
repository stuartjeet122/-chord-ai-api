from __future__ import annotations

from fastapi.testclient import TestClient

from music_api.music import audio_api
from music_api.music.audio_analysis import parse_lrc


def test_audio_analysis_accepts_mp3_and_timestamped_lyrics(
    client: TestClient,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        audio_api,
        "analyze_mp3",
        lambda _path: {
            "duration_seconds": 30.0,
            "tempo_bpm": 120.0,
            "time_signature": None,
            "estimated_key": "C major",
            "beats": [{"timestamp_seconds": 0.5, "beat_index": 1}],
            "chords": [
                {
                    "start_seconds": 0.0,
                    "end_seconds": 2.0,
                    "chord": "C",
                    "confidence": 0.8,
                }
            ],
            "estimated_note_events": [
                {"start_seconds": 0.0, "end_seconds": 1.0, "notes": ["C", "E", "G"]}
            ],
        },
    )

    response = client.post(
        "/api/v1/audio/analyze",
        files={"file": ("song.mp3", b"ID3fake-mp3-content", "audio/mpeg")},
        data={"lyrics_lrc": "[00:01.25]First line\n[00:02.5]Second line"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["tempo_bpm"] == 120.0
    assert body["chords"][0]["chord"] == "C"
    assert body["lyrics_status"] == "timestamped"
    assert body["lyrics"] == [
        {"start_seconds": 1.25, "text": "First line", "beat_index": 1, "chords": ["C"]},
        {"start_seconds": 2.5, "text": "Second line", "beat_index": 1, "chords": []},
    ]
    assert body["transcription_status"] == "not_configured"


def test_audio_analysis_rejects_non_mp3(client: TestClient) -> None:
    response = client.post(
        "/api/v1/audio/analyze",
        files={"file": ("song.wav", b"audio", "audio/wav")},
    )

    assert response.status_code == 415


def test_audio_analysis_rejects_invalid_mp3_content(client: TestClient) -> None:
    response = client.post(
        "/api/v1/audio/analyze",
        files={"file": ("song.mp3", b"not an mp3", "audio/mpeg")},
    )

    assert response.status_code == 422


def test_audio_analysis_enforces_upload_size_limit(client: TestClient, monkeypatch) -> None:
    monkeypatch.setattr(audio_api, "MAX_UPLOAD_BYTES", 4)
    response = client.post(
        "/api/v1/audio/analyze",
        files={"file": ("song.mp3", b"ID3more", "audio/mpeg")},
    )

    assert response.status_code == 413


def test_lrc_requires_timestamps_within_audio_duration() -> None:
    assert parse_lrc("[00:01.2]Hello", 5) == [{"start_seconds": 1.2, "text": "Hello"}]

    try:
        parse_lrc("[00:10.00]Too late", 5)
    except ValueError as exc:
        assert "beyond the end" in str(exc)
    else:
        raise AssertionError("LRC timestamps beyond the audio duration should be rejected.")
