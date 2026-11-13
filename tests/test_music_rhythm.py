from __future__ import annotations

import pytest

from music_api.music.rhythm import (
    beat_duration_seconds,
    build_count_in_schedule,
    build_schedule_window,
    generate_tempo_ramp,
    measure_duration_seconds,
    validate_bpm,
)


def test_validate_bpm_rejects_invalid_ranges() -> None:
    assert validate_bpm(80.0) == pytest.approx(80.0)
    with pytest.raises(ValueError):
        validate_bpm(0)
    with pytest.raises(ValueError):
        validate_bpm(999)


@pytest.mark.parametrize(
    ("bpm", "time_signature", "expected"),
    [
        (120, "4/4", 2.0),
        (90, "3/4", 2.0),
        (120, "6/8", 1.5),
    ],
)
def test_time_signature_measure_durations_are_consistent(bpm: float, time_signature: str, expected: float) -> None:
    assert measure_duration_seconds(bpm, time_signature=time_signature) == pytest.approx(expected)


def test_schedule_generation_contains_expected_beat_and_subdivision_events() -> None:
    schedule = build_schedule_window(
        bpm=120,
        time_signature="4/4",
        duration_seconds=1.5,
        subdivisions="eighth",
        count_in_bars=1,
        max_events=200,
    )
    assert schedule["truncated"] is False
    events = schedule["events"]
    assert events[0]["phase"] == "count_in"
    assert any(event["event_type"] == "downbeat" and event["measure_index"] == 0 for event in events)
    assert any(event["subdivision_index"] == 1 and event["event_type"] == "subdivision" for event in events)


def test_tempo_ramp_and_count_in_are_predictable() -> None:
    ramp = generate_tempo_ramp(70, 110, duration_seconds=8.0, steps=5)
    assert ramp["start_bpm"] == pytest.approx(70)
    assert ramp["end_bpm"] == pytest.approx(110)
    assert len(ramp["points"]) == 5

    count_in = build_count_in_schedule(bpm=80, time_signature="4/4", bar_count=2, subdivision="beat")
    assert len(count_in) == 8
    assert count_in[0]["event_type"] == "count-in"


def test_route_generates_schedule(client) -> None:
    response = client.post(
        "/api/v1/music/metronome/schedule",
        json={"bpm": 120, "time_signature": "4/4", "duration_seconds": 1.0, "subdivision": "eighth"},
        headers={"X-Dev-User-ID": "00000000-0000-0000-0000-000000000001"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["total_events"] > 0
    assert body["events"][0]["bpm"] == pytest.approx(120)


def test_beat_duration_uses_selected_unit() -> None:
    assert beat_duration_seconds(120, beat_unit="quarter") == pytest.approx(0.5)
    assert beat_duration_seconds(120, beat_unit="eighth") == pytest.approx(0.25)
    assert beat_duration_seconds(120, beat_unit="dotted-quarter") == pytest.approx(0.75)
