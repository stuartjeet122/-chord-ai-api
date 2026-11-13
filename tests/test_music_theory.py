from __future__ import annotations

import asyncio
import uuid

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from music_api.jobs.models import ProcessingJob
from music_api.music.models import MusicAnalysisRecord
from music_api.music.theory import (
    Interval,
    analyze_progression,
    capo_calculation,
    chord_inversion,
    chord_notes,
    diatonic_harmony,
    identify_interval,
    identify_scales,
    invert_interval,
    key_signature,
    parse_chord,
    parse_note,
    pitch_to_frequency,
    scale_notes,
    transpose_by_interval,
    transpose_chord,
    transpose_note,
    transpose_progression,
)
from music_api.music.theory import (
    compare_progressions as theory_compare,
)
from music_api.storage.models import AudioAsset


@pytest.mark.parametrize(
    ("name", "pitch", "midi"),
    [
        ("C#4", 1, 61),
        ("Db4", 1, 61),
        ("B#4", 0, 72),
        ("Cb4", 11, 59),
        ("F##4", 7, 67),
        ("Gbb4", 5, 65),
        ("Cbb4", 10, 58),
        ("B##4", 1, 73),
    ],
)
def test_note_spelling_pitch_midi_and_enharmonics(name: str, pitch: int, midi: int) -> None:
    note = parse_note(name)
    assert note.pitch_class == pitch
    assert note.midi == midi


def test_note_frequency_and_invalid_octave_input() -> None:
    assert pitch_to_frequency("A4") == pytest.approx(440.0)
    assert parse_note("C#").frequency_hz is None
    with pytest.raises(ValueError, match="octave"):
        pitch_to_frequency("C#")
    with pytest.raises(ValueError, match="Invalid note"):
        parse_note("H#4")


@pytest.mark.parametrize(
    ("first", "second", "interval_name", "semitones"),
    [
        ("C4", "E4", "major third", 4),
        ("C4", "Eb4", "minor third", 3),
        ("C4", "F#4", "augmented fourth", 6),
        ("C4", "Gb4", "diminished fifth", 6),
        ("C4", "D5", "major ninth", 14),
        ("C4", "B3", "minor second", 1),
        ("C", "B#", "augmented seventh", 12),
    ],
)
def test_spelled_interval_identification(
    first: str, second: str, interval_name: str, semitones: int
) -> None:
    interval = identify_interval(first, second)
    assert interval.name == interval_name
    assert interval.semitones == semitones
    if second == "B3":
        assert interval.direction == -1
    if first == "C" and second == "B#":
        assert interval.name == "augmented seventh"
        assert interval.semitones == 12


def test_interval_inversion_and_theoretical_transposition() -> None:
    tritone = identify_interval("C4", "F#4")
    assert invert_interval(tritone).name == "diminished fifth"
    assert transpose_by_interval("C4", Interval(4, "A", 6)) == "F#4"
    assert transpose_note("C#", 0, prefer_flats=True) == "Db"


@pytest.mark.parametrize(
    ("root", "scale", "expected"),
    [
        ("G", "major", ["G", "A", "B", "C", "D", "E", "F#"]),
        ("C", "harmonic_minor", ["C", "D", "Eb", "F", "G", "Ab", "B"]),
        ("D", "dorian", ["D", "E", "F", "G", "A", "B", "C"]),
        ("A", "minor_pentatonic", ["A", "C", "D", "E", "G"]),
        ("C", "blues", ["C", "Eb", "F", "Gb", "G", "Bb"]),
    ],
)
def test_scale_construction_preserves_degree_spelling(
    root: str, scale: str, expected: list[str]
) -> None:
    assert [note.name for note in scale_notes(root, scale)] == expected


def test_scale_identification_reports_ambiguity_and_custom_scale_validation() -> None:
    result = identify_scales(["C", "E", "G"])
    assert result["status"] == "ambiguous"
    assert result["candidates"]
    with pytest.raises(ValueError, match="start with tonic"):
        scale_notes("C", pattern=(1, 2, 4))


@pytest.mark.parametrize(
    ("symbol", "canonical", "quality", "notes"),
    [
        ("C", "C", "major", ["C", "E", "G"]),
        ("Cmin", "Cm", "minor", ["C", "Eb", "G"]),
        ("Cmaj7", "Cmaj7", "major7", ["C", "E", "G", "B"]),
        ("C°", "Cdim", "diminished", ["C", "Eb", "Gb"]),
        ("Cø7", "Cm7b5", "half_diminished7", ["C", "Eb", "Gb", "Bb"]),
        ("C7b9", "C7b9", "dominant7_flat9", ["C", "E", "G", "Bb", "Db"]),
        ("C7#9", "C7#9", "dominant7_sharp9", ["C", "E", "G", "Bb", "D#"]),
        ("Cmaj7#11", "Cmaj7#11", "major7", ["C", "E", "G", "B", "F#"]),
        ("C6/9", "C6/9", "six_nine", ["C", "E", "G", "A", "D"]),
    ],
)
def test_chord_parser_normalizes_aliases_and_preserves_structure(
    symbol: str, canonical: str, quality: str, notes: list[str]
) -> None:
    chord = parse_chord(symbol)
    assert chord.symbol == canonical
    assert chord.quality == quality
    assert list(chord_notes(chord)) == notes
    reparsed = parse_chord(chord.symbol)
    assert reparsed.quality == chord.quality
    assert reparsed.tones == chord.tones
    assert reparsed.pitch_classes == chord.pitch_classes


def test_chord_parser_modifiers_slash_chords_and_invalid_symbols() -> None:
    chord = parse_chord(" Dm7 / F ")
    assert chord.symbol == "Dm7/F"
    assert chord.bass is not None and chord.bass.name == "F"
    altered = parse_chord("C7no5add9")
    assert 5 in altered.omitted_tones
    assert 9 in altered.added_tones
    assert parse_chord("Cmajor7").symbol == "Cmaj7"
    assert parse_chord("C7(b9)").symbol == "C7b9"
    assert parse_chord("Cm(maj7)").symbol == "Cm(maj7)"
    assert parse_chord("C7sus4").symbol == "C7sus4"
    assert parse_chord("Cmaj7").extensions == (7,)
    assert parse_chord("C7b9").alterations == ("b9",)
    with pytest.raises(ValueError, match="Unsupported"):
        parse_chord("Cmaj13foo")
    with pytest.raises(ValueError, match="Invalid"):
        parse_chord("H7")
    with pytest.raises(ValueError, match="slash"):
        parse_chord("C/E/G")


def test_chord_inversion_does_not_infer_instrument_voicing() -> None:
    chord = parse_chord("G7/B")
    result = chord_inversion(chord)
    assert result["inversion"] == "first_inversion"
    assert result["bass_is_chord_tone"] is True
    assert result["voicing_inferred"] is False
    assert chord_inversion(parse_chord("C"), "D")["inversion"] == "non_chord_bass"
    assert analyze_progression(["C/E"], "C major")["chords"][0]["roman_numeral"] == "I6"


def test_key_signature_relationships_and_diatonic_harmony() -> None:
    assert key_signature("C major")["key_signature"] == []
    assert key_signature("Ab major")["signature_count"] == -4
    assert key_signature("A minor")["relative_key"] == "C major"
    assert key_signature("A harmonic_minor")["scale_notes"] == [
        "A",
        "B",
        "C",
        "D",
        "E",
        "F",
        "G#",
    ]
    major = diatonic_harmony("C major")
    assert [item["symbol"] for item in major] == ["C", "Dm", "Em", "F", "G", "Am", "Bdim"]
    minor = diatonic_harmony("A minor")
    assert [item["symbol"] for item in minor] == ["Am", "Bdim", "C", "Dm", "Em", "F", "G"]
    sevenths = diatonic_harmony("C major", sevenths=True)
    assert [item["quality"] for item in sevenths] == [
        "major7",
        "minor7",
        "minor7",
        "major7",
        "dominant7",
        "minor7",
        "half_diminished7",
    ]


def test_progression_roman_numerals_secondary_dominants_borrowing_and_cadences() -> None:
    result = analyze_progression(["C", "G", "Am", "F"], "C major")
    assert [item["roman_numeral"] for item in result["chords"]] == ["I", "V", "vi", "IV"]
    assert result["patterns"] == [{"name": "I–V–vi–IV", "start_index": 0}]
    cadence = analyze_progression(
        ["G", "C"],
        "C major",
        soprano_notes=["D4", "C4"],
    )["cadences"]
    assert cadence[0]["type"] == "perfect_authentic"
    secondary = analyze_progression(["D7", "G"], "C major")
    assert secondary["chords"][0]["roman_numeral"] == "V7/V"
    borrowed = analyze_progression(["Fm", "C"], "C major")
    assert borrowed["chords"][0]["roman_numeral"] == "borrowed iv"
    borrowed_flat_seventh = analyze_progression(["Bb", "C"], "C major")
    assert borrowed_flat_seventh["chords"][0]["classification"] == "borrowed"
    slash_chord = analyze_progression(["C/E"], "C major")
    assert slash_chord["chords"][0]["roman_numeral"] == "I6"


def test_progression_timestamps_and_transposition_invariants() -> None:
    event_ranges = [
        {"start_seconds": 0.0, "end_seconds": 2.0},
        {"start_seconds": 2.0, "end_seconds": 4.0},
    ]
    analyzed = analyze_progression(["C", "G"], "C major", timestamps=event_ranges)
    assert analyzed["harmonic_rhythm"]["mean_duration_seconds"] == 2.0
    transposed = transpose_progression(["C", "Am", "F", "G"], 2)
    assert transposed["symbols"] == ["D", "Bm", "G", "A"]
    source = parse_chord("Dm7/F")
    roundtrip = transpose_chord(transpose_chord(source, 5), -5)
    assert roundtrip.pitch_classes == source.pitch_classes
    assert roundtrip.quality == source.quality
    assert roundtrip.bass is not None
    assert roundtrip.bass.pitch_class == source.bass.pitch_class
    comparison = theory_compare(["C", "Am", "F", "G"], ["D", "Bm", "G", "A"])
    assert comparison["equivalent_under_transposition"] is True
    assert comparison["semitone_shift"] == 2


def test_capo_direction_is_explicit_and_invertible() -> None:
    written = capo_calculation(["C", "Am"], 1, key="C major", prefer_flats=True)
    assert written["sounding_chords"] == ["Db", "Bbm"]
    assert written["written_key"] == "C major"
    assert written["sounding_key"] == "Db major"
    reverse = capo_calculation(["Db", "Bbm"], 1, direction="sounding_to_written", key="Db major")
    assert reverse["written_chords"] == ["C", "Am"]
    assert reverse["written_key"] == "C major"


def test_stateless_theory_routes_validate_and_return_real_results(client: TestClient) -> None:
    chord = client.post("/api/v1/theory/chords/parse", json={"symbol": "Dm7/F"})
    assert chord.status_code == 200
    assert chord.json()["symbol"] == "Dm7/F"
    assert chord.json()["bass"] == "F"

    interval = client.post(
        "/api/v1/theory/intervals",
        json={"from_pitch": "C4", "to_pitch": "F#4"},
    )
    assert interval.status_code == 200
    assert interval.json()["name"] == "augmented fourth"

    transposed = client.post(
        "/api/v1/theory/transpose",
        json={"chord_symbols": ["C", "Am", "F", "G"], "semitones": 2},
    )
    assert transposed.status_code == 200
    assert transposed.json()["symbols"] == ["D", "Bm", "G", "A"]

    key_transposed = client.post(
        "/api/v1/theory/transpose-between-keys",
        json={
            "chord_symbols": ["C", "Am", "F", "G"],
            "source_key": "C major",
            "target_key": "D major",
        },
    )
    assert key_transposed.status_code == 200
    assert key_transposed.json()["symbols"] == ["D", "Bm", "G", "A"]

    capo = client.post(
        "/api/v1/theory/capo",
        json={"chord_symbols": ["C"], "capo_fret": 2, "key": "C major"},
    )
    assert capo.status_code == 200
    assert capo.json()["sounding_key"] == "D major"

    invalid = client.post("/api/v1/theory/chords/parse", json={"symbol": "Cmaj7xyz"})
    assert invalid.status_code == 422
    assert invalid.json()["code"] == "invalid_theory_input"


def test_analysis_theory_is_persisted_separately_and_owner_scoped(
    client: TestClient,
    settings,
) -> None:
    owner_id = uuid.uuid4()
    asset_id = uuid.uuid4()
    job_id = uuid.uuid4()
    analysis_id = uuid.uuid4()

    async def insert_records() -> None:
        engine = create_async_engine(settings.database_url)
        factory = async_sessionmaker(engine, expire_on_commit=False)
        async with factory.begin() as session:
            asset = AudioAsset(
                id=asset_id,
                owner_id=owner_id,
                asset_type="original_audio",
                original_filename="theory.wav",
                storage_key=f"assets/{owner_id}/{asset_id}.wav",
                storage_provider="local",
                mime_type="audio/wav",
                file_extension="wav",
                size_bytes=1,
                sha256="a" * 64,
                status="available",
            )
            job = ProcessingJob(
                id=job_id,
                owner_id=owner_id,
                input_asset_id=asset_id,
                task_type="music_analysis",
                status="succeeded",
                stage="SUCCEEDED",
            )
            analysis = MusicAnalysisRecord(
                id=analysis_id,
                owner_id=owner_id,
                source_asset_id=asset_id,
                job_id=job_id,
                version=1,
                status="succeeded",
                tasks=["chords", "key"],
                source_sha256="a" * 64,
                engine_version="test",
                model_references=[],
                execution_device="cpu",
                parameters={},
                result={
                    "chords": [
                        {"symbol": "C", "start_seconds": 0.0, "end_seconds": 1.0},
                        {"symbol": "G7", "start_seconds": 1.0, "end_seconds": 2.0},
                    ],
                    "key": {"selected": {"tonic": "C", "mode": "major"}},
                },
                stage_errors=[],
            )
            session.add_all([asset, job, analysis])
        await engine.dispose()

    asyncio.run(insert_records())
    path = f"/api/v1/music/{asset_id}/analysis/{analysis_id}/theory"
    headers = {"X-Dev-User-ID": str(owner_id)}
    response = client.post(path, headers=headers, json={})
    assert response.status_code == 201
    assert response.json()["source"] == "detected_chords"
    assert response.json()["key_context"] == {"tonic": "C", "mode": "major"}
    assert response.json()["result"]["status"] == "interpreted"
    assert response.json()["result"]["chords"][1]["roman_numeral"] == "V7"

    latest = client.get(path, headers=headers)
    assert latest.status_code == 200
    assert latest.json()["version"] == 1
    assert latest.json()["id"] == response.json()["id"]
    forbidden = client.get(path, headers={"X-Dev-User-ID": str(uuid.uuid4())})
    assert forbidden.status_code == 404
