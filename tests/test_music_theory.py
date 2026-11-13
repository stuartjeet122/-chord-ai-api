from __future__ import annotations

from fastapi.testclient import TestClient

from music_api.music.theory import analyze_progression, parse_chord


def test_chord_parser_and_progression_analysis() -> None:
    assert parse_chord("Dm7/F").symbol == "Dm7/F"
    result = analyze_progression(["C", "G", "Am", "F"], "C major")
    assert [item["roman_numeral"] for item in result["chords"]] == ["I", "V", "vi", "IV"]


def test_theory_routes_are_not_exposed(client: TestClient) -> None:
    response = client.post("/api/v1/theory/chords/parse", json={"symbol": "Dm7/F"})
    assert response.status_code == 404
