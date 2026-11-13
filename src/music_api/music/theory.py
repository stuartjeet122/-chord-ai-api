from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any

NOTE_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
FLAT_NOTE_NAMES = ("C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B")
ENHARMONIC_NAMES = {
    "B#": 0,
    "C": 0,
    "C#": 1,
    "Cbb": 10,
    "Db": 1,
    "D": 2,
    "D#": 3,
    "Eb": 3,
    "E": 4,
    "Fb": 4,
    "E#": 5,
    "F": 5,
    "F#": 6,
    "Gb": 6,
    "G": 7,
    "G#": 8,
    "Ab": 8,
    "A": 9,
    "A#": 10,
    "Bb": 10,
    "B": 11,
    "B##": 1,
    "Cb": 11,
    "C##": 2,
    "Dbb": 0,
    "D##": 4,
    "Ebb": 2,
    "E##": 6,
    "Fbb": 3,
    "F##": 7,
    "Gbb": 5,
    "G##": 9,
    "Abb": 7,
    "A##": 11,
    "Bbb": 9,
}

# Kept stable because the Phase 7 DSP chord candidate engine consumes this mapping.
CHORD_INTERVALS: dict[str, tuple[int, ...]] = {
    "major": (0, 4, 7),
    "minor": (0, 3, 7),
    "diminished": (0, 3, 6),
    "augmented": (0, 4, 8),
    "sus2": (0, 2, 7),
    "sus4": (0, 5, 7),
    "power": (0, 7),
    "major6": (0, 4, 7, 9),
    "minor6": (0, 3, 7, 9),
    "major7": (0, 4, 7, 11),
    "minor7": (0, 3, 7, 10),
    "dominant7": (0, 4, 7, 10),
    "diminished7": (0, 3, 6, 9),
    "half_diminished7": (0, 3, 6, 10),
    "minor_major7": (0, 3, 7, 11),
    "augmented_major7": (0, 4, 8, 11),
    "six_nine": (0, 2, 4, 7, 9),
    "add9": (0, 2, 4, 7),
    "add11": (0, 4, 5, 7),
    "add13": (0, 4, 7, 9),
    "major9": (0, 2, 4, 7, 11),
    "minor9": (0, 2, 3, 7, 10),
    "dominant9": (0, 2, 4, 7, 10),
    "major11": (0, 2, 4, 5, 7, 11),
    "minor11": (0, 2, 3, 5, 7, 10),
    "dominant11": (0, 2, 4, 5, 7, 10),
    "major13": (0, 2, 4, 7, 9, 11),
    "minor13": (0, 2, 3, 7, 9, 10),
    "dominant13": (0, 2, 4, 7, 9, 10),
    "dominant7_flat5": (0, 4, 6, 10),
    "dominant7_sharp5": (0, 4, 8, 10),
    "dominant7_flat9": (0, 1, 4, 7, 10),
    "dominant7_sharp9": (0, 3, 4, 7, 10),
    "dominant7_sharp11": (0, 4, 6, 7, 10),
    "dominant7_flat13": (0, 4, 7, 8, 10),
}

QUALITY_SUFFIXES = {
    "major": "",
    "minor": "m",
    "diminished": "dim",
    "augmented": "aug",
    "sus2": "sus2",
    "sus4": "sus4",
    "power": "5",
    "major6": "6",
    "minor6": "m6",
    "major7": "maj7",
    "minor7": "m7",
    "dominant7": "7",
    "diminished7": "dim7",
    "half_diminished7": "m7b5",
    "minor_major7": "m(maj7)",
    "augmented_major7": "augmaj7",
    "six_nine": "6/9",
    "add9": "add9",
    "add11": "add11",
    "add13": "add13",
    "major9": "maj9",
    "minor9": "m9",
    "dominant9": "9",
    "major11": "maj11",
    "minor11": "m11",
    "dominant11": "11",
    "major13": "maj13",
    "minor13": "m13",
    "dominant13": "13",
    "dominant7_flat5": "7b5",
    "dominant7_sharp5": "7#5",
    "dominant7_flat9": "7b9",
    "dominant7_sharp9": "7#9",
    "dominant7_sharp11": "7#11",
    "dominant7_flat13": "7b13",
}

MAJOR_PROFILE = (6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88)
MINOR_PROFILE = (6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17)

_LETTERS = "CDEFGAB"
_NATURAL_PCS = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
_NOTE_PATTERN = re.compile(r"^([A-Ga-g])((?:#{1,2}|b{1,2})?)(-?\d+)?$")
_NATURAL_INTERVALS = {1: 0, 2: 2, 3: 4, 4: 5, 5: 7, 6: 9, 7: 11}
_SCALE_PATTERNS: dict[str, tuple[int, ...]] = {
    "major": (0, 2, 4, 5, 7, 9, 11),
    "natural_minor": (0, 2, 3, 5, 7, 8, 10),
    "harmonic_minor": (0, 2, 3, 5, 7, 8, 11),
    "melodic_minor": (0, 2, 3, 5, 7, 9, 11),
    "chromatic": tuple(range(12)),
    "major_pentatonic": (0, 2, 4, 7, 9),
    "minor_pentatonic": (0, 3, 5, 7, 10),
    "blues": (0, 3, 5, 6, 7, 10),
    "ionian": (0, 2, 4, 5, 7, 9, 11),
    "dorian": (0, 2, 3, 5, 7, 9, 10),
    "phrygian": (0, 1, 3, 5, 7, 8, 10),
    "lydian": (0, 2, 4, 6, 7, 9, 11),
    "mixolydian": (0, 2, 4, 5, 7, 9, 10),
    "aeolian": (0, 2, 3, 5, 7, 8, 10),
    "locrian": (0, 1, 3, 5, 6, 8, 10),
}
_SCALE_ALIASES = {"minor": "natural_minor", "aeolian_minor": "natural_minor"}
_SCALE_DEGREE_BY_SEMITONE = (1, 2, 2, 3, 3, 4, 5, 5, 6, 6, 7, 7)
_FORMULAS: dict[str, tuple[tuple[int, int], ...]] = {
    "major": ((1, 0), (3, 0), (5, 0)),
    "minor": ((1, 0), (3, -1), (5, 0)),
    "diminished": ((1, 0), (3, -1), (5, -1)),
    "augmented": ((1, 0), (3, 0), (5, 1)),
    "sus2": ((1, 0), (2, 0), (5, 0)),
    "sus4": ((1, 0), (4, 0), (5, 0)),
    "power": ((1, 0), (5, 0)),
    "major6": ((1, 0), (3, 0), (5, 0), (6, 0)),
    "minor6": ((1, 0), (3, -1), (5, 0), (6, 0)),
    "major7": ((1, 0), (3, 0), (5, 0), (7, 0)),
    "minor7": ((1, 0), (3, -1), (5, 0), (7, -1)),
    "dominant7": ((1, 0), (3, 0), (5, 0), (7, -1)),
    "diminished7": ((1, 0), (3, -1), (5, -1), (7, -2)),
    "half_diminished7": ((1, 0), (3, -1), (5, -1), (7, -1)),
    "minor_major7": ((1, 0), (3, -1), (5, 0), (7, 0)),
    "augmented_major7": ((1, 0), (3, 0), (5, 1), (7, 0)),
    "six_nine": ((1, 0), (3, 0), (5, 0), (6, 0), (9, 0)),
    "add9": ((1, 0), (3, 0), (5, 0), (9, 0)),
    "add11": ((1, 0), (3, 0), (5, 0), (11, 0)),
    "add13": ((1, 0), (3, 0), (5, 0), (13, 0)),
    "major9": ((1, 0), (3, 0), (5, 0), (7, 0), (9, 0)),
    "minor9": ((1, 0), (3, -1), (5, 0), (7, -1), (9, 0)),
    "dominant9": ((1, 0), (3, 0), (5, 0), (7, -1), (9, 0)),
    "major11": ((1, 0), (3, 0), (5, 0), (7, 0), (9, 0), (11, 0)),
    "minor11": ((1, 0), (3, -1), (5, 0), (7, -1), (9, 0), (11, 0)),
    "dominant11": ((1, 0), (3, 0), (5, 0), (7, -1), (9, 0), (11, 0)),
    "major13": ((1, 0), (3, 0), (5, 0), (7, 0), (9, 0), (13, 0)),
    "minor13": ((1, 0), (3, -1), (5, 0), (7, -1), (9, 0), (13, 0)),
    "dominant13": ((1, 0), (3, 0), (5, 0), (7, -1), (9, 0), (13, 0)),
    "dominant7_flat5": ((1, 0), (3, 0), (5, -1), (7, -1)),
    "dominant7_sharp5": ((1, 0), (3, 0), (5, 1), (7, -1)),
    "dominant7_flat9": ((1, 0), (3, 0), (5, 0), (7, -1), (9, -1)),
    "dominant7_sharp9": ((1, 0), (3, 0), (5, 0), (7, -1), (9, 1)),
    "dominant7_sharp11": ((1, 0), (3, 0), (5, 0), (7, -1), (11, 1)),
    "dominant7_flat13": ((1, 0), (3, 0), (5, 0), (7, -1), (13, -1)),
}
_QUALITY_ALIASES = {
    "M": "major",
    "maj": "major",
    "major": "major",
    "m": "minor",
    "min": "minor",
    "minor": "minor",
    "dim": "diminished",
    "diminished": "diminished",
    "aug": "augmented",
    "augmented": "augmented",
    "sus2": "sus2",
    "sus4": "sus4",
    "sus": "sus4",
    "5": "power",
    "power": "power",
    "6": "major6",
    "m6": "minor6",
    "maj7": "major7",
    "M7": "major7",
    "m7": "minor7",
    "7": "dominant7",
    "dim7": "diminished7",
    "o7": "diminished7",
    "°7": "diminished7",
    "m7b5": "half_diminished7",
    "ø7": "half_diminished7",
    "m(maj7)": "minor_major7",
    "mmaj7": "minor_major7",
    "augmaj7": "augmented_major7",
    "6/9": "six_nine",
    "add9": "add9",
    "add11": "add11",
    "add13": "add13",
    "maj9": "major9",
    "m9": "minor9",
    "9": "dominant9",
    "maj11": "major11",
    "m11": "minor11",
    "11": "dominant11",
    "maj13": "major13",
    "m13": "minor13",
    "13": "dominant13",
    "7b5": "dominant7_flat5",
    "7#5": "dominant7_sharp5",
    "7b9": "dominant7_flat9",
    "7#9": "dominant7_sharp9",
    "7#11": "dominant7_sharp11",
    "7b13": "dominant7_flat13",
}
_KEY_SIGNATURES: dict[str, tuple[str, ...]] = {
    "C": (),
    "G": ("F#",),
    "D": ("F#", "C#"),
    "A": ("F#", "C#", "G#"),
    "E": ("F#", "C#", "G#", "D#"),
    "B": ("F#", "C#", "G#", "D#", "A#"),
    "F#": ("F#", "C#", "G#", "D#", "A#", "E#"),
    "C#": ("F#", "C#", "G#", "D#", "A#", "E#", "B#"),
    "F": ("Bb",),
    "Bb": ("Bb", "Eb"),
    "Eb": ("Bb", "Eb", "Ab"),
    "Ab": ("Bb", "Eb", "Ab", "Db"),
    "Db": ("Bb", "Eb", "Ab", "Db", "Gb"),
    "Gb": ("Bb", "Eb", "Ab", "Db", "Gb", "Cb"),
    "Cb": ("Bb", "Eb", "Ab", "Db", "Gb", "Cb", "Fb"),
}
_FIFTH_ORDER_SHARPS = ("F", "C", "G", "D", "A", "E", "B")
_FIFTH_ORDER_FLATS = ("B", "E", "A", "D", "G", "C", "F")
_ROMAN = ("I", "II", "III", "IV", "V", "VI", "VII")
_MAJOR_ROMANS = ("I", "ii", "iii", "IV", "V", "vi", "vii°")
_MINOR_ROMANS = ("i", "ii°", "III", "iv", "v", "VI", "VII")


@dataclass(frozen=True)
class Note:
    letter: str
    accidental: int = 0
    octave: int | None = None

    @property
    def pitch_class(self) -> int:
        return (_NATURAL_PCS[self.letter] + self.accidental) % 12

    @property
    def midi(self) -> int | None:
        if self.octave is None:
            return None
        return (self.octave + 1) * 12 + _NATURAL_PCS[self.letter] + self.accidental

    @property
    def frequency_hz(self) -> float | None:
        midi = self.midi
        return None if midi is None else 440.0 * 2.0 ** ((midi - 69) / 12.0)

    @property
    def name(self) -> str:
        accidental = "#" * self.accidental if self.accidental > 0 else "b" * -self.accidental
        return f"{self.letter}{accidental}"

    def __str__(self) -> str:
        suffix = "" if self.octave is None else str(self.octave)
        return f"{self.name}{suffix}"


@dataclass(frozen=True)
class Interval:
    number: int
    quality: str
    semitones: int
    direction: int = 1

    @property
    def name(self) -> str:
        names = {
            1: "unison",
            2: "second",
            3: "third",
            4: "fourth",
            5: "fifth",
            6: "sixth",
            7: "seventh",
            8: "octave",
            9: "ninth",
            10: "tenth",
            11: "eleventh",
            12: "twelfth",
            13: "thirteenth",
            14: "fourteenth",
            15: "two octaves",
        }
        qualities = {
            "P": "perfect",
            "M": "major",
            "m": "minor",
            "A": "augmented",
            "d": "diminished",
        }
        base = names.get(self.number, f"{self.number}th")
        if self.number == 1 and self.quality == "P":
            return qualities[self.quality] + " " + base
        return f"{qualities[self.quality]} {base}"


@dataclass(frozen=True)
class KeyCandidate:
    tonic: str
    mode: str
    score: float
    roman_numeral_scale: tuple[int, ...]


@dataclass(frozen=True)
class ChordTone:
    degree: int
    alteration: int
    semitones: int

    @property
    def label(self) -> str:
        accidental = "#" * self.alteration if self.alteration > 0 else "b" * -self.alteration
        return f"{accidental}{self.degree}"


@dataclass(frozen=True)
class Chord:
    root: Note
    quality: str
    tones: tuple[ChordTone, ...]
    bass: Note | None = None
    extensions: tuple[int, ...] = ()
    alterations: tuple[str, ...] = ()
    added_tones: tuple[int, ...] = ()
    omitted_tones: tuple[int, ...] = ()

    @property
    def symbol(self) -> str:
        suffix = QUALITY_SUFFIXES.get(self.quality)
        if suffix is None:
            raise ValueError(f"Unsupported chord quality: {self.quality}")
        if self.quality == "major" and self.extensions:
            suffix = "maj" + str(max(self.extensions))
        elif self.quality == "minor" and self.extensions:
            suffix = "m" + str(max(self.extensions))
        elif self.quality == "dominant7" and self.extensions:
            suffix = str(max(self.extensions))
        extras = [f"add{degree}" for degree in self.added_tones]
        if self.quality not in {
            "add9",
            "add11",
            "add13",
            "dominant7_flat5",
            "dominant7_sharp5",
            "dominant7_flat9",
            "dominant7_sharp9",
            "dominant7_sharp11",
            "dominant7_flat13",
            "half_diminished7",
            "augmented_major7",
        }:
            extras.extend(self.alterations)
        if self.quality in {"add9", "add11", "add13"}:
            extras = [item for item in extras if item != self.quality]
        extras.extend(f"no{degree}" for degree in self.omitted_tones)
        if extras:
            suffix += "".join(extras)
        bass = "" if self.bass is None else f"/{self.bass.name}"
        return f"{self.root.name}{suffix}{bass}"

    @property
    def pitch_classes(self) -> tuple[int, ...]:
        return tuple(sorted({(self.root.pitch_class + tone.semitones) % 12 for tone in self.tones}))

    def as_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "root": self.root.name,
            "root_pitch_class": self.root.pitch_class,
            "quality": self.quality,
            "extensions": list(self.extensions),
            "alterations": list(self.alterations),
            "added_tones": list(self.added_tones),
            "omitted_tones": list(self.omitted_tones),
            "bass": None if self.bass is None else self.bass.name,
            "bass_pitch_class": None if self.bass is None else self.bass.pitch_class,
            "tones": [
                {
                    "degree": tone.degree,
                    "alteration": tone.alteration,
                    "semitones": tone.semitones,
                    "note": _spell_degree(self.root, tone.degree, tone.semitones).name,
                }
                for tone in self.tones
            ],
            "pitch_classes": list(self.pitch_classes),
        }


def parse_note(value: str) -> Note:
    normalized = value.strip().replace("♯", "#").replace("♭", "b")
    normalized = normalized.replace("𝄪", "##").replace("𝄫", "bb")
    match = _NOTE_PATTERN.fullmatch(normalized)
    if match is None:
        raise ValueError(
            f"Invalid note spelling: {value!r}. Expected A-G with up to two accidentals."
        )
    letter = match.group(1).upper()
    token = match.group(2)
    accidental = len(token) if token.startswith("#") else -len(token) if token else 0
    octave = None if match.group(3) is None else int(match.group(3))
    return Note(letter, accidental, octave)


def pitch_class(note: str) -> int:
    return parse_note(note).pitch_class


def pitch_to_frequency(note: str) -> float:
    parsed = parse_note(note)
    if parsed.frequency_hz is None:
        raise ValueError("An octave is required to convert a note to frequency.")
    return parsed.frequency_hz


def midi_to_note(midi: int, *, prefer_flats: bool = False) -> str:
    if not isinstance(midi, int):
        raise ValueError("MIDI note number must be an integer.")
    if not 0 <= midi <= 127:
        raise ValueError("MIDI note number must be between 0 and 127.")
    name = (FLAT_NOTE_NAMES if prefer_flats else NOTE_NAMES)[midi % 12]
    octave = midi // 12 - 1
    return f"{name}{octave}"


def frequency_to_midi(frequency_hz: float) -> float:
    if not math.isfinite(frequency_hz) or frequency_hz <= 0:
        raise ValueError("Frequency must be a finite positive value.")
    return 69.0 + 12.0 * math.log2(frequency_hz / 440.0)


def _spell_pitch_class(
    value: int,
    *,
    prefer_flats: bool = False,
    key: str | None = None,
) -> Note:
    target = value % 12
    if key is not None:
        try:
            key_root, key_mode = parse_key(key)
            scale_name = _key_scale_name(key) if key_mode != "major" else "major"
            if scale_name == "minor":
                scale_name = "natural_minor"
            notes = scale_notes(key_root, scale_name)
            for note in notes:
                if note.pitch_class == target:
                    return note
        except ValueError:
            raise
    name = (FLAT_NOTE_NAMES if prefer_flats else NOTE_NAMES)[target]
    return parse_note(name)


def _spell_degree(root: Note, degree: int, semitones: int, octave: int | None = None) -> Note:
    if degree < 1:
        raise ValueError("Scale/chord degree must be at least 1.")
    steps = degree - 1
    root_index = _LETTERS.index(root.letter)
    target_letter = _LETTERS[(root_index + steps) % 7]
    target_pc = (root.pitch_class + semitones) % 12
    accidental = (target_pc - _NATURAL_PCS[target_letter]) % 12
    if accidental > 6:
        accidental -= 12
    if accidental not in {-2, -1, 0, 1, 2}:
        raise ValueError("The requested spelling requires more than a double accidental.")
    target_octave = None
    if octave is not None:
        target_octave = octave + (root_index + steps) // 7
    elif root.octave is not None:
        target_octave = root.octave + (root_index + steps) // 7
    return Note(target_letter, accidental, target_octave)


def transpose_note(
    note: str,
    semitones: int,
    *,
    prefer_flats: bool = False,
    key: str | None = None,
) -> str:
    source = parse_note(note)
    if source.octave is None:
        return _spell_pitch_class(
            source.pitch_class + semitones, prefer_flats=prefer_flats, key=key
        ).name
    target_midi = source.midi
    assert target_midi is not None
    target = target_midi + semitones
    spelled = _spell_pitch_class(target, prefer_flats=prefer_flats, key=key)
    if spelled.midi is None:
        midi_name = midi_to_note(target, prefer_flats=prefer_flats)
        return midi_name
    octave = target // 12 - 1
    return f"{spelled.name}{octave}"


def _expected_interval_semitones(number: int, quality: str) -> int:
    simple = ((number - 1) % 7) + 1
    octaves = (number - 1) // 7
    base = _NATURAL_INTERVALS[simple] + 12 * octaves
    perfect_class = simple in {1, 4, 5}
    if quality == "d" and perfect_class:
        adjustment = -1
    elif quality == "d" and not perfect_class:
        adjustment = -2
    elif quality == "P":
        adjustment = 0
    elif quality == "m":
        adjustment = -1
    elif quality == "M":
        adjustment = 0
    elif quality == "A":
        adjustment = 1
    else:
        raise ValueError(f"Unsupported interval quality: {quality}")
    if perfect_class and quality in {"M", "m"}:
        raise ValueError(f"Quality {quality} is not valid for interval number {number}.")
    if not perfect_class and quality == "P":
        raise ValueError(f"Perfect quality is not valid for interval number {number}.")
    return base + adjustment


def identify_interval(from_pitch: str, to_pitch: str) -> Interval:
    first = parse_note(from_pitch)
    second = parse_note(to_pitch)
    if first.octave is None or second.octave is None:
        direction = 1
        diatonic = (_LETTERS.index(second.letter) - _LETTERS.index(first.letter)) % 7
        number = diatonic + 1
        natural_delta = _NATURAL_PCS[second.letter] - _NATURAL_PCS[first.letter]
        if _LETTERS.index(second.letter) < _LETTERS.index(first.letter):
            natural_delta += 12
        semitones = natural_delta + second.accidental - first.accidental
    else:
        first_midi = first.midi
        second_midi = second.midi
        assert first_midi is not None and second_midi is not None
        direction = 1 if second_midi >= first_midi else -1
        semitones = abs(second_midi - first_midi)
        first_index = first.octave * 7 + _LETTERS.index(first.letter)
        second_index = second.octave * 7 + _LETTERS.index(second.letter)
        number = abs(second_index - first_index) + 1
    simple = ((number - 1) % 7) + 1
    base = _NATURAL_INTERVALS[simple] + 12 * ((number - 1) // 7)
    difference = semitones - base
    perfect_class = simple in {1, 4, 5}
    if difference == 0:
        quality = "P" if perfect_class else "M"
    elif difference == -1 and not perfect_class:
        quality = "m"
    elif difference > 0:
        quality = "A"
    elif perfect_class and difference < 0:
        quality = "d"
    elif difference <= -2:
        quality = "d"
    else:
        raise ValueError("The spelled interval cannot be represented by a standard quality.")
    return Interval(number, quality, semitones, direction)


def invert_interval(interval: Interval) -> Interval:
    simple = ((interval.number - 1) % 7) + 1
    inverse_number = 1 if interval.number == 8 else 9 - simple
    if interval.number == 1:
        inverse_number = 8
    if interval.quality == "M":
        quality = "m"
    elif interval.quality == "m":
        quality = "M"
    elif interval.quality == "A":
        quality = "d"
    elif interval.quality == "d":
        quality = "A"
    else:
        quality = "P"
    inverse_semitones = 0 if interval.number == 8 else 12 - interval.semitones % 12
    return Interval(inverse_number, quality, inverse_semitones, interval.direction)


def transpose_by_interval(note: str, interval: Interval) -> str:
    source = parse_note(note)
    steps = (interval.number - 1) * interval.direction
    destination_letter = _LETTERS[(_LETTERS.index(source.letter) + steps) % 7]
    target_pc = (source.pitch_class + interval.semitones * interval.direction) % 12
    alteration = (target_pc - _NATURAL_PCS[destination_letter]) % 12
    if alteration > 6:
        alteration -= 12
    if alteration not in {-2, -1, 0, 1, 2}:
        raise ValueError("The requested interval requires more than a double accidental.")
    octave = None
    if source.octave is not None:
        octave = source.octave + (_LETTERS.index(source.letter) + steps) // 7
    return str(Note(destination_letter, alteration, octave))


def scale_notes(
    root: str | Note, scale: str = "major", pattern: tuple[int, ...] | None = None
) -> tuple[Note, ...]:
    tonic = parse_note(root) if isinstance(root, str) else root
    scale_name = _SCALE_ALIASES.get(scale.lower(), scale.lower())
    intervals = pattern if pattern is not None else _SCALE_PATTERNS.get(scale_name)
    if intervals is None:
        raise ValueError(f"Unsupported scale: {scale}")
    if not intervals or intervals[0] != 0:
        raise ValueError("A scale pattern must be non-empty and start with tonic interval 0.")
    if any(not isinstance(value, int) or value < 0 or value > 24 for value in intervals):
        raise ValueError("Scale intervals must be integer semitone values from 0 to 24.")
    if tuple(sorted(set(intervals))) != intervals:
        raise ValueError("Scale intervals must be unique and strictly ascending.")
    notes: list[Note] = []
    for interval in intervals:
        octave, within_octave = divmod(interval, 12)
        degree = _SCALE_DEGREE_BY_SEMITONE[within_octave] + 7 * octave
        notes.append(_spell_degree(tonic, degree, interval))
    return tuple(notes)


def scale_information(
    root: str, scale: str = "major", pattern: tuple[int, ...] | None = None
) -> dict[str, Any]:
    scale_name = _SCALE_ALIASES.get(scale.lower(), scale.lower())
    intervals = pattern if pattern is not None else _SCALE_PATTERNS.get(scale_name)
    notes = scale_notes(root, scale, pattern)
    return {
        "root": parse_note(root).name,
        "name": scale_name if pattern is None else "custom",
        "interval_pattern": list(intervals or ()),
        "notes": [note.name for note in notes],
        "pitch_classes": [note.pitch_class for note in notes],
        "degrees": [_degree_label(interval) for interval in (intervals or ())],
        "tonal_context": (
            {
                "tonic": parse_note(root).name,
                "mode": "major" if scale_name in {"major", "ionian"} else "minor",
            }
            if scale_name in {"major", "ionian", "natural_minor", "aeolian"}
            else None
        ),
    }


def _degree_label(semitones: int) -> str:
    octave, within_octave = divmod(semitones, 12)
    degree = _SCALE_DEGREE_BY_SEMITONE[within_octave] + 7 * octave
    alteration = semitones - _natural_semitones(degree)
    accidental = "#" * alteration if alteration > 0 else "b" * -alteration
    return f"{accidental}{degree}"


def identify_scales(notes: list[str]) -> dict[str, Any]:
    if not notes:
        raise ValueError("At least one note is required for scale identification.")
    observed = {parse_note(note).pitch_class for note in notes}
    candidates: list[dict[str, Any]] = []
    seen: set[tuple[int, str]] = set()
    for tonic_pc in range(12):
        tonic = _spell_pitch_class(tonic_pc, prefer_flats=any("b" in note for note in notes)).name
        for scale_name, pattern in _SCALE_PATTERNS.items():
            candidate_pcs = {(tonic_pc + interval) % 12 for interval in pattern}
            covered = len(observed & candidate_pcs)
            extra = len(candidate_pcs - observed)
            score = covered / len(observed) - extra / (12 * len(candidate_pcs))
            key = tonic_pc, scale_name
            if covered == len(observed) and key not in seen:
                seen.add(key)
                candidates.append(
                    {
                        "root": tonic,
                        "scale": scale_name,
                        "matched_notes": covered,
                        "input_note_count": len(observed),
                        "score": round(score, 6),
                    }
                )
    candidates.sort(key=lambda item: (-item["score"], item["root"], item["scale"]))
    return {
        "status": "unique" if len(candidates) == 1 else "ambiguous" if candidates else "no_match",
        "candidates": candidates[:50],
    }


def _natural_semitones(degree: int) -> int:
    octaves, simple_index = divmod(degree - 1, 7)
    return _NATURAL_INTERVALS[simple_index + 1] + octaves * 12


def _chord_tones(formula: tuple[tuple[int, int], ...]) -> tuple[ChordTone, ...]:
    return tuple(
        ChordTone(degree, alteration, _natural_semitones(degree) + alteration)
        for degree, alteration in formula
    )


def _suffix_base(suffix: str) -> tuple[str, str, int | None]:
    token = suffix
    lower = token.lower()
    extension: int | None
    major_extension = re.match(r"(?i)^major(13|11|9|7)?", token)
    if major_extension is not None:
        extension = int(major_extension.group(1)) if major_extension.group(1) else None
        quality = {
            None: "major",
            7: "major7",
            9: "major9",
            11: "major11",
            13: "major13",
        }[extension]
        return quality, QUALITY_SUFFIXES[quality], extension
    minor_extension = re.match(r"(?i)^(?:minor|min)(13|11|9|7|6)", token)
    if minor_extension is not None:
        extension = int(minor_extension.group(1))
        quality = {6: "minor6", 7: "minor7", 9: "minor9", 11: "minor11", 13: "minor13"}[extension]
        return quality, QUALITY_SUFFIXES[quality], extension
    exact_aliases = sorted(_QUALITY_ALIASES, key=len, reverse=True)
    for alias in exact_aliases:
        if token.startswith(alias):
            quality = _QUALITY_ALIASES[alias]
            canonical = QUALITY_SUFFIXES[quality]
            extension = next((degree for degree in (6, 7, 9, 11, 13) if str(degree) in alias), None)
            # M7 is major seventh, while m7 is minor seventh.
            if alias == "M7":
                quality, canonical, extension = "major7", "maj7", 7
            return quality, canonical, extension
    if lower.startswith("m(maj7)"):
        return "minor_major7", "m(maj7)", 7
    if lower.startswith("minor-major7"):
        return "minor_major7", "m(maj7)", 7
    if lower.startswith("maj"):
        match = re.match(r"(?i)^maj(13|11|9|7)?", token)
        if match:
            extension = int(match.group(1)) if match.group(1) else None
            quality = {None: "major", 7: "major7", 9: "major9", 11: "major11", 13: "major13"}[
                extension
            ]
            return quality, QUALITY_SUFFIXES[quality], extension
    if token.startswith("M"):
        match = re.match(r"^M(13|11|9|7)?", token)
        if match:
            extension = int(match.group(1)) if match.group(1) else None
            quality = {None: "major", 7: "major7", 9: "major9", 11: "major11", 13: "major13"}[
                extension
            ]
            return quality, QUALITY_SUFFIXES[quality], extension
    for prefix in ("minor", "min", "m"):
        if lower.startswith(prefix):
            rest = token[len(prefix) :]
            match = re.match(r"^(13|11|9|7|6)?", rest)
            if match:
                extension = int(match.group(1)) if match.group(1) else None
                quality = {
                    None: "minor",
                    6: "minor6",
                    7: "minor7",
                    9: "minor9",
                    11: "minor11",
                    13: "minor13",
                }[extension]
                return quality, QUALITY_SUFFIXES[quality], extension
    for extension, quality in (
        (13, "dominant13"),
        (11, "dominant11"),
        (9, "dominant9"),
        (7, "dominant7"),
        (6, "major6"),
    ):
        if lower.startswith(str(extension)):
            return quality, QUALITY_SUFFIXES[quality], extension
    if lower.startswith(("dim", "o", "°")):
        if lower.startswith(("dim7", "o7")) or token.startswith("°7"):
            return "diminished7", "dim7", 7
        return "diminished", "dim", None
    if lower.startswith(("aug", "+")):
        if lower.startswith(("augmaj7", "augm7")):
            return "augmented_major7", "augmaj7", 7
        return "augmented", "aug", None
    if lower.startswith(("sus2", "sus4", "sus")):
        quality = "sus2" if lower.startswith("sus2") else "sus4"
        return quality, QUALITY_SUFFIXES[quality], None
    if token == "":
        return "major", "", None
    raise ValueError(f"Unsupported chord quality or extension in {suffix!r}.")


def parse_chord(symbol: str) -> Chord:
    if not isinstance(symbol, str) or not symbol.strip():
        raise ValueError("Chord symbol must be a non-empty string.")
    text = (
        symbol.strip()
        .replace("♯", "#")
        .replace("♭", "b")
        .replace("Δ", "maj")
        .replace("∆", "maj")
        .replace("ø", "ø")
        .replace("°", "°")
        .replace("(", "")
        .replace(")", "")
    )
    if len(text) > 64:
        raise ValueError("Chord symbol cannot exceed 64 characters.")
    bass: Note | None = None
    if "/" in text:
        head, tail = text.rsplit("/", 1)
        try:
            bass = parse_note(tail)
        except ValueError as exc:
            if tail != "9" or not head.endswith("6"):
                raise ValueError(f"Invalid slash-chord bass note in {symbol!r}.") from exc
        else:
            text = head
    match = re.match(r"^([A-Ga-g](?:#{1,2}|b{1,2})?)(.*)$", text)
    if match is None:
        raise ValueError(f"Invalid chord root in {symbol!r}.")
    root = parse_note(match.group(1))
    suffix = match.group(2).replace(" ", "")
    if "/" in suffix and suffix != "6/9":
        raise ValueError(f"Malformed slash-chord notation in {symbol!r}.")
    quality, canonical_base, extension = _suffix_base(suffix)
    # Prefix aliases may have different length from their canonical output.
    prefix_len = _base_prefix_length(suffix, quality)
    remainder = suffix[prefix_len:]
    formula = list(_FORMULAS.get(quality, ()))
    if not formula:
        raise ValueError(f"Chord quality {quality!r} has no supported chord formula.")
    extensions: set[int] = set()
    if extension in {6, 7, 9, 11, 13}:
        extensions.add(extension)
    alterations: list[str] = []
    added_tones: set[int] = set()
    omitted_tones: set[int] = set()
    if quality in {
        "dominant7_flat5",
        "dominant7_sharp5",
        "dominant7_flat9",
        "dominant7_sharp9",
        "dominant7_sharp11",
        "dominant7_flat13",
    }:
        alterations.append(canonical_base[1:])
    elif quality == "half_diminished7":
        alterations.append("b5")
    elif quality == "augmented_major7":
        alterations.append("#5")
    elif quality in {"add9", "add11", "add13"}:
        added_tones.add(int(quality[-2:]))
    if quality == "six_nine":
        extensions.update((6, 9))
    cursor = 0
    while cursor < len(remainder):
        mod = re.match(r"(?i)(add|omit|no|sus|b|#)(\d+)?", remainder[cursor:])
        if mod is None:
            raise ValueError(
                f"Unsupported chord modifier near {remainder[cursor:]!r} in {symbol!r}."
            )
        operation = mod.group(1).lower()
        degree_token = mod.group(2)
        degree = int(degree_token) if degree_token else 4
        if operation == "sus":
            if degree not in {2, 4}:
                raise ValueError("A suspended chord modifier must be sus2 or sus4.")
            formula = [item for item in formula if item[0] != 3]
            formula.append((degree, 0))
            alterations.append(f"sus{degree}")
            cursor += len(mod.group(0))
            continue
        if degree not in {3, 5, 9, 11, 13}:
            raise ValueError(f"Unsupported chord degree {degree} in {symbol!r}.")
        cursor += len(mod.group(0))
        if operation in {"omit", "no"}:
            formula = [item for item in formula if item[0] != degree]
            omitted_tones.add(degree)
            continue
        if operation == "add":
            if degree not in {9, 11, 13}:
                raise ValueError("Only 9, 11, and 13 may be added with the add modifier.")
            if not any(item[0] == degree for item in formula):
                formula.append((degree, 0))
            added_tones.add(degree)
            continue
        alteration = -1 if operation == "b" else 1
        formula = [item for item in formula if item[0] != degree]
        formula.append((degree, alteration))
        alteration_label = f"{operation}{degree}"
        if alteration_label not in alterations:
            alterations.append(alteration_label)
    if not formula:
        raise ValueError("A chord must contain at least one chord tone.")
    formula.sort(key=lambda item: item[0])
    return Chord(
        root=root,
        quality=quality,
        tones=_chord_tones(tuple(formula)),
        bass=bass,
        extensions=tuple(sorted(extensions)),
        alterations=tuple(alterations),
        added_tones=tuple(sorted(added_tones)),
        omitted_tones=tuple(sorted(omitted_tones)),
    )


def _base_prefix_length(suffix: str, quality: str) -> int:
    if not suffix:
        return 0
    prefixes = {
        "minor_major7": ("m(maj7)", "minor-major7", "mmaj7"),
        "half_diminished7": ("m7b5", "ø7", "halfdim7"),
        "diminished7": ("dim7", "o7", "°7"),
        "augmented_major7": ("augmaj7", "augm7"),
        "dominant7_flat5": ("7b5",),
        "dominant7_sharp5": ("7#5",),
        "dominant7_flat9": ("7b9",),
        "dominant7_sharp9": ("7#9",),
        "dominant7_sharp11": ("7#11",),
        "dominant7_flat13": ("7b13",),
        "major13": ("major13", "maj13", "M13"),
        "major11": ("major11", "maj11", "M11"),
        "major9": ("major9", "maj9", "M9"),
        "major7": ("major7", "maj7", "M7"),
        "minor13": ("minor13", "min13", "m13"),
        "minor": ("minor", "min", "m"),
        "minor11": ("minor11", "min11", "m11"),
        "minor9": ("minor9", "min9", "m9"),
        "minor7": ("minor7", "min7", "m7"),
        "minor6": ("minor6", "min6", "m6"),
        "dominant13": ("13",),
        "dominant11": ("11",),
        "dominant9": ("9",),
        "dominant7": ("7",),
        "major6": ("6",),
        "six_nine": ("6/9",),
        "add13": ("add13",),
        "add11": ("add11",),
        "add9": ("add9",),
        "diminished": ("diminished", "dim", "o", "°"),
        "augmented": ("augmented", "aug", "+"),
        "sus2": ("sus2",),
        "sus4": ("sus4", "sus"),
        "power": ("power", "5"),
        "major": ("major", "maj", "M"),
    }
    for prefix in prefixes.get(quality, ()):
        if suffix.lower().startswith(prefix.lower()):
            return len(prefix)
    return 0


def spell_chord(root_pitch_class: int, quality: str) -> str:
    if not 0 <= root_pitch_class < 12:
        raise ValueError("Pitch class must be between 0 and 11.")
    suffix = QUALITY_SUFFIXES.get(quality)
    if suffix is None:
        raise ValueError(f"Unsupported chord quality: {quality}")
    return f"{NOTE_NAMES[root_pitch_class]}{suffix}"


def construct_chord(root: str, quality: str) -> Chord:
    canonical = _QUALITY_ALIASES.get(quality, quality)
    if canonical not in _FORMULAS:
        raise ValueError(f"Unsupported chord quality: {quality}")
    return Chord(parse_note(root), canonical, _chord_tones(_FORMULAS[canonical]))


def chord_notes(chord: Chord) -> tuple[str, ...]:
    return tuple(
        _spell_degree(chord.root, tone.degree, tone.semitones).name for tone in chord.tones
    )


def chord_inversion(chord: Chord, bass: str | None = None) -> dict[str, Any]:
    selected_bass = parse_note(bass) if bass is not None else chord.bass or chord.root
    bass_interval = (selected_bass.pitch_class - chord.root.pitch_class) % 12
    tone = next(
        (item for item in chord.tones if item.semitones % 12 == bass_interval),
        None,
    )
    inversion_names = {
        1: "root_position",
        3: "first_inversion",
        5: "second_inversion",
        7: "third_inversion",
    }
    inversion = (
        "root_position"
        if bass_interval == 0
        else (inversion_names.get(tone.degree if tone is not None else -1, "non_chord_bass"))
    )
    return {
        "symbol": chord.symbol
        if bass is None
        else (
            f"{chord.root.name}"
            f"{chord.symbol[len(chord.root.name) :].split('/')[0]}/{selected_bass.name}"
        ),
        "root": chord.root.name,
        "bass": selected_bass.name,
        "inversion": inversion,
        "bass_is_chord_tone": tone is not None,
        "chord_tones": list(chord_notes(chord)),
        "pitch_classes": list(chord.pitch_classes),
        "voicing_inferred": False,
    }


def _major_tonic_for_minor(tonic: Note) -> Note:
    return _spell_pitch_class(tonic.pitch_class + 3, prefer_flats=tonic.accidental < 0)


def parse_key(value: str) -> tuple[Note, str]:
    match = re.fullmatch(
        r"\s*([A-Ga-g](?:#{1,2}|b{1,2})?)\s*(major|minor|natural_minor|harmonic_minor|melodic_minor|maj|min|m)?\s*",
        value,
        re.I,
    )
    if match is None:
        raise ValueError(f"Invalid key: {value!r}. Use a tonic followed by major or minor.")
    tonic = parse_note(match.group(1))
    mode_token = (match.group(2) or "major").lower()
    mode = (
        "minor"
        if mode_token in {"minor", "natural_minor", "harmonic_minor", "melodic_minor", "min", "m"}
        else "major"
    )
    return tonic, mode


def _key_scale_name(key: str) -> str:
    match = re.fullmatch(
        r"\s*[A-Ga-g](?:#{1,2}|b{1,2})?\s*"
        r"(major|minor|natural_minor|harmonic_minor|melodic_minor|maj|min|m)?\s*",
        key,
        re.I,
    )
    if match is None or match.group(1) is None:
        return "major"
    mode = match.group(1).lower()
    if mode in {"harmonic_minor", "melodic_minor"}:
        return mode
    return "major" if mode in {"major", "maj"} else "minor"


def _minor_tonic_for_major(tonic: Note) -> Note:
    return _spell_pitch_class(tonic.pitch_class - 3, prefer_flats=tonic.accidental < 0)


def key_signature(key: str) -> dict[str, Any]:
    tonic, mode = parse_key(key)
    scale_mode = _key_scale_name(key)
    mode_name = scale_mode if scale_mode in {"harmonic_minor", "melodic_minor"} else mode
    major_tonic = tonic if mode == "major" else _major_tonic_for_minor(tonic)
    signature = _KEY_SIGNATURES.get(major_tonic.name)
    if signature is None:
        raise ValueError(f"No conventional key signature is defined for {key!r}.")
    relative = _minor_tonic_for_major(major_tonic) if mode == "major" else major_tonic
    parallel = _spell_pitch_class(tonic.pitch_class, prefer_flats=tonic.accidental < 0)
    scale = (
        "major"
        if mode == "major"
        else mode_name
        if mode_name in {"harmonic_minor", "melodic_minor"}
        else "natural_minor"
    )
    return {
        "tonic": tonic.name,
        "mode": mode_name,
        "key_signature": list(signature),
        "signature_count": len(signature) if signature and "#" in signature[0] else -len(signature),
        "relative_key": f"{relative.name} {'minor' if mode == 'major' else 'major'}",
        "parallel_key": f"{parallel.name} {'minor' if mode == 'major' else 'major'}",
        "scale_notes": [note.name for note in scale_notes(tonic, scale)],
        "circle_of_fifths_position": _circle_position(major_tonic.name),
    }


def _circle_position(major_tonic: str) -> int | None:
    sharp_order = ("C", "G", "D", "A", "E", "B", "F#", "C#")
    flat_order = ("C", "F", "Bb", "Eb", "Ab", "Db", "Gb", "Cb")
    if major_tonic in sharp_order:
        return sharp_order.index(major_tonic)
    if major_tonic in flat_order:
        return -flat_order.index(major_tonic)
    return None


def diatonic_degree(root: str, tonic: str, mode: str) -> str | None:
    scale = "major" if mode == "major" else "natural_minor" if mode == "minor" else None
    if scale is None:
        raise ValueError("Mode must be 'major' or 'minor'.")
    degrees = _MAJOR_ROMANS if mode == "major" else _MINOR_ROMANS
    root_pc = pitch_class(root)
    notes = scale_notes(tonic, scale)
    for index, note in enumerate(notes):
        if note.pitch_class == root_pc:
            return degrees[index]
    return None


def _quality_from_pitch_classes(notes: tuple[Note, ...]) -> str:
    if len(notes) < 3:
        return "power" if len(notes) == 2 else "unknown"
    root = notes[0].pitch_class
    relative = {(note.pitch_class - root) % 12 for note in notes}
    triads = {
        frozenset({0, 4, 7}): "major",
        frozenset({0, 3, 7}): "minor",
        frozenset({0, 3, 6}): "diminished",
        frozenset({0, 4, 8}): "augmented",
    }
    return triads.get(frozenset(relative), "unknown")


def diatonic_harmony(key: str, *, sevenths: bool = False) -> list[dict[str, Any]]:
    tonic, mode = parse_key(key)
    scale_name = "major" if mode == "major" else "natural_minor"
    notes = scale_notes(tonic, scale_name)
    romans = _MAJOR_ROMANS if mode == "major" else _MINOR_ROMANS
    output: list[dict[str, Any]] = []
    for index, root in enumerate(notes):
        count = 4 if sevenths else 3
        chord_notes_in_scale = tuple(
            notes[(index + 2 * offset) % len(notes)] for offset in range(count)
        )
        if sevenths:
            semitones = tuple(
                (item.pitch_class - root.pitch_class) % 12 for item in chord_notes_in_scale
            )
            quality_by_intervals: dict[tuple[int, ...], str] = {
                (0, 4, 7, 11): "major7",
                (0, 4, 7, 10): "dominant7",
                (0, 3, 7, 10): "minor7",
                (0, 3, 6, 10): "half_diminished7",
                (0, 3, 6, 9): "diminished7",
                (0, 3, 7, 11): "minor_major7",
            }
            quality = quality_by_intervals.get(semitones, "unknown")
        else:
            quality = _quality_from_pitch_classes(chord_notes_in_scale)
        output.append(
            {
                "degree": index + 1,
                "roman_numeral": romans[index],
                "root": root.name,
                "quality": quality,
                "symbol": f"{root.name}{QUALITY_SUFFIXES.get(quality, '')}",
                "chord_tones": [note.name for note in chord_notes_in_scale],
                "function": harmonic_function(index + 1, mode),
            }
        )
    return output


def harmonic_function(degree: int, mode: str = "major") -> str | None:
    if mode == "major":
        return {
            1: "tonic",
            2: "predominant",
            3: "tonic",
            4: "predominant",
            5: "dominant",
            6: "tonic",
            7: "dominant",
        }.get(degree)
    if mode == "minor":
        return {
            1: "tonic",
            2: "predominant",
            3: "tonic",
            4: "predominant",
            5: "dominant",
            6: "predominant",
            7: "dominant",
        }.get(degree)
    return None


def harmonize_scale_degree(key: str, degree: int, *, sevenths: bool = False) -> dict[str, Any]:
    if not 1 <= degree <= 7:
        raise ValueError("Scale degree must be between 1 and 7.")
    return diatonic_harmony(key, sevenths=sevenths)[degree - 1]


def harmonize_melody_note(key: str, note: str, *, sevenths: bool = False) -> dict[str, Any]:
    target = parse_note(note)
    candidates = []
    for chord in diatonic_harmony(key, sevenths=sevenths):
        chord_tones = [parse_note(item) for item in chord["chord_tones"]]
        if any(tone.pitch_class == target.pitch_class for tone in chord_tones):
            candidates.append(chord)
    return {
        "note": target.name,
        "key": key,
        "status": "possible_harmonizations" if candidates else "no_diatonic_harmonization",
        "unique": len(candidates) == 1,
        "candidates": candidates,
    }


def _is_dominant_quality(quality: str) -> bool:
    return quality in {
        "dominant7",
        "dominant9",
        "dominant11",
        "dominant13",
        "dominant7_flat5",
        "dominant7_sharp5",
        "dominant7_flat9",
        "dominant7_sharp9",
        "dominant7_sharp11",
        "dominant7_flat13",
    }


def _quality_matches_diatonic(quality: str, reference: str, degree: int, mode: str) -> bool:
    accepted = {
        "major": {"major", "major6", "major7", "major9", "major11", "major13"},
        "minor": {"minor", "minor6", "minor7", "minor9", "minor11", "minor13"},
        "diminished": {"diminished", "diminished7", "half_diminished7"},
        "augmented": {"augmented", "augmented_major7"},
    }
    if quality in accepted.get(reference, set()):
        return True
    return degree == 5 and quality in (
        {"major", *_dominant_qualities()} if mode == "minor" else _dominant_qualities()
    )


def _dominant_qualities() -> set[str]:
    return {
        "dominant7",
        "dominant9",
        "dominant11",
        "dominant13",
        "dominant7_flat5",
        "dominant7_sharp5",
        "dominant7_flat9",
        "dominant7_sharp9",
        "dominant7_sharp11",
        "dominant7_flat13",
    }


def _inversion_figure(chord: Chord) -> str:
    bass = chord.bass or chord.root
    if bass.pitch_class == chord.root.pitch_class:
        degree = 1
    else:
        tone = next(
            (
                item
                for item in chord.tones
                if item.semitones % 12 == (bass.pitch_class - chord.root.pitch_class) % 12
            ),
            None,
        )
        if tone is None:
            return ""
        degree = tone.degree
    has_seventh = any(tone.degree == 7 for tone in chord.tones)
    if has_seventh:
        return {1: "7", 3: "65", 5: "43", 7: "42"}.get(degree, "")
    return {1: "", 3: "6", 5: "64"}.get(degree, "")


def _roman_for_letter(root: Note, tonic: Note, mode: str) -> tuple[str, int, str]:
    degree_index = (_LETTERS.index(root.letter) - _LETTERS.index(tonic.letter)) % 7
    roman = (_MAJOR_ROMANS if mode == "major" else _MINOR_ROMANS)[degree_index]
    scale = scale_notes(tonic, "major" if mode == "major" else "natural_minor")
    difference = (root.pitch_class - scale[degree_index].pitch_class) % 12
    accidental = "#" if difference == 1 else "b" if difference == 11 else ""
    if difference not in {0, 1, 11}:
        accidental = f"{difference:+d}"
    return f"{accidental}{roman}", degree_index, accidental


def _roman_for_chord(chord: Chord, tonic: Note, mode: str) -> dict[str, Any]:
    scale_name = "major" if mode == "major" else "natural_minor"
    scale = scale_notes(tonic, scale_name)
    index = next(
        (i for i, note in enumerate(scale) if note.pitch_class == chord.root.pitch_class), None
    )
    if index is None:
        parallel_mode = "minor" if mode == "major" else "major"
        parallel_scale = scale_notes(
            tonic, "natural_minor" if parallel_mode == "minor" else "major"
        )
        parallel_index = next(
            (
                i
                for i, note in enumerate(parallel_scale)
                if note.pitch_class == chord.root.pitch_class
            ),
            None,
        )
        if parallel_index is not None:
            root_roman, _, _ = _roman_for_letter(chord.root, tonic, mode)
            parallel_harmony = diatonic_harmony(f"{tonic.name} {parallel_mode}")[parallel_index]
            if _quality_matches_diatonic(
                chord.quality, parallel_harmony["quality"], parallel_index + 1, parallel_mode
            ):
                return {
                    "roman_numeral": f"borrowed {root_roman}{_inversion_figure(chord)}",
                    "inversion_figure": _inversion_figure(chord),
                    "classification": "borrowed",
                    "function": harmonic_function(parallel_index + 1, parallel_mode),
                    "explanation": (
                        f"Chord root and quality fit the parallel {parallel_mode} harmony."
                    ),
                }
            return {
                "roman_numeral": f"chromatic {root_roman}{_inversion_figure(chord)}",
                "inversion_figure": _inversion_figure(chord),
                "classification": "chromatic",
                "function": None,
                "explanation": (
                    "Root is borrowed, but chord quality does not match the parallel scale."
                ),
            }
        root_roman, _, _ = _roman_for_letter(chord.root, tonic, mode)
        return {
            "roman_numeral": f"chromatic {root_roman}{_inversion_figure(chord)}",
            "inversion_figure": _inversion_figure(chord),
            "classification": "chromatic",
            "function": None,
            "explanation": "The chord root is outside both the key and its parallel mode.",
        }
    roman = (_MAJOR_ROMANS if mode == "major" else _MINOR_ROMANS)[index]
    if mode == "minor" and index == 4 and _is_dominant_quality(chord.quality):
        roman = "V"
    diatonic = diatonic_harmony(f"{tonic.name} {mode}")[index]
    inversion = _inversion_figure(chord)
    if _quality_matches_diatonic(chord.quality, diatonic["quality"], index + 1, mode):
        return {
            "roman_numeral": f"{roman}{inversion}",
            "inversion_figure": inversion,
            "classification": "diatonic",
            "function": harmonic_function(index + 1, mode),
            "explanation": "Root and chord quality agree with the diatonic harmony.",
        }
    parallel_mode = "minor" if mode == "major" else "major"
    parallel_scale = scale_notes(tonic, "natural_minor" if parallel_mode == "minor" else "major")
    parallel_index = next(
        (i for i, note in enumerate(parallel_scale) if note.pitch_class == chord.root.pitch_class),
        None,
    )
    if parallel_index is not None:
        parallel_roman = (_MINOR_ROMANS if parallel_mode == "minor" else _MAJOR_ROMANS)[
            parallel_index
        ]
        parallel_harmony = diatonic_harmony(f"{tonic.name} {parallel_mode}")[parallel_index]
        if _quality_matches_diatonic(
            chord.quality, parallel_harmony["quality"], parallel_index + 1, parallel_mode
        ):
            return {
                "roman_numeral": f"borrowed {parallel_roman}{inversion}",
                "inversion_figure": inversion,
                "classification": "borrowed",
                "function": harmonic_function(parallel_index + 1, parallel_mode),
                "explanation": f"Chord quality fits the parallel {parallel_mode} harmony.",
            }
    root_roman, _, _ = _roman_for_letter(chord.root, tonic, mode)
    return {
        "roman_numeral": f"chromatic {root_roman}{inversion}",
        "inversion_figure": inversion,
        "classification": "chromatic",
        "function": None,
        "explanation": (
            "The root is diatonic, but chord quality differs from diatonic triad/seventh."
        ),
    }


def analyze_progression(
    symbols: list[str],
    key: str,
    *,
    timestamps: list[dict[str, float]] | None = None,
    soprano_notes: list[str] | None = None,
) -> dict[str, Any]:
    if not symbols:
        raise ValueError("At least one chord is required.")
    if soprano_notes is not None and len(soprano_notes) != len(symbols):
        raise ValueError("Provide one soprano pitch for each chord when soprano_notes is supplied.")
    tonic, mode = parse_key(key)
    chords = [parse_chord(symbol) for symbol in symbols]
    interpretations = [_roman_for_chord(chord, tonic, mode) for chord in chords]
    for index in range(len(chords) - 1):
        current = chords[index]
        following = chords[index + 1]
        if _is_dominant_quality(current.quality):
            if (current.root.pitch_class - following.root.pitch_class) % 12 == 7:
                target = _roman_for_chord(following, tonic, mode)["roman_numeral"]
                if target is not None:
                    interpretations[index] = {
                        "roman_numeral": f"V{_inversion_figure(current)}/{_base_roman(target)}",
                        "inversion_figure": _inversion_figure(current),
                        "classification": "secondary_dominant",
                        "function": "dominant",
                        "explanation": (
                            "Dominant-quality chord resolves by fifth to the following chord root."
                        ),
                    }
    root_movements = [
        {
            "from": chords[index].root.name,
            "to": chords[index + 1].root.name,
            "ascending_semitones": (
                chords[index + 1].root.pitch_class - chords[index].root.pitch_class
            )
            % 12,
            "common_tones": sorted(
                set(chords[index].pitch_classes) & set(chords[index + 1].pitch_classes)
            ),
        }
        for index in range(len(chords) - 1)
    ]
    numerals = [item["roman_numeral"] for item in interpretations]
    functions = [item["function"] for item in interpretations]
    return {
        "key": {"tonic": tonic.name, "mode": mode},
        "chords": [
            {"symbol": chord.symbol, **interpretation}
            for chord, interpretation in zip(chords, interpretations, strict=True)
        ],
        "root_movements": root_movements,
        "diatonic_chords": [item["classification"] == "diatonic" for item in interpretations],
        "harmonic_functions": functions,
        "cadences": detect_cadences(chords, interpretations, mode, soprano_notes=soprano_notes),
        "patterns": _progression_patterns(numerals, mode),
        "repeated_patterns": _repeated_patterns(chords),
        "harmonic_rhythm": _harmonic_rhythm(timestamps),
        "confidence": None,
        "confidence_methodology": (
            "No calibrated probability model is used; "
            "interpretations are deterministic rule matches."
        ),
    }


def _progression_patterns(numerals: list[str | None], mode: str) -> list[dict[str, Any]]:
    normalized = [_base_roman(numeral) for numeral in numerals]
    patterns = [
        ("I–V–vi–IV", ["I", "V", "vi", "IV"]),
        ("vi–IV–I–V", ["vi", "IV", "I", "V"]),
        ("ii–V–I", ["ii", "V", "I"]),
        ("I–IV–V", ["I", "IV", "V"]),
    ]
    matches = [
        {"name": name, "start_index": start}
        for name, pattern in patterns
        for start in range(len(numerals) - len(pattern) + 1)
        if normalized[start : start + len(pattern)] == pattern
    ]
    if len(numerals) == 12:
        blues_major = ["I", "I", "I", "I", "IV", "IV", "I", "I", "V", "IV", "I", "I"]
        blues_minor = ["i", "i", "i", "i", "iv", "iv", "i", "i", "v", "iv", "i", "i"]
        if normalized == (blues_minor if mode == "minor" else blues_major):
            matches.append({"name": "12-bar blues", "start_index": 0})
    return matches


def _base_roman(numeral: str | None) -> str | None:
    if numeral is None or numeral.startswith(("borrowed ", "chromatic ")):
        return None
    if "/" in numeral:
        numerator, denominator = numeral.split("/", 1)
        numerator = re.sub(r"(?:64|65|43|42|7|6)$", "", numerator)
        denominator = re.sub(r"(?:64|65|43|42|7|6)$", "", denominator)
        return f"{numerator}/{denominator}"
    return re.sub(r"(?:64|65|43|42|7|6)$", "", numeral)


def _repeated_patterns(chords: list[Chord]) -> list[dict[str, Any]]:
    signatures = [
        (
            chord.root.pitch_class,
            chord.quality,
            chord.pitch_classes,
            chord.bass.pitch_class if chord.bass else None,
        )
        for chord in chords
    ]
    results: list[dict[str, Any]] = []
    seen: set[tuple[int, int, tuple[tuple[int, str, tuple[int, ...], int | None], ...]]] = set()
    for length in range(1, min(4, len(chords) // 2) + 1):
        for start in range(len(chords) - length * 2 + 1):
            pattern = tuple(signatures[start : start + length])
            for repeated_at in range(start + length, len(chords) - length + 1):
                if tuple(signatures[repeated_at : repeated_at + length]) != pattern:
                    continue
                identity = length, start, pattern
                if identity not in seen:
                    results.append(
                        {
                            "pattern_symbols": [
                                chord.symbol for chord in chords[start : start + length]
                            ],
                            "positions": [start, repeated_at],
                        }
                    )
                    seen.add(identity)
    return results


def detect_cadences(
    chords: list[Chord],
    interpretations: list[dict[str, Any]] | None = None,
    mode: str = "major",
    *,
    soprano_notes: list[str] | None = None,
) -> list[dict[str, Any]]:
    if len(chords) < 2:
        return []
    first, final = chords[-2:]
    first_roman = (
        None if interpretations is None else _base_roman(interpretations[-2]["roman_numeral"])
    )
    final_roman = (
        None if interpretations is None else _base_roman(interpretations[-1]["roman_numeral"])
    )
    result: list[dict[str, Any]] = []
    if first_roman in {"V", "v"} and final_roman in {"I", "i"}:
        root_position = first.bass is None or first.bass.pitch_class == first.root.pitch_class
        tonic_root = final.bass is None or final.bass.pitch_class == final.root.pitch_class
        soprano_ends_on_tonic = (
            None
            if soprano_notes is None
            else parse_note(soprano_notes[-1]).pitch_class == final.root.pitch_class
        )
        perfect = (
            mode == "major"
            and first.quality in {"major", "dominant7"}
            and final.quality == "major"
            and root_position
            and tonic_root
            and soprano_ends_on_tonic is True
        )
        cadence_type = (
            "perfect_authentic"
            if perfect
            else "imperfect_authentic"
            if soprano_ends_on_tonic is False
            else "authentic_candidate"
        )
        result.append(
            {
                "type": cadence_type,
                "start_index": len(chords) - 2,
                "conditions": {
                    "dominant_to_tonic": True,
                    "root_position_dominant": root_position,
                    "root_position_tonic": tonic_root,
                    "major_dominant_and_tonic": first.quality in {"major", "dominant7"}
                    and final.quality == "major",
                    "soprano_ends_on_tonic": soprano_ends_on_tonic,
                },
                "possible_types": (
                    ["perfect_authentic", "imperfect_authentic"]
                    if soprano_ends_on_tonic is None
                    else [cadence_type]
                ),
            }
        )
    elif first_roman in {"IV", "iv"} and final_roman in {"I", "i"}:
        result.append(
            {
                "type": "plagal",
                "start_index": len(chords) - 2,
                "conditions": {"subdominant_to_tonic": True},
            }
        )
    elif first_roman == "V" and final_roman in ({"vi"} if mode == "major" else {"VI"}):
        result.append(
            {
                "type": "deceptive",
                "start_index": len(chords) - 2,
                "conditions": {"dominant_to_submediant": True},
            }
        )
    elif final_roman in {"V", "v"}:
        result.append(
            {
                "type": "half",
                "start_index": len(chords) - 1,
                "conditions": {"ends_on_dominant": True},
            }
        )
    return result


def _harmonic_rhythm(timestamps: list[dict[str, float]] | None) -> dict[str, Any] | None:
    if timestamps is None:
        return None
    if not timestamps:
        return {"event_count": 0, "mean_duration_seconds": None, "changes_per_minute": None}
    durations = []
    for item in timestamps:
        start = item.get("start_seconds")
        end = item.get("end_seconds")
        if start is None or end is None or start < 0 or end <= start:
            raise ValueError(
                "Each timestamp must have non-negative start_seconds and a later end_seconds."
            )
        durations.append(end - start)
    total = max(item["end_seconds"] for item in timestamps) - min(
        item["start_seconds"] for item in timestamps
    )
    return {
        "event_count": len(timestamps),
        "mean_duration_seconds": sum(durations) / len(durations),
        "changes_per_minute": None if total <= 0 else max(0, len(timestamps) - 1) * 60 / total,
    }


def transpose_chord(
    chord: Chord,
    semitones: int,
    *,
    prefer_flats: bool = False,
    key: str | None = None,
) -> Chord:
    root_name = transpose_note(chord.root.name, semitones, prefer_flats=prefer_flats, key=key)
    bass = None
    if chord.bass is not None:
        bass_name = transpose_note(chord.bass.name, semitones, prefer_flats=prefer_flats, key=key)
        bass = parse_note(bass_name)
    return Chord(
        root=parse_note(root_name),
        quality=chord.quality,
        tones=chord.tones,
        bass=bass,
        extensions=chord.extensions,
        alterations=chord.alterations,
        added_tones=chord.added_tones,
        omitted_tones=chord.omitted_tones,
    )


def transpose_progression(
    symbols: list[str],
    semitones: int,
    *,
    prefer_flats: bool = False,
    key: str | None = None,
) -> dict[str, Any]:
    if not symbols:
        raise ValueError("At least one chord is required.")
    transposed = [
        transpose_chord(parse_chord(symbol), semitones, prefer_flats=prefer_flats, key=key)
        for symbol in symbols
    ]
    return {
        "semitones": semitones,
        "target_key": key,
        "source_symbols": symbols,
        "symbols": [chord.symbol for chord in transposed],
        "chords": [chord.as_dict() for chord in transposed],
        "preserved_structure": True,
        "direction": "positive semitones raise sounding pitch; negative semitones lower it",
    }


def transpose_key(key: str, semitones: int, *, prefer_flats: bool = False) -> str:
    tonic, mode = parse_key(key)
    shifted = transpose_note(tonic.name, semitones, prefer_flats=prefer_flats)
    scale_mode = _key_scale_name(key)
    output_mode = scale_mode if scale_mode in {"harmonic_minor", "melodic_minor"} else mode
    return f"{shifted} {output_mode}"


def transpose_between_keys(
    symbols: list[str],
    source_key: str,
    target_key: str,
    *,
    prefer_flats: bool = False,
) -> dict[str, Any]:
    source_tonic, _ = parse_key(source_key)
    target_tonic, _ = parse_key(target_key)
    semitones = (target_tonic.pitch_class - source_tonic.pitch_class) % 12
    if semitones > 6:
        semitones -= 12
    result = transpose_progression(
        symbols,
        semitones,
        prefer_flats=prefer_flats,
        key=target_key,
    )
    return {
        **result,
        "source_key": source_key,
        "target_key": target_key,
    }


def compare_progressions(
    first_symbols: list[str],
    second_symbols: list[str],
    *,
    key: str | None = None,
) -> dict[str, Any]:
    if not first_symbols or not second_symbols:
        raise ValueError("Both progressions must contain at least one chord.")
    first = [parse_chord(symbol) for symbol in first_symbols]
    second = [parse_chord(symbol) for symbol in second_symbols]
    same_length = len(first) == len(second)
    same_harmonic_structure = same_length and all(
        a.quality == b.quality
        and tuple((tone.degree, tone.alteration) for tone in a.tones)
        == tuple((tone.degree, tone.alteration) for tone in b.tones)
        and (None if a.bass is None else (a.bass.pitch_class - a.root.pitch_class) % 12)
        == (None if b.bass is None else (b.bass.pitch_class - b.root.pitch_class) % 12)
        for a, b in zip(first, second, strict=True)
    )
    shifts = (
        [(b.root.pitch_class - a.root.pitch_class) % 12 for a, b in zip(first, second, strict=True)]
        if same_length
        else []
    )
    same_transposition = same_harmonic_structure and len(set(shifts)) == 1
    result: dict[str, Any] = {
        "same_length": same_length,
        "same_symbols": same_length
        and [chord.symbol for chord in first] == [chord.symbol for chord in second],
        "same_harmonic_structure": same_harmonic_structure,
        "equivalent_under_transposition": same_transposition,
        "semitone_shift": shifts[0] if same_transposition else None,
    }
    if key is not None:
        result["first_analysis"] = analyze_progression(first_symbols, key)
        result["second_analysis"] = analyze_progression(second_symbols, key)
    return result


def capo_calculation(
    chord_symbols: list[str],
    capo_fret: int,
    *,
    direction: str = "written_to_sounding",
    prefer_flats: bool = False,
    key: str | None = None,
) -> dict[str, Any]:
    if not 0 <= capo_fret <= 24:
        raise ValueError("Capo fret must be between 0 and 24.")
    if direction not in {"written_to_sounding", "sounding_to_written"}:
        raise ValueError("Direction must be 'written_to_sounding' or 'sounding_to_written'.")
    delta = capo_fret if direction == "written_to_sounding" else -capo_fret
    written_key = None
    sounding_key = None
    if key is not None:
        if direction == "written_to_sounding":
            written_key = key
            sounding_key = transpose_key(key, capo_fret, prefer_flats=prefer_flats)
        else:
            sounding_key = key
            written_key = transpose_key(key, -capo_fret, prefer_flats=prefer_flats)
    target_key = sounding_key if direction == "written_to_sounding" else written_key
    result = transpose_progression(
        chord_symbols,
        delta,
        prefer_flats=prefer_flats,
        key=target_key,
    )
    return {
        "capo_fret": capo_fret,
        "direction": direction,
        "written_chords": chord_symbols
        if direction == "written_to_sounding"
        else result["symbols"],
        "sounding_chords": result["symbols"]
        if direction == "written_to_sounding"
        else chord_symbols,
        "written_key": written_key,
        "sounding_key": sounding_key,
        "semitone_delta": delta,
        "convention": (
            "A capo at fret n raises sounding pitch n semitones above "
            "the written open-position chord."
        ),
    }
