# Deterministic music theory engine

## Scope and status

The theory engine adds stateless, deterministic calculations and an optional,
versioned interpretation of Phase 7 chord-analysis results. It does not call an
AI provider, modify an existing analysis, or make claims about audio recognition
accuracy. Theory calculations operate on caller-supplied symbols and pitches;
persisted interpretations are linked to an immutable music-analysis version.

## API

All stateless routes are under `/api/v1/theory`:

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/notes/parse` | Parse a note spelling and return pitch class, optional MIDI number, and frequency. |
| `POST` | `/intervals` | Identify a spelled interval, including quality, direction, and compound interval number. |
| `POST` | `/scales` | Construct a named or custom scale, retaining diatonic spelling. |
| `POST` | `/scales/identify` | Return matching scale candidates; results can be ambiguous. |
| `POST` | `/chords/parse` | Parse and normalize a chord symbol. |
| `POST` | `/chords/construct` | Construct a chord from a root and supported quality. |
| `POST` | `/chords/inversion` | Classify the specified bass relative to chord tones; does not infer a full voicing. |
| `POST` | `/keys` | Return key signature, scale notes, relative/parallel keys, and circle-of-fifths position. |
| `POST` | `/keys/transpose` | Transpose a key by signed semitones. |
| `POST` | `/harmony` | Generate diatonic triads or seventh chords. |
| `POST` | `/harmony/melody` | List diatonic chords containing a supplied melody pitch. |
| `POST` | `/progressions/analyze` | Return Roman numerals, harmonic functions, cadence candidates, patterns, and optional timing analysis. |
| `POST` | `/progressions/compare` | Compare chord symbols and transposition-equivalent structure. |
| `POST` | `/transpose` | Transpose a chord progression by signed semitones. |
| `POST` | `/transpose-between-keys` | Transpose a progression between explicit source and target keys. |
| `POST` | `/capo` | Convert written and sounding chords/keys under an explicit capo direction. |

Persisted interpretations use the existing development owner identity:

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/api/v1/music/{asset_id}/analysis/{analysis_id}/theory` | Create the next immutable theory interpretation version. |
| `GET` | `/api/v1/music/{asset_id}/analysis/{analysis_id}/theory` | Retrieve the latest owner-scoped interpretation. |

Stateless calculation errors use the API's standard error envelope with code
`invalid_theory_input` and HTTP status `422`. Persisted resources owned by
another development identity are returned as `404`. Production authentication
is not implemented by this phase.

## Supported conventions

- Note spellings accept A-G, natural/sharp/flat, and double-sharp/double-flat
  accidentals; an optional signed octave is accepted where applicable.
- Pitch classes use MIDI convention C=0 through B=11. Frequency uses A4=440 Hz
  and equal temperament.
- Intervals use letter spelling as well as semitone distance, so enharmonic
  intervals can have different names. Compound intervals are not reduced to
  their simple form. The interval API reports the directed interval from the
  first pitch to the second.
- Scale patterns are ascending semitone offsets starting at zero. Named scales
  include major/modes, natural/harmonic/melodic minor, pentatonic scales, blues,
  and chromatic. Scale identification reports candidates rather than selecting
  a unique interpretation when multiple patterns fit.
- Chord symbols support triads, suspended and power chords, common sixth,
  seventh, ninth, eleventh, and thirteenth families, common altered dominants,
  added/omitted tones, and slash bass notes. The parser normalizes recognized
  aliases and rejects unrecognized trailing syntax; this is not a complete
  implementation of every publishing convention for chord notation.
- Key-aware operations use the requested target key to spell roots when the
  target scale contains that pitch. Otherwise spelling follows the
  `prefer_flats` option. Where an enharmonic spelling choice is ambiguous,
  output remains deterministic but may not match an editor's house style.
- Capo conversion follows the written-to-sounding convention: a capo at fret
  `n` raises the sounding pitch `n` semitones. `sounding_to_written` reverses
  this mapping.

## Analysis and evidence limits

Roman numerals and harmonic functions are rule-based classifications against
the supplied key. They are explanatory labels, not evidence that the audio is
in that key. Cadence classifications use chord sequence and optional soprano
notes; without sufficient melodic evidence, the result remains a cadence
candidate instead of asserting an authentic cadence. The response deliberately
does not provide a calibrated progression confidence score.

Progression timestamp ranges must correspond one-to-one with chords and be
ordered. Harmonic rhythm is derived from these supplied ranges; it is not
detected from audio. Similarly, melody harmonization enumerates compatible
diatonic chords and does not infer accompaniment or voice leading.

Chord inversion classification only checks whether a supplied bass pitch is
the root or another chord tone. Instrument-specific voicings, octave placement,
guitar/piano fingering, and inversion inference from a recording are outside
scope.

## Persistence and versioning

`music_theory_analyses` stores an owner-scoped record associated with a
`music_analyses` version. `(analysis_id, version)` is unique; each new
interpretation creates another version. The payload records whether source
chords came from the existing detected result or from request-supplied chord
symbols. Source analysis results and events are not updated by theory
interpretation. Deleting a source analysis cascades to its interpretations.

## Validation status

Unit and API tests cover deterministic calculations, parsing, spelling,
transposition, persistence, and owner isolation. These tests are not annotated
audio benchmarks and do not establish chord, key, cadence, or transcription
accuracy. The engine is independent of AI inference and adds no model or
third-party runtime dependency.
