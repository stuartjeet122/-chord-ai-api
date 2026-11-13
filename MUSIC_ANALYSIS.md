# Phase 7 music analysis

> Historical phase report. Analysis persistence and background processing are not part of the current stateless application.

## Status

This phase adds an asynchronous, versioned analysis path using the existing audio validation, storage, job, and DSP infrastructure. The locally available analysis is a **DSP candidate generator**, not a trained or accuracy-certified chord recognizer. Phase 6 did not validate or enable any model; no model output is produced or implied here. The project environment also has no chord or note model runtime installed.

The results expose uncertainty rather than inventing certainty: event `confidence` is `null`, estimates retain evidence and candidates, unclear harmonic frames are represented as `N`/unknown, and beat meter/downbeats are `null` when no validated estimator is available. Model-dependent melody transcription and structural section labeling explicitly report unavailable. The completed-scope condition in the Phase 7 brief is therefore **not met**: no annotated real-recording benchmark or model inference has run.

## Architecture and execution

1. The API validates caller ownership and audio availability, then creates the ordinary processing job and a new immutable analysis version in one transaction.
2. The audio worker downloads and integrity-checks the source, probes it, and performs FFmpeg full-stream decode validation using the existing ingestion path.
3. The analysis service streams bounded PCM blocks, computes harmonic-partial salience and spectral-flux features, and derives chord candidates, key profiles, local tempo estimates, beat-grid candidates, and low-register pitch candidates.
4. Analysis events and result evidence are written transactionally. A retry reuses the job's analysis record; a new request creates a new version instead of replacing previous results.
5. API clients use the existing job status/cancel endpoints and poll the version/result endpoints. No WebSocket event architecture existed in the approved Phase 1–6 work, so this phase does not invent an unaudited event channel.

AI inference stays behind the Phase 6 provider boundary. Since no provider factory is configured, the analysis worker does not call an untested external model or infer a model prediction from DSP features.

## DSP chord candidates

- Audio is read in blocks; channel spectra are combined by magnitude so opposite-polarity stereo channels do not cancel during mono downmix.
- Each 460 ms Hann-windowed frame advances by approximately 100 ms. Spectral magnitude at note fundamentals and harmonics contributes to a MIDI salience track.
- Pitch-class evidence is scored against explicit interval templates. Candidate records retain template score, runner-up margin, and the top three alternatives.
- A low score or narrow top-candidate margin yields an ambiguous event; low-energy frames yield `N`/unknown. These thresholds are heuristic, not probability thresholds.
- Adjacent identical decisions are merged on the source timeline. `start_seconds`/`end_seconds` are clipped to the source duration; event IDs are persisted separately. The frame/window resolution limits boundary precision and has not been evaluated against human annotations.
- Bass candidates use the strongest low-register harmonic-salience pitch. This is not a polyphonic bass transcription model and can be wrong when fundamentals are weak or the arrangement is dense.
- Roman numerals are only attached when the profile margin clears a conservative heuristic and the chord root/quality matches the selected key's simple diatonic vocabulary. The key margin and numeral are not calibrated.

The code contains templates for major, minor, diminished, augmented, sus2/sus4, power, 6/7/9/11/13 families, add tones, half-diminished, minor-major, and several altered dominant forms. A template being present does **not** mean the chord is reliably identifiable in a full mix. DSP analysis output still uses sharp pitch-class spellings. The separate Phase 8 theory API supports key-aware chord spelling and bass-relative chord inversion classification, but it does not infer a complete instrument voicing or alter the DSP analysis record.

## Key and rhythm estimates

- Global and 20-second local key candidates use correlation between pitch-class salience and major/minor Krumhansl-Schmuckler profiles. Output includes alternatives and local candidate changes. These are uncalibrated profile scores, not confidence probabilities. Short or silent audio may yield no key.
- Tempo candidates use autocorrelation of positive log-spectrum flux over a 50–200 BPM range. Half/double-time alternatives are returned when in range.
- A local tempo map is estimated over 20-second windows and used to construct candidate beat times. These are estimates, not validated beat/downbeat annotations. Meter and downbeats remain null; an average BPM must not be used as a claim of constant tempo.

No separate CQT, HPSS, learned sequence model, source separation, or melody model has been introduced. The implemented DSP methods require comparison on legally accessible annotated recordings before selection or tuning.

## Persistence

- `music_analyses` stores owner, source asset/checksum, job, monotonically increasing asset version, requested tasks, engine version, model references, execution device, parameters, result, stage errors, and timestamps.
- `music_analysis_events` stores timestamp-indexed chord, beat, key-change, bass, melody, and section event kinds, with query indexes. Core DSP emits chord, beat, key-change, and bass rows.
- Database uniqueness prevents two committed versions with the same owner/asset/version and prevents multiple analysis records for one job. Source asset deletion is restricted while analyses reference it.
- Corrections are not yet exposed as an API; source events include `manually_corrected=false`, and result history is never overwritten by a subsequent analysis.

## API

All routes are under `/api/v1/music`:

| Method and path | Responsibility |
|---|---|
| `POST /{asset_id}/analyze` | Queue requested `chords`, `key`, `tempo`, `beats`, or `full` work; supports the existing `Idempotency-Key`. |
| `GET /{asset_id}/analysis` | Latest completed or partial version. |
| `GET /{asset_id}/analysis/{analysis_id}` | Specific completed version. |
| `GET /{asset_id}/analysis/compare` | Recent completed versions for comparison. |
| `GET /{asset_id}/chords` | Chord events with time range, minimum-confidence, quality, and pagination filters. Current confidence is null, so a minimum-confidence filter returns no matches. |
| `GET /{asset_id}/tempo` | Tempo candidates, local tempo map, beat candidates, and explicit unknown meter/downbeats. |
| `GET /{asset_id}/key` | Key candidate, alternatives, and local key changes. |
| `GET /{asset_id}/bass` | Low-register pitch candidates. |
| `GET /{asset_id}/melody` | Explicitly unavailable until a validated provider exists. |
| `GET /{asset_id}/sections` | Explicitly unavailable until a validated structure model exists. |

Analysis is queued through the shared job system; no duplicate job-status or cancellation endpoints are added. Production authentication remains outside the existing development identity implementation.

## Validation and benchmark status

Synthetic test audio checks basic triad progression ordering, timeline bounds, silence-to-unknown behavior, and enharmonic/theory helpers. End-to-end tests exercise upload, job processing, persistence, results, idempotency, and immutable version comparison. These tests validate plumbing and controlled behavior only.

No real annotated audio dataset was supplied or evaluated in this phase. Accordingly, no chord accuracy, key accuracy, beat accuracy, boundary error, throughput, or memory-performance claims are reported. Do not treat synthetic fixtures or the Phase 6 metric functions as benchmark results.

## Known limitations and next validation gates

- No Phase 6 model was tested or selected for production; chord/model reconciliation, note transcription, learned section analysis, and model-backed confidence remain unavailable.
- Time-signature and downbeat detection are not implemented. Local tempo is a heuristic periodicity estimate.
- Independent bass tracking, melody/accompaniment separation, audio-derived harmonic transitions, and manual correction endpoints remain incomplete. The separate Phase 8 theory API supplies deterministic chord parsing and progression pattern analysis for provided chord symbols; these calculations are not DSP detections.
- The spectral front end is bounded in audio-block memory, but feature tracks are retained proportional to duration; long-recording resource limits and throughput require measurements before production sizing.
- No WebSocket architecture was approved/implemented. Clients poll jobs and results.
- No real-data benchmark, supported Python 3.12+ run, PostgreSQL deployment, Redis/Celery production worker, or GPU deployment was verified in this environment.

Before calling Phase 7 complete, validate a licensed representative train/validation/test set, run the exact selected model in the supported Python 3.12+ worker image, calibrate confidence against held-out annotations, measure long-file performance, and implement or explicitly exclude the remaining requested analysis types.
