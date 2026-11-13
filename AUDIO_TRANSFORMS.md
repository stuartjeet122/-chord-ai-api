# Audio pitch and tempo transformations

## Engine decision

Phase 9 uses the SoundTouch `SoundStretch` command-line utility behind the
`music_api.audio.transforms` adapter. It processes decoded WAV intermediates,
keeps tempo and pitch as independent controls, accepts fractional semitones,
and does not require the API process to import a native audio library.

The choice is based on verified capabilities and deployment constraints, not a
claim that SoundTouch is the best-sounding option for every recording:

| Candidate | Verified capabilities and constraints | Decision |
|---|---|---|
| SoundTouch / SoundStretch | Upstream documents independent tempo, pitch, and playback-rate controls; fractional tempo/pitch arguments; WAV input/output and standard pipes. It documents mono/stereo, 8–48 kHz support and describes a WSOLA-like time-domain stretcher. The library is LGPL v2.1; the vendor also offers a commercial license. Debian Bookworm provides `soundstretch` 2.3.2. | Selected for a contained CPU worker adapter and LGPL-compatible deployment. The container pins Debian's 2.3.2 package. No third-party Python binding is needed. |
| Rubber Band | The upstream project documents independent pitch/time scaling and a block-based phase-vocoder approach with transient resets; it says the library is GPL v2-or-later and requires a commercial license for proprietary distribution. FFmpeg's Rubber Band filter is available only in FFmpeg builds compiled with the library. | Not selected for the default proprietary deployment because commercial licensing and FFmpeg build capability would need separate approval. It may be evaluated behind the same adapter if licensed. |
| FFmpeg `atempo` | FFmpeg documents a tempo filter. It is not a standalone pitch-shifting algorithm; pitch and tempo must not be conflated. | Existing FFmpeg remains responsible for decode and export, not the selected pitch/time algorithm. |
| librosa | The project is ISC-licensed. Its documented `effects.time_stretch` accepts multi-channel NumPy arrays and uses a phase-vocoder approach. Array-oriented processing requires retaining decoded audio in memory, and phase-vocoder artifacts are a concern for transient-heavy material. | Not used in the long-file worker path; it remains a potential research/benchmark candidate. |

Official references:

- [SoundTouch project and releases](https://codeberg.org/soundtouch/soundtouch)
- [SoundTouch processing, formats, algorithms, and limits](https://soundtouch.surina.net/README.html)
- [SoundStretch CLI and supported controls](https://soundtouch.surina.net/soundstretch.html)
- [SoundTouch license](https://soundtouch.surina.net/license.html)
- [Rubber Band project](https://breakfastquay.com/rubberband/)
- [Rubber Band technical notes](https://breakfastquay.com/rubberband/technical.html)
- [Rubber Band licensing](https://breakfastquay.com/rubberband/license.html)
- [FFmpeg filter documentation](https://ffmpeg.org/ffmpeg-filters.html)
- [librosa time stretching](https://librosa.org/doc/0.11.0/generated/librosa.effects.time_stretch.html)
- [librosa license](https://github.com/librosa/librosa/blob/main/LICENSE.md)

The selected algorithm is not perceptually transparent for every source or
ratio. No listening panel or annotated quality benchmark was run for this
implementation; output metrics are technical validation only.

## API

`POST /api/v1/audio/{asset_id}/transform` returns `202 Accepted` and the normal
job resource. Use `Idempotency-Key` for safe request retries. Poll
`GET /api/v1/jobs/{job_id}` for state/progress and
`GET /api/v1/audio/{asset_id}/processing-history` for processing results and
derived output asset IDs. Download a completed derivative through the existing
asset download route.

Example: transpose C major to D major and slow from 120 to 90 BPM:

```http
POST /api/v1/audio/{asset_id}/transform
Idempotency-Key: practice-d-major-slow
Content-Type: application/json

{
  "source_key": "C major",
  "target_key": "D major",
  "source_bpm": 120,
  "target_bpm": 90,
  "output_format": "flac"
}
```

Equivalent direct controls are `semitones` (including fractional values) and
`tempo_factor`. A factor of `0.75` slows tempo to 75% and makes duration about
`1 / 0.75` times longer; a factor of `1.25` speeds tempo to 125%. `source_bpm`
and `target_bpm` are caller-supplied; the API does not silently trust an
uncertain Phase 7 tempo estimate. Key transposition requires explicit source
and target keys or an explicit semitone value. Positive semitones raise
sounding pitch.

To process a selected source range, provide both `start_seconds` and
`end_seconds`. The resulting asset contains that excerpt, not the rest of the
song; its processing record retains the source time range. A 10 ms edge fade is
applied by default to reduce clicks and can be set from 0 to 50 ms. Invalid
ranges fail rather than being clamped.

The request supports optional pitch, optional tempo, or both in one job.
Supported output formats are WAV, FLAC, MP3, AAC, and Opus; `bitrate_kbps` is
used by the lossy encoders. Output is encoded once from the processed WAV
intermediate. Audio input is decoded through the existing FFmpeg path.

## Limits and validation

- Pitch shift: -24 through +24 semitones, including fractional values.
- Tempo factor: 0.25 through 4.0. BPM-based factors outside this range are
  rejected. Result duration must also remain within
  `MAX_AUDIO_DURATION_SECONDS`.
- SoundTouch input: mono or stereo, 8 kHz–48 kHz. Higher sample rates are
  downsampled to 48 kHz for processing; output sample rate is recorded. Rates
  below 8 kHz and more than two channels are rejected.
- SoundTouch uses WAV intermediates and classic RIFF size limits. Processing
  rejects requests whose intermediate would approach the 4 GiB RIFF boundary.
- Excerpts are cut during FFmpeg decode; long source files are not copied as
  whole decoded intermediates when a smaller selected section was requested.
- Exported output passes FFprobe inspection and full FFmpeg decode validation.
  Duration, sample rate, channel count, bit depth when reported, peak, RMS,
  sample clipping, and non-finite floating-point samples are recorded. Peak
  above -1 dBFS is attenuated before export. Encoded duration is checked with a
  codec-appropriate tolerance. A measurable, non-silent input becoming silent
  is treated as an error.
- Output bytes are staged through the existing bounded storage API as a new
  derived asset. The source remains available and is never overwritten.
- A worker cancellation terminates the SoundStretch child process; the existing
  per-job temporary directory is removed when processing exits. Retry and
  job ownership continue to use the established durable job system.

The SoundStretch CLI does not expose a documented frame-progress callback.
During its execution, the worker reports an estimate from bytes actually
written to the output file, then reports completion only after file and WAV
checks pass. Durable job polling remains the supported notification mechanism:
this codebase has no existing job WebSocket endpoint.

## Persistence

No new tables are needed. `audio_processing_records` stores operation
parameters, including resolved semitones, key/BPM inputs, output format, engine
and version, status, duration, and technical validation result. The existing
ordered processing-output link points to a new child audio asset, whose asset
record stores its measured properties and checksum. The source asset ID remains
the processing record's source and is not modified.

## Deployment and quality gates

The Docker image installs Debian's pinned `soundstretch=2.3.2+ds1-1` package.
`SOUNDTOUCH_PATH` and `SOUNDTOUCH_VERSION` configure the worker executable and
version recorded with results; ensure the version value matches the deployed
binary when overriding either. For redistributions, retain the package's LGPL
notices and source/relinking materials as required by LGPL v2.1; obtain legal
review or the vendor's commercial license if the deployment model requires it.

The existing audio worker timeout and concurrency settings also bound this
operation. The processing runs on the CPU queue; no GPU is used. There is no
claimed real-time ratio or throughput target. Representative piano, voice,
guitar, bass, drums, and dense-mix listening/benchmark validation remains a
release gate, especially for large shifts and slow/fast extremes.
