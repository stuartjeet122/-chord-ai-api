# Audio DSP and preprocessing

> Historical phase report. Audio processing endpoints and background workers are not part of the current stateless application.

Phase 5 adds reusable, non-AI audio analysis and processing. The FastAPI process validates requests and queues durable jobs; CPU DSP and FFmpeg work runs in Celery workers, independently of request handling. Results and processing history are persisted with source/output asset references.

## Endpoints

All routes are under `/api/v1/audio`. Routes that start work return `202 Accepted` and a job resource; poll `GET /api/v1/jobs/{job_id}` for progress and terminal status. `POST /api/v1/jobs/{job_id}/cancel` requests cancellation. Use `Idempotency-Key` to make job submission retries safe.

| Method and route | Responsibility |
|---|---|
| `POST /{asset_id}/waveform` | Generate a bounded per-channel waveform with min/max, peak, RMS, and timestamps. |
| `GET /{asset_id}/waveform` | Retrieve the latest completed waveform JSON artifact. |
| `POST /{asset_id}/silence-detection` | Find silence intervals using absolute or relative thresholds and configurable windows. |
| `POST /{asset_id}/quality` | Queue peak, RMS, clipping, DC-offset, silence, crest-factor, and rough noise-floor measurements. |
| `GET /{asset_id}/quality` | Retrieve the latest completed quality report. |
| `POST /{asset_id}/trim` | Trim by start/end seconds, quantized to source sample frames. |
| `POST /{asset_id}/split` | Split at timestamps, equal intervals, or detected silence; optional overlap and duration bounds apply to equal splits. |
| `POST /{asset_id}/chunks` | Create time-aligned chunks with optional output sample-rate/channel preparation and a caller-supplied profile ID. |
| `POST /{asset_id}/normalize` | Peak, RMS, or EBU R128 loudness normalization. |
| `POST /{asset_id}/gain` | Apply linear, dB, or per-channel gain. |
| `POST /{asset_id}/fade` | Apply linear or equal-power fade-in/fade-out. |
| `POST /{asset_id}/filter` | Apply Butterworth high/low/band-pass or notch filtering. |
| `POST /{asset_id}/channels` | Convert to mono/stereo, extract/swap channels, or split channels into assets. |
| `POST /{asset_id}/resample` | Resample to a supported sample rate. |
| `POST /{asset_id}/pipeline` | Run an ordered, typed sequence of supported DSP operations as one job. |
| `POST /{asset_id}/transform` | Queue pitch shift, key transposition, tempo change, or combined audio transformation, optionally for a selected range. |
| `POST /concatenate` | Concatenate 2–32 owned audio assets, optionally with an equal-power crossfade. |
| `GET /{asset_id}/processing-history` | List the latest 100 processing records and output references. |

Existing inspection, conversion, metadata, conversion-profile, and derived-asset routes are described in [AUDIO_INGESTION.md](./AUDIO_INGESTION.md).
Transformation engine selection, request schema, limits, and quality caveats are documented in [AUDIO_TRANSFORMS.md](./AUDIO_TRANSFORMS.md).

### Example requests

Generate waveform data and retrieve it after the job succeeds:

```http
POST /api/v1/audio/{asset_id}/waveform
Idempotency-Key: waveform-v1
Content-Type: application/json

{"samples_per_second": 100, "max_points": 2000000}
```

Prepare chunks for a downstream analysis model:

```http
POST /api/v1/audio/{asset_id}/chunks
Content-Type: application/json

{
  "chunk_seconds": 30,
  "overlap_seconds": 2,
  "sample_rate": 44100,
  "channels": "mono",
  "profile_id": "analysis-model-v1"
}
```

Apply an ordered pipeline:

```http
POST /api/v1/audio/{asset_id}/pipeline
Content-Type: application/json

{
  "operations": [
    {"type": "trim", "start_seconds": 3.2, "end_seconds": 42.0},
    {"type": "channels", "mode": "mono"},
    {"type": "normalize", "method": "peak", "target_dbfs": -1, "ceiling_dbfs": -1}
  ]
}
```

## Signal and output behavior

- The reusable DSP functions use floating-point NumPy signals in `(frames, channels)` order. Mono input is normalized to `(frames, 1)`. Sample rate and frame offsets remain explicit.
- File-based PCM operations stream bounded blocks where practical. Resampling and zero-phase filtering require full-signal memory and are rejected when the estimated working set exceeds 256 MiB.
- Generated audio assets are WAV PCM 24-bit. Samples are clipped to `[-1, 1]` at that output boundary; operations report clipping where relevant. Source assets are not modified.
- Segmentation is capped at 1,000 output assets per job. Concatenation accepts at most 32 source assets. Waveform generation is capped at 2,000,000 points.
- Chunks preserve source start/end sample indices and timestamps even when the prepared output is resampled or downmixed. `profile_id` is descriptive metadata; it does not select or invoke a model.
- Causal filtering is streamable and introduces phase shift. Zero-phase filtering is offline, memory-bounded, and can introduce boundary transients.
- Peak/RMS normalization reports sample-domain quantities. EBU R128 loudness normalization uses FFmpeg's `loudnorm` filter and reports integrated LUFS and true peak separately. A quality report does not estimate LUFS or true peak from sample peak/RMS; those fields are explicitly `null`.
- Crossfades use equal-power curves and subtract their overlap from the resulting duration. All concatenation sources must be owned and available to the requesting user. Different source sample rates require an explicit target sample rate; incompatible channel counts require mono/stereo conversion.

## Worker and storage requirements

DSP tasks use the existing durable job/outbox system and `audio_cpu` queue. API and workers require compatible settings and access to the same local storage volume, or the same S3-compatible bucket. FFmpeg and FFprobe must be available to workers. The PostgreSQL processing record contains operation parameters, status, result metrics, timings/errors, and ordered output/source references; output bytes are stored through the configured asset storage provider.

Configure `AUDIO_CONVERSION_TIMEOUT_SECONDS`, `AUDIO_CONVERSION_MAX_CONCURRENCY`, `MAX_AUDIO_DURATION_SECONDS`, and `MAX_AUDIO_OUTPUT_SIZE_BYTES` as described in [AUDIO_INGESTION.md](./AUDIO_INGESTION.md). API request size, storage quota, and ownership policies continue to apply. Apply database changes with `python -m alembic upgrade head`.

## Limits

This phase provides generic DSP and preprocessing only. It does not include chord/note recognition, BPM/key/time-signature inference, lyrics, source separation, MIDI/notation, or model invocation. Integrated loudness/true-peak analysis is available when the loudness-normalization operation runs, not as part of the generic quality scan. GPU worker management and live PostgreSQL/Redis dispatch still require deployment verification.
