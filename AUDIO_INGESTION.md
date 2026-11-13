# Audio ingestion and conversion

Phase 4 adds server-side audio inspection, metadata/artwork extraction, conversion, and durable background jobs. It does not implement chord recognition, beat/key analysis, lyrics, source separation, or AI inference.

## Runtime requirements

- FFmpeg and FFprobe must be installed and executable by both API and worker processes. Configure `FFMPEG_PATH` and `FFPROBE_PATH` when they are not on `PATH`. The Docker image installs the distribution FFmpeg package.
- PostgreSQL stores assets, metadata, jobs, and the transactional outbox. Redis is the Celery broker. The worker runs on Linux in the provided container setup; Celery workers are not supported as native Windows processes.
- The worker needs access to the same local storage volume as the API when `STORAGE_PROVIDER=local`. With S3-compatible storage, configure the same bucket/provider settings for both processes.
- FFmpeg is an external executable whose enabled components and license depend on the exact build. Mutagen is distributed under GPL-2.0-or-later. Review the licenses of the exact dependencies and FFmpeg build for the intended commercial distribution and deployment model; this note is not legal advice.

## Supported audio flow

1. Upload an audio file through `POST /api/v1/assets` with `asset_type=original_audio`. Upload validation recognizes MP3, WAV, FLAC, Ogg, MP4/M4A, AIFF, and AAC signatures. A recognized signature is only a preliminary check.
2. Queue full inspection with `POST /api/v1/audio/{asset_id}/inspect`. FFprobe extracts stream/container properties, then FFmpeg decodes the complete first audio stream to validate it. A successful job stores normalized and raw tags, technical metadata, and safe embedded artwork assets.
3. Poll `GET /api/v1/jobs/{job_id}` for status, progress, and errors. Retrieve successful inspection output at `GET /api/v1/audio/{asset_id}/metadata`.
4. Optionally queue conversion with `POST /api/v1/audio/{asset_id}/convert`, then list derived assets with `GET /api/v1/audio/{asset_id}/assets`. Converted assets retain a parent reference to their source.

`GET /api/v1/audio/conversion-profiles` lists the built-in profiles. With the default WAV output, `source_preserving_wav` preserves the source sample rate and channel layout. `music_analysis_wav` targets 44.1 kHz; it does not upsample a lower-rate source unless `allow_upsampling` is true. Both default to 24-bit PCM.

Conversion accepts `wav`, `flac`, `mp3`, `aac`, or `opus`. Optional settings include sample rate (22.05, 32, 44.1, 48, 88.2, or 96 kHz), original/mono/stereo channels, PCM bit depth, WAV float32, lossy bitrate, loudness normalization, and trim boundaries. Unsupported profile/option combinations are rejected during request validation. Arbitrary FFmpeg arguments or filters are not accepted.

Example conversion request:

```json
{
  "profile": "music_analysis_wav",
  "output_format": "wav",
  "sample_rate": 44100,
  "channels": "original",
  "bit_depth": 24
}
```

Inspection and conversion endpoints return `202 Accepted` with a job resource. Supply `Idempotency-Key` to make retries safe; reusing a key with different job inputs returns a conflict. `POST /api/v1/jobs/{job_id}/cancel` requests cancellation. Running FFmpeg processes are terminated cooperatively; terminal jobs cannot be cancelled. Job status is persisted in PostgreSQL and exposed through polling; no WebSocket event stream is currently provided.

## Worker and recovery model

The API writes the job and outbox event in one database transaction. A Celery Beat dispatcher publishes pending outbox events to the `audio_cpu` queue; workers update PostgreSQL as the source of truth. A periodic recovery task detects stale running jobs and retries or marks them failed. Celery task results are not used as product job state.

The Docker Compose worker starts Celery with Beat and listens on the `audio_cpu` and `maintenance` queues. Configure the worker process count using `AUDIO_CONVERSION_MAX_CONCURRENCY` (default `2`). Inspection and conversion have explicit timeouts; output size and source duration limits are configured separately. The API readiness endpoint checks PostgreSQL, enabled Redis, FFmpeg, and FFprobe. Worker/broker connectivity and actual task execution are not proven by the API readiness check.

## Relevant settings

| Setting | Default | Purpose |
|---|---:|---|
| `MAX_UPLOAD_SIZE_BYTES` | `104857600` | Maximum input upload size. |
| `MAX_AUDIO_DURATION_SECONDS` | `21600` | Maximum source duration accepted by inspection. |
| `MAX_AUDIO_OUTPUT_SIZE_BYTES` | `1073741824` | Maximum converted output size. |
| `FFMPEG_PATH`, `FFPROBE_PATH` | `ffmpeg`, `ffprobe` | Executable paths used for validation and processing. |
| `AUDIO_CONVERSION_TIMEOUT_SECONDS` | `21600` | Per-operation time limit for decode/inspection/conversion. |
| `AUDIO_CONVERSION_MAX_CONCURRENCY` | `2` | Celery worker concurrency. |

Use the existing storage configuration for local or S3-compatible object storage and the Redis/PostgreSQL settings for task processing. Apply migrations with `python -m alembic upgrade head` before starting the API and worker.

## Verification and limitations

The integration suite generates audio fixtures and exercises real FFmpeg/FFprobe processing for WAV, MP3, FLAC, M4A, AAC, Ogg, Opus, and AIFF. Run it with FFmpeg on `PATH` or explicitly set `FFMPEG_PATH` and `FFPROBE_PATH`, then run `python -m pytest`.

The API still uses development-only owner identity outside production; real user authentication and authorization are not implemented. Run the provided Docker Compose stack and live PostgreSQL/Redis dispatch in the target deployment environment before production use. Local and S3 storage behavior, external licensing, resource quotas, ingress request limits, and observability must also be reviewed for the deployment.
