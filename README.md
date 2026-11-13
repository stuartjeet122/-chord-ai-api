# Music Intelligence API

The project has completed the **Phase 2 foundation**, **Phase 3: asset storage**, **Phase 4: audio ingestion**, **Phase 5: reusable audio DSP**, the **Phase 6 AI infrastructure scaffold**, an initial **Phase 7 DSP music-analysis path**, a deterministic **Phase 8 music-theory engine**, and **Phase 9 audio pitch/tempo transformations**. It provides the versioned FastAPI application, PostgreSQL/SQLAlchemy and Alembic plumbing, provider-independent asset storage, FFmpeg/FFprobe validation and conversion, metadata/artwork extraction, bounded signal analysis and transformations, durable Celery jobs, isolated model/provider/benchmark infrastructure, versioned DSP harmonic/rhythm estimates, independent music-theory calculations, and derived audio pitch/time processing. No production AI model is enabled or validated, and music estimates have not been benchmarked against real annotated recordings. Real authentication is not implemented. See [AI_INFRASTRUCTURE.md](./AI_INFRASTRUCTURE.md), [AI_MODEL_EVALUATION.md](./AI_MODEL_EVALUATION.md), [MUSIC_ANALYSIS.md](./MUSIC_ANALYSIS.md), [MUSIC_THEORY.md](./MUSIC_THEORY.md), and [AUDIO_TRANSFORMS.md](./AUDIO_TRANSFORMS.md) for precise limitations.

## Requirements

- Python 3.12 or newer
- PostgreSQL 16/17
- Redis 7 when `REDIS_ENABLED=true`
- FFmpeg and FFprobe for audio inspection and conversion (included in the Docker image)
- Docker Engine with Docker Compose v2 for the container workflow

The dependency versions are pinned in `pyproject.toml`. The selected environment must satisfy `requires-python = ">=3.12"`.

## Local development

PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
Copy-Item .env.example .env
```

Set `DATABASE_URL` and `REDIS_URL` in the local `.env` to point to running services. The values in `.env.example` are local-only examples; never use them in deployment.

Start the API:

```powershell
python -m alembic upgrade head
music-api
```

The server uses the configured `API_HOST` and `API_PORT` (defaults `127.0.0.1:8000`) and listens under `/api/v1`. In development, interactive docs are at `/api/v1/docs`, ReDoc at `/api/v1/redoc`, and the OpenAPI JSON at `/api/v1/openapi.json`.

## Docker Compose

Build and start the API, audio worker, PostgreSQL, and Redis:

```powershell
docker compose up --build
```

The API container waits for healthy database and Redis services, applies Alembic migrations, and starts Uvicorn without development reload. Local-only database credentials are declared in Compose and must be replaced for any deployed environment.

Stop containers while preserving persistent data:

```powershell
docker compose down
```

Remove persistent volumes only when intentionally discarding local development data:

```powershell
docker compose down --volumes
```

## Configuration

Settings are loaded from environment variables and, for local development, an optional `.env` file. Environment variable names are case-insensitive.

| Variable | Default | Description |
|---|---|---|
| `APP_NAME` | `Music Intelligence API` | Service name shown by the root endpoint and OpenAPI. |
| `APP_ENV` | `development` | `development`, `testing`, `staging`, or `production`. |
| `DEBUG` | `false` | Debug flag; production rejects `true`. |
| `API_PREFIX` | `/api/v1` | Versioned route and documentation prefix. |
| `API_VERSION` | `v1` | Public API version identifier. |
| `API_HOST`, `API_PORT` | `127.0.0.1`, `8000` | Suggested bind settings for local serving. |
| `DATABASE_URL` | Local PostgreSQL URL | Async SQLAlchemy URL using `postgresql+asyncpg`. It must be explicitly set in production. |
| `REDIS_ENABLED` | `true` | Enables Redis health checks and client lifecycle. |
| `REDIS_URL` | `redis://localhost:6379/0` | Redis URL; explicitly required in production when enabled. |
| `JWT_SECRET_KEY` | unset | Reserved for the later authentication phase; no token issuance is implemented here. |
| `JWT_ALGORITHM` | `HS256` | Reserved JWT algorithm setting. |
| `JWT_ACCESS_TOKEN_MINUTES`, `JWT_REFRESH_TOKEN_DAYS` | `15`, `30` | Reserved token lifetime settings. |
| `CORS_ORIGINS` | `[]` | JSON array of allowed origins. |
| `MAX_UPLOAD_SIZE_BYTES` | `104857600` | Maximum size accepted while streaming an uploaded file to staging. |
| `MAX_AUDIO_DURATION_SECONDS` | `21600` | Maximum duration accepted by FFprobe for an audio asset. |
| `MAX_AUDIO_OUTPUT_SIZE_BYTES` | `1073741824` | Maximum size for each converted audio output. |
| `FFMPEG_PATH`, `FFPROBE_PATH` | `ffmpeg`, `ffprobe` | Executable names or configured paths used by API readiness checks and audio workers. |
| `SOUNDTOUCH_PATH`, `SOUNDTOUCH_VERSION` | `soundstretch`, `2.3.2+ds1-1` | SoundTouch CLI executable and version recorded for pitch/tempo transform jobs. |
| `AUDIO_CONVERSION_TIMEOUT_SECONDS` | `21600` | Maximum time allowed for conversion or full-stream decode verification. |
| `AUDIO_CONVERSION_MAX_CONCURRENCY` | `2` | Celery worker concurrency for audio tasks. |
| `TEMP_DIR`, `STORAGE_DIR` | `./tmp`, `./storage` | Private upload staging and local provider root. |
| `STORAGE_PROVIDER` | `local` | `local` or `s3`; application code uses a provider interface. |
| `STORAGE_S3_ENDPOINT_URL` | unset | Optional S3-compatible endpoint, such as a private MinIO endpoint. |
| `STORAGE_S3_REGION`, `STORAGE_S3_BUCKET` | `us-east-1`, `music-api` | S3 region and private bucket. |
| `STORAGE_S3_ACCESS_KEY_ID`, `STORAGE_S3_SECRET_ACCESS_KEY` | unset | Optional development credentials; required in production S3 configuration. Prefer workload identity/roles where available. |
| `STORAGE_S3_USE_TLS` | `true` | Whether the S3 connection uses TLS; endpoint scheme must match. |
| `STORAGE_S3_PRESIGNED_URL_TTL_SECONDS` | `300` | Maximum lifetime for generated S3 download URLs. |
| `USER_STORAGE_QUOTA_BYTES` | `5368709120` | Default per-owner durable storage quota; active reservations count against it. |
| `UPLOAD_STAGING_TTL_SECONDS` | `86400` | Age after which reconciliation may remove staging files and expire abandoned reservations. |
| `STORAGE_DEV_AUTH_ENABLED` | `true` | Enables `X-Dev-User-ID` only outside production. Production requires it to be `false`; real auth integration is not implemented yet. |
| `LOG_LEVEL` | `INFO` | `CRITICAL`, `ERROR`, `WARNING`, `INFO`, or `DEBUG`. |
| `LOG_FORMAT` | `console` | `console` or `json`; production always uses JSON logs. |
| `DOCS_ENABLED` | environment-dependent | Defaults on outside production; production rejects enabling public docs. |
| `EXTERNAL_SERVICE_BASE_URL`, `EXTERNAL_SERVICE_API_KEY` | unset | Reserved external integration settings. Secrets are masked by Pydantic and are not logged. |

Production settings fail validation when required database/Redis URLs are not explicitly supplied, when debug is on, or when public documentation is enabled. Put secret values in the deployment secret manager, not in source control or image build arguments.

## Database and migrations

The SQLAlchemy engine uses asyncpg, connection pre-ping, and a bounded pool. Request sessions are dependency-injected and closed on completion. Schema creation is never performed at application startup.

Apply migrations explicitly:

```powershell
python -m alembic upgrade head
```

Create a migration after adding mapped domain models:

```powershell
python -m alembic revision --autogenerate -m "describe schema change"
```

The initial migration establishes a clean baseline without creating premature music-domain tables. In production, run migrations as a controlled deployment step before rolling out API replicas.

## Storage API

Uploads stream in bounded chunks to private staging, enforce the maximum size, calculate SHA-256, check recognized file signatures, reserve owner quota, and verify the stored object's size and checksum before recording it. The API does not buffer the complete audio body in application memory. Multipart parsing occurs before the route receives the upload, so deployments must also set an ingress/request-body limit to protect temporary disk.

The local provider writes atomically beneath its private root; the S3-compatible provider uses managed transfers and can issue short-lived presigned downloads. Object keys are generated internally and filenames are display metadata only. Reconciliation reports missing/orphaned objects without deleting permanent objects; it also cleans stale staging files and expired reservations. Storage reconciliation is not scheduled by the audio worker and must be run separately. For provider settings, lifecycle, integrity, quota behavior, and limitations, see [STORAGE.md](./STORAGE.md).

Production requests fail closed because JWT authentication is not yet implemented. In non-production, set `X-Dev-User-ID: <uuid>` to exercise ownership. Supported signatures indicate a likely file/container type, not full decodability; use the audio inspection job to validate the full stream and extract technical metadata.

Audio ingestion, conversion profiles, job operation, and limitations are documented in [AUDIO_INGESTION.md](./AUDIO_INGESTION.md).

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/api/v1/assets` | Multipart upload (`file`, optional `asset_type`, optional `parent_asset_id`); optional `Idempotency-Key`. |
| `GET` | `/api/v1/assets` | List owned assets with `limit`, `offset`, and optional `asset_type` filter. |
| `GET` | `/api/v1/assets/{asset_id}` | Retrieve owned asset metadata. |
| `GET` | `/api/v1/assets/{asset_id}/children` | List derived assets. |
| `GET` | `/api/v1/assets/{asset_id}/download` | Stream owned bytes; `signed=true` uses an S3 signed redirect when supported. |
| `DELETE` | `/api/v1/assets/{asset_id}` | Delete an asset unless it has live children. |
| `GET` | `/api/v1/storage/usage` | Used/reserved/quota/available bytes and per-category totals. |
| `GET` | `/api/v1/audio/conversion-profiles` | List built-in WAV conversion profiles. |
| `POST` | `/api/v1/audio/{asset_id}/inspect` | Queue full audio validation and metadata/artwork extraction. |
| `GET` | `/api/v1/audio/{asset_id}/metadata` | Retrieve extracted normalized/raw tags, technical metadata, and artwork references. |
| `POST` | `/api/v1/audio/{asset_id}/convert` | Queue conversion to WAV, FLAC, MP3, AAC, or Opus. |
| `GET` | `/api/v1/audio/{asset_id}/assets` | List derived audio assets for the source asset. |
| `GET` | `/api/v1/jobs/{job_id}` | Poll owned job status and progress. |
| `POST` | `/api/v1/jobs/{job_id}/cancel` | Request cooperative job cancellation. |

## Foundation API endpoints

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/api/v1/` | Application name, API version, and non-secret environment identifier. |
| `GET` | `/api/v1/health` | Basic process health. |
| `GET` | `/api/v1/health/live` | Liveness probe; does not depend on external services. |
| `GET` | `/api/v1/health/ready` | PostgreSQL, enabled Redis, FFmpeg, and FFprobe readiness checks; returns `503` if a required dependency is unavailable. |

Errors use `application/problem+json` with a stable `code` and `trace_id`. Every HTTP response includes an `X-Request-ID`. Supplied IDs are accepted only if they match a restricted safe character set; otherwise a UUID is generated.

## Tests and checks

Install development dependencies and run:

```powershell
python -m pytest
python -m ruff check .
python -m ruff format --check .
python -m mypy src
```

Unit tests use isolated SQLite databases and temporary local storage, and disable Redis. PostgreSQL/Redis integration checks require explicitly configured test service URLs; never point test configuration at a development or production database.

## Structure

```text
src/music_api/
├── api/                 # HTTP dependencies, errors, request middleware, versioned routes
├── db/                  # SQLAlchemy base and engine/session lifecycle
├── audio/               # Ingestion, metadata, DSP, conversion profiles, and API routes
│   └── dsp/             # Signal operations, analysis, schemas, chunking, and workers
├── jobs/                # Durable job/outbox records, Celery tasks, and status APIs
├── music/               # DSP analysis, deterministic theory engine, schemas, and routes
├── storage/             # Asset models, local/S3 providers, storage service, and API routes
├── config.py            # Typed environment configuration
├── errors.py            # Reusable application exceptions
├── main.py              # FastAPI factory and resource lifespan
├── observability.py     # Structured logging
└── resources.py         # Application-scoped DB and Redis clients
migrations/              # Alembic environment and versioned schema history
tests/                   # Unit and integration tests
```

Future application/domain/infrastructure modules should follow the boundaries in [TECHNICAL_DESIGN.md](./TECHNICAL_DESIGN.md).

## Current limitations

- Real authentication, user/song tables, music understanding, and AI model inference are future phases.
- DSP endpoints, output formats, processing limits, and signal semantics are described in [AUDIO_DSP.md](./AUDIO_DSP.md).
- Pitch/time transformation engine, formats, limits, API examples, and licensing notes are described in [AUDIO_TRANSFORMS.md](./AUDIO_TRANSFORMS.md).
- Deterministic theory routes, notation conventions, interpretation persistence, and known limits are described in [MUSIC_THEORY.md](./MUSIC_THEORY.md).
- Audio workers use FFmpeg/FFprobe and Redis-backed Celery; job state and dispatch outbox are persisted in PostgreSQL. Job updates are polled; WebSockets are not implemented.
- Local storage requires API and worker containers to share the configured storage volume. S3-compatible production integration requires deployment-environment validation.
- The Docker Compose stack, live PostgreSQL/Redis broker dispatch, and production deployment have not been verified in this environment.
- Review licensing for the exact Mutagen package and deployed FFmpeg build/distribution model before commercial distribution.
- Python 3.12+ is the supported runtime.
