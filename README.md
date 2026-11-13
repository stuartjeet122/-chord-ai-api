# Music Intelligence API

A lightweight, stateless FastAPI service for music-theory calculations and metronome schedule generation. It does not use a database, message broker, `.env` file, or background workers.

## Run locally

Requires Python 3.12 or newer.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
music-api
```

The service listens at `http://127.0.0.1:8000` by default:

- `GET /` — simple service status
- `GET /api/v1/` — API and version information
- `GET /api/v1/docs` — Swagger UI
- `GET /api/v1/openapi.json` — OpenAPI document
- `GET /api/v1/health/live` — liveness probe
- `GET /api/v1/health/ready` — readiness probe

Stateless endpoints include music-theory operations under `/api/v1/theory` and metronome schedule generation at `POST /api/v1/music/metronome/schedule`.

## Configuration

Defaults are defined in `src/music_api/config.py`. `.env` files are not loaded. Set environment variables only when a deployment needs to override a default:

```powershell
$env:API_HOST = "0.0.0.0"
$env:API_PORT = "8000"
```

Swagger and ReDoc are enabled by default outside production. They can be enabled or disabled with `DOCS_ENABLED`.

## Docker

```powershell
docker compose up --build
```

Compose runs only the API container.

## Tests

```powershell
python -m pytest
```
