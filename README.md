# Music Intelligence API

A stateless FastAPI service for MP3 metadata extraction and audio analysis, including estimated tempo, beat positions, key, chord changes, and chord-tone events. It also provides a metronome schedule endpoint. It does not expose music-theory utility routes or use a database, message broker, `.env` file, or background workers.

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

The public API documents only the audio workflow and metronome schedule:
- `POST /api/v1/audio/analyze` — estimate BPM, beat timestamps, key, chords, and chord tones; optionally match timestamped LRC lyrics
- `POST /api/v1/audio/metadata` — extract media properties and available ID3 tags
- `POST /api/v1/music/metronome/schedule` — generate a metronome event schedule

Internal API information and health-check routes remain available to service monitoring but are hidden from Swagger.

## Analyze an MP3

Open Swagger at `http://127.0.0.1:8000/api/v1/docs`, choose either audio endpoint, click **Try it out**, choose an MP3, then click **Execute**. Use **Analyze an MP3** for musical features and optional timestamped lyrics, or **Extract metadata from an MP3** for duration, codec, bitrate, and tags.

The endpoint accepts MP3 files up to 50 MB and 10 minutes. It returns estimated BPM, beat timestamps, key, chord events, and estimated chord-tone pitch classes. These are signal-based estimates and may be inaccurate for dense mixes or unusual tuning; they are not isolated instrument or vocal transcriptions. Timestamped lyrics are matched to the nearest beat and active chord. Time-signature detection and automatic singing/lyrics transcription are not implemented. To include lyrics with timestamps, provide LRC text such as:

```text
[00:12.50]First lyric line
[00:16.00]Second lyric line
```

For example, from PowerShell:

```powershell
curl.exe -X POST "http://127.0.0.1:8000/api/v1/audio/analyze" `
  -F "file=@C:\music\song.mp3;type=audio/mpeg" `
  -F "lyrics_lrc=[00:12.50]First lyric line"
```

FFmpeg is required for decoding. It is installed in the provided Docker image.

## Configuration

Defaults are defined in `src/music_api/config.py`. `.env` files are not loaded. Set environment variables only when a deployment needs to override a default:

```powershell
$env:API_HOST = "0.0.0.0"
$env:API_PORT = "8000"
```

Swagger and ReDoc are enabled by default outside production. They can be enabled or disabled with `DOCS_ENABLED`.
When deployed behind App Manager under a path such as `/chord-ai-api/`, the `APP_BASE_PATH` supplied by the manager is included in the OpenAPI server URL so Swagger's “Try it out” requests use that path.

## Docker

```powershell
docker compose up --build
```

Compose runs only the API container.
The deployment mount prefix is configured as `/chord-ai-api` so Swagger "Try it out" requests use the same public route as the deployed API.

## Tests

```powershell
python -m pytest
```
