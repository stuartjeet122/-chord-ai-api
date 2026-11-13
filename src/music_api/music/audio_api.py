from __future__ import annotations

import asyncio
import tempfile
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Literal

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from music_api.music.audio_analysis import (
    MAX_DURATION_SECONDS,
    AnalysisCapacityError,
    BeatEventData,
    ChordEventData,
    analyze_mp3,
    parse_lrc,
    probe_mp3,
)

router = APIRouter(prefix="/audio", tags=["Audio analysis"])
MAX_UPLOAD_BYTES = 50 * 1024 * 1024
UPLOAD_CHUNK_BYTES = 1024 * 1024


class BeatEvent(BaseModel):
    timestamp_seconds: float
    beat_index: int


class ChordEvent(BaseModel):
    start_seconds: float
    end_seconds: float
    chord: str
    confidence: float


class NoteEvent(BaseModel):
    start_seconds: float
    end_seconds: float
    notes: list[str]


class LyricLine(BaseModel):
    start_seconds: float
    text: str
    beat_index: int | None
    chords: list[str]


class AudioAnalysisResponse(BaseModel):
    filename: str
    duration_seconds: float
    tempo_bpm: float | None
    time_signature: None
    estimated_key: str | None
    beats: list[BeatEvent]
    chords: list[ChordEvent]
    estimated_note_events: list[NoteEvent]
    lyrics_status: Literal["timestamped", "not_provided"]
    lyrics: list[LyricLine]
    transcription_status: Literal["not_configured"]
    analysis_notes: list[str]


class AudioMetadataResponse(BaseModel):
    filename: str
    duration_seconds: float
    codec: str | None
    sample_rate_hz: int | None
    channels: int | None
    bitrate_kbps: float | None
    title: str | None
    artist: str | None
    album: str | None


@asynccontextmanager
async def _temporary_mp3_upload(file: UploadFile) -> AsyncIterator[tuple[str, Path]]:
    filename = (file.filename or "").replace("\\", "/").rsplit("/", 1)[-1]
    if Path(filename).suffix.lower() != ".mp3":
        await file.close()
        raise HTTPException(status_code=415, detail="Only MP3 uploads are supported.")

    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_file:
            temp_path = Path(temp_file.name)
            size = 0
            while chunk := await file.read(UPLOAD_CHUNK_BYTES):
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail=(
                            "Audio file must be no larger than "
                            f"{MAX_UPLOAD_BYTES // (1024 * 1024)} MB."
                        ),
                    )
                temp_file.write(chunk)
        if size == 0:
            raise HTTPException(status_code=400, detail="The uploaded audio file is empty.")
        yield filename, temp_path
    finally:
        await file.close()
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


@router.post(
    "/analyze",
    response_model=AudioAnalysisResponse,
    summary="Analyze an MP3 for musical structure",
    description=(
        "Upload an MP3 to estimate tempo, beat positions, key, chord changes, and active "
        "pitch classes. Optionally supply timestamped LRC lyrics. This endpoint does not "
        "transcribe singing or infer time signature."
    ),
)
async def analyze_audio(
    file: Annotated[
        UploadFile,
        File(description="MP3 audio file (maximum 50 MB and 10 minutes)."),
    ],
    lyrics_lrc: Annotated[
        str | None,
        Form(
        description="Optional timestamped LRC lyrics, e.g. [00:12.50]First lyric line",
        ),
    ] = None,
) -> AudioAnalysisResponse:
    if lyrics_lrc is not None:
        try:
            parse_lrc(lyrics_lrc, float(MAX_DURATION_SECONDS))
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    async with _temporary_mp3_upload(file) as (filename, temp_path):
        try:
            analysis = await asyncio.to_thread(analyze_mp3, temp_path)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except AnalysisCapacityError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

        try:
            lyric_lines = parse_lrc(lyrics_lrc, analysis["duration_seconds"]) if lyrics_lrc else []
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        lyrics = [
            {
                **line,
                "beat_index": _nearest_beat_index(
                    float(line["start_seconds"]),
                    analysis["beats"],
                ),
                "chords": _chords_at(
                    float(line["start_seconds"]),
                    analysis["chords"],
                ),
            }
            for line in lyric_lines
        ]
        return AudioAnalysisResponse(
            filename=filename,
            **analysis,
            lyrics_status="timestamped" if lyrics_lrc else "not_provided",
            lyrics=lyrics,
            transcription_status="not_configured",
            analysis_notes=[
                (
                    "Tempo, key, chords, and pitch classes are signal-based estimates, "
                    "not guaranteed transcriptions."
                ),
                "Time signature is not estimated by this service yet.",
                (
                    "Singing transcription is not configured. Supply timestamped LRC "
                    "lyrics to include lyric timing."
                ),
            ],
        )


@router.post(
    "/metadata",
    response_model=AudioMetadataResponse,
    summary="Extract metadata from an MP3",
    description=(
        "Upload an MP3 to read its duration, codec, sample rate, channel count, bitrate, "
        "and available title, artist, and album tags."
    ),
)
async def extract_audio_metadata(
    file: Annotated[
        UploadFile,
        File(description="MP3 audio file (maximum 50 MB and 10 minutes)."),
    ],
) -> AudioMetadataResponse:
    async with _temporary_mp3_upload(file) as (filename, temp_path):
        try:
            metadata = await asyncio.to_thread(probe_mp3, temp_path)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return AudioMetadataResponse(filename=filename, **metadata)


def _nearest_beat_index(
    timestamp: float,
    beats: list[BeatEventData],
) -> int | None:
    if not beats:
        return None
    return min(
        beats,
        key=lambda beat: abs(float(beat["timestamp_seconds"]) - timestamp),
    )["beat_index"]


def _chords_at(timestamp: float, chords: list[ChordEventData]) -> list[str]:
    return [
        str(event["chord"])
        for event in chords
        if float(event["start_seconds"]) <= timestamp < float(event["end_seconds"])
    ]
