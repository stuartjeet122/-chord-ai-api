from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class AudioChunk:
    id: uuid.UUID
    source_asset_id: uuid.UUID
    start_sample: int
    end_sample: int
    sample_rate: int
    processing_config: dict[str, Any]

    @property
    def start_seconds(self) -> float:
        return self.start_sample / self.sample_rate

    @property
    def end_seconds(self) -> float:
        return self.end_sample / self.sample_rate

    @property
    def duration_seconds(self) -> float:
        return (self.end_sample - self.start_sample) / self.sample_rate


def plan_chunks(
    source_asset_id: uuid.UUID,
    *,
    frame_count: int,
    sample_rate: int,
    chunk_seconds: float,
    overlap_seconds: float = 0.0,
    min_chunk_seconds: float | None = None,
    max_chunk_seconds: float | None = None,
) -> list[AudioChunk]:
    if frame_count <= 0 or sample_rate <= 0 or chunk_seconds <= 0:
        raise ValueError("Frame count, sample rate, and chunk duration must be positive.")
    if overlap_seconds < 0 or overlap_seconds >= chunk_seconds:
        raise ValueError("Overlap must be non-negative and shorter than a chunk.")
    minimum = min_chunk_seconds or 0.0
    maximum = max_chunk_seconds or chunk_seconds
    if minimum < 0 or maximum <= 0 or minimum > maximum:
        raise ValueError("Chunk duration limits are invalid.")
    chunk_frames = round(min(chunk_seconds, maximum) * sample_rate)
    step_frames = round((min(chunk_seconds, maximum) - overlap_seconds) * sample_rate)
    minimum_frames = math_ceil_samples(minimum, sample_rate)
    chunks: list[AudioChunk] = []
    start = 0
    config = {
        "chunk_seconds": chunk_seconds,
        "overlap_seconds": overlap_seconds,
        "min_chunk_seconds": min_chunk_seconds,
        "max_chunk_seconds": max_chunk_seconds,
    }
    while start < frame_count:
        end = min(frame_count, start + chunk_frames)
        if end - start < minimum_frames:
            if chunks:
                previous = chunks.pop()
                chunks.append(
                    AudioChunk(
                        id=previous.id,
                        source_asset_id=source_asset_id,
                        start_sample=previous.start_sample,
                        end_sample=frame_count,
                        sample_rate=sample_rate,
                        processing_config=config,
                    )
                )
            elif frame_count >= minimum_frames:
                end = frame_count
            else:
                raise ValueError("Source is shorter than min_chunk_seconds.")
            if chunks:
                break
        chunks.append(
            AudioChunk(
                id=uuid.uuid4(),
                source_asset_id=source_asset_id,
                start_sample=start,
                end_sample=end,
                sample_rate=sample_rate,
                processing_config=config,
            )
        )
        if end == frame_count:
            break
        start += step_frames
    return chunks


def math_ceil_samples(seconds: float, sample_rate: int) -> int:
    value = seconds * sample_rate
    whole = int(value)
    return whole if whole == value else whole + 1
