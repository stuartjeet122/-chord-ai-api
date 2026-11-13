from __future__ import annotations

import asyncio
import uuid
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from music_api.ai.schemas import ExecutionDevice, InferenceResult, InferenceTask, ModelDefinition

ProgressCallback = Callable[[float], Awaitable[bool]]


@dataclass(frozen=True)
class AudioInput:
    asset_id: uuid.UUID
    path: Path
    duration_seconds: float
    sample_rate: int
    channel_count: int


@dataclass(frozen=True)
class ProviderOutput:
    predictions: list[dict[str, Any]]
    raw_output: dict[str, Any] | None = None


@runtime_checkable
class ProviderLifecycle(Protocol):
    async def initialize(
        self,
        definition: ModelDefinition,
        device: ExecutionDevice,
    ) -> None: ...

    async def warmup(self) -> None: ...

    async def unload(self) -> None: ...


@runtime_checkable
class ChordAnalysisProvider(ProviderLifecycle, Protocol):
    async def analyze_chords(
        self,
        audio: AudioInput,
        *,
        parameters: dict[str, Any],
        progress: ProgressCallback,
        cancelled: asyncio.Event,
    ) -> ProviderOutput: ...


@runtime_checkable
class NoteTranscriptionProvider(ProviderLifecycle, Protocol):
    async def transcribe_notes(
        self,
        audio: AudioInput,
        *,
        parameters: dict[str, Any],
        progress: ProgressCallback,
        cancelled: asyncio.Event,
    ) -> ProviderOutput: ...


@runtime_checkable
class SourceSeparationProvider(ProviderLifecycle, Protocol):
    async def separate_sources(
        self,
        audio: AudioInput,
        *,
        parameters: dict[str, Any],
        progress: ProgressCallback,
        cancelled: asyncio.Event,
    ) -> ProviderOutput: ...


@runtime_checkable
class MusicUnderstandingProvider(ProviderLifecycle, Protocol):
    async def understand_music(
        self,
        audio: AudioInput,
        *,
        parameters: dict[str, Any],
        progress: ProgressCallback,
        cancelled: asyncio.Event,
    ) -> ProviderOutput: ...


@runtime_checkable
class AudioEmbeddingProvider(ProviderLifecycle, Protocol):
    async def embed_audio(
        self,
        audio: AudioInput,
        *,
        parameters: dict[str, Any],
        progress: ProgressCallback,
        cancelled: asyncio.Event,
    ) -> ProviderOutput: ...


ProviderFactory = Callable[[], ProviderLifecycle]


@dataclass(frozen=True)
class InferenceExecution:
    result: InferenceResult
    metrics: dict[str, float | int | str | None]


async def invoke_provider(
    provider: ProviderLifecycle,
    task: InferenceTask,
    audio: AudioInput,
    *,
    parameters: dict[str, Any],
    progress: ProgressCallback,
    cancelled: asyncio.Event,
) -> ProviderOutput:
    if task == InferenceTask.CHORD_RECOGNITION and isinstance(provider, ChordAnalysisProvider):
        return await provider.analyze_chords(
            audio,
            parameters=parameters,
            progress=progress,
            cancelled=cancelled,
        )
    if task == InferenceTask.NOTE_TRANSCRIPTION and isinstance(provider, NoteTranscriptionProvider):
        return await provider.transcribe_notes(
            audio,
            parameters=parameters,
            progress=progress,
            cancelled=cancelled,
        )
    if task == InferenceTask.SOURCE_SEPARATION and isinstance(provider, SourceSeparationProvider):
        return await provider.separate_sources(
            audio,
            parameters=parameters,
            progress=progress,
            cancelled=cancelled,
        )
    if task == InferenceTask.MUSIC_UNDERSTANDING and isinstance(
        provider, MusicUnderstandingProvider
    ):
        return await provider.understand_music(
            audio,
            parameters=parameters,
            progress=progress,
            cancelled=cancelled,
        )
    if task == InferenceTask.AUDIO_EMBEDDING and isinstance(provider, AudioEmbeddingProvider):
        return await provider.embed_audio(
            audio,
            parameters=parameters,
            progress=progress,
            cancelled=cancelled,
        )
    raise ProviderCapabilityError(
        f"Provider {type(provider).__name__} does not implement task {task.value}."
    )


class ProviderCapabilityError(Exception):
    pass


def validate_provider_definition(
    definition: ModelDefinition,
    task: InferenceTask,
    available_devices: Sequence[ExecutionDevice],
) -> None:
    if task not in definition.supported_tasks:
        raise ProviderCapabilityError(
            f"Model {definition.model_id} does not support task {task.value}."
        )
    if not any(device in definition.supported_devices for device in available_devices):
        raise ProviderCapabilityError(
            f"Model {definition.model_id} has no supported device available on this worker."
        )
