from __future__ import annotations

import asyncio
import importlib.util
import logging
import os
import time
from collections.abc import Awaitable, Callable
from typing import Any

from pydantic import ValidationError

from music_api.ai.hardware import HardwareCapabilities
from music_api.ai.lifecycle import ModelLifecycleError, ModelLifecycleManager
from music_api.ai.providers import AudioInput, ProviderCapabilityError, invoke_provider
from music_api.ai.schemas import (
    ExecutionDevice,
    InferenceResult,
    InferenceTask,
    ModelDefinition,
)

logger = logging.getLogger(__name__)
ProgressCallback = Callable[[float], Awaitable[bool]]


class InferenceError(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class InferenceValidationError(InferenceError):
    pass


class InferenceEngine:
    def __init__(
        self,
        *,
        lifecycle: ModelLifecycleManager,
        hardware: HardwareCapabilities,
        max_concurrency: int,
        timeout_seconds: int,
    ) -> None:
        if max_concurrency < 1 or timeout_seconds < 1:
            raise ValueError("Inference concurrency and timeout must be positive.")
        self._lifecycle = lifecycle
        self._hardware = hardware
        self._timeout_seconds = timeout_seconds
        self._cpu_slots = asyncio.Semaphore(max_concurrency)
        self._accelerator_slots = asyncio.Semaphore(1)

    async def execute(
        self,
        definition: ModelDefinition,
        task: InferenceTask,
        audio: AudioInput,
        *,
        parameters: dict[str, Any],
        progress: ProgressCallback,
        cancelled: asyncio.Event,
    ) -> tuple[InferenceResult, dict[str, float | int | str | None]]:
        if task not in definition.supported_tasks:
            raise InferenceError(
                "unsupported_model_task",
                f"Model '{definition.model_id}' does not support task '{task.value}'.",
            )
        if definition.max_duration_seconds is not None and (
            audio.duration_seconds > definition.max_duration_seconds
        ):
            raise InferenceError(
                "audio_duration_limit",
                f"Audio exceeds the {definition.max_duration_seconds:g}-second model limit.",
            )
        if definition.sample_rate is not None and audio.sample_rate != definition.sample_rate:
            raise InferenceError(
                "audio_preprocessing_required",
                f"Model requires {definition.sample_rate} Hz audio; "
                "no provider adapter changed it.",
            )
        if definition.channels is not None and audio.channel_count != definition.channels:
            raise InferenceError(
                "audio_preprocessing_required",
                f"Model requires {definition.channels} channel(s); no provider adapter changed it.",
            )

        queued_at = time.monotonic()
        try:
            provider, device = await self._lifecycle.load(definition)
        except ModelLifecycleError as exc:
            raise InferenceError(exc.code, str(exc)) from exc
        except Exception as exc:
            logger.exception(
                "AI model lifecycle operation failed",
                extra={"model_id": definition.model_id},
            )
            raise InferenceError(
                "model_lifecycle_failed", "The model could not be prepared."
            ) from exc

        scheduler = self._cpu_slots if device == ExecutionDevice.CPU else self._accelerator_slots
        async with scheduler:
            queue_wait_seconds = time.monotonic() - queued_at
            if cancelled.is_set():
                raise InferenceError("job_cancelled", "Inference was cancelled before execution.")
            started = time.monotonic()
            memory_before = _process_memory_bytes()
            inference = asyncio.create_task(
                invoke_provider(
                    provider,
                    task,
                    audio,
                    parameters=parameters,
                    progress=progress,
                    cancelled=cancelled,
                )
            )
            cancel_waiter = asyncio.create_task(cancelled.wait())
            try:
                done, _ = await asyncio.wait(
                    {inference, cancel_waiter},
                    timeout=self._timeout_seconds,
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if inference not in done:
                    inference.cancel()
                    await asyncio.gather(inference, return_exceptions=True)
                    if cancel_waiter in done:
                        raise InferenceError("job_cancelled", "Inference was cancelled.")
                    raise InferenceError(
                        "inference_timeout", "Model inference exceeded its time limit."
                    )
                output = inference.result()
            except asyncio.CancelledError:
                inference.cancel()
                await asyncio.gather(inference, return_exceptions=True)
                raise
            except ProviderCapabilityError as exc:
                raise InferenceError("provider_task_unsupported", str(exc)) from exc
            except RuntimeError as exc:
                if _is_out_of_memory_error(exc):
                    _release_cuda_cache(device)
                    raise InferenceError(
                        "inference_out_of_memory",
                        "Model inference exceeded available device memory; "
                        "reduce batch or chunk size.",
                    ) from exc
                logger.exception(
                    "AI provider failed during inference",
                    extra={"model_id": definition.model_id, "device": device.value},
                )
                raise InferenceError(
                    "provider_inference_failed", "The AI provider failed."
                ) from exc
            except InferenceError:
                raise
            except Exception as exc:
                logger.exception(
                    "AI provider failed during inference",
                    extra={"model_id": definition.model_id, "device": device.value},
                )
                raise InferenceError(
                    "provider_inference_failed", "The AI provider failed."
                ) from exc
            finally:
                cancel_waiter.cancel()
                await asyncio.gather(cancel_waiter, return_exceptions=True)
            duration_seconds = time.monotonic() - started
            duration_ms = max(0, int(duration_seconds * 1000))
            try:
                result = InferenceResult.model_validate(
                    {
                        "model_id": definition.model_id,
                        "model_version": definition.version,
                        "provider": definition.provider,
                        "task": task,
                        "source_asset_id": audio.asset_id,
                        "start_seconds": 0.0,
                        "end_seconds": audio.duration_seconds,
                        "processing_duration_ms": duration_ms,
                        "execution_device": device,
                        "predictions": output.predictions,
                        "raw_output": output.raw_output,
                    }
                )
                _validate_prediction_ranges(result, audio.duration_seconds)
            except (ValidationError, ValueError) as exc:
                raise InferenceValidationError(
                    "invalid_model_output",
                    "Model output failed normalized schema or timestamp validation.",
                ) from exc
            memory_after = _process_memory_bytes()
            metrics: dict[str, float | int | str | None] = {
                "inference_duration_seconds": duration_seconds,
                "audio_duration_seconds": audio.duration_seconds,
                "real_time_factor": (
                    duration_seconds / audio.duration_seconds
                    if audio.duration_seconds > 0
                    else None
                ),
                "queue_wait_seconds": queue_wait_seconds,
                "execution_device": device.value,
                "process_memory_before_bytes": memory_before,
                "process_memory_after_bytes": memory_after,
                "process_memory_delta_bytes": (
                    memory_after - memory_before
                    if memory_before is not None and memory_after is not None
                    else None
                ),
                "gpu_memory_allocated_bytes": _gpu_memory_bytes(device),
            }
            return result, metrics


def _validate_prediction_ranges(result: InferenceResult, duration_seconds: float) -> None:
    if result.source_asset_id.int == 0:
        raise InferenceError("invalid_model_output", "Provider returned an empty source asset ID.")
    if result.start_seconds != 0 or result.end_seconds > duration_seconds:
        raise InferenceError(
            "invalid_model_output", "Provider returned invalid source time bounds."
        )


def _process_memory_bytes() -> int | None:
    if importlib.util.find_spec("psutil") is None:
        return None
    import psutil

    return int(psutil.Process(os.getpid()).memory_info().rss)


def _gpu_memory_bytes(device: ExecutionDevice) -> int | None:
    if device != ExecutionDevice.CUDA or importlib.util.find_spec("torch") is None:
        return None
    import torch

    return int(torch.cuda.max_memory_allocated())


def _release_cuda_cache(device: ExecutionDevice) -> None:
    if device == ExecutionDevice.CUDA and importlib.util.find_spec("torch") is not None:
        import torch

        torch.cuda.empty_cache()


def _is_out_of_memory_error(error: RuntimeError) -> bool:
    message = str(error).casefold()
    return "out of memory" in message or "cuda error: out of memory" in message
