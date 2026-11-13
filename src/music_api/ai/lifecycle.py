from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from music_api.ai.hardware import HardwareCapabilities
from music_api.ai.providers import ProviderFactory, ProviderLifecycle
from music_api.ai.schemas import (
    ExecutionDevice,
    ModelDefinition,
    ModelLifecycleState,
    ModelStatusResponse,
)

logger = logging.getLogger(__name__)


@dataclass
class _ManagedModel:
    state: ModelLifecycleState = ModelLifecycleState.REGISTERED
    provider: ProviderLifecycle | None = None
    device: ExecutionDevice | None = None
    initialized_at: datetime | None = None
    initialization_duration_ms: int | None = None
    last_error_code: str | None = None
    last_error_message: str | None = None

    def status(self, model_id: str) -> ModelStatusResponse:
        return ModelStatusResponse(
            model_id=model_id,
            state=self.state,
            selected_device=self.device,
            initialized_at=self.initialized_at,
            initialization_duration_ms=self.initialization_duration_ms,
            last_error_code=self.last_error_code,
            last_error_message=self.last_error_message,
        )


class ModelLifecycleManager:
    def __init__(
        self,
        *,
        factories: dict[str, ProviderFactory],
        hardware: HardwareCapabilities,
        configured_device: str,
        model_root: Path,
    ) -> None:
        self._factories = factories
        self._hardware = hardware
        self._configured_device = configured_device
        self._model_root = model_root.resolve()
        self._models: dict[str, _ManagedModel] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    def status(self, definition: ModelDefinition) -> ModelStatusResponse:
        managed = self._models.get(definition.model_id, _ManagedModel())
        if not definition.enabled:
            managed = _ManagedModel(state=ModelLifecycleState.DISABLED)
        elif definition.provider not in self._factories and managed.state in {
            ModelLifecycleState.REGISTERED,
            ModelLifecycleState.UNAVAILABLE,
        }:
            managed = _ManagedModel(
                state=ModelLifecycleState.UNAVAILABLE,
                last_error_code="provider_unavailable",
                last_error_message=(
                    f"Provider '{definition.provider}' is not installed or configured."
                ),
            )
        return managed.status(definition.model_id)

    @property
    def provider_names(self) -> frozenset[str]:
        return frozenset(self._factories)

    async def load(self, definition: ModelDefinition) -> tuple[ProviderLifecycle, ExecutionDevice]:
        managed = self._models.setdefault(definition.model_id, _ManagedModel())
        lock = self._locks.setdefault(definition.model_id, asyncio.Lock())
        async with lock:
            if not definition.enabled:
                managed.state = ModelLifecycleState.DISABLED
                raise ModelLifecycleError(
                    "model_disabled", f"Model '{definition.model_id}' is disabled."
                )
            if managed.provider is not None and managed.state == ModelLifecycleState.READY:
                if managed.device is None:
                    raise ModelLifecycleError("invalid_model_state", "Loaded model has no device.")
                return managed.provider, managed.device
            factory = self._factories.get(definition.provider)
            if factory is None:
                managed.state = ModelLifecycleState.UNAVAILABLE
                managed.last_error_code = "provider_unavailable"
                managed.last_error_message = (
                    f"Provider '{definition.provider}' is not installed or configured."
                )
                raise ModelLifecycleError(managed.last_error_code, managed.last_error_message)
            device = self._select_device(definition)
            self._verify_weights(definition)
            provider = factory()
            started = time.monotonic()
            managed.state = ModelLifecycleState.INITIALIZING
            managed.last_error_code = None
            managed.last_error_message = None
            try:
                await provider.initialize(definition, device)
                await provider.warmup()
            except Exception as exc:
                managed.state = ModelLifecycleState.FAILED
                managed.provider = None
                managed.device = None
                managed.last_error_code = "model_initialization_failed"
                managed.last_error_message = str(exc)[:255]
                try:
                    await provider.unload()
                except Exception:
                    logger.exception(
                        "Could not release resources after model initialization failed",
                        extra={"model_id": definition.model_id},
                    )
                logger.exception(
                    "AI model initialization failed",
                    extra={"model_id": definition.model_id, "device": device.value},
                )
                raise ModelLifecycleError(
                    managed.last_error_code, managed.last_error_message
                ) from exc
            managed.provider = provider
            managed.device = device
            managed.state = ModelLifecycleState.READY
            managed.initialized_at = datetime.now(UTC)
            managed.initialization_duration_ms = int((time.monotonic() - started) * 1000)
            return provider, device

    async def unload(self, definition: ModelDefinition) -> None:
        managed = self._models.setdefault(definition.model_id, _ManagedModel())
        lock = self._locks.setdefault(definition.model_id, asyncio.Lock())
        async with lock:
            if managed.provider is not None:
                await managed.provider.unload()
            self._models[definition.model_id] = _ManagedModel()

    def _select_device(self, definition: ModelDefinition) -> ExecutionDevice:
        available = self._hardware.available_devices()
        configured = self._configured_device
        if configured == "auto":
            preferences = (
                ExecutionDevice.CUDA,
                ExecutionDevice.MPS,
                ExecutionDevice.CPU,
            )
            selected = next(
                (
                    device
                    for device in preferences
                    if device in available and device in definition.supported_devices
                ),
                None,
            )
        else:
            try:
                selected = ExecutionDevice(configured)
            except ValueError as exc:
                raise ModelLifecycleError(
                    "invalid_execution_device", f"Unsupported inference device: {configured}."
                ) from exc
            if selected not in available:
                raise ModelLifecycleError(
                    "execution_device_unavailable",
                    f"Configured device '{selected.value}' is not available on this worker.",
                )
            if selected not in definition.supported_devices:
                raise ModelLifecycleError(
                    "execution_device_unsupported",
                    f"Model '{definition.model_id}' does not support '{selected.value}'.",
                )
        if selected is None:
            raise ModelLifecycleError(
                "execution_device_unavailable",
                f"No supported inference device is available for '{definition.model_id}'.",
            )
        return selected

    def _verify_weights(self, definition: ModelDefinition) -> None:
        if definition.weights_path is None:
            return
        configured = Path(definition.weights_path)
        weight_path = (
            configured.resolve()
            if configured.is_absolute()
            else (self._model_root / configured).resolve()
        )
        if not weight_path.is_relative_to(self._model_root):
            raise ModelLifecycleError(
                "unsafe_model_path",
                "Model weights path must remain inside the configured model directory.",
            )
        if not weight_path.is_file():
            raise ModelLifecycleError(
                "model_weights_missing",
                f"Configured model weights are missing: {weight_path.name}.",
            )
        if definition.weight_sha256 is not None:
            digest = hashlib.sha256()
            with weight_path.open("rb") as model_file:
                for block in iter(lambda: model_file.read(1024 * 1024), b""):
                    digest.update(block)
            if digest.hexdigest().casefold() != definition.weight_sha256.casefold():
                raise ModelLifecycleError(
                    "model_checksum_mismatch", "Model weight SHA-256 checksum did not match."
                )


class ModelLifecycleError(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
