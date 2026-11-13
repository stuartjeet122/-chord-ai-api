from __future__ import annotations

from pathlib import Path

from music_api.ai.catalog import build_model_registry
from music_api.ai.engine import InferenceEngine
from music_api.ai.hardware import HardwareCapabilities, detect_hardware
from music_api.ai.lifecycle import ModelLifecycleManager
from music_api.ai.providers import ProviderFactory
from music_api.ai.registry import ModelRegistry
from music_api.config import Settings


class AIInferenceRuntime:
    def __init__(
        self,
        settings: Settings,
        *,
        registry: ModelRegistry | None = None,
        factories: dict[str, ProviderFactory] | None = None,
        hardware: HardwareCapabilities | None = None,
    ) -> None:
        self.registry = registry or build_model_registry()
        self.hardware = hardware or detect_hardware()
        self.lifecycle = ModelLifecycleManager(
            factories=factories if factories is not None else build_provider_factories(),
            hardware=self.hardware,
            configured_device=settings.ai_execution_device.value,
            model_root=Path(settings.ai_model_dir),
        )
        self.engine = InferenceEngine(
            lifecycle=self.lifecycle,
            hardware=self.hardware,
            max_concurrency=settings.ai_inference_max_concurrency,
            timeout_seconds=settings.ai_inference_timeout_seconds,
        )

    def provider_is_configured(self, model_id: str) -> bool:
        definition = self.registry.get(model_id)
        return definition.enabled and definition.provider in self.lifecycle.provider_names


def build_provider_factories() -> dict[str, ProviderFactory]:
    """Return only model providers installed and explicitly configured for this service."""
    return {}
