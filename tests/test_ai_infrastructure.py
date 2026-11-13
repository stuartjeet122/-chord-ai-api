from __future__ import annotations

import asyncio
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from music_api.ai.benchmarking import (
    TimedChord,
    TimedNote,
    chord_metrics,
    note_metrics,
    source_separation_metrics,
)
from music_api.ai.engine import InferenceEngine, InferenceError, InferenceValidationError
from music_api.ai.hardware import HardwareCapabilities
from music_api.ai.lifecycle import ModelLifecycleError, ModelLifecycleManager
from music_api.ai.providers import (
    AudioInput,
    ChordAnalysisProvider,
    ProgressCallback,
    ProviderOutput,
)
from music_api.ai.registry import ModelNotFoundError, ModelRegistry
from music_api.ai.schemas import (
    BenchmarkDatasetRequest,
    ChordPrediction,
    ExecutionDevice,
    InferenceTask,
    ModelCategory,
    ModelDefinition,
    ModelLifecycleState,
    NotePrediction,
)


def _definition(**overrides: Any) -> ModelDefinition:
    data: dict[str, Any] = {
        "model_id": "test-chords",
        "name": "Test chord provider",
        "provider": "test",
        "category": ModelCategory.CHORD_RECOGNITION,
        "version": "1.0",
        "architecture": "test-only fixture",
        "supported_tasks": [InferenceTask.CHORD_RECOGNITION],
        "supported_devices": [ExecutionDevice.CPU],
        "license": "test fixture",
        "commercial_use": "test only",
    }
    data.update(overrides)
    return ModelDefinition.model_validate(data)


def _hardware() -> HardwareCapabilities:
    return HardwareCapabilities(
        python_version="3.12.0",
        operating_system="test",
        cpu_count=4,
        memory_total_bytes=8_000_000_000,
        memory_available_bytes=4_000_000_000,
        torch_available=False,
        torch_version=None,
        cuda_available=False,
        cuda_devices=(),
        mps_available=False,
    )


class _ChordProvider(ChordAnalysisProvider):
    def __init__(self) -> None:
        self.initialize_count = 0
        self.unload_count = 0

    async def initialize(
        self,
        _definition: ModelDefinition,
        _device: ExecutionDevice,
    ) -> None:
        self.initialize_count += 1
        await asyncio.sleep(0)

    async def warmup(self) -> None:
        return None

    async def unload(self) -> None:
        self.unload_count += 1

    async def analyze_chords(
        self,
        _audio: AudioInput,
        *,
        parameters: dict[str, Any],
        progress: ProgressCallback,
        cancelled: asyncio.Event,
    ) -> ProviderOutput:
        assert parameters == {"fixture": True}
        assert not cancelled.is_set()
        await progress(1.0)
        return ProviderOutput(
            [
                {
                    "start_seconds": 0.0,
                    "end_seconds": 1.0,
                    "symbol": "C",
                    "root": "C",
                    "quality": "major",
                    "bass": "C",
                    "confidence": None,
                }
            ]
        )


@pytest.mark.asyncio
async def test_model_lifecycle_is_lazy_and_initializes_once_concurrently(tmp_path: Path) -> None:
    provider = _ChordProvider()
    manager = ModelLifecycleManager(
        factories={"test": lambda: provider},
        hardware=_hardware(),
        configured_device="auto",
        model_root=tmp_path,
    )
    definition = _definition()

    loaded = await asyncio.gather(manager.load(definition), manager.load(definition))

    assert provider.initialize_count == 1
    assert loaded[0][0] is loaded[1][0]
    assert loaded[0][1] == ExecutionDevice.CPU
    assert manager.status(definition).state == ModelLifecycleState.READY
    await manager.unload(definition)
    assert provider.unload_count == 1
    assert manager.status(definition).state == ModelLifecycleState.REGISTERED


@pytest.mark.asyncio
async def test_model_lifecycle_reports_unavailable_provider_without_downloading(
    tmp_path: Path,
) -> None:
    definition = _definition(provider="not-installed")
    manager = ModelLifecycleManager(
        factories={},
        hardware=_hardware(),
        configured_device="auto",
        model_root=tmp_path,
    )

    assert manager.status(definition).state == ModelLifecycleState.UNAVAILABLE
    with pytest.raises(ModelLifecycleError, match="not installed"):
        await manager.load(definition)
    assert not list(tmp_path.iterdir())


@pytest.mark.asyncio
async def test_lifecycle_rejects_missing_or_mismatched_model_weights(tmp_path: Path) -> None:
    weights = tmp_path / "weights.bin"
    weights.write_bytes(b"verified test weights")
    definition = _definition(weights_path="weights.bin", weight_sha256="0" * 64)
    manager = ModelLifecycleManager(
        factories={"test": _ChordProvider},
        hardware=_hardware(),
        configured_device="cpu",
        model_root=tmp_path,
    )

    with pytest.raises(ModelLifecycleError, match="checksum"):
        await manager.load(definition)


@pytest.mark.asyncio
async def test_inference_engine_normalizes_provider_output_and_records_metrics(
    tmp_path: Path,
) -> None:
    provider = _ChordProvider()
    definition = _definition()
    lifecycle = ModelLifecycleManager(
        factories={"test": lambda: provider},
        hardware=_hardware(),
        configured_device="cpu",
        model_root=tmp_path,
    )
    engine = InferenceEngine(
        lifecycle=lifecycle,
        hardware=_hardware(),
        max_concurrency=1,
        timeout_seconds=2,
    )
    audio = AudioInput(uuid.uuid4(), tmp_path / "input.wav", 1.0, 44_100, 2)

    async def progress(_fraction: float) -> bool:
        return True

    result, metrics = await engine.execute(
        definition,
        InferenceTask.CHORD_RECOGNITION,
        audio,
        parameters={"fixture": True},
        progress=progress,
        cancelled=asyncio.Event(),
    )

    assert result.source_asset_id == audio.asset_id
    assert result.execution_device == ExecutionDevice.CPU
    assert result.predictions[0]["symbol"] == "C"
    assert metrics["real_time_factor"] >= 0
    assert metrics["execution_device"] == "cpu"


@pytest.mark.asyncio
async def test_inference_engine_rejects_required_preprocessing_instead_of_guessing(
    tmp_path: Path,
) -> None:
    definition = _definition(sample_rate=22_050)
    lifecycle = ModelLifecycleManager(
        factories={"test": _ChordProvider},
        hardware=_hardware(),
        configured_device="cpu",
        model_root=tmp_path,
    )
    engine = InferenceEngine(
        lifecycle=lifecycle,
        hardware=_hardware(),
        max_concurrency=1,
        timeout_seconds=1,
    )

    async def progress(_fraction: float) -> bool:
        return True

    with pytest.raises(InferenceError, match="no provider adapter"):
        await engine.execute(
            definition,
            InferenceTask.CHORD_RECOGNITION,
            AudioInput(uuid.uuid4(), tmp_path / "input.wav", 1.0, 44_100, 2),
            parameters={},
            progress=progress,
            cancelled=asyncio.Event(),
        )


@pytest.mark.asyncio
async def test_inference_engine_validates_malformed_provider_results(tmp_path: Path) -> None:
    class MalformedProvider(_ChordProvider):
        async def analyze_chords(
            self,
            _audio: AudioInput,
            *,
            parameters: dict[str, Any],
            progress: ProgressCallback,
            cancelled: asyncio.Event,
        ) -> ProviderOutput:
            return ProviderOutput(
                [
                    {
                        "start_seconds": 0.0,
                        "end_seconds": 5.0,
                        "symbol": "C",
                        "root": "C",
                        "quality": "major",
                        "confidence": 1.5,
                    }
                ]
            )

    definition = _definition()
    lifecycle = ModelLifecycleManager(
        factories={"test": MalformedProvider},
        hardware=_hardware(),
        configured_device="cpu",
        model_root=tmp_path,
    )
    engine = InferenceEngine(
        lifecycle=lifecycle,
        hardware=_hardware(),
        max_concurrency=1,
        timeout_seconds=1,
    )

    async def progress(_fraction: float) -> bool:
        return True

    with pytest.raises(InferenceValidationError, match="schema"):
        await engine.execute(
            definition,
            InferenceTask.CHORD_RECOGNITION,
            AudioInput(uuid.uuid4(), tmp_path / "input.wav", 1.0, 44_100, 2),
            parameters={},
            progress=progress,
            cancelled=asyncio.Event(),
        )


def test_registry_rejects_duplicate_ids_and_reports_unknown_ids() -> None:
    definition = _definition()
    registry = ModelRegistry([definition])
    with pytest.raises(ValueError, match="already registered"):
        registry.register(definition)
    with pytest.raises(ModelNotFoundError):
        registry.get("unknown")


def test_chord_metrics_report_symbol_accuracy_and_change_timing() -> None:
    references = [
        TimedChord(0, 1, "C", "C", "major"),
        TimedChord(1, 2, "G7", "G", "dominant7"),
    ]
    predictions = [
        ChordPrediction(
            start_seconds=0,
            end_seconds=1.02,
            symbol="C",
            root="C",
            quality="major",
        ),
        ChordPrediction(
            start_seconds=1.02,
            end_seconds=2,
            symbol="G7",
            root="G",
            quality="dominant7",
        ),
    ]

    metrics = chord_metrics(references, predictions, boundary_tolerance_seconds=0.05)

    assert metrics["symbol_accuracy"] == pytest.approx(0.99)
    assert metrics["chord_change_f1"] == pytest.approx(1.0)
    assert metrics["boundary_timing_error_mean_seconds"] == pytest.approx(0.02)
    assert metrics["extended_chord_symbol_accuracy"] == pytest.approx(0.98)


def test_note_metrics_match_pitch_and_onset_without_claiming_unmatched_offsets() -> None:
    references = [TimedNote(0.1, 0.3, 60), TimedNote(0.5, 0.7, 64)]
    predictions = [
        NotePrediction(start_seconds=0.11, end_seconds=0.31, pitch_midi=60),
        NotePrediction(start_seconds=0.8, end_seconds=0.9, pitch_midi=67),
    ]

    metrics = note_metrics(references, predictions)

    assert metrics["precision"] == pytest.approx(0.5)
    assert metrics["recall"] == pytest.approx(0.5)
    assert metrics["f1"] == pytest.approx(0.5)
    assert metrics["mean_onset_error_seconds"] == pytest.approx(0.01)


def test_source_separation_metrics_are_finite_for_near_perfect_output() -> None:
    reference = np.sin(np.linspace(0, 2 * np.pi, 256, endpoint=False))
    metrics = source_separation_metrics(reference, reference.copy())
    assert all(np.isfinite(value) for value in metrics.values())
    assert metrics["si_sdr_db"] > 100


def test_timed_prediction_rejects_invalid_duration_and_confidence() -> None:
    with pytest.raises(ValueError):
        ChordPrediction(
            start_seconds=1,
            end_seconds=1,
            symbol="C",
            root="C",
            quality="major",
        )
    with pytest.raises(ValueError):
        NotePrediction(start_seconds=0, end_seconds=1, pitch_midi=60, confidence=float("nan"))


def test_model_definition_validates_checksum_format() -> None:
    with pytest.raises(ValueError):
        _definition(weight_sha256="invalid")


def test_benchmark_dataset_validates_task_specific_annotations() -> None:
    asset_id = uuid.uuid4()
    request = BenchmarkDatasetRequest.model_validate(
        {
            "dataset_id": "test-set",
            "version": "1",
            "task": "chord_recognition",
            "license": "CC0-1.0",
            "samples": [
                {
                    "asset_id": str(asset_id),
                    "references": [
                        {
                            "start_seconds": 0,
                            "end_seconds": 1,
                            "symbol": "C",
                            "root": "C",
                            "quality": "major",
                        }
                    ],
                }
            ],
        }
    )

    assert request.samples[0].asset_id == asset_id
    with pytest.raises(ValueError, match="pitch_midi"):
        BenchmarkDatasetRequest.model_validate(
            {
                "dataset_id": "test-set",
                "version": "2",
                "task": "note_transcription",
                "license": "CC0-1.0",
                "samples": [
                    {
                        "asset_id": str(asset_id),
                        "references": [
                            {
                                "start_seconds": 0,
                                "end_seconds": 1,
                                "symbol": "C",
                            }
                        ],
                    }
                ],
            }
        )
