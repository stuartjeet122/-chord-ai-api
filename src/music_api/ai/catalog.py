from __future__ import annotations

from music_api.ai.registry import ModelRegistry
from music_api.ai.schemas import (
    ExecutionDevice,
    InferenceTask,
    ModelCategory,
    ModelDefinition,
)


def build_model_registry() -> ModelRegistry:
    return ModelRegistry(
        [
            ModelDefinition(
                model_id="moss-music-8b-instruct",
                name="MOSS-Music 8B Instruct",
                provider="moss_music",
                category=ModelCategory.MUSIC_UNDERSTANDING,
                version="upstream-unpinned",
                architecture="MOSS-Audio-Encoder + modality adapter + Qwen3-8B",
                supported_tasks=(InferenceTask.MUSIC_UNDERSTANDING,),
                sample_rate=None,
                channels=None,
                max_duration_seconds=None,
                supported_devices=(ExecutionDevice.CUDA,),
                license="Apache-2.0 (project; verify exact weight terms)",
                download_source="https://huggingface.co/OpenMOSS-Team/MOSS-Music-8B-Instruct",
                weight_format="safetensors",
                commercial_use=(
                    "Not approved: verify checkpoint and training-data terms; "
                    "chord transcription accuracy is unbenchmarked."
                ),
                enabled=False,
            ),
            ModelDefinition(
                model_id="music-x-lab-ismir2019-chord",
                name="Large-Vocabulary Chord Transcription (ISMIR 2019)",
                provider="music_x_lab_chord",
                category=ModelCategory.CHORD_RECOGNITION,
                version="upstream-unpinned",
                architecture="Chord structure decomposition network with HMM decoding",
                supported_tasks=(InferenceTask.CHORD_RECOGNITION,),
                supported_devices=(),
                license="MIT (source code; checkpoint terms unverified)",
                download_source=(
                    "https://github.com/music-x-lab/ISMIR2019-Large-Vocabulary-Chord-Recognition"
                ),
                weight_format="PyTorch checkpoint (exact format/terms require review)",
                commercial_use=(
                    "Not approved: pretrained checkpoint terms, runtime compatibility, "
                    "and target accuracy require verification."
                ),
                enabled=False,
            ),
            ModelDefinition(
                model_id="spotify-basic-pitch",
                name="Spotify Basic Pitch",
                provider="basic_pitch",
                category=ModelCategory.NOTE_TRANSCRIPTION,
                version="upstream-unpinned",
                architecture="Lightweight polyphonic note transcription neural network",
                supported_tasks=(InferenceTask.NOTE_TRANSCRIPTION,),
                supported_devices=(ExecutionDevice.CPU,),
                license="Apache-2.0",
                download_source="https://github.com/spotify/basic-pitch",
                weight_format="ONNX / TensorFlow / TFLite / CoreML",
                commercial_use=(
                    "Code/model repository is Apache-2.0; verify package/weight provenance "
                    "and supported Python runtime before deployment."
                ),
                enabled=False,
            ),
            ModelDefinition(
                model_id="demucs-htdemucs",
                name="Demucs HTDemucs",
                provider="demucs",
                category=ModelCategory.SOURCE_SEPARATION,
                version="upstream-unpinned",
                architecture="Hybrid spectrogram/waveform U-Net with cross-domain Transformer",
                supported_tasks=(InferenceTask.SOURCE_SEPARATION,),
                supported_devices=(ExecutionDevice.CPU, ExecutionDevice.CUDA),
                license="MIT (source code; checkpoint terms unverified)",
                download_source="https://github.com/facebookresearch/demucs",
                weight_format="PyTorch checkpoint",
                commercial_use=(
                    "Not approved: verify exact checkpoint terms and compatible "
                    "PyTorch/torchaudio environment."
                ),
                enabled=False,
            ),
        ]
    )
