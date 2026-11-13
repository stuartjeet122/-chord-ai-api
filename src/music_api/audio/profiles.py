from __future__ import annotations

from typing import Any

CONVERSION_PROFILES: tuple[dict[str, Any], ...] = (
    {
        "id": "source_preserving_wav",
        "label": "Source-preserving WAV",
        "container": "wav",
        "codec": "pcm_s24le",
        "sample_rate": None,
        "channels": "original",
        "bit_depth": 24,
        "description": "Uncompressed PCM WAV; preserves source rate and channel layout.",
    },
    {
        "id": "music_analysis_wav",
        "label": "Music analysis WAV",
        "container": "wav",
        "codec": "pcm_s24le",
        "sample_rate": 44100,
        "channels": "original",
        "bit_depth": 24,
        "description": (
            "44.1 kHz PCM WAV; sources below 44.1 kHz remain at their native rate unless "
            "upsampling is explicitly requested."
        ),
    },
)
