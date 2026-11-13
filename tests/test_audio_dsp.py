from __future__ import annotations

import json
import uuid
from pathlib import Path

import numpy as np
import pytest
import soundfile

from music_api.audio.dsp.analysis import detect_silence, generate_waveform, inspect_quality
from music_api.audio.dsp.chunking import plan_chunks
from music_api.audio.dsp.operations import (
    apply_fade,
    apply_filter,
    apply_gain,
    change_channels,
    normalize_peak,
    normalize_rms,
    resample_audio,
    trim_audio,
)
from music_api.audio.dsp.pipeline import apply_pipeline
from music_api.audio.dsp.signal import AudioSignal
from music_api.audio.dsp.worker import process_audio
from music_api.audio.toolchain import ProbeResult


def _write(path: Path, samples: np.ndarray, sample_rate: int) -> None:
    soundfile.write(path, samples, sample_rate, subtype="FLOAT")


def _probe(path: Path) -> ProbeResult:
    info = soundfile.info(path)
    return ProbeResult(
        container="wav",
        codec="pcm_f32le",
        codec_profile=None,
        duration_seconds=info.frames / info.samplerate,
        bitrate=None,
        sample_rate=info.samplerate,
        channels=info.channels,
        channel_layout="stereo" if info.channels == 2 else "mono",
        bit_depth=32,
        audio_stream_count=1,
        file_size_bytes=path.stat().st_size,
        raw_format_tags={},
        variable_bitrate=None,
    )


def test_audio_signal_uses_frames_channels_and_float_representation() -> None:
    mono = AudioSignal(np.ones(100, dtype=np.float32), 10)
    assert mono.samples.shape == (100, 1)
    assert mono.duration_seconds == 10
    with pytest.raises(ValueError, match="NaN"):
        AudioSignal(np.asarray([[float("nan")]], dtype=np.float64), 10)


def test_waveform_reports_channel_peak_rms_and_original_timestamps(tmp_path: Path) -> None:
    sample_rate = 48_000
    time = np.arange(sample_rate) / sample_rate
    samples = np.column_stack(
        (0.5 * np.sin(2 * np.pi * 440 * time), 0.25 * np.sin(2 * np.pi * 440 * time))
    )
    path = tmp_path / "tone.wav"
    _write(path, samples, sample_rate)

    result = generate_waveform(path, samples_per_second=100)

    assert result["duration_seconds"] == pytest.approx(1.0)
    assert result["channel_count"] == 2
    assert result["timestamps_seconds"][0] == pytest.approx(0.005)
    assert result["timestamps_seconds"][-1] == pytest.approx(0.995)
    assert max(result["channels"][0]["peak"]) == pytest.approx(0.5, abs=1e-4)
    assert np.mean(result["channels"][0]["rms"]) == pytest.approx(0.5 / np.sqrt(2), abs=1e-3)
    with pytest.raises(ValueError, match="maximum"):
        generate_waveform(path, samples_per_second=100, max_points=50)


def test_silence_detection_preserves_quiet_audio_and_finds_edges(tmp_path: Path) -> None:
    sample_rate = 10_000
    silence = np.zeros(3000)
    quiet_tone = 0.02 * np.sin(2 * np.pi * 440 * np.arange(4000) / sample_rate)
    samples = np.concatenate((silence, quiet_tone, silence))
    path = tmp_path / "silence.wav"
    _write(path, samples, sample_rate)

    result = detect_silence(
        path,
        threshold_dbfs=-50,
        window_ms=20,
        min_silence_ms=200,
    )

    assert len(result["regions"]) == 2
    assert result["regions"][0]["start_seconds"] == pytest.approx(0.0)
    assert result["regions"][0]["end_seconds"] == pytest.approx(0.3)
    assert result["regions"][1]["start_seconds"] == pytest.approx(0.7)
    assert result["silence_percentage"] == pytest.approx(60.0)


def test_quality_measurements_are_separate_and_mathematically_grounded(
    tmp_path: Path,
) -> None:
    samples = np.column_stack((np.full(8000, 0.25), np.full(8000, -0.5)))
    path = tmp_path / "dc.wav"
    _write(path, samples, 8000)

    result = inspect_quality(path)

    assert result["peak_amplitude"] == pytest.approx([0.25, 0.5])
    assert result["rms_amplitude"] == pytest.approx([0.25, 0.5])
    assert result["dc_offset"] == pytest.approx([0.25, -0.5])
    assert result["clipped_sample_percentage"] == 0
    assert result["integrated_loudness_lufs"] is None
    assert result["true_peak_dbfs"] is None


def test_sample_accurate_trim_gain_and_normalization() -> None:
    sample_rate = 10
    samples = np.column_stack(
        (np.arange(10, dtype=np.float64) / 20, np.arange(10, dtype=np.float64) / 40)
    )
    signal_data = AudioSignal(samples, sample_rate)

    trimmed = trim_audio(signal_data, 0.2, 0.8)
    gained, report = apply_gain(trimmed, gain_db=6.020599913)
    normalized, normalization = normalize_peak(gained, target_dbfs=-6)
    rms_normalized, rms_report = normalize_rms(signal_data, target_dbfs=-12, ceiling_dbfs=-1)

    assert trimmed.samples.shape == (6, 2)
    assert trimmed.samples[0, 0] == pytest.approx(0.1)
    assert report["clipping_prevented"] is False
    assert np.max(np.abs(normalized.samples)) == pytest.approx(10 ** (-6 / 20), abs=1e-8)
    assert normalization["output_peak_dbfs"] == pytest.approx(-6)
    assert rms_report["output_rms_dbfs"] <= -1
    assert np.max(np.abs(rms_normalized.samples)) <= 10 ** (-1 / 20) + 1e-12
    silent, silence_report = normalize_peak(AudioSignal(np.zeros((10, 1)), 10))
    assert np.count_nonzero(silent.samples) == 0
    assert silence_report["input_peak_dbfs"] is None


def test_fades_channels_resampling_and_filters() -> None:
    sample_rate = 8000
    time = np.arange(sample_rate) / sample_rate
    left = np.sin(2 * np.pi * 100 * time)
    right = np.sin(2 * np.pi * 1000 * time)
    stereo = AudioSignal(np.column_stack((left, right)), sample_rate)

    faded = apply_fade(stereo, fade_in_seconds=0.1, fade_out_seconds=0.1)
    mono = change_channels(stereo, "mono")
    swapped = change_channels(stereo, "swap")
    converted = resample_audio(stereo, 16_000)
    filtered = apply_filter(stereo, "lowpass", cutoff_hz=200, order=4)

    assert faded.samples[0].tolist() == [0.0, 0.0]
    assert faded.samples[-1].tolist() == pytest.approx([0.0, 0.0], abs=1e-12)
    assert mono.channel_count == 1
    assert swapped.samples[0].tolist() == pytest.approx(stereo.samples[0, ::-1])
    assert converted.sample_rate == 16_000
    assert converted.frame_count == 16_000
    assert np.std(filtered.samples[:, 1]) < 0.02
    with pytest.raises(ValueError, match="Nyquist"):
        apply_filter(stereo, "lowpass", cutoff_hz=4000)


def test_chunk_planning_keeps_offsets_and_overlap_reconstructable() -> None:
    source_id = uuid.uuid4()
    chunks = plan_chunks(
        source_id,
        frame_count=1000,
        sample_rate=100,
        chunk_seconds=4,
        overlap_seconds=1,
        min_chunk_seconds=2,
    )

    assert [(chunk.start_sample, chunk.end_sample) for chunk in chunks] == [
        (0, 400),
        (300, 700),
        (600, 1000),
    ]
    assert all(chunk.source_asset_id == source_id for chunk in chunks)
    assert chunks[1].start_seconds == 3
    assert chunks[1].duration_seconds == 4


def test_pipeline_applies_operations_in_order_and_reports_each_result() -> None:
    source = AudioSignal(np.full((100, 1), 0.1), 100)

    processed, reports = apply_pipeline(
        source,
        [
            {"type": "trim", "start_seconds": 0.1, "end_seconds": 0.9},
            {"type": "gain", "gain_db": 6.020599913},
            {"type": "fade", "fade_in_seconds": 0.1},
        ],
    )

    assert processed.frame_count == 80
    assert processed.samples[0, 0] == 0
    assert processed.samples[10, 0] == pytest.approx(0.2, abs=1e-8)
    assert [report["type"] for report in reports] == ["trim", "gain", "fade"]


@pytest.mark.asyncio
async def test_worker_trims_and_generates_stored_json_artifact(tmp_path: Path) -> None:
    source = tmp_path / "source.wav"
    sample_rate = 8000
    samples = np.column_stack(
        (
            np.sin(2 * np.pi * 440 * np.arange(sample_rate) / sample_rate),
            np.sin(2 * np.pi * 220 * np.arange(sample_rate) / sample_rate),
        )
    )
    _write(source, samples, sample_rate)
    probe = ProbeResult(
        container="wav",
        codec="pcm_f32le",
        codec_profile=None,
        duration_seconds=1.0,
        bitrate=None,
        sample_rate=sample_rate,
        channels=2,
        channel_layout="stereo",
        bit_depth=32,
        audio_stream_count=1,
        file_size_bytes=source.stat().st_size,
        raw_format_tags={},
        variable_bitrate=None,
    )

    async def report(_fraction: float) -> bool:
        return True

    trim_result = await process_audio(
        source,
        tmp_path,
        source_asset_id=uuid.uuid4(),
        operation="trim",
        options={"start_seconds": 0.25, "end_seconds": 0.75},
        probe=probe,
        progress=report,
    )
    waveform_result = await process_audio(
        source,
        tmp_path,
        source_asset_id=uuid.uuid4(),
        operation="waveform",
        options={"samples_per_second": 10, "max_points": 100},
        probe=probe,
        progress=report,
    )
    trimmed, trim_rate = soundfile.read(trim_result.outputs[0].path, always_2d=True)
    waveform = json.loads(waveform_result.outputs[0].path.read_text(encoding="utf-8"))

    assert trim_rate == sample_rate
    assert trimmed.shape == (4000, 2)
    assert waveform["duration_seconds"] == pytest.approx(1.0)
    assert waveform_result.outputs[0].asset_type == "waveform"


@pytest.mark.asyncio
async def test_worker_prepares_analysis_chunks_and_preserves_source_offsets(
    tmp_path: Path,
) -> None:
    sample_rate = 8000
    source = tmp_path / "source.wav"
    samples = np.column_stack(
        (
            np.sin(2 * np.pi * 440 * np.arange(sample_rate) / sample_rate),
            np.sin(2 * np.pi * 220 * np.arange(sample_rate) / sample_rate),
        )
    )
    _write(source, samples, sample_rate)
    source_id = uuid.uuid4()

    async def report(_fraction: float) -> bool:
        return True

    result = await process_audio(
        source,
        tmp_path,
        source_asset_id=source_id,
        operation="chunks",
        options={
            "chunk_seconds": 0.5,
            "overlap_seconds": 0.1,
            "sample_rate": 16000,
            "channels": "mono",
            "profile_id": "analysis-v1",
        },
        probe=_probe(source),
        progress=report,
    )

    first, first_rate = soundfile.read(result.outputs[0].path, always_2d=True)
    assert result.result["chunk_count"] == 3
    assert first.shape == (8000, 1)
    assert first_rate == 16000
    assert result.outputs[1].metadata["start_sample"] == 3200
    assert result.outputs[0].metadata["processing_config"]["profile_id"] == "analysis-v1"


@pytest.mark.asyncio
async def test_worker_concatenates_pcm_with_sample_accurate_crossfade(tmp_path: Path) -> None:
    sample_rate = 8000
    first = tmp_path / "first.wav"
    second = tmp_path / "second.wav"
    _write(first, np.full((4000, 1), 0.2), sample_rate)
    _write(second, np.full((4000, 1), 0.4), sample_rate)
    source_id = uuid.uuid4()

    async def report(_fraction: float) -> bool:
        return True

    result = await process_audio(
        first,
        tmp_path,
        source_asset_id=source_id,
        operation="concatenate",
        options={
            "asset_ids": [str(source_id), str(uuid.uuid4())],
            "_source_paths": [str(first), str(second)],
            "crossfade_seconds": 0.1,
        },
        probe=_probe(first),
        progress=report,
    )
    output, output_rate = soundfile.read(result.outputs[0].path, always_2d=True)

    assert output_rate == sample_rate
    assert output.shape == (7200, 1)
    assert result.result["duration_seconds"] == pytest.approx(0.9)
    phase = 400 / 799 * np.pi / 2
    expected = 0.2 * np.cos(phase) + 0.4 * np.sin(phase)
    assert output[3600, 0] == pytest.approx(expected, abs=1e-6)
