from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from music_api.ai.schemas import ChordPrediction, NotePrediction


@dataclass(frozen=True)
class TimedChord:
    start_seconds: float
    end_seconds: float
    symbol: str
    root: str
    quality: str
    bass: str | None = None
    inversion: str | None = None

    def __post_init__(self) -> None:
        _validate_interval(self.start_seconds, self.end_seconds)
        if not self.symbol.strip() or not self.root.strip() or not self.quality.strip():
            raise ValueError("Chord symbol, root, and quality are required.")


@dataclass(frozen=True)
class TimedNote:
    start_seconds: float
    end_seconds: float
    pitch_midi: int

    def __post_init__(self) -> None:
        _validate_interval(self.start_seconds, self.end_seconds)
        if not 0 <= self.pitch_midi <= 127:
            raise ValueError("MIDI pitch must be between 0 and 127.")


def chord_metrics(
    references: Sequence[TimedChord],
    predictions: Sequence[ChordPrediction],
    *,
    boundary_tolerance_seconds: float = 0.1,
) -> dict[str, float | int | None]:
    if not math.isfinite(boundary_tolerance_seconds) or boundary_tolerance_seconds < 0:
        raise ValueError("Boundary tolerance must be finite and non-negative.")
    if not references:
        raise ValueError("Chord benchmark requires at least one reference event.")
    overlap = _chord_overlap(references, predictions)
    total_reference_duration = sum(event.end_seconds - event.start_seconds for event in references)
    root_correct = sum(
        amount
        for (reference_index, prediction_index), amount in overlap.items()
        if references[reference_index].root.casefold()
        == (predictions[prediction_index].root or "").casefold()
    )
    quality_correct = sum(
        amount
        for (reference_index, prediction_index), amount in overlap.items()
        if references[reference_index].quality.casefold()
        == (predictions[prediction_index].quality or "").casefold()
    )
    symbol_correct = sum(
        amount
        for (reference_index, prediction_index), amount in overlap.items()
        if references[reference_index].symbol.casefold()
        == predictions[prediction_index].symbol.casefold()
    )
    duration_by_symbol: dict[str, float] = {}
    correct_by_symbol: dict[str, float] = {}
    for (reference_index, prediction_index), amount in overlap.items():
        symbol = references[reference_index].symbol
        duration_by_symbol[symbol] = duration_by_symbol.get(symbol, 0.0) + amount
        if (
            references[reference_index].symbol.casefold()
            == predictions[prediction_index].symbol.casefold()
        ):
            correct_by_symbol[symbol] = correct_by_symbol.get(symbol, 0.0) + amount
    weighted_recall = (
        sum(
            duration * correct_by_symbol.get(symbol, 0.0) / duration
            for symbol, duration in duration_by_symbol.items()
            if duration > 0
        )
        / total_reference_duration
    )
    reference_changes = [event.start_seconds for event in references[1:]]
    predicted_changes = [event.start_seconds for event in predictions[1:]]
    matched, timing_errors = _match_times(
        reference_changes,
        predicted_changes,
        boundary_tolerance_seconds,
    )
    precision = matched / len(predicted_changes) if predicted_changes else 0.0
    recall = matched / len(reference_changes) if reference_changes else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "root_accuracy": root_correct / total_reference_duration,
        "quality_accuracy": quality_correct / total_reference_duration,
        "symbol_accuracy": symbol_correct / total_reference_duration,
        "weighted_chord_recall": weighted_recall,
        "chord_change_precision": precision,
        "chord_change_recall": recall,
        "chord_change_f1": f1,
        "boundary_timing_error_mean_seconds": (
            float(np.mean(timing_errors)) if timing_errors else None
        ),
        "reference_chord_count": len(references),
        "predicted_chord_count": len(predictions),
        "extended_chord_symbol_accuracy": _subset_symbol_accuracy(
            references, predictions, overlap, _is_extended
        ),
        "slash_chord_symbol_accuracy": _subset_symbol_accuracy(
            references, predictions, overlap, lambda event: event.bass is not None
        ),
        "inversion_symbol_accuracy": _subset_symbol_accuracy(
            references, predictions, overlap, lambda event: event.inversion is not None
        ),
    }


def note_metrics(
    references: Sequence[TimedNote],
    predictions: Sequence[NotePrediction],
    *,
    onset_tolerance_seconds: float = 0.05,
    offset_tolerance_seconds: float = 0.05,
) -> dict[str, float | int | None]:
    for tolerance in (onset_tolerance_seconds, offset_tolerance_seconds):
        if not math.isfinite(tolerance) or tolerance < 0:
            raise ValueError("Note timing tolerances must be finite and non-negative.")
    matches = _match_notes(references, predictions, onset_tolerance_seconds)
    true_positives = len(matches)
    false_positives = len(predictions) - true_positives
    false_negatives = len(references) - true_positives
    precision = (
        true_positives / (true_positives + false_positives)
        if true_positives + false_positives
        else 0.0
    )
    recall = (
        true_positives / (true_positives + false_negatives)
        if true_positives + false_negatives
        else 0.0
    )
    onset_errors = [
        abs(references[reference_index].start_seconds - predictions[prediction_index].start_seconds)
        for reference_index, prediction_index in matches
    ]
    offset_errors = [
        abs(references[reference_index].end_seconds - predictions[prediction_index].end_seconds)
        for reference_index, prediction_index in matches
    ]
    return {
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "pitch_accuracy": recall,
        "onset_accuracy": (
            sum(error <= onset_tolerance_seconds for error in onset_errors) / len(references)
            if references
            else 0.0
        ),
        "offset_accuracy": (
            sum(error <= offset_tolerance_seconds for error in offset_errors) / len(references)
            if references
            else 0.0
        ),
        "mean_onset_error_seconds": float(np.mean(onset_errors)) if onset_errors else None,
        "mean_offset_error_seconds": float(np.mean(offset_errors)) if offset_errors else None,
        "reference_note_count": len(references),
        "predicted_note_count": len(predictions),
    }


def source_separation_metrics(
    reference: NDArray[np.float64],
    estimate: NDArray[np.float64],
) -> dict[str, float]:
    if reference.shape != estimate.shape or reference.ndim != 1 or reference.size == 0:
        raise ValueError("Reference and estimate must be non-empty, equal-length mono arrays.")
    if not np.all(np.isfinite(reference)) or not np.all(np.isfinite(estimate)):
        raise ValueError("Reference and estimate samples must be finite.")
    reference_energy = float(np.dot(reference, reference))
    if reference_energy == 0:
        raise ValueError("Reference signal must have non-zero energy.")
    error = reference - estimate
    estimate_energy = float(np.dot(estimate, estimate))
    error_energy = float(np.dot(error, error))
    scale = float(np.dot(estimate, reference) / reference_energy)
    projection = scale * reference
    residual = estimate - projection
    return {
        "sdr_db": _ratio_db(reference_energy, error_energy),
        "si_sdr_db": _ratio_db(
            float(np.dot(projection, projection)),
            float(np.dot(residual, residual)),
        ),
        "sir_db": _ratio_db(reference_energy, float(np.dot(residual, residual))),
        "sar_db": _ratio_db(estimate_energy, float(np.dot(residual, residual))),
    }


def _chord_overlap(
    references: Sequence[TimedChord],
    predictions: Sequence[ChordPrediction],
) -> dict[tuple[int, int], float]:
    overlap: dict[tuple[int, int], float] = {}
    for reference_index, reference in enumerate(references):
        for prediction_index, prediction in enumerate(predictions):
            duration = max(
                0.0,
                min(reference.end_seconds, prediction.end_seconds)
                - max(reference.start_seconds, prediction.start_seconds),
            )
            if duration:
                overlap[(reference_index, prediction_index)] = duration
    return overlap


def _match_times(
    references: Sequence[float],
    predictions: Sequence[float],
    tolerance: float,
) -> tuple[int, list[float]]:
    available = set(range(len(predictions)))
    errors: list[float] = []
    for reference in references:
        candidates = [(abs(reference - predictions[index]), index) for index in available]
        if candidates:
            error, index = min(candidates)
            if error <= tolerance:
                available.remove(index)
                errors.append(error)
    return len(errors), errors


def _match_notes(
    references: Sequence[TimedNote],
    predictions: Sequence[NotePrediction],
    tolerance: float,
) -> list[tuple[int, int]]:
    available = set(range(len(predictions)))
    matches: list[tuple[int, int]] = []
    for reference_index, reference in enumerate(references):
        candidates = [
            (
                abs(reference.start_seconds - predictions[index].start_seconds),
                index,
            )
            for index in available
            if reference.pitch_midi == predictions[index].pitch_midi
        ]
        if candidates:
            onset_error, prediction_index = min(candidates)
            if onset_error <= tolerance:
                available.remove(prediction_index)
                matches.append((reference_index, prediction_index))
    return matches


def _subset_symbol_accuracy(
    references: Sequence[TimedChord],
    predictions: Sequence[ChordPrediction],
    overlap: dict[tuple[int, int], float],
    predicate: Any,
) -> float | None:
    duration = 0.0
    correct = 0.0
    for (reference_index, prediction_index), amount in overlap.items():
        reference = references[reference_index]
        if predicate(reference):
            duration += amount
            if reference.symbol.casefold() == predictions[prediction_index].symbol.casefold():
                correct += amount
    return correct / duration if duration else None


def _is_extended(event: TimedChord) -> bool:
    lowered = event.quality.casefold()
    return any(token in lowered for token in ("7", "9", "11", "13", "add"))


def _validate_interval(start: float, end: float) -> None:
    if not math.isfinite(start) or not math.isfinite(end) or start < 0 or end <= start:
        raise ValueError("Annotation intervals must be finite and have positive duration.")


def _ratio_db(numerator: float, denominator: float) -> float:
    floor = float(np.finfo(np.float64).tiny)
    numerator_value = max(float(numerator), floor)
    denominator_value = max(float(denominator), floor)
    return 10.0 * (math.log10(numerator_value) - math.log10(denominator_value))
