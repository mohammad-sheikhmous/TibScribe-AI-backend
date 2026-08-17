"""Confidence calibration (IMPLEMENTATION.md P4-03).

The problem this fixes: `confidence` was a raw softmax score, and the "low confidence"
threshold was the round number 0.5. Neural classifiers are systematically
overconfident, and this one is trained with label smoothing, which distorts the scores
further. So 0.5 meant nothing in particular — yet 29% of a real report was flagged by it.

Temperature scaling divides the logits by a single scalar T fitted on the validation
set. It cannot change any prediction (dividing by a positive constant preserves the
argmax), only how sure the model claims to be — which is exactly the property wanted:
accuracy is untouched, the number becomes honest.

Then `threshold_for_precision` replaces the guess: it reports the confidence level at
which the accepted predictions are actually right 90% of the time. That is a number a
clinician can reason about.

Pure numpy so it stays testable without torch, and so the same code runs at train time
(fitting) and at serve time (applying).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

DEFAULT_BINS = 15
SEARCH_RANGE = (0.05, 10.0)  # T below 1 sharpens, above 1 softens


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    scaled = np.asarray(logits, dtype=np.float64) / max(temperature, 1e-6)
    scaled -= scaled.max(axis=-1, keepdims=True)  # stability
    exp = np.exp(scaled)
    return exp / exp.sum(axis=-1, keepdims=True)


def negative_log_likelihood(logits: np.ndarray, labels: np.ndarray, temperature: float) -> float:
    probs = softmax(logits, temperature)
    picked = probs[np.arange(len(labels)), labels]
    return float(-np.mean(np.log(np.clip(picked, 1e-12, None))))


def expected_calibration_error(
    probs: np.ndarray, labels: np.ndarray, n_bins: int = DEFAULT_BINS
) -> float:
    """Mean gap between claimed confidence and observed accuracy, weighted by bin size.

    ECE = 0 means "when it says 80%, it is right 80% of the time".
    """
    probs = np.asarray(probs, dtype=np.float64)
    confidence = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    correct = (predictions == np.asarray(labels)).astype(np.float64)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    error = 0.0
    for low, high in zip(edges[:-1], edges[1:]):
        in_bin = (confidence > low) & (confidence <= high)
        if not in_bin.any():
            continue
        weight = in_bin.mean()
        error += weight * abs(correct[in_bin].mean() - confidence[in_bin].mean())
    return float(error)


def fit_temperature(
    logits: np.ndarray,
    labels: np.ndarray,
    *,
    iterations: int = 60,
    objective: str = "nll",
) -> float:
    """Find the best T by ternary search on a single scalar.

    One parameter over a bounded objective — a search is more robust here than gradient
    descent, needs no optimiser, and keeps this module dependency-free.

    `objective="nll"` is the textbook choice, but NLL and ECE are not the same target:
    when miscalibration is not a pure temperature effect (e.g. the model is confident
    in two distinct regimes), the NLL-optimal T can leave ECE unchanged or slightly
    worse. `calibrate()` therefore fits both and keeps whichever actually calibrates.
    """
    logits = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels)
    if logits.ndim != 2 or len(logits) != len(labels):
        raise ValueError("logits must be (n, classes) and align with labels")

    if objective == "nll":
        def cost(t: float) -> float:
            return negative_log_likelihood(logits, labels, t)
    elif objective == "ece":
        def cost(t: float) -> float:
            return expected_calibration_error(softmax(logits, t), labels)
    else:
        raise ValueError("objective must be 'nll' or 'ece'")

    low, high = SEARCH_RANGE
    for _ in range(iterations):
        third = (high - low) / 3.0
        mid_low, mid_high = low + third, high - third
        if cost(mid_low) < cost(mid_high):
            high = mid_high
        else:
            low = mid_low
    return round((low + high) / 2.0, 4)


def threshold_for_precision(
    probs: np.ndarray,
    labels: np.ndarray,
    *,
    target_precision: float = 0.90,
    min_coverage: float = 0.30,
) -> Optional[float]:
    """Lowest confidence at which accepted predictions reach `target_precision`.

    Returns None when the model cannot reach that precision at any usable coverage —
    an honest "this model is not good enough to auto-accept anything", instead of
    silently returning 1.0.
    """
    probs = np.asarray(probs, dtype=np.float64)
    confidence = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == np.asarray(labels)).astype(np.float64)

    for candidate in np.unique(np.round(confidence, 3))[::-1]:
        accepted = confidence >= candidate
        coverage = accepted.mean()
        if coverage < min_coverage:
            continue
        if correct[accepted].mean() >= target_precision:
            return float(candidate)
    return None


@dataclass(frozen=True)
class CalibrationResult:
    temperature: float
    ece_before: float
    ece_after: float
    nll_before: float
    nll_after: float
    accuracy: float
    threshold_90: Optional[float]
    n_samples: int

    @property
    def improved(self) -> bool:
        return self.ece_after < self.ece_before

    def to_dict(self) -> dict:
        return {
            "temperature": self.temperature,
            "ece_before": round(self.ece_before, 4),
            "ece_after": round(self.ece_after, 4),
            "nll_before": round(self.nll_before, 4),
            "nll_after": round(self.nll_after, 4),
            "accuracy": round(self.accuracy, 4),
            "low_confidence_threshold": self.threshold_90,
            "n_samples": self.n_samples,
            "note": (
                "T>1 means the model was overconfident and its scores were softened; "
                "predictions are unchanged."
            ),
        }


def calibrate(logits: np.ndarray, labels: np.ndarray) -> CalibrationResult:
    """Fit T and report what it changed — the record written into model_config.json.

    Guarded on purpose: three candidates (NLL-optimal, ECE-optimal, and T=1) are
    compared on ECE, and the winner is kept. A calibration step that does not improve
    calibration must not be applied — silently making the numbers *less* honest is
    worse than leaving them alone.
    """
    logits = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels)

    candidates = {
        1.0: expected_calibration_error(softmax(logits, 1.0), labels),
        fit_temperature(logits, labels, objective="nll"): None,
        fit_temperature(logits, labels, objective="ece"): None,
    }
    for temperature in list(candidates):
        if candidates[temperature] is None:
            candidates[temperature] = expected_calibration_error(
                softmax(logits, temperature), labels
            )
    # Ties go to the identity: do not scale without a reason.
    temperature = min(candidates, key=lambda t: (candidates[t], abs(t - 1.0)))

    before, after = softmax(logits, 1.0), softmax(logits, temperature)
    return CalibrationResult(
        temperature=temperature,
        ece_before=expected_calibration_error(before, labels),
        ece_after=expected_calibration_error(after, labels),
        nll_before=negative_log_likelihood(logits, labels, 1.0),
        nll_after=negative_log_likelihood(logits, labels, temperature),
        accuracy=float((before.argmax(axis=1) == labels).mean()),
        threshold_90=threshold_for_precision(after, labels),
        n_samples=len(labels),
    )
