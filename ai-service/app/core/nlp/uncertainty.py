"""Uncertainty and out-of-distribution scoring (IMPLEMENTATION.md P4-06).

Softmax confidence answers "which class is most likely *given that it is one of my
classes*". It cannot say "this sentence is unlike anything I was trained on" — and it
is famously confident on exactly such inputs. That is the wrong failure mode for a
clinical tool fed Whisper output, where hallucinations and off-topic chatter arrive
regularly.

Three complementary signals, none sufficient alone:

* **Entropy** — how spread the distribution is. Catches genuine ambiguity between
  known classes ("is this `plan` or `treatment`?").
* **MC-dropout variance** — run the same input several times with dropout still active
  and see whether the answer wobbles. This is *epistemic* uncertainty: what the model
  does not know, as opposed to what is inherently ambiguous. The head already carries
  `dropout=0.3`, so this costs nothing but forward passes.
* **Mahalanobis distance** — how far the sentence embedding sits from the training
  distribution. This is the only one that detects a genuinely foreign input, and it is
  the one softmax cannot express.

Pure numpy: the torch-side sampling lives in `classifier.py`, so the maths stays
testable in a bare environment.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class TrainStats:
    """Where the training data lives in embedding space (written at train time).

    ``distance_quantiles`` calibrate raw Mahalanobis distance against the model's own
    training distribution.  Older checkpoints lack these fields; loading remains
    backward compatible and falls back to the historical dimensionality heuristic.
    """

    mean: np.ndarray                 # (dim,)
    precision: np.ndarray            # (dim, dim) inverse covariance (legacy/global)
    dim: int
    distance_quantiles: Optional[dict[str, float]] = None
    # v2: class-conditional diagonal Mahalanobis.  A single global Gaussian is a
    # poor approximation for a 20-class classifier and can mark perfectly normal
    # samples as OOD simply because the representation is multimodal.
    class_labels: Optional[list[str]] = None
    class_means: Optional[np.ndarray] = None          # (classes, dim)
    class_diag_precision: Optional[np.ndarray] = None # (dim,) shared within-class
    class_distance_quantiles: Optional[dict[str, float]] = None
    ood_version: str = "global_mahalanobis_v1"
    calibration_source: Optional[str] = None

    @classmethod
    def fit(
        cls,
        embeddings: np.ndarray,
        *,
        labels: Optional[np.ndarray | list] = None,
        shrinkage: float = 1e-3,
    ) -> "TrainStats":
        """Estimate the training representation distribution.

        When labels are supplied we additionally fit a class-conditional diagonal
        Mahalanobis model.  The diagonal/shared within-class variance is deliberately
        conservative and much more stable than inverting 768x768 per-class
        covariances.  Runtime OOD is then distance to the *nearest known class*, not
        distance to one unrealistic global centroid.
        """
        embeddings = np.asarray(embeddings, dtype=np.float64)
        if embeddings.ndim != 2:
            raise ValueError("embeddings must be (n, dim)")
        mean = embeddings.mean(axis=0)
        centered = embeddings - mean
        covariance = (centered.T @ centered) / max(len(embeddings) - 1, 1)
        covariance += shrinkage * np.eye(covariance.shape[0]) * np.trace(covariance) / covariance.shape[0]
        precision = np.linalg.pinv(covariance)

        stats = cls(mean=mean, precision=precision, dim=embeddings.shape[1])
        if labels is None:
            distances = mahalanobis_distance(embeddings, stats)
            return cls(
                mean=mean, precision=precision, dim=embeddings.shape[1],
                distance_quantiles=_distance_quantiles(distances),
            )

        labels_arr = np.asarray(labels)
        if labels_arr.shape[0] != embeddings.shape[0]:
            raise ValueError("labels length must match embeddings")
        class_labels = [str(x) for x in sorted(np.unique(labels_arr), key=lambda x: str(x))]
        class_means = []
        residual_parts = []
        for label in class_labels:
            mask = np.asarray([str(x) == label for x in labels_arr], dtype=bool)
            if not mask.any():
                continue
            mu = embeddings[mask].mean(axis=0)
            class_means.append(mu)
            residual_parts.append(embeddings[mask] - mu)

        means = np.vstack(class_means)
        residuals = np.vstack(residual_parts)
        variance = residuals.var(axis=0, ddof=1)
        finite_positive = variance[np.isfinite(variance) & (variance > 0)]
        target = float(np.median(finite_positive)) if finite_positive.size else 1.0
        variance = (1.0 - shrinkage) * variance + shrinkage * target
        variance = np.clip(variance, max(target * 1e-6, 1e-12), None)
        diag_precision = 1.0 / variance

        stats = cls(
            mean=mean, precision=precision, dim=embeddings.shape[1],
            class_labels=class_labels, class_means=means,
            class_diag_precision=diag_precision,
            ood_version="class_conditional_diag_v2",
            calibration_source="training",
        )
        distances = class_conditional_distance(embeddings, stats)
        return cls(
            **{**stats.__dict__, "class_distance_quantiles": _distance_quantiles(distances)}
        )

    def calibrate_class_distances(
        self, embeddings: np.ndarray, *, source: str = "validation"
    ) -> "TrainStats":
        """Calibrate v2 OOD scores on held-out in-distribution embeddings."""
        if self.class_means is None or self.class_diag_precision is None:
            return self
        distances = class_conditional_distance(embeddings, self)
        return TrainStats(
            **{
                **self.__dict__,
                "class_distance_quantiles": _distance_quantiles(distances),
                "calibration_source": source,
            }
        )

    def to_payload(self) -> dict:
        payload = {
            "mean": self.mean.tolist(),
            "precision": self.precision.tolist(),
            "dim": self.dim,
        }
        if self.distance_quantiles:
            payload["distance_quantiles"] = dict(self.distance_quantiles)
        if self.class_labels and self.class_means is not None and self.class_diag_precision is not None:
            payload.update({
                "class_labels": list(self.class_labels),
                "class_means": self.class_means.tolist(),
                "class_diag_precision": self.class_diag_precision.tolist(),
                "ood_version": self.ood_version,
                "calibration_source": self.calibration_source,
            })
            if self.class_distance_quantiles:
                payload["class_distance_quantiles"] = dict(self.class_distance_quantiles)
        return payload

    @classmethod
    def from_payload(cls, payload: dict) -> "TrainStats":
        class_means = payload.get("class_means")
        class_diag_precision = payload.get("class_diag_precision")
        return cls(
            mean=np.asarray(payload["mean"], dtype=np.float64),
            precision=np.asarray(payload["precision"], dtype=np.float64),
            dim=int(payload["dim"]),
            distance_quantiles=(
                {str(k): float(v) for k, v in payload.get("distance_quantiles", {}).items()}
                or None
            ),
            class_labels=[str(x) for x in payload.get("class_labels", [])] or None,
            class_means=(np.asarray(class_means, dtype=np.float64) if class_means is not None else None),
            class_diag_precision=(
                np.asarray(class_diag_precision, dtype=np.float64)
                if class_diag_precision is not None else None
            ),
            class_distance_quantiles=(
                {str(k): float(v) for k, v in payload.get("class_distance_quantiles", {}).items()}
                or None
            ),
            ood_version=str(payload.get("ood_version", "global_mahalanobis_v1")),
            calibration_source=payload.get("calibration_source"),
        )

def _distance_quantiles(distances: np.ndarray) -> dict[str, float]:
    distances = np.asarray(distances, dtype=np.float64)
    return {
        "p50": float(np.quantile(distances, 0.50)),
        "p90": float(np.quantile(distances, 0.90)),
        "p95": float(np.quantile(distances, 0.95)),
        "p99": float(np.quantile(distances, 0.99)),
        "p995": float(np.quantile(distances, 0.995)),
    }


def class_conditional_distance(embeddings: np.ndarray, stats: TrainStats) -> np.ndarray:
    """Distance to the nearest known class under shared within-class variance."""
    if stats.class_means is None or stats.class_diag_precision is None:
        return mahalanobis_distance(embeddings, stats)
    x = np.atleast_2d(np.asarray(embeddings, dtype=np.float64))
    # (n, classes, dim); datasets here are small enough that this is cheap and it
    # avoids coupling OOD to whichever class softmax happened to choose.
    delta = x[:, None, :] - stats.class_means[None, :, :]
    squared = np.sum(delta * delta * stats.class_diag_precision[None, None, :], axis=2)
    return np.sqrt(np.maximum(squared.min(axis=1), 0.0))


def predictive_entropy(probs: np.ndarray, *, normalized: bool = True) -> np.ndarray:
    """Shannon entropy per row; normalized to 0..1 so it is comparable across models."""
    probs = np.atleast_2d(np.asarray(probs, dtype=np.float64))
    entropy = -np.sum(probs * np.log(np.clip(probs, 1e-12, None)), axis=1)
    if normalized:
        entropy = entropy / np.log(probs.shape[1])
    return entropy


def mc_dropout_stats(samples: np.ndarray) -> dict[str, np.ndarray]:
    """Summarise repeated stochastic forward passes.

    `samples` is (passes, n, classes). `variance` is the mean per-class variance —
    high when the prediction depends on which neurons happened to be dropped, i.e. the
    model is guessing. `disagreement` is the fraction of passes that did NOT vote for
    the majority class, which is the same idea in a form a human can read.
    """
    samples = np.asarray(samples, dtype=np.float64)
    if samples.ndim != 3:
        raise ValueError("samples must be (passes, n, classes)")
    mean = samples.mean(axis=0)
    variance = samples.var(axis=0).mean(axis=1)

    votes = samples.argmax(axis=2)                      # (passes, n)
    majority = mean.argmax(axis=1)
    disagreement = (votes != majority[None, :]).mean(axis=0)
    return {"mean_probs": mean, "variance": variance, "disagreement": disagreement}


def mahalanobis_distance(embeddings: np.ndarray, stats: TrainStats) -> np.ndarray:
    """Distance from the training distribution, in units that account for its shape."""
    embeddings = np.atleast_2d(np.asarray(embeddings, dtype=np.float64))
    centered = embeddings - stats.mean
    return np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", centered, stats.precision, centered), 0.0))


def ood_score(
    embeddings: np.ndarray, stats: TrainStats, *, reference_distance: Optional[float] = None
) -> np.ndarray:
    """Map Mahalanobis distance to 0..1 using empirical training calibration.

    New checkpoints store distance quantiles from the training embeddings.  A score
    around 0.5 therefore corresponds roughly to the upper 5% of training distances,
    0.85 to ~p99 and 1.0 to ~p99.5+.  This is far more meaningful than assuming every
    768-dimensional embedding cloud is a perfect unit Gaussian.

    Older checkpoints remain supported through the historical sqrt(dim) fallback.
    """
    use_class_conditional = (
        stats.class_means is not None
        and stats.class_diag_precision is not None
        and stats.class_distance_quantiles is not None
    )
    distances = (
        class_conditional_distance(embeddings, stats)
        if use_class_conditional
        else mahalanobis_distance(embeddings, stats)
    )
    q = (stats.class_distance_quantiles if use_class_conditional else stats.distance_quantiles) or {}
    required = ("p50", "p90", "p95", "p99", "p995")
    if all(k in q for k in required):
        xp = np.asarray([q[k] for k in required], dtype=np.float64)
        # Degenerate quantiles are possible only on pathological tiny datasets.
        if np.all(np.diff(xp) > 0):
            fp = np.asarray([0.0, 0.25, 0.50, 0.85, 1.0], dtype=np.float64)
            return np.interp(distances, xp, fp, left=0.0, right=1.0)

    scale = reference_distance or np.sqrt(stats.dim)
    return np.clip(distances / (2.0 * scale), 0.0, 1.0)

def auroc(scores: np.ndarray, is_positive: np.ndarray) -> float:
    """Rank-based AUROC — the metric for "does this score separate OOD from normal?".

    Implemented directly (Mann-Whitney U) to keep the module dependency-free.
    """
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(is_positive).astype(bool)
    positives, negatives = scores[labels], scores[~labels]
    if len(positives) == 0 or len(negatives) == 0:
        return float("nan")

    order = np.argsort(np.concatenate([positives, negatives]), kind="mergesort")
    ranks = np.empty(len(order), dtype=np.float64)
    ranks[order] = np.arange(1, len(order) + 1)
    # average ranks for ties, otherwise identical scores skew the statistic
    all_scores = np.concatenate([positives, negatives])
    for value in np.unique(all_scores):
        tied = all_scores == value
        if tied.sum() > 1:
            ranks[tied] = ranks[tied].mean()

    rank_sum = ranks[: len(positives)].sum()
    u = rank_sum - len(positives) * (len(positives) + 1) / 2
    return float(u / (len(positives) * len(negatives)))


@dataclass(frozen=True)
class UncertaintyScores:
    confidence: float          # calibrated max probability
    entropy: float             # 0..1
    variance: float            # MC-dropout epistemic uncertainty
    ood: float                 # 0..1
    review_priority: float     # combined ranking score for active learning

    def to_dict(self) -> dict:
        return {
            "confidence": round(self.confidence, 4),
            "entropy": round(self.entropy, 4),
            "mc_variance": round(self.variance, 6),
            "ood_score": round(self.ood, 4),
            "review_priority": round(self.review_priority, 4),
        }


def review_priority(
    confidence: float, entropy: float, variance: float, ood: float
) -> float:
    """How much a human reviewing THIS sentence would teach the model (P7-04).

    Weighted toward what the model does not know (variance, OOD) over what is simply
    ambiguous (entropy): an inherently borderline sentence stays borderline however
    many times it is labelled.
    """
    return float(
        0.35 * (1.0 - confidence)
        + 0.20 * entropy
        + 0.25 * min(1.0, variance * 50)
        + 0.20 * ood
    )
