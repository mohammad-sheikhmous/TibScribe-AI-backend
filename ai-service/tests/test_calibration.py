"""Confidence calibration + uncertainty maths (IMPLEMENTATION.md P4-03 / P4-06).

Synthetic logits with a known amount of overconfidence: that lets the tests assert the
direction and magnitude of the correction, which real data could not do repeatably.
"""
import numpy as np
import pytest

from app.core.nlp.calibration import (
    calibrate,
    expected_calibration_error,
    fit_temperature,
    negative_log_likelihood,
    softmax,
    threshold_for_precision,
)
from app.core.nlp.uncertainty import (
    TrainStats,
    auroc,
    mahalanobis_distance,
    mc_dropout_stats,
    ood_score,
    predictive_entropy,
    review_priority,
)

rng = np.random.default_rng(11)
N_CLASSES = 20


def overconfident_logits(n=800, sharpness=6.0, accuracy=0.75):
    """Logits that are right `accuracy` of the time but claim far more certainty.

    Two properties are modelled deliberately, because both matter:
      * average claimed confidence exceeds accuracy -> temperature must rise above 1;
      * confidence still correlates with correctness (wrong answers get a smaller
        margin) -> a precision threshold exists. A generator where right and wrong
        predictions are equally confident has no usable threshold at all, and an
        earlier version of this fixture made exactly that mistake.
    """
    labels = rng.integers(0, N_CLASSES, size=n)
    logits = rng.normal(0, 0.5, size=(n, N_CLASSES))
    for i, label in enumerate(labels):
        if rng.random() < accuracy:
            logits[i, label] += sharpness
        else:
            wrong = (label + 1 + rng.integers(0, N_CLASSES - 1)) % N_CLASSES
            logits[i, wrong] += sharpness * 0.6
    return logits, labels


# --- temperature scaling ---------------------------------------------------------------

def test_overconfidence_is_corrected_upward_in_temperature():
    logits, labels = overconfident_logits()
    temperature = fit_temperature(logits, labels)
    assert temperature > 1.0  # >1 softens an overconfident model


def test_calibration_never_makes_calibration_worse():
    """The guard: three candidate temperatures are compared and the best kept, so the
    step can only help. NLL-optimal is not always ECE-optimal."""
    logits, labels = overconfident_logits()
    result = calibrate(logits, labels)
    assert result.ece_after <= result.ece_before + 1e-9


def test_pure_temperature_distortion_is_largely_undone():
    """The textbook case: one honest distribution, then scaled logits that overstate it.

    Labels are SAMPLED from the honest probabilities, so the honest model is perfectly
    calibrated by construction and the only error is the scaling. (Taking the argmax
    instead would make the model 100% accurate — and a model that is never wrong
    *should* be maximally confident, so there would be no overconfidence to undo.)
    """
    honest = rng.normal(0, 1.2, size=(1500, N_CLASSES))
    probs = softmax(honest, 1.0)
    labels = np.array([rng.choice(N_CLASSES, p=row) for row in probs])
    distorted = honest * 3.0  # the model shouts an opinion it does not hold that firmly

    result = calibrate(distorted, labels)
    assert result.temperature > 2.0          # recovers roughly the factor applied
    assert result.ece_after < result.ece_before / 2


def test_calibration_never_changes_a_prediction():
    """The property that makes this safe: accuracy is untouched, only the claim."""
    logits, labels = overconfident_logits()
    result = calibrate(logits, labels)
    before = softmax(logits, 1.0).argmax(axis=1)
    after = softmax(logits, result.temperature).argmax(axis=1)
    assert np.array_equal(before, after)


def test_temperature_minimises_negative_log_likelihood():
    logits, labels = overconfident_logits()
    best = fit_temperature(logits, labels)
    at_best = negative_log_likelihood(logits, labels, best)
    for other in (best * 0.5, best * 1.5, 1.0):
        assert at_best <= negative_log_likelihood(logits, labels, other) + 1e-6


def test_already_calibrated_model_is_left_alone():
    logits = rng.normal(0, 1.0, size=(500, N_CLASSES))
    labels = softmax(logits).argmax(axis=1)  # perfectly consistent
    assert fit_temperature(logits, labels) < 1.5


def test_ece_is_zero_for_a_perfectly_calibrated_toy_case():
    probs = np.array([[0.5, 0.5]] * 100)
    labels = np.array([0, 1] * 50)
    assert expected_calibration_error(probs, labels) < 0.02


def test_mismatched_shapes_are_rejected():
    with pytest.raises(ValueError):
        fit_temperature(np.zeros((10, 5)), np.zeros(3))


# --- the threshold that replaces 0.5 ----------------------------------------------------

def test_threshold_is_derived_from_observed_precision():
    logits, labels = overconfident_logits(n=1500, accuracy=0.8)
    result = calibrate(logits, labels)
    threshold = result.threshold_90
    assert threshold is not None

    probs = softmax(logits, result.temperature)
    accepted = probs.max(axis=1) >= threshold
    precision = (probs.argmax(axis=1) == labels)[accepted].mean()
    assert precision >= 0.88  # the promised precision holds on the data it was fitted on


def test_threshold_is_none_when_the_model_is_too_weak():
    """An honest refusal beats returning 1.0 and pretending."""
    logits = rng.normal(0, 0.1, size=(400, N_CLASSES))
    labels = rng.integers(0, N_CLASSES, size=400)  # pure noise
    assert threshold_for_precision(softmax(logits), labels) is None


# --- entropy + MC dropout -----------------------------------------------------------------

def test_entropy_is_high_for_a_flat_distribution_and_low_for_a_peak():
    flat = np.full((1, N_CLASSES), 1 / N_CLASSES)
    peaked = np.zeros((1, N_CLASSES))
    peaked[0, 3] = 1.0
    assert predictive_entropy(flat)[0] == pytest.approx(1.0, abs=1e-6)
    assert predictive_entropy(peaked)[0] == pytest.approx(0.0, abs=1e-6)


def test_mc_dropout_flags_a_wobbling_prediction():
    stable = np.stack([np.array([[0.9, 0.1]])] * 10)
    wobbling = np.stack([
        np.array([[0.9, 0.1]]), np.array([[0.2, 0.8]]), np.array([[0.6, 0.4]]),
        np.array([[0.3, 0.7]]), np.array([[0.8, 0.2]]),
    ])
    assert mc_dropout_stats(stable)["variance"][0] < 1e-9
    assert mc_dropout_stats(wobbling)["variance"][0] > 0.01
    assert mc_dropout_stats(wobbling)["disagreement"][0] > 0.3


def test_mc_dropout_rejects_the_wrong_shape():
    with pytest.raises(ValueError):
        mc_dropout_stats(np.zeros((5, 3)))


# --- out-of-distribution ---------------------------------------------------------------

def test_mahalanobis_grows_with_distance_from_training_data():
    training = rng.normal(0, 1, size=(500, 16))
    stats = TrainStats.fit(training)
    near = mahalanobis_distance(training[:20], stats).mean()
    far = mahalanobis_distance(rng.normal(8, 1, size=(20, 16)), stats).mean()
    assert far > near * 3


def test_ood_score_separates_foreign_inputs():
    training = rng.normal(0, 1, size=(600, 16))
    stats = TrainStats.fit(training)
    in_dist = ood_score(rng.normal(0, 1, size=(100, 16)), stats)
    out_dist = ood_score(rng.normal(6, 1, size=(100, 16)), stats)

    scores = np.concatenate([out_dist, in_dist])
    is_ood = np.concatenate([np.ones(100), np.zeros(100)])
    assert auroc(scores, is_ood) > 0.95  # the P4-07 gate is 0.80


def test_train_stats_round_trip():
    stats = TrainStats.fit(rng.normal(0, 1, size=(200, 8)))
    restored = TrainStats.from_payload(stats.to_payload())
    assert restored.dim == 8
    assert np.allclose(restored.mean, stats.mean)


def test_covariance_is_invertible_even_with_few_samples():
    """More dimensions than samples — the shrinkage term is what saves this."""
    stats = TrainStats.fit(rng.normal(0, 1, size=(10, 64)))
    assert np.isfinite(mahalanobis_distance(rng.normal(0, 1, size=(3, 64)), stats)).all()


def test_auroc_is_half_for_a_useless_score():
    scores = rng.normal(size=200)
    labels = rng.integers(0, 2, size=200)
    assert 0.35 < auroc(scores, labels) < 0.65


# --- review priority ---------------------------------------------------------------------

def test_review_priority_ranks_the_unknown_above_the_merely_ambiguous():
    ambiguous = review_priority(confidence=0.55, entropy=0.9, variance=0.0001, ood=0.05)
    unfamiliar = review_priority(confidence=0.55, entropy=0.2, variance=0.02, ood=0.9)
    assert unfamiliar > ambiguous


def test_confident_familiar_prediction_has_low_priority():
    assert review_priority(0.98, 0.05, 0.0001, 0.02) < 0.1


def test_class_conditional_ood_handles_multimodal_known_classes():
    """Two distant known classes should both be in-distribution, not global OOD."""
    local_rng = np.random.default_rng(1234)
    a = local_rng.normal(-5.0, 0.6, size=(400, 12))
    b = local_rng.normal(+5.0, 0.6, size=(400, 12))
    training = np.vstack([a, b])
    labels = np.array(["a"] * len(a) + ["b"] * len(b))
    stats = TrainStats.fit(training, labels=labels)

    validation = np.vstack([
        local_rng.normal(-5.0, 0.6, size=(100, 12)),
        local_rng.normal(+5.0, 0.6, size=(100, 12)),
    ])
    stats = stats.calibrate_class_distances(validation)

    known = np.vstack([
        local_rng.normal(-5.0, 0.6, size=(50, 12)),
        local_rng.normal(+5.0, 0.6, size=(50, 12)),
    ])
    foreign = local_rng.normal(0.0, 0.6, size=(100, 12))
    known_scores = ood_score(known, stats)
    foreign_scores = ood_score(foreign, stats)

    assert np.median(known_scores) < 0.5
    assert np.median(foreign_scores) > 0.85
    assert stats.ood_version == "class_conditional_diag_v2"
    assert stats.calibration_source == "validation"


def test_class_conditional_train_stats_round_trip():
    local_rng = np.random.default_rng(55)
    training = np.vstack([
        local_rng.normal(-2, 1, size=(50, 6)),
        local_rng.normal(+2, 1, size=(50, 6)),
    ])
    labels = np.array([0] * 50 + [1] * 50)
    stats = TrainStats.fit(training, labels=labels).calibrate_class_distances(training)
    restored = TrainStats.from_payload(stats.to_payload())
    assert restored.class_means is not None
    assert restored.class_diag_precision is not None
    assert restored.class_distance_quantiles is not None
    assert np.allclose(restored.class_means, stats.class_means)
