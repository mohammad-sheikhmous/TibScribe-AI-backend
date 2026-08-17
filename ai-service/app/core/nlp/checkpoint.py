"""Checkpoint integrity validation (IMPLEMENTATION.md P0-08 / P0-09).

A silent mismatch between `best_model.pt`, `label_mapping.json` and
`model_config.json` produces a report where EVERY label is wrong, with no error and
full confidence. That is the worst possible failure mode for a clinical tool, so it
is turned into a loud, fail-fast error here.

This module is deliberately **pure logic** — no torch, no transformers, no file I/O —
for the same reason as the rest of `app/core/`: it can be unit-tested in an
environment without the heavy ML stack installed (`tests/test_checkpoint.py`).
`classifier.py` collects the facts and calls `validate_checkpoint()`.

Severity model
--------------
FATAL (always)      shape/count disagreements — the checkpoint cannot be served at all.
FATAL (strict only) trained/runtime preprocessing skew, missing fine-tuned weights,
                    missing label mapping — servable but not trustworthy.
WARN                informational drift that the loader already resolves.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


class CheckpointMismatchError(RuntimeError):
    """Raised when a checkpoint cannot be trusted to produce correct labels."""


@dataclass(frozen=True)
class CheckpointFacts:
    """What the loader observed on disk, normalized into plain values.

    All fields are optional because a partially-populated `model_output/` is exactly
    the situation this check exists to catch.
    """

    config_num_classes: Optional[int] = None  # model_config.json -> num_classes
    mapping_num_labels: Optional[int] = None  # label_mapping.json -> len(id2label)
    head_out_features: Optional[int] = None   # best_model.pt -> classifier.weight.shape[0]
    trained_preprocessing: Optional[str] = None  # model_config.json -> preprocessing
    runtime_preprocessing: Optional[str] = None  # normalize.get_preprocessing_mode()
    config_model_name: Optional[str] = None
    runtime_model_name: Optional[str] = None
    has_config: bool = False                  # model_config.json present
    has_weights: bool = False                 # best_model.pt present
    has_mapping: bool = False                 # label_mapping.json present
    has_local_backbone: bool = False          # model_output/bert/ present
    has_local_tokenizer: bool = False         # model_output/tokenizer/ present


def _class_count_problems(f: CheckpointFacts) -> list[str]:
    """Disagreements about how many classes the model predicts. Always fatal.

    Every pair is checked independently so the error message names *all* the
    inconsistent sources, not just the first one found.
    """
    problems: list[str] = []
    pairs = (
        ("best_model.pt classifier head", f.head_out_features,
         "label_mapping.json", f.mapping_num_labels),
        ("best_model.pt classifier head", f.head_out_features,
         "model_config.json num_classes", f.config_num_classes),
        ("label_mapping.json", f.mapping_num_labels,
         "model_config.json num_classes", f.config_num_classes),
    )
    for a_name, a_val, b_name, b_val in pairs:
        if a_val is not None and b_val is not None and a_val != b_val:
            problems.append(
                f"class-count mismatch: {a_name} says {a_val} but {b_name} says {b_val}"
            )
    return problems


def validate_checkpoint(facts: CheckpointFacts, *, strict: bool = False) -> list[str]:
    """Validate a checkpoint. Returns warnings; raises on anything fatal.

    Args:
        facts: what the loader found on disk.
        strict: production mode — trust-level problems become fatal too.

    Raises:
        CheckpointMismatchError: with every fatal problem listed at once, so a broken
            checkpoint is fixed in one pass instead of one error at a time.
    """
    fatal: list[str] = _class_count_problems(facts)
    warnings: list[str] = []

    def _flag(message: str) -> None:
        (fatal if strict else warnings).append(message)

    if not facts.has_config:
        _flag(
            "model_config.json is missing — max_len/model/preprocessing provenance is "
            "unknown, so the checkpoint cannot be served safely."
        )

    if not facts.has_mapping:
        _flag(
            "label_mapping.json is missing — predictions can only be reported as "
            "opaque class ids (class_0, class_1, ...)."
        )

    if not facts.has_weights:
        _flag(
            "best_model.pt is missing — the classification head is UNTRAINED and its "
            "output is meaningless."
        )

    trained, runtime = facts.trained_preprocessing, facts.runtime_preprocessing
    if trained and runtime and trained != runtime:
        _flag(
            f"preprocessing skew: the model was trained with '{trained}' but this "
            f"environment normalizes with '{runtime}'. Align the environment "
            f"(e.g. `pip install arabert`) or retrain."
        )

    # Informational only: the loader already prefers the config's model name, and the
    # locally-saved backbone/tokenizer, so these never change behaviour silently.
    if (
        facts.config_model_name
        and facts.runtime_model_name
        and facts.config_model_name != facts.runtime_model_name
    ):
        warnings.append(
            f"model name drift: checkpoint was trained on '{facts.config_model_name}', "
            f"settings request '{facts.runtime_model_name}'; the checkpoint wins."
        )
    if facts.has_weights and not (facts.has_local_backbone and facts.has_local_tokenizer):
        warnings.append(
            "model_output/bert/ or model_output/tokenizer/ is missing — falling back to "
            "downloading the backbone from the HF Hub (needs network at every startup). "
            "Re-run training to write them locally."
        )

    if fatal:
        raise CheckpointMismatchError(
            "Refusing to load an inconsistent checkpoint:\n  - "
            + "\n  - ".join(fatal)
            + "\nSee IMPLEMENTATION.md P0-08."
        )
    return warnings
