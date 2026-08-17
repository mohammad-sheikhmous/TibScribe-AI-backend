"""Checkpoint integrity guard (IMPLEMENTATION.md P0-08).

Runs without torch/transformers — validate_checkpoint() is pure logic.
"""
import pytest

from app.core.nlp.checkpoint import (
    CheckpointFacts,
    CheckpointMismatchError,
    validate_checkpoint,
)


def _healthy(**overrides) -> CheckpointFacts:
    base = dict(
        config_num_classes=21,
        mapping_num_labels=21,
        head_out_features=21,
        trained_preprocessing="manual",
        runtime_preprocessing="manual",
        config_model_name="aubmindlab/bert-base-arabertv02",
        runtime_model_name="aubmindlab/bert-base-arabertv02",
        has_config=True,
        has_weights=True,
        has_mapping=True,
        has_local_backbone=True,
        has_local_tokenizer=True,
    )
    base.update(overrides)
    return CheckpointFacts(**base)


def test_healthy_checkpoint_passes_silently():
    assert validate_checkpoint(_healthy(), strict=True) == []


# --- class-count disagreements: fatal in BOTH modes -------------------------------

@pytest.mark.parametrize("strict", [False, True])
def test_head_vs_mapping_mismatch_is_always_fatal(strict):
    """The real-world trap: weights trained on 22 classes, mapping rewritten to 21."""
    with pytest.raises(CheckpointMismatchError, match="class-count mismatch"):
        validate_checkpoint(_healthy(head_out_features=22), strict=strict)


@pytest.mark.parametrize("strict", [False, True])
def test_config_vs_mapping_mismatch_is_always_fatal(strict):
    with pytest.raises(CheckpointMismatchError):
        validate_checkpoint(_healthy(config_num_classes=22), strict=strict)


def test_error_lists_every_inconsistency_at_once():
    with pytest.raises(CheckpointMismatchError) as exc:
        validate_checkpoint(_healthy(head_out_features=22, config_num_classes=20))
    # head!=mapping, head!=config, mapping!=config
    assert exc.value.args[0].count("class-count mismatch") == 3


def test_partial_facts_do_not_false_positive():
    """Unknown values (None) must never be treated as a mismatch."""
    facts = _healthy(head_out_features=None, config_num_classes=None)
    assert validate_checkpoint(facts, strict=True) == []


# --- trust-level problems: warn by default, fatal under STRICT_MODEL_CHECKS -------

def test_preprocessing_skew_warns_then_blocks_in_strict():
    facts = _healthy(trained_preprocessing="manual", runtime_preprocessing="arabert")
    warnings = validate_checkpoint(facts, strict=False)
    assert any("preprocessing skew" in w for w in warnings)
    with pytest.raises(CheckpointMismatchError, match="preprocessing skew"):
        validate_checkpoint(facts, strict=True)


def test_missing_weights_warns_then_blocks_in_strict():
    facts = _healthy(has_weights=False)
    assert any("UNTRAINED" in w for w in validate_checkpoint(facts, strict=False))
    with pytest.raises(CheckpointMismatchError, match="UNTRAINED"):
        validate_checkpoint(facts, strict=True)


def test_missing_mapping_warns_then_blocks_in_strict():
    facts = _healthy(has_mapping=False)
    assert validate_checkpoint(facts, strict=False)
    with pytest.raises(CheckpointMismatchError, match="label_mapping.json"):
        validate_checkpoint(facts, strict=True)


# --- informational drift: never fatal --------------------------------------------

def test_missing_local_backbone_is_warning_only_even_in_strict():
    """Today's model_output/ state: weights present, bert/ + tokenizer/ absent."""
    facts = _healthy(has_local_backbone=False, has_local_tokenizer=False)
    warnings = validate_checkpoint(facts, strict=True)  # must not raise
    assert any("HF Hub" in w for w in warnings)


def test_model_name_drift_is_warning_only():
    facts = _healthy(runtime_model_name="some/other-model")
    warnings = validate_checkpoint(facts, strict=True)
    assert any("model name drift" in w for w in warnings)


def test_missing_model_config_blocks_strict_loading():
    facts = _healthy(has_config=False)
    assert any("model_config.json" in w for w in validate_checkpoint(facts, strict=False))
    with pytest.raises(CheckpointMismatchError, match="model_config.json"):
        validate_checkpoint(facts, strict=True)
