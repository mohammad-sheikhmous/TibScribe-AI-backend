"""Whole-word masking used by the DAPT run (train_mlm.py, P0-06).

Skipped automatically where torch is absent, so the suite still runs in a bare env.
The collator is fed plain id sets, so no real tokenizer/model is needed.
"""
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from train_mlm import WholeWordMaskCollator  # noqa: E402

# Toy vocabulary:
#   0 = [PAD], 1 = [CLS], 2 = [SEP], 3 = [MASK]  (special)
#   10, 11, 12 = word starts;  100, 101 = "##" continuations
SPECIAL = {0, 1, 2, 3}
CONTINUATION = {100, 101}


def make(prob=0.15, seed=0):
    return WholeWordMaskCollator(
        mask_token_id=3, vocab_size=200, special_ids=SPECIAL,
        continuation_ids=CONTINUATION, mlm_probability=prob, seed=seed,
    )


def test_word_groups_bind_continuations_to_their_start():
    ids = [1, 10, 100, 101, 11, 2]  # [CLS] word(3 pieces) word [SEP]
    assert make().word_groups(ids) == [[1, 2, 3], [4]]


def test_special_tokens_are_never_grouped():
    groups = make().word_groups([1, 10, 2, 11, 0])
    assert all(pos not in {0, 2, 4} for g in groups for pos in g)


def test_continuation_after_a_special_token_starts_a_new_word():
    """A '##' piece right after [CLS] cannot continue anything — it must open a group."""
    assert make().word_groups([1, 100, 10]) == [[1], [2]]


def test_masking_keeps_whole_words_together():
    """Either every piece of a word is selected, or none is."""
    block = [1] + [10, 100, 101] * 20 + [2]
    out = make(prob=0.5, seed=7)([{"input_ids": block}])
    labels = out["labels"][0].tolist()
    for start in range(1, len(block) - 1, 3):
        picked = {labels[start + k] != -100 for k in range(3)}
        assert len(picked) == 1, f"word at {start} was partially masked"


def test_only_selected_positions_carry_labels():
    block = [1] + list(range(10, 90)) + [2]
    out = make(prob=0.15, seed=3)([{"input_ids": block}])
    labels = out["labels"][0]
    assert (labels != -100).sum().item() > 0
    # every specials position stays unlabelled
    assert labels[0].item() == -100 and labels[-1].item() == -100


def test_masking_rate_is_near_the_requested_probability():
    block = [1] + list(range(10, 120)) + [2]  # 110 single-piece words
    out = make(prob=0.15, seed=11)([{"input_ids": block}])
    masked = (out["labels"][0] != -100).sum().item()
    assert 0.10 * 110 <= masked <= 0.22 * 110


def test_batch_shapes_and_attention_mask():
    blocks = [{"input_ids": [1] + list(range(10, 40)) + [2]} for _ in range(4)]
    out = make()(blocks)
    assert out["input_ids"].shape == (4, 32)
    assert out["labels"].shape == (4, 32)
    assert out["attention_mask"].sum().item() == 4 * 32  # no padding in packed blocks


def test_unmasked_positions_are_left_untouched():
    block = [1] + list(range(10, 60)) + [2]
    out = make(prob=0.15, seed=5)([{"input_ids": block}])
    ids, labels = out["input_ids"][0].tolist(), out["labels"][0].tolist()
    for pos, (token, label) in enumerate(zip(ids, labels)):
        if label == -100:
            assert token == block[pos]
