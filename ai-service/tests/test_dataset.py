"""Data layer: de-duplication, label maps and the reproducible stratified split.

The split is the contract between training and evaluation, so its properties
(determinism, disjointness, no text leakage) are pinned here.
"""
import json

from app.core.nlp.dataset import (
    build_label_maps,
    class_distribution,
    dedupe_texts,
    find_conflicts,
    load_jsonl,
    stratified_split,
)


def _rows(*pairs):
    return [{"text": t, "label": l} for t, l in pairs]


# --- loading ---------------------------------------------------------------------

def test_load_jsonl_skips_blank_lines(tmp_path):
    p = tmp_path / "d.jsonl"
    p.write_text(
        json.dumps({"text": "صداع", "label": "symptom"}, ensure_ascii=False)
        + "\n\n"
        + json.dumps({"text": "دوخة", "label": "symptom"}, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )
    assert len(load_jsonl(p)) == 2


# --- conflicts / dedupe ----------------------------------------------------------

def test_find_conflicts_reports_only_multi_label_texts():
    rows = _rows(
        ("أكلي صحي", "nutrition"),
        ("أكلي صحي", "pregnancy_nutrition"),   # same text, different label
        ("صداع", "symptom"),
        ("صداع", "symptom"),                    # same text, same label -> not a conflict
    )
    conflicts = find_conflicts(rows)
    assert list(conflicts) == ["أكلي صحي"]
    assert sorted(conflicts["أكلي صحي"]) == ["nutrition", "pregnancy_nutrition"]


def test_dedupe_keeps_first_occurrence_and_counts_removals():
    result = dedupe_texts(_rows(("صداع", "symptom"), ("صداع", "diagnosis"), ("دوخة", "symptom")))
    assert len(result.rows) == 2
    assert result.removed == 1
    assert result.rows[0]["label"] == "symptom"  # first occurrence wins
    assert "صداع" in result.conflicts


# --- label maps ------------------------------------------------------------------

def test_label_maps_are_alphabetically_stable():
    names, label2id, id2label = build_label_maps(["symptom", "allergy", "symptom"])
    assert names == ["allergy", "symptom"]
    assert label2id == {"allergy": 0, "symptom": 1}
    assert id2label[0] == "allergy"


def test_class_distribution_sorted_desc():
    assert list(class_distribution(["a", "b", "a", "c", "a", "b"])) == ["a", "b", "c"]


# --- the split -------------------------------------------------------------------

LABELS = ["symptom"] * 100 + ["lab"] * 60 + ["allergy"] * 40


def test_split_is_disjoint_and_total():
    train, val, test = stratified_split(LABELS)
    assert set(train) & set(val) == set()
    assert set(train) & set(test) == set()
    assert set(val) & set(test) == set()
    assert len(train) + len(val) + len(test) == len(LABELS)


def test_split_is_deterministic_for_a_given_seed():
    assert stratified_split(LABELS, seed=42) == stratified_split(LABELS, seed=42)
    assert stratified_split(LABELS, seed=42) != stratified_split(LABELS, seed=7)


def test_split_preserves_class_proportions():
    train, val, test = stratified_split(LABELS)
    for part in (train, val, test):
        counts = {}
        for i in part:
            counts[LABELS[i]] = counts.get(LABELS[i], 0) + 1
        assert set(counts) == {"symptom", "lab", "allergy"}  # every class in every split
    assert 0.13 <= len(test) / len(LABELS) <= 0.17


def test_tiny_class_still_appears_in_val_and_test():
    train, val, test = stratified_split(["rare"] * 3 + ["common"] * 50)
    rare_positions = {i for i, l in enumerate(["rare"] * 3 + ["common"] * 50) if l == "rare"}
    assert rare_positions & set(val) and rare_positions & set(test)
    assert rare_positions & set(train)


def test_no_text_leakage_after_dedupe():
    """The whole point of dedupe-before-split: a text cannot be in two splits."""
    rows = dedupe_texts(_rows(*[(f"جملة {i % 40}", "symptom") for i in range(120)])).rows
    texts = [r["text"] for r in rows]
    train, val, test = stratified_split([r["label"] for r in rows])
    train_texts = {texts[i] for i in train}
    assert not (train_texts & {texts[i] for i in test})
    assert not (train_texts & {texts[i] for i in val})
