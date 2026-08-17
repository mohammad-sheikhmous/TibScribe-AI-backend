"""Dataset loading, de-duplication and splitting — single source of truth.

Shared by `train_arabert.py` (training), `eval/eval_classifier.py` (evaluation),
`train_mlm.py` (DAPT corpus) and `scripts/check_conflicts.py`. Keeping the split in
ONE place is what guarantees a model is always scored on data it never saw: an
evaluator that re-implements the split will silently score on training rows.

Pure python on purpose — no torch, no sklearn — so it stays testable in a bare
environment (same rationale as `normalize.py` / `checkpoint.py`).
"""
from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence


@dataclass(frozen=True)
class DedupeResult:
    rows: list[dict]                       # unique-text rows, first occurrence kept
    removed: int                           # duplicate rows dropped
    conflicts: dict[str, list[str]] = field(default_factory=dict)  # text -> labels seen


def load_jsonl(path: str | Path) -> list[dict]:
    """Read a {"text": ..., "label": ...} per line file, skipping blank lines."""
    rows: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:  # pragma: no cover - corrupt input
                raise ValueError(f"{path}:{line_no}: invalid JSON — {exc}") from exc
    return rows


def find_conflicts(rows: Sequence[dict]) -> dict[str, list[str]]:
    """Texts that appear with MORE THAN ONE label — an unresolved annotation decision.

    These cap achievable accuracy: the model is asked to learn two answers for the
    same input. Resolved in P0-04/P0-05 by a documented human decision.
    """
    seen: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        label = row["label"]
        labels = seen[row["text"]]
        if label not in labels:
            labels.append(label)
    return {text: labels for text, labels in seen.items() if len(labels) > 1}


def dedupe_texts(rows: Sequence[dict]) -> DedupeResult:
    """Drop exact-duplicate texts, keeping the first occurrence.

    Must happen BEFORE splitting: an identical sentence in both train and test
    inflates test scores without any real generalization.
    """
    conflicts = find_conflicts(rows)
    seen: set[str] = set()
    kept: list[dict] = []
    for row in rows:
        text = row["text"]
        if text in seen:
            continue
        seen.add(text)
        kept.append(row)
    return DedupeResult(rows=kept, removed=len(rows) - len(kept), conflicts=conflicts)


def build_label_maps(labels: Iterable[str]) -> tuple[list[str], dict[str, int], dict[int, str]]:
    """Stable label<->id maps. Sorted alphabetically so ids never shift between runs."""
    names = sorted(set(labels))
    label2id = {name: i for i, name in enumerate(names)}
    id2label = {i: name for name, i in label2id.items()}
    return names, label2id, id2label


def stratified_split(
    labels: Sequence[str | int],
    *,
    seed: int = 42,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
) -> tuple[list[int], list[int], list[int]]:
    """Deterministic, class-stratified index split.

    Returns (train_idx, val_idx, test_idx) as sorted index lists. Per class the
    ordering is shuffled with a seeded RNG, so the same seed always reproduces the
    same split — which is what lets an evaluator recreate the exact test set months
    later without storing it.

    Classes with at least 3 members are guaranteed at least one row in val and test.
    """
    if not 0 < val_frac + test_frac < 1:
        raise ValueError("val_frac + test_frac must be strictly between 0 and 1")

    by_label: dict[str | int, list[int]] = defaultdict(list)
    for idx, label in enumerate(labels):
        by_label[label].append(idx)

    train: list[int] = []
    val: list[int] = []
    test: list[int] = []
    for label in sorted(by_label, key=str):  # sorted: order must not depend on dict order
        idx = list(by_label[label])
        random.Random(f"{seed}:{label}").shuffle(idx)
        n = len(idx)
        n_test = round(n * test_frac)
        n_val = round(n * val_frac)
        if n >= 3:  # never let a class vanish from val/test
            n_test = max(1, n_test)
            n_val = max(1, n_val)
        n_test = min(n_test, max(0, n - 1))
        n_val = min(n_val, max(0, n - n_test - 1))
        test.extend(idx[:n_test])
        val.extend(idx[n_test:n_test + n_val])
        train.extend(idx[n_test + n_val:])

    return sorted(train), sorted(val), sorted(test)


def class_distribution(labels: Iterable[str]) -> dict[str, int]:
    """Label -> count, ordered from most to least frequent (for reports/metrics.json)."""
    return dict(sorted(Counter(labels).items(), key=lambda kv: (-kv[1], kv[0])))
