"""Classifier evaluation harness (IMPLEMENTATION.md P0-12).

Scores one or more checkpoints on two very different sets:

  1. **held-out split** of data.jsonl — recreated exactly via the shared
     `stratified_split(seed=...)`, so it is genuinely unseen data, not a re-scored
     training set;
  2. **gold set** (`eval/gold/*.jsonl`) — real Whisper transcripts labelled by hand.
     This is the only number that predicts field behaviour; the gap between (1) and
     (2) IS the distribution shift.

Also the adoption gate for domain-adaptive pretraining: `--compare base adapted`
prints both and the delta, which P0-12 requires to be ≥ +1.5 macro-F1 points before
the adapted encoder replaces the baseline.

    python eval/eval_classifier.py --model-dir model_output
    python eval/eval_classifier.py --compare model_output model_output_adapted
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sklearn.metrics import classification_report, f1_score  # noqa: E402

from app.core.nlp.dataset import dedupe_texts, load_jsonl, stratified_split  # noqa: E402

BATCH = 64


def load_synthetic_test(data_path: str, seed: int) -> tuple[list[str], list[str]]:
    """Recreate the exact test split train_arabert.py held out."""
    rows = dedupe_texts(load_jsonl(data_path)).rows
    labels_raw = [r["label"] for r in rows]
    _, _, test_idx = stratified_split(labels_raw, seed=seed)
    return [rows[i]["text"] for i in test_idx], [labels_raw[i] for i in test_idx]


def load_gold(gold_dir: str) -> tuple[list[str], list[str]]:
    """Every eval/gold/*.jsonl file: {"text": ..., "label": ...} per line."""
    root = Path(gold_dir)
    texts: list[str] = []
    labels: list[str] = []
    if not root.exists():
        return texts, labels
    for path in sorted(root.glob("*.jsonl")):
        for row in load_jsonl(path):
            if row.get("text") and row.get("label"):
                texts.append(str(row["text"]))
                labels.append(str(row["label"]))
    return texts, labels


def score(model_dir: str, texts: list[str], gold: list[str]) -> dict:
    from app.core.nlp.classifier import MedicalSentenceClassifier

    clf = MedicalSentenceClassifier(model_dir=model_dir)
    preds: list[str] = []
    for start in range(0, len(texts), BATCH):
        preds.extend(label for label, _ in clf.predict_batch(texts[start:start + BATCH]))

    known = sorted(set(gold) | set(preds))
    return {
        "n": len(texts),
        "macro_f1": round(float(f1_score(gold, preds, average="macro", zero_division=0)), 4),
        "weighted_f1": round(
            float(f1_score(gold, preds, average="weighted", zero_division=0)), 4
        ),
        "accuracy": round(sum(p == g for p, g in zip(preds, gold)) / max(len(gold), 1), 4),
        "per_class": classification_report(
            gold, preds, labels=known, output_dict=True, zero_division=0
        ),
    }


def evaluate_dir(model_dir: str, sets: dict[str, tuple[list[str], list[str]]]) -> dict:
    out: dict = {"model_dir": model_dir}
    for name, (texts, gold) in sets.items():
        if not texts:
            print(f"  [{name}] empty — skipped")
            out[name] = None
            continue
        print(f"  [{name}] scoring {len(texts)} sentences...")
        out[name] = score(model_dir, texts, gold)
    return out


def _fmt(result: dict | None, key: str) -> str:
    return "—" if not result or not result.get(key) else f"{result[key]['macro_f1']:.4f}"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Evaluate classifier checkpoint(s).")
    p.add_argument("--model-dir", default="model_output")
    p.add_argument("--compare", nargs=2, metavar=("BASE", "ADAPTED"),
                   help="score two checkpoints and print the delta (DAPT adoption gate)")
    p.add_argument("--data", default="data.jsonl")
    p.add_argument("--gold", default="eval/gold")
    p.add_argument("--seed", type=int, default=42, help="must match the training seed")
    p.add_argument("--gate", type=float, default=1.5,
                   help="minimum macro-F1 point gain for --compare to recommend adoption")
    p.add_argument("--out", default=None, help="write full results JSON here")
    args = p.parse_args(argv)

    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # pragma: no cover
        pass

    sets = {
        "synthetic_test": load_synthetic_test(args.data, args.seed),
        "gold_real": load_gold(args.gold),
    }
    print(f"synthetic held-out: {len(sets['synthetic_test'][0])} sentences")
    print(f"gold (real audio) : {len(sets['gold_real'][0])} sentences")
    if not sets["gold_real"][0]:
        print("  NOTE: no gold set yet (P0-11). Only the synthetic number is meaningful,\n"
              "        and it systematically overstates field accuracy.")

    dirs = list(args.compare) if args.compare else [args.model_dir]
    results = []
    for model_dir in dirs:
        if not os.path.isdir(model_dir):
            raise SystemExit(f"model dir not found: {model_dir}")
        print(f"\n=== {model_dir} ===")
        results.append(evaluate_dir(model_dir, sets))

    print("\n" + "=" * 72)
    print(f"{'checkpoint':<34}{'synthetic macro-F1':>20}{'gold macro-F1':>18}")
    print("-" * 72)
    for r in results:
        print(f"{r['model_dir']:<34}{_fmt(r, 'synthetic_test'):>20}{_fmt(r, 'gold_real'):>18}")

    if args.compare:
        base, adapted = results
        for name, label in (("gold_real", "gold"), ("synthetic_test", "synthetic")):
            if base.get(name) and adapted.get(name):
                delta = (adapted[name]["macro_f1"] - base[name]["macro_f1"]) * 100
                verdict = "ADOPT" if delta >= args.gate else "keep baseline"
                print(f"\n{label} delta: {delta:+.2f} macro-F1 points "
                      f"(gate {args.gate:+.1f}) -> {verdict}")

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(
                {"created_at": datetime.now(timezone.utc).isoformat(),
                 "seed": args.seed, "results": results},
                ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
