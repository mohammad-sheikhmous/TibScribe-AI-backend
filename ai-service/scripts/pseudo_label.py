"""Self-training on real transcripts, with guards (IMPLEMENTATION.md P7-06).

The model labels sentences it was not trained on, and the confident ones become extra
training data. Done carelessly this is a machine that amplifies its own mistakes: the
model is most confident exactly where it is systematically wrong, so naive
pseudo-labelling entrenches the error and the metrics look fine because the same bias
scores them.

Four guards, each closing one way that fails:

  1. **Real transcripts only.** Pseudo-labelling the synthetic corpus would double down
     on the distribution the model already over-fits, which is the opposite of the point.
  2. **Calibrated confidence.** The threshold is applied to the calibrated score (P4-03);
     a raw softmax of 0.95 from an uncalibrated model means very little.
  3. **Out-of-distribution veto.** A confident answer about a sentence unlike anything in
     training is precisely the case not to trust (P4-06).
  4. **Human corrections are an anchor.** If a doctor corrected a sentence, the model's
     opinion about that same text is discarded, never the other way round.

And the batch is provisional by construction: rows are tagged `source: pseudo`, weighted
0.3 at training time, and the whole batch can be dropped by source if the gate in
P7-09 shows it hurt.

    python scripts/pseudo_label.py --dry-run
    python scripts/pseudo_label.py --out data/gold/pseudo.jsonl --min-confidence 0.9
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.db import init_db, session_scope  # noqa: E402
from app.db.models import Correction, ReportItemRow  # noqa: E402

DEFAULT_MIN_CONFIDENCE = 0.90
DEFAULT_MAX_OOD = 0.40
DEFAULT_MAX_PER_LABEL = 200


def collect_candidates(limit: int) -> list[dict]:
    """Un-corrected report items — i.e. real transcript sentences (guard 1)."""
    with session_scope() as session:
        corrected_texts = {
            text for (text,) in session.query(ReportItemRow.text)
            .join(Correction, Correction.item_id == ReportItemRow.item_id).all()
        }
        rows = (
            session.query(ReportItemRow)
            .filter(ReportItemRow.is_corrected.is_(False),
                    ReportItemRow.is_asr_suspect.is_(False))
            .limit(limit)
            .all()
        )
        return [
            {
                "text": row.text, "label": row.label, "item_id": row.item_id,
                "job_id": row.job_id, "confidence": row.combined_confidence
                or row.confidence,
                "ood": row.ood_score,
            }
            for row in rows
            if row.text.strip() and row.text not in corrected_texts  # guard 4
        ]


def filter_candidates(candidates: list[dict], *, min_confidence: float, max_ood: float,
                      max_per_label: int) -> tuple[list[dict], Counter]:
    rejected: Counter = Counter()
    kept: list[dict] = []
    per_label: Counter = Counter()

    for candidate in sorted(candidates, key=lambda c: -(c["confidence"] or 0)):
        if (candidate["confidence"] or 0) < min_confidence:      # guard 2
            rejected["low_confidence"] += 1
            continue
        if candidate["ood"] is not None and candidate["ood"] > max_ood:  # guard 3
            rejected["out_of_distribution"] += 1
            continue
        if per_label[candidate["label"]] >= max_per_label:
            # A model that over-predicts one class would otherwise flood the batch with
            # it and make the imbalance worse.
            rejected["label_quota"] += 1
            continue
        per_label[candidate["label"]] += 1
        kept.append(candidate)
    return kept, rejected


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate guarded pseudo-labels.")
    parser.add_argument("--out", default="data/gold/pseudo.jsonl")
    parser.add_argument("--min-confidence", type=float, default=DEFAULT_MIN_CONFIDENCE)
    parser.add_argument("--max-ood", type=float, default=DEFAULT_MAX_OOD)
    parser.add_argument("--max-per-label", type=int, default=DEFAULT_MAX_PER_LABEL)
    parser.add_argument("--limit", type=int, default=20000)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # pragma: no cover
        pass

    init_db()
    candidates = collect_candidates(args.limit)
    print(f"candidates (real transcript sentences): {len(candidates)}")
    if not candidates:
        print("\nNothing to label yet — the archive has no processed reports.\n"
              "Pseudo-labelling only ever runs on real transcripts (guard 1).")
        return 0

    kept, rejected = filter_candidates(
        candidates, min_confidence=args.min_confidence, max_ood=args.max_ood,
        max_per_label=args.max_per_label,
    )

    print(f"accepted : {len(kept)}")
    for reason, count in rejected.most_common():
        print(f"  rejected — {reason}: {count}")
    distribution = Counter(row["label"] for row in kept)
    print(f"labels   : {dict(distribution.most_common(10))}")

    if args.dry_run:
        print("\n--dry-run: nothing written")
        return 0

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as handle:
        for row in kept:
            handle.write(json.dumps({
                "text": row["text"], "label": row["label"], "source": "pseudo",
                "confidence": row["confidence"], "item_id": row["item_id"],
            }, ensure_ascii=False) + "\n")

    print(f"\nwrote {out}")
    print("These rows are PROVISIONAL: weighted 0.3 at training time, and the batch is "
          "dropped if the activation gate (scripts/retrain.sh) shows a regression.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
