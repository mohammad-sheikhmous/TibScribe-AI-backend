"""Out-of-distribution detection benchmark (IMPLEMENTATION.md P4-07).

Measures whether the model can tell "I have never seen anything like this" from "I know
this". The comparison that matters is against softmax confidence: the claim is that
softmax *cannot* do this and the embedding distance can, so both are scored on the same
data and the AUROCs are printed side by side.

In-distribution sentences come from the held-out split. Out-of-distribution ones come
from `eval/ood/*.jsonl` (real off-topic or hallucinated transcript lines) and, failing
that, from a small built-in set of clearly non-clinical Arabic sentences.

    python eval/eval_ood.py --model-dir model_output
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.nlp.dataset import dedupe_texts, load_jsonl, stratified_split  # noqa: E402
from app.core.nlp.uncertainty import auroc  # noqa: E402

# Obviously non-clinical Arabic, plus the shapes Whisper hallucinates on silence.
BUILTIN_OOD = [
    "شكرا لمشاهدتكم ولا تنسوا الاشتراك بالقناة",
    "الطقس اليوم مشمس ودرجة الحرارة معتدلة",
    "فريق كرة القدم فاز بالمباراة أمس بثلاثة أهداف",
    "أرجو تحويل المبلغ إلى الحساب البنكي غدا",
    "الفيلم كان طويلا لكن القصة كانت ممتعة",
    "ترجمة نانسي قنقر",
    "اشترك بالقناة اشترك بالقناة اشترك بالقناة",
    "السيارة تحتاج تغيير زيت المحرك والفلاتر",
    "الاجتماع سيعقد الساعة العاشرة في قاعة المؤتمرات",
    "أضف الملح والفلفل ثم اتركه على نار هادئة",
]


def load_ood(directory: str) -> list[str]:
    root = Path(directory)
    texts: list[str] = []
    if root.exists():
        for path in sorted(root.glob("*.jsonl")):
            texts.extend(str(row["text"]) for row in load_jsonl(path) if row.get("text"))
    return texts or list(BUILTIN_OOD)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Score OOD detection.")
    parser.add_argument("--model-dir", default="model_output")
    parser.add_argument("--data", default="data.jsonl")
    parser.add_argument("--ood", default="eval/ood")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # pragma: no cover
        pass

    from app.core.nlp.classifier import MedicalSentenceClassifier

    rows = dedupe_texts(load_jsonl(args.data)).rows
    _, _, test_idx = stratified_split([r["label"] for r in rows], seed=args.seed)
    in_dist = [rows[i]["text"] for i in test_idx][: args.limit]
    out_dist = load_ood(args.ood)[: args.limit]

    print(f"in-distribution : {len(in_dist)} sentences (held-out split)")
    print(f"out-of-distribution: {len(out_dist)} sentences "
          f"({'from ' + args.ood if Path(args.ood).exists() else 'built-in set'})")

    classifier = MedicalSentenceClassifier(model_dir=args.model_dir)
    if classifier.train_stats is None:
        print("\nThis checkpoint has no train_stats.json, so OOD scoring is unavailable.\n"
              "Retrain with the current train_arabert.py (it writes the statistics).")
        return 1

    scored_in = classifier.predict_with_uncertainty(in_dist, mc_passes=1, mc_only_below=0.0)
    scored_out = classifier.predict_with_uncertainty(out_dist, mc_passes=1, mc_only_below=0.0)

    ood_scores = np.array([r["uncertainty"].ood for r in scored_out + scored_in])
    # Softmax's own attempt at the same job: low confidence should mean "unfamiliar".
    softmax_scores = np.array([1 - r["confidence"] for r in scored_out + scored_in])
    is_ood = np.array([1] * len(scored_out) + [0] * len(scored_in))

    ood_auroc = auroc(ood_scores, is_ood)
    softmax_auroc = auroc(softmax_scores, is_ood)

    print("\n" + "=" * 60)
    print(f"AUROC — Mahalanobis (embedding distance): {ood_auroc:.4f}")
    print(f"AUROC — 1 - softmax confidence         : {softmax_auroc:.4f}")
    print("=" * 60)
    print(f"P4-07 gate: OOD AUROC >= 0.80  ->  "
          f"{'PASS' if ood_auroc >= 0.80 else 'FAIL'}")
    if ood_auroc > softmax_auroc:
        print("Embedding distance beats softmax, as expected: a confident wrong answer\n"
              "on a foreign input is exactly what softmax cannot flag.")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps({
            "created_at": datetime.now(timezone.utc).isoformat(),
            "model_dir": args.model_dir,
            "n_in": len(in_dist), "n_out": len(out_dist),
            "auroc_mahalanobis": round(ood_auroc, 4),
            "auroc_softmax": round(softmax_auroc, 4),
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
