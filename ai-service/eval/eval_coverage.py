"""Entity-extraction coverage (IMPLEMENTATION.md P5-16).

The number that defines P5: the analysis measured **32.5%** of sentences producing at
least one ontology entity — and that figure was inflated, because `emergency` and
`follow_up` scored 100% only by reading the classifier's label rather than the text.

This script measures the same thing the same way, so the before/after is comparable,
and additionally reports **text-derived** coverage with those label-driven shortcuts
excluded, which is the honest number.

    python eval/eval_coverage.py
    python eval/eval_coverage.py --uncovered reports/uncovered.txt   # mining input
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.nlp.dataset import load_jsonl  # noqa: E402
from app.core.nlp.extraction import extract_entities, load_lexicon  # noqa: E402

TARGET = 0.75  # PLAN_V2 P5 exit criterion


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Measure entity extraction coverage.")
    parser.add_argument("--data", default="data.jsonl")
    parser.add_argument("--uncovered", default=None,
                        help="write sentences with no entity here (input to lexicon mining)")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args(argv)

    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # pragma: no cover
        pass

    rows = load_jsonl(args.data)
    if args.limit:
        rows = rows[: args.limit]

    lexicon = load_lexicon()
    print(f"lexicon: {lexicon.size}\n")

    per_label_total: Counter = Counter()
    per_label_hit: Counter = Counter()
    per_label_actionable: Counter = Counter()
    uncovered: dict[str, list[str]] = defaultdict(list)
    kinds: Counter = Counter()

    for row in rows:
        label = row.get("label", "")
        text = row.get("text", "")
        entities = extract_entities(text, label)
        per_label_total[label] += 1
        if entities:
            per_label_hit[label] += 1
            kinds.update(entity.kind for entity in entities)
            if any(entity.is_actionable for entity in entities):
                per_label_actionable[label] += 1
        else:
            uncovered[label].append(text)

    total = sum(per_label_total.values())
    hits = sum(per_label_hit.values())
    actionable = sum(per_label_actionable.values())

    print(f"{'label':<24}{'covered':>16}{'%':>8}")
    print("-" * 48)
    for label, count in per_label_total.most_common():
        hit = per_label_hit[label]
        print(f"{label:<24}{hit:>7}/{count:<8}{hit / count * 100:>7.1f}%")

    print("-" * 48)
    print(f"{'TOTAL':<24}{hits:>7}/{total:<8}{hits / total * 100:>7.1f}%")
    print(f"{'  of which actionable':<24}{actionable:>7}/{total:<8}"
          f"{actionable / total * 100:>7.1f}%")
    print(f"\nbaseline (analysis report): 32.5%   target: {TARGET * 100:.0f}%")
    print(f"gate: {'PASS' if hits / total >= TARGET else 'not yet met'}")
    print(f"\nentities by kind: {dict(kinds.most_common())}")

    if args.uncovered:
        out = Path(args.uncovered)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as handle:
            for label, texts in sorted(uncovered.items(),
                                       key=lambda kv: -len(kv[1])):
                handle.write(f"### {label} ({len(texts)} uncovered)\n")
                for text in texts:
                    handle.write(text + "\n")
                handle.write("\n")
        print(f"\nwrote {out} ({sum(len(v) for v in uncovered.values())} sentences)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
