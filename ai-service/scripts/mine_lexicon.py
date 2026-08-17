"""Unsupervised lexicon mining (IMPLEMENTATION.md P5-06, PLAN_V2 §5).

Writing a gazetteer by hand means guessing which phrasings patients use. The data
already knows. This reads the sentences the current lexicon does NOT cover, and reports
what they repeatedly say — so the term list grows from evidence instead of imagination.

Two modes, both proposing and never deciding (PLAN_V2 §5.3):

  * **phrases** (default, no torch) — frequent n-grams in uncovered sentences, filtered
    against terms already known. This is where dialect forms and Whisper's systematic
    mistranscriptions surface: the misspellings already in `medications.yaml`
    ("بروفن", "أمل وديبين") were found exactly this way, by hand.
  * **expand** — semantic neighbours of a seed term, from the project's own encoder.
    "صداع" pulls in the phrasings that mean the same thing without sharing a substring.

    python scripts/mine_lexicon.py --uncovered reports/uncovered.txt --top 60
    python scripts/mine_lexicon.py --expand صداع --data data.jsonl
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.nlp.dataset import load_jsonl  # noqa: E402
from app.core.nlp.extraction import extract_entities, load_lexicon  # noqa: E402

# Words that carry no clinical content; an n-gram made only of these is noise.
STOPWORDS = {
    "من", "في", "على", "عن", "إلى", "الى", "مع", "أن", "ان", "هذا", "هذه", "التي",
    "الذي", "كان", "كانت", "قد", "لا", "ما", "هو", "هي", "بعد", "قبل", "عند", "كل",
    "أو", "او", "ثم", "لكن", "بس", "يجب", "لازم", "ممكن", "المريضة", "المريض", "الدكتور",
    "و", "ب", "ل", "هناك", "يوجد", "عندها", "عنده", "لديها", "لديه", "يكون", "تكون",
    "جدا", "جداً", "كثير", "شوي", "حاليا", "حالياً", "اليوم", "أمس", "غدا",
}
TOKEN_RE = re.compile(r"[ء-ي]{2,}")
MIN_COUNT = 8


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text)


def ngrams(tokens: list[str], n: int) -> list[str]:
    return [" ".join(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def known_surface_forms() -> set[str]:
    lexicon = load_lexicon()
    forms: set[str] = set()
    for group in (lexicon.symptoms, lexicon.conditions, lexicon.tests,
                  lexicon.labs, lexicon.drugs):
        forms.update(surface for _, surface in group)
    return forms


def read_uncovered(path: Path) -> list[str]:
    """The report written by eval/eval_coverage.py --uncovered."""
    lines = path.read_text(encoding="utf-8").splitlines()
    return [line.strip() for line in lines if line.strip() and not line.startswith("###")]


def mine_phrases(sentences: list[str], *, top: int, min_count: int) -> list[dict]:
    known = known_surface_forms()
    counts: Counter = Counter()
    examples: dict[str, str] = {}

    for sentence in sentences:
        tokens = [t for t in tokenize(sentence)]
        for n in (2, 3, 4):
            for phrase in ngrams(tokens, n):
                words = phrase.split()
                if all(word in STOPWORDS for word in words):
                    continue
                if words[0] in STOPWORDS and words[-1] in STOPWORDS:
                    continue
                if any(known_form in phrase for known_form in known if len(known_form) > 4):
                    continue
                counts[phrase] += 1
                examples.setdefault(phrase, sentence)

    # Prefer the longest phrase among near-duplicates: "ألم أسفل البطن" over "أسفل البطن".
    ranked = [(phrase, count) for phrase, count in counts.most_common()
              if count >= min_count]
    kept: list[tuple[str, int]] = []
    for phrase, count in ranked:
        if any(phrase in longer and phrase != longer for longer, _ in kept):
            continue
        kept.append((phrase, count))
        if len(kept) >= top:
            break

    return [
        {"phrase": phrase, "count": count, "example": examples[phrase],
         "proposed_code": "", "kind": "", "decision": ""}
        for phrase, count in kept
    ]


def expand_seed(seed: str, sentences: list[str], *, top: int) -> list[dict]:
    """Nearest neighbours of a seed phrase in embedding space."""
    try:
        import numpy as np
        from app.core.nlp.embeddings import embed_texts
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(f"--expand needs torch + transformers ({type(exc).__name__})")

    candidates = sorted({s for s in sentences if len(s) > 15})[:4000]
    vectors = embed_texts([seed] + candidates)
    seed_vector, rest = vectors[0], vectors[1:]
    scores = rest @ seed_vector
    order = np.argsort(-scores)[:top]
    return [
        {"phrase": candidates[i], "count": "", "example": "",
         "similarity": round(float(scores[i]), 4), "proposed_code": "", "decision": ""}
        for i in order
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Mine lexicon candidates from the data.")
    parser.add_argument("--uncovered", default="reports/uncovered.txt")
    parser.add_argument("--data", default="data.jsonl")
    parser.add_argument("--expand", default=None, help="seed term to expand semantically")
    parser.add_argument("--top", type=int, default=60)
    parser.add_argument("--min-count", type=int, default=MIN_COUNT)
    parser.add_argument("--out", default="reports/lexicon_candidates.csv")
    args = parser.parse_args(argv)

    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # pragma: no cover
        pass

    if args.expand:
        sentences = [row["text"] for row in load_jsonl(args.data)]
        rows = expand_seed(args.expand, sentences, top=args.top)
        print(f"nearest neighbours of «{args.expand}»:\n")
        for row in rows[:20]:
            print(f"  {row['similarity']:.3f}  {row['phrase'][:80]}")
    else:
        path = Path(args.uncovered)
        if not path.exists():
            raise SystemExit(
                f"{path} not found — run: python eval/eval_coverage.py --uncovered {path}"
            )
        sentences = read_uncovered(path)
        print(f"uncovered sentences: {len(sentences)}")
        rows = mine_phrases(sentences, top=args.top, min_count=args.min_count)
        print(f"candidate phrases (count >= {args.min_count}):\n")
        for row in rows:
            print(f"  {row['count']:5d}  {row['phrase']}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else
                                ["phrase", "decision"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {out} — fill `proposed_code`/`kind`, then add the accepted rows to "
          f"app/core/nlp/lexicon/*.yaml")
    print("Candidates are PROPOSALS: nothing enters the lexicon without human review "
          "(PLAN_V2 §5.3).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
