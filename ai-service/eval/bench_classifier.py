"""Classifier throughput benchmark (IMPLEMENTATION.md P4-02).

Measures what the batching change actually bought, on this machine, with this
checkpoint — the number that goes in the metrics board. Correctness is irrelevant here,
so it runs against whatever checkpoint is present.

    python eval/bench_classifier.py --n 40 --repeats 3
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

SAMPLES = [
    "المريضة حامل بالأسبوع 34 وتشتكي من صداع شديد",
    "ضغط الدم 150 على 90 والنبض 88",
    "التشخيص تسمم حمل مبكر يحتاج متابعة",
    "سنعطيها سلفات المغنيسيوم ونراقب الضغط",
    "تحليل البول يبين زلال إيجابي",
    "المريضة عندها حساسية من البنسلين",
    "ننصح بالراحة التامة وتقليل الملح",
    "موعد المراجعة بعد أسبوع مع إعادة التحاليل",
]


def make_texts(n: int) -> list[str]:
    return [SAMPLES[i % len(SAMPLES)] for i in range(n)]


def timed(fn, repeats: int) -> float:
    """Median of `repeats` runs, after one warm-up (the first pass allocates buffers)."""
    fn()
    durations = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        durations.append(time.perf_counter() - start)
    return statistics.median(durations)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark classifier throughput.")
    parser.add_argument("--model-dir", default="model_output")
    parser.add_argument("--n", type=int, default=40, help="sentences per run (a typical visit)")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--mc-passes", type=int, default=8)
    args = parser.parse_args(argv)

    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # pragma: no cover
        pass

    from app.core.nlp.classifier import MedicalSentenceClassifier

    classifier = MedicalSentenceClassifier(model_dir=args.model_dir)
    texts = make_texts(args.n)

    one_by_one = timed(lambda: [classifier.predict(t) for t in texts], args.repeats)
    batched = timed(lambda: classifier.predict_batch(texts), args.repeats)
    with_uncertainty = timed(
        lambda: classifier.predict_with_uncertainty(texts, mc_passes=args.mc_passes),
        args.repeats,
    )

    print(f"\nsentences per run : {args.n}")
    print(f"device            : {classifier.device}")
    print("-" * 56)
    print(f"one call per sentence : {one_by_one:7.3f} s")
    print(f"batched (current)     : {batched:7.3f} s   -> {one_by_one / batched:.1f}x faster")
    print(f"+ uncertainty ({args.mc_passes} passes): {with_uncertainty:7.3f} s")
    print("-" * 56)
    print(f"target for P4-02: batched < 0.2 s  ->  "
          f"{'PASS' if batched < 0.2 else 'not met on this hardware'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
