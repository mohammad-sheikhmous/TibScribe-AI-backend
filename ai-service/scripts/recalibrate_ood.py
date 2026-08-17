"""Post-hoc OOD calibration for an already-trained TibScribe AraBERT checkpoint.

No retraining. Recreates the deterministic training split, collects CLS embeddings
with the final best_model.pt, fits TrainStats including empirical Mahalanobis-distance
quantiles, and overwrites model_output/train_stats.json atomically.

Run from ai-service/:
  python scripts/recalibrate_ood.py --data-path data.jsonl --model-dir model_output
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.core.nlp.classifier import MedicalSentenceClassifier  # noqa: E402
from app.core.nlp.dataset import dedupe_texts, load_jsonl, stratified_split  # noqa: E402
from app.core.nlp.uncertainty import TrainStats, mahalanobis_distance, ood_score  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-path", default="data.jsonl")
    ap.add_argument("--model-dir", default="model_output")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rows = dedupe_texts(load_jsonl(args.data_path)).rows
    texts = [str(r["text"]) for r in rows]
    labels = [str(r["label"]) for r in rows]
    train_idx, val_idx, _ = stratified_split(labels, seed=args.seed)
    train_texts = [texts[i] for i in train_idx]
    train_labels = [labels[i] for i in train_idx]
    val_texts = [texts[i] for i in val_idx]

    clf = MedicalSentenceClassifier(model_dir=args.model_dir, strict=True)
    def collect(batch_texts: list[str], label: str) -> np.ndarray:
        parts: list[np.ndarray] = []
        clf.model.eval()
        with torch.no_grad():
            for start in range(0, len(batch_texts), args.batch_size):
                batch = batch_texts[start:start + args.batch_size]
                input_ids, attention_mask = clf._encode(batch)
                _, emb = clf.model(input_ids, attention_mask, return_embedding=True)
                parts.append(emb.cpu().numpy())
                done = min(start + len(batch), len(batch_texts))
                if done % 1000 < args.batch_size or done == len(batch_texts):
                    print(f"Collected {label} {done}/{len(batch_texts)} embeddings")
        return np.vstack(parts)

    train_matrix = collect(train_texts, "train")
    val_matrix = collect(val_texts, "validation")
    stats = TrainStats.fit(train_matrix, labels=train_labels)
    stats = stats.calibrate_class_distances(val_matrix, source="validation")
    distances = mahalanobis_distance(train_matrix, stats)
    out = Path(args.model_dir) / "train_stats.json"
    tmp = out.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(stats.to_payload()), encoding="utf-8")
    os.replace(tmp, out)

    print("OOD calibration updated:", out)
    print("OOD version:", stats.ood_version)
    print("calibration source:", stats.calibration_source)
    val_scores = ood_score(val_matrix, stats)
    print("class-distance quantiles:", stats.class_distance_quantiles)
    print("validation OOD median/p95/max:",
          float(np.median(val_scores)), float(np.quantile(val_scores, 0.95)), float(val_scores.max()))
    print("validation saturation rate (score >= 0.999):",
          float(np.mean(val_scores >= 0.999)))
    print("legacy global train max distance:", float(distances.max()))


if __name__ == "__main__":
    main()
