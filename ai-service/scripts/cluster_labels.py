"""Unsupervised cluster analysis of the label taxonomy (P0-03, PLAN_V2 §5).

Answers two questions that no supervised metric can:

  1. **Is a class real?** Cluster one label's sentences: clean sub-clusters mean it is
     several classes merged (the `info` case — 709 rows, 1.8% KBS coverage); a
     shapeless blob means it is a semantic dumping ground.
  2. **Are two classes separable at all?** If `nutrition` and `pregnancy_nutrition`
     sentences interleave in embedding space, no amount of training will separate
     them — the taxonomy is wrong, not the model.

The output is EVIDENCE FOR A HUMAN DECISION, never an automatic relabel
(PLAN_V2 §5.3). Decisions get recorded in data/LABELING_GUIDE.md.

    python scripts/cluster_labels.py --label info --clusters 8
    python scripts/cluster_labels.py --separability nutrition pregnancy_nutrition

Embeddings: `--method bert` uses the project's own backbone (best signal, needs
torch); `--method tfidf` uses character n-grams (no torch, good enough to spot
gross overlap). Default: bert when importable, else tfidf.
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from app.core.nlp.dataset import load_jsonl  # noqa: E402
from app.core.nlp.normalize import normalize_arabic  # noqa: E402


def _bert_available() -> bool:
    """Both halves must import: torch can be installed but blocked by an OS policy."""
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
        return True
    except Exception:
        return False


def embed_tfidf(texts: list[str]) -> np.ndarray:
    """Character n-grams: robust to Arabic morphology and to Whisper misspellings."""
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer

    normalized = [normalize_arabic(t) for t in texts]
    matrix = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=2).fit_transform(
        normalized
    )
    n_components = min(64, matrix.shape[1] - 1, max(2, matrix.shape[0] - 1))
    return TruncatedSVD(n_components=n_components, random_state=42).fit_transform(matrix)


def embed_bert(texts: list[str], model_dir: str) -> np.ndarray:
    """Mean-pooled hidden states from the project's own encoder."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    local_tok = Path(model_dir) / "tokenizer"
    local_bert = Path(model_dir) / "bert"
    name = "aubmindlab/bert-base-arabertv02"
    tokenizer = AutoTokenizer.from_pretrained(str(local_tok) if local_tok.is_dir() else name)
    model = AutoModel.from_pretrained(str(local_bert) if local_bert.is_dir() else name).eval()

    vectors: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(texts), 64):
            batch = [normalize_arabic(t) for t in texts[start:start + 64]]
            enc = tokenizer(batch, padding=True, truncation=True, max_length=128,
                            return_tensors="pt")
            hidden = model(**enc).last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1).float()
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            vectors.append(pooled.cpu().numpy())
    return np.vstack(vectors)


def embed(texts: list[str], method: str, model_dir: str) -> tuple[np.ndarray, str]:
    if method == "auto":
        method = "bert" if _bert_available() else "tfidf"
    if method == "bert":
        return embed_bert(texts, model_dir), "bert"
    return embed_tfidf(texts), "tfidf"


def cluster_one_label(rows, label, k, method, model_dir, out_path):
    from sklearn.cluster import KMeans

    texts = [r["text"] for r in rows if r["label"] == label]
    if len(texts) < k * 2:
        raise SystemExit(f"not enough rows for label {label!r}: {len(texts)}")
    print(f"label '{label}': {len(texts)} sentences")

    vectors, used = embed(texts, method, model_dir)
    print(f"embeddings: {used} ({vectors.shape[1]} dims)")
    km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(vectors)

    print(f"\n{k} clusters — representative sentences (nearest to each centre):\n")
    report_rows = []
    for c in range(k):
        idx = np.where(km.labels_ == c)[0]
        if not len(idx):
            continue
        centre = km.cluster_centers_[c]
        order = idx[np.argsort(np.linalg.norm(vectors[idx] - centre, axis=1))]
        print(f"  ── cluster {c}  ({len(idx)} sentences)")
        for i in order[:6]:
            print(f"       {texts[i][:90]}")
        print()
        for i in order:
            report_rows.append({"cluster": c, "text": texts[i]})

    if out_path:
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["cluster", "text"])
            w.writeheader()
            w.writerows(report_rows)
        print(f"wrote {out}")
    print("Interpretation: clean, distinct clusters => the label hides several classes.\n"
          "Overlapping, unreadable clusters => it is a dumping ground; exclude it from "
          "reasoning.\nRecord the decision in data/LABELING_GUIDE.md.")


def separability(rows, label_a, label_b, method, model_dir):
    """How well can the two labels be told apart from the sentence alone?

    Reports 1-nearest-neighbour purity: for each sentence, does its closest neighbour
    (excluding itself) carry the same label? Near 50% == indistinguishable.
    """
    subset = [r for r in rows if r["label"] in (label_a, label_b)]
    texts = [r["text"] for r in subset]
    labels = np.array([r["label"] for r in subset])
    counts = Counter(labels.tolist())
    print(f"{label_a}: {counts.get(label_a, 0)} | {label_b}: {counts.get(label_b, 0)}")

    vectors, used = embed(texts, method, model_dir)
    print(f"embeddings: {used}")
    norm = vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-9)
    sim = norm @ norm.T
    np.fill_diagonal(sim, -np.inf)
    nearest = np.argmax(sim, axis=1)
    same = float(np.mean(labels[nearest] == labels))

    print(f"\n1-NN label purity: {same * 100:.1f}%")
    if same < 0.60:
        print("  => The two labels are NOT separable from sentence text alone.\n"
              "     Recommend merging them and deriving the distinction from patient\n"
              "     context instead (PLAN_V2 P2-06). See data/LABELING_GUIDE.md §2.")
    elif same < 0.80:
        print("  => Weak separation: expect systematic confusion between these labels.")
    else:
        print("  => Separable: the boundary is learnable from text.")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Unsupervised analysis of the label taxonomy.")
    p.add_argument("--data", default="data.jsonl")
    p.add_argument("--label", help="cluster the sentences of this single label")
    p.add_argument("--separability", nargs=2, metavar=("LABEL_A", "LABEL_B"))
    p.add_argument("--clusters", type=int, default=8)
    p.add_argument("--method", choices=["auto", "bert", "tfidf"], default="auto")
    p.add_argument("--model-dir", default="model_output")
    p.add_argument("--out", default=None, help="CSV of cluster assignments")
    args = p.parse_args(argv)

    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # pragma: no cover
        pass

    rows = load_jsonl(args.data)
    if args.separability:
        separability(rows, *args.separability, args.method, args.model_dir)
    elif args.label:
        cluster_one_label(rows, args.label, args.clusters, args.method,
                          args.model_dir, args.out)
    else:
        p.error("choose --label or --separability")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
