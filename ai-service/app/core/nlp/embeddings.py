"""Sentence embeddings over the project's own encoder (IMPLEMENTATION.md P5-05).

Used by the unsupervised layer (PLAN_V2 §5): lexicon mining by semantic neighbourhood,
label-taxonomy clustering, and later the patient-history search in P8. Deliberately the
SAME backbone the classifier uses — a separate embedding model would measure similarity
in a space the rest of the system does not live in.

torch is imported lazily so this module can be imported anywhere; only calling
`embed_texts` requires the ML stack.
"""
from __future__ import annotations

import functools
import logging
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "aubmindlab/bert-base-arabertv02"
BATCH = 32
MAX_LEN = 128


@functools.lru_cache(maxsize=2)
def _load(model_dir: str, model_name: str):
    """Local-first, exactly like the classifier: prefer the adapted encoder on disk."""
    import torch  # noqa: F401
    from transformers import AutoModel, AutoTokenizer

    root = Path(model_dir)
    for candidate in (root / "bert_adapted", root / "bert"):
        if candidate.is_dir():
            source = str(candidate)
            break
    else:
        source = model_name

    tokenizer_dir = root / "tokenizer"
    tokenizer = AutoTokenizer.from_pretrained(
        str(tokenizer_dir) if tokenizer_dir.is_dir() else model_name
    )
    model = AutoModel.from_pretrained(source).eval()
    logger.info("embeddings: using %s", source)
    return tokenizer, model


def embed_texts(
    texts: Sequence[str],
    *,
    model_dir: str = "model_output",
    model_name: str = DEFAULT_MODEL,
    normalize: bool = True,
    preprocess: bool = True,
) -> np.ndarray:
    """Mean-pooled contextual embeddings, one row per text.

    Mean pooling over real tokens rather than the CLS vector: CLS is trained for the
    classification objective, while mean pooling is the better general-purpose sentence
    representation for similarity work.
    """
    import torch

    from .normalize import normalize_arabic

    if not texts:
        return np.zeros((0, 768))

    tokenizer, model = _load(model_dir, model_name)
    prepared = [normalize_arabic(str(t)) if preprocess else str(t) for t in texts]

    vectors: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(prepared), BATCH):
            batch = prepared[start:start + BATCH]
            encoded = tokenizer(batch, padding=True, truncation=True,
                                max_length=MAX_LEN, return_tensors="pt")
            hidden = model(**encoded).last_hidden_state
            mask = encoded["attention_mask"].unsqueeze(-1).float()
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            vectors.append(pooled.cpu().numpy())

    stacked = np.vstack(vectors)
    if normalize:
        norms = np.linalg.norm(stacked, axis=1, keepdims=True)
        stacked = stacked / np.clip(norms, 1e-9, None)
    return stacked


def nearest_neighbours(
    seed: str, candidates: Sequence[str], *, top: int = 50, **kwargs
) -> list[tuple[str, float]]:
    """Candidates most similar to `seed`, best first — the seed-expansion step."""
    vectors = embed_texts([seed, *candidates], **kwargs)
    scores = vectors[1:] @ vectors[0]
    order = np.argsort(-scores)[:top]
    return [(candidates[i], float(scores[i])) for i in order]
