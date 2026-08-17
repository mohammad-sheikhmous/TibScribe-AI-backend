"""Domain-Adaptive Pre-Training (DAPT) — continue AraBERT's MLM on our own text.

Why (PLAN_V2.md §5, IMPLEMENTATION.md P0-06):
`aubmindlab/bert-base-arabertv02` was pretrained on general Arabic web text, then
fine-tuned here on clean synthetic clinical sentences. What it actually serves is
dialectal OB/GYN speech transcribed by Whisper. This script closes part of that gap
WITHOUT any labels: it keeps training the same masked-language-model objective on our
own unlabeled text, then hands the adapted encoder to `train_arabert.py`.

    python train_mlm.py                                   # corpus = data.jsonl texts
    python train_mlm.py --extra-texts data/transcripts    # + every accumulated transcript
    python train_arabert.py --model-name model_output/bert_adapted   # then fine-tune

Two properties this shares with the fine-tuning script on purpose:
  * the SAME `normalize_arabic()` — DAPT on differently-normalized text would teach the
    model a distribution it never sees at fine-tune or serve time;
  * block packing instead of per-sentence padding — our sentences average ~15 tokens,
    so padding each to 128 would waste ~85% of the compute.

Output (`model_output/bert_adapted/`): a plain encoder + tokenizer, loadable by
`AraBERTClassifier(path, num_classes)`, plus `dapt_info.json` (corpus size, loss
history) — the evidence for the ≥ +1.5 macro-F1 adoption gate in P0-12.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from app.core.nlp.dataset import load_jsonl  # noqa: E402
from app.core.nlp.normalize import get_preprocessing_mode, normalize_arabic  # noqa: E402

DEFAULT_MODEL_NAME = "aubmindlab/bert-base-arabertv02"
DEFAULT_OUTPUT_DIR = "model_output/bert_adapted"
DEFAULT_BLOCK_SIZE = 128
DEFAULT_EPOCHS = 5
DEFAULT_BATCH_SIZE = 16
DEFAULT_LR = 5e-5
DEFAULT_MLM_PROB = 0.15
DEFAULT_SEED = 42
MIN_USEFUL_TOKENS = 100_000  # below this, DAPT rarely moves the needle — we warn


class BlockDataset(Dataset):
    """Fixed-length token blocks — no padding, so every position carries signal."""

    def __init__(self, blocks: list[list[int]]):
        self.blocks = blocks

    def __len__(self) -> int:
        return len(self.blocks)

    def __getitem__(self, idx: int) -> dict:
        return {"input_ids": self.blocks[idx]}


class WholeWordMaskCollator:
    """Mask whole words rather than individual wordpieces.

    Masking one piece of a word the model can trivially complete from its neighbours
    ("صد ##اع" -> mask "##اع") teaches almost nothing; AraBERT itself was pretrained
    with whole-word masking. Transformers' own `DataCollatorForWholeWordMask` now
    requires per-example `offset_mapping`, which does not survive block packing (the
    offsets refer to the original sentences, not the concatenated stream), so word
    boundaries are recovered here from the `##` continuation prefix instead — which
    IS still valid after packing.

    The core takes plain id sets, so it can be unit-tested without a real tokenizer.
    """

    def __init__(
        self,
        mask_token_id: int,
        vocab_size: int,
        special_ids: set[int],
        continuation_ids: set[int],
        mlm_probability: float = 0.15,
        seed: int | None = None,
    ):
        self.mask_token_id = mask_token_id
        self.vocab_size = vocab_size
        self.special_ids = special_ids
        self.continuation_ids = continuation_ids
        self.mlm_probability = mlm_probability
        self.rng = random.Random(seed)

    @classmethod
    def from_tokenizer(cls, tokenizer, mlm_probability: float = 0.15, seed: int | None = None):
        vocab = tokenizer.get_vocab()
        return cls(
            mask_token_id=tokenizer.mask_token_id,
            vocab_size=len(vocab),
            special_ids=set(tokenizer.all_special_ids),
            continuation_ids={i for tok, i in vocab.items() if tok.startswith("##")},
            mlm_probability=mlm_probability,
            seed=seed,
        )

    def word_groups(self, ids: list[int]) -> list[list[int]]:
        """Split one sequence into maskable word groups (special tokens excluded)."""
        groups: list[list[int]] = []
        for pos, token_id in enumerate(ids):
            if token_id in self.special_ids:
                groups.append([])  # barrier: never mask, never continue across it
                continue
            if token_id in self.continuation_ids and groups and groups[-1]:
                groups[-1].append(pos)
            else:
                groups.append([pos])
        return [g for g in groups if g]

    def __call__(self, features: list[dict]) -> dict:
        input_ids = torch.tensor([f["input_ids"] for f in features], dtype=torch.long)
        labels = torch.full_like(input_ids, -100)

        for row in range(input_ids.size(0)):
            ids = input_ids[row].tolist()
            groups = self.word_groups(ids)
            self.rng.shuffle(groups)
            budget = max(1, round(sum(len(g) for g in groups) * self.mlm_probability))

            chosen: list[int] = []
            for group in groups:
                if len(chosen) >= budget:
                    break
                chosen.extend(group)

            for pos in chosen:
                labels[row, pos] = input_ids[row, pos]
                draw = self.rng.random()
                if draw < 0.8:  # 80% [MASK]
                    input_ids[row, pos] = self.mask_token_id
                elif draw < 0.9:  # 10% random token
                    input_ids[row, pos] = self.rng.randrange(self.vocab_size)
                # 10%: leave the original token in place

        # Blocks are exactly block_size with no padding, so attention is all-ones.
        return {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
            "labels": labels,
        }


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def collect_texts(data_path: str | None, extra_dirs: list[str]) -> list[str]:
    """Gather every available unlabeled sentence: the dataset + any transcript files.

    Labels are ignored on purpose — this stage learns the language of the domain,
    not the task.
    """
    texts: list[str] = []
    if data_path and Path(data_path).exists():
        texts.extend(str(row["text"]) for row in load_jsonl(data_path) if row.get("text"))

    for directory in extra_dirs:
        root = Path(directory)
        if not root.exists():
            print(f"  (skipping missing --extra-texts dir: {root})")
            continue
        for path in sorted(root.rglob("*")):
            if path.suffix == ".txt":
                texts.extend(
                    line.strip() for line in path.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                )
            elif path.suffix == ".jsonl":
                texts.extend(
                    str(row["text"]) for row in load_jsonl(path) if row.get("text")
                )
            elif path.suffix == ".json":  # a saved Report: pull its item texts
                try:
                    report = json.loads(path.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    continue
                for section in (report.get("soap") or {}).values():
                    texts.extend(
                        str(item.get("text", "")) for item in section.get("items", [])
                    )

    # De-duplicate: a sentence repeated across files (boilerplate advice, the same
    # recording processed twice) would otherwise be over-weighted in the MLM objective.
    # Order is preserved so the corpus stays reproducible.
    seen: set[str] = set()
    unique: list[str] = []
    for text in texts:
        text = text.strip()
        if text and text not in seen:
            seen.add(text)
            unique.append(text)
    return unique


def build_blocks(texts: list[str], tokenizer, block_size: int) -> tuple[list[list[int]], int]:
    """Normalize -> tokenize -> concatenate -> chunk into fixed-size blocks."""
    all_ids: list[int] = []
    for text in texts:
        ids = tokenizer(normalize_arabic(text), add_special_tokens=True)["input_ids"]
        all_ids.extend(ids)
    total = len(all_ids)
    usable = (total // block_size) * block_size
    blocks = [all_ids[i:i + block_size] for i in range(0, usable, block_size)]
    return blocks, total


@torch.no_grad()
def evaluate(model, loader, device) -> float:
    model.eval()
    total, batches = 0.0, 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        total += float(model(**batch).loss.item())
        batches += 1
    return total / max(batches, 1)


def train(args) -> None:
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Collecting corpus...")
    texts = collect_texts(args.data_path, args.extra_texts)
    if not texts:
        raise SystemExit("No text found — nothing to adapt on.")
    print(f"  unique sentences: {len(texts)}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    blocks, total_tokens = build_blocks(texts, tokenizer, args.block_size)
    print(f"  tokens   : {total_tokens}  ->  {len(blocks)} blocks of {args.block_size}")
    if total_tokens < MIN_USEFUL_TOKENS:
        print(
            f"  WARNING: {total_tokens} tokens is a small corpus for DAPT "
            f"(<{MIN_USEFUL_TOKENS}). Expect a small gain; re-run once transcripts "
            f"accumulate (PLAN_V2 §5.1)."
        )
    if len(blocks) < 10:
        raise SystemExit("Corpus too small to train on.")

    random.Random(args.seed).shuffle(blocks)
    n_val = max(1, int(len(blocks) * args.val_frac))
    val_ds, train_ds = BlockDataset(blocks[:n_val]), BlockDataset(blocks[n_val:])
    print(f"  train blocks: {len(train_ds)} | val blocks: {len(val_ds)}")

    collator = WholeWordMaskCollator.from_tokenizer(
        tokenizer, mlm_probability=args.mlm_prob, seed=args.seed
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collator)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, collate_fn=collator)

    model = AutoModelForMaskedLM.from_pretrained(args.model_name).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = max(1, len(train_loader) * args.epochs)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(total_steps * args.warmup_ratio), total_steps
    )
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    history: list[dict] = []
    best_val = float("inf")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\nStarting domain-adaptive pretraining...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            if scaler:
                with torch.amp.autocast("cuda"):
                    loss = model(**batch).loss
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss = model(**batch).loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            scheduler.step()
            running += float(loss.item())

        train_loss = running / max(len(train_loader), 1)
        val_loss = evaluate(model, val_loader, device)
        history.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "val_loss": round(val_loss, 4),
            "val_perplexity": round(math.exp(min(val_loss, 20)), 2),
        })
        print(f"Epoch {epoch:2d}/{args.epochs} | train {train_loss:.4f} | "
              f"val {val_loss:.4f} | ppl {math.exp(min(val_loss, 20)):.2f}")

        if val_loss < best_val:
            best_val = val_loss
            # Save the ENCODER (not the MLM head): this is what AraBERTClassifier loads.
            model.bert.save_pretrained(out_dir)
            tokenizer.save_pretrained(out_dir)
            print(f"  -> saved adapted encoder to {out_dir}")

    info = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "base_model": args.model_name,
        "device": str(device),
        "corpus": {
            "sentences": len(texts),
            "tokens": total_tokens,
            "blocks": len(blocks),
            "block_size": args.block_size,
            "sources": [args.data_path, *args.extra_texts],
        },
        "hyperparameters": {
            "epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr,
            "mlm_probability": args.mlm_prob, "warmup_ratio": args.warmup_ratio,
            "seed": args.seed, "preprocessing": get_preprocessing_mode(),
        },
        "best_val_loss": round(best_val, 4),
        "best_val_perplexity": round(math.exp(min(best_val, 20)), 2),
        "history": history,
        "next_step": (
            f"python train_arabert.py --model-name {args.output_dir} "
            f"--output-dir model_output_adapted  # then compare with eval/eval_classifier.py"
        ),
    }
    (out_dir / "dapt_info.json").write_text(
        json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nDone. Adapted encoder + dapt_info.json in {out_dir}")
    print(f"Next: {info['next_step']}")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Domain-adaptive MLM pretraining for AraBERT.")
    p.add_argument("--data-path", default="data.jsonl", help="labelled file; labels ignored")
    p.add_argument("--extra-texts", action="append", default=[],
                   help="dir of .txt/.jsonl/.json (report) files — repeatable")
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    p.add_argument("--block-size", type=int, default=DEFAULT_BLOCK_SIZE)
    p.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    p.add_argument("--lr", type=float, default=DEFAULT_LR)
    p.add_argument("--mlm-prob", type=float, default=DEFAULT_MLM_PROB)
    p.add_argument("--warmup-ratio", type=float, default=0.1)
    p.add_argument("--val-frac", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return p


def main() -> None:
    train(build_arg_parser().parse_args())


if __name__ == "__main__":
    main()
