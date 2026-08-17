"""Fine-tune AraBERT for Arabic medical-sentence classification (OB/GYN).

Reads data.jsonl ({"text": ..., "label": ...} per line), fine-tunes
`aubmindlab/bert-base-arabertv02`, and writes a self-describing checkpoint to
model_output/ that app/core/nlp/classifier.py loads as-is:

    model_output/
      best_model.pt        full model state_dict (backbone + head)
      bert/                AraBERT backbone (save_pretrained) — for local-first load
      tokenizer/           tokenizer (save_pretrained)      — for local-first load
      label_mapping.json   id2label / label2id
      model_config.json    model_name / max_len / dropout / num_classes  (anti-drift)
      metrics.json         test macro/weighted F1, per-class report, confusion matrix, history
      classification_report.txt   human-readable per-class report

Training recipe: stratified 70/15/15 split, class-weighted + label-smoothed
cross-entropy (handles class imbalance), linear warmup, gradient clipping, AMP on
CUDA, early stopping on validation macro-F1. Reproducible via a single SEED.

Run:  python train_arabert.py            (sensible defaults)
      python train_arabert.py --epochs 15 --batch-size 32 --max-len 192
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import warnings
from collections import Counter, defaultdict
from datetime import datetime, timezone

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

warnings.filterwarnings("ignore")

# Allow `python train_arabert.py` from the repo root to import the shared app package.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from app.core.nlp.dataset import (  # noqa: E402
    build_label_maps,
    class_distribution,
    dedupe_texts,
    load_jsonl,
    stratified_split,
)
from app.core.nlp.calibration import calibrate  # noqa: E402
from app.core.nlp.model import AraBERTClassifier  # noqa: E402
from app.core.nlp.normalize import get_preprocessing_mode, normalize_arabic  # noqa: E402
from app.core.nlp.uncertainty import TrainStats  # noqa: E402

# --- defaults (all overridable via argparse) ---
DEFAULT_MODEL_NAME = "aubmindlab/bert-base-arabertv02"
DEFAULT_DATA_PATH = "data.jsonl"
DEFAULT_OUTPUT_DIR = "model_output"
DEFAULT_MAX_LEN = 128  # data max is ~135 chars (<128 tokens); 192 wasted compute/memory
DEFAULT_BATCH_SIZE = 16
DEFAULT_EPOCHS = 20
DEFAULT_LR = 2e-5
DEFAULT_WARMUP_RATIO = 0.1
DEFAULT_DROPOUT = 0.3
DEFAULT_LABEL_SMOOTHING = 0.1
DEFAULT_PATIENCE = 5
DEFAULT_LLRD_DECAY = 0.95  # layer-wise LR decay; 1.0 == uniform LR
DEFAULT_SEED = 42


class MedicalDataset(Dataset):
    """Sentences + labels, each row carrying a training weight.

    The weight is how the active-learning loop pays off: a sentence a doctor corrected
    on a real transcript is worth several synthetic ones, and a self-labelled
    (pseudo) sentence is worth a fraction. Without per-row weights, 16k synthetic rows
    simply drown the few hundred real ones.
    """

    def __init__(self, texts, labels, tokenizer, max_len, weights=None):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.weights = weights if weights is not None else [1.0] * len(texts)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = normalize_arabic(str(self.texts[idx]))
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
            "weight": torch.tensor(float(self.weights[idx]), dtype=torch.float),
        }


# Relative worth of one row, by where it came from (IMPLEMENTATION.md P7-08).
SOURCE_WEIGHTS = {
    "human": 3.0,       # a doctor corrected it, on a real transcript
    "confirmed": 2.0,   # a doctor reviewed it and agreed
    "pseudo": 0.3,      # the model labelled it itself — provisional
    "synthetic": 1.0,   # the original dataset
}


def load_extra_data(paths, weight_overrides=None):
    """Load exported corrections / pseudo-labels, weighted by `source`."""
    texts, labels, weights, counts = [], [], [], Counter()
    overrides = weight_overrides or {}
    for path in paths or []:
        if not os.path.exists(path):
            print(f"  WARNING: --extra-data file not found: {path}")
            continue
        for row in load_jsonl(path):
            if not row.get("text") or not row.get("label"):
                continue
            source = row.get("source", "human")
            texts.append(row["text"])
            labels.append(row["label"])
            weights.append(overrides.get(source, SOURCE_WEIGHTS.get(source, 1.0)))
            counts[source] += 1
    if counts:
        print(f"  extra data: {dict(counts)}")
    return texts, labels, weights


class LabelSmoothingCrossEntropy(nn.Module):
    """Cross-entropy with label smoothing and optional per-class weighting."""

    def __init__(self, smoothing=0.1, weight=None):
        super().__init__()
        self.smoothing = smoothing
        self.weight = weight

    def forward(self, logits, target, sample_weight=None):
        log_probs = torch.log_softmax(logits, dim=-1)
        nll_loss = -log_probs.gather(dim=-1, index=target.unsqueeze(1)).squeeze(1)
        smooth_loss = -log_probs.mean(dim=-1)
        if self.weight is not None:
            w = self.weight[target]          # class weight (imbalance)
            nll_loss = nll_loss * w
            smooth_loss = smooth_loss * w
        loss = (1 - self.smoothing) * nll_loss + self.smoothing * smooth_loss
        if sample_weight is not None:
            # Per-ROW weight (data provenance) — multiplies with the class weight, then
            # the mean is re-normalised so a batch of high-weight rows does not silently
            # act as a larger learning rate.
            loss = loss * sample_weight
            return loss.sum() / sample_weight.sum().clamp(min=1e-8)
        return loss.mean()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_data(path: str):
    """Load + de-duplicate via the shared data layer (app/core/nlp/dataset.py).

    De-duplication happens BEFORE splitting so an identical sentence can never leak
    into both train and test. The loading/splitting logic is shared with
    eval/eval_classifier.py so evaluation always reproduces the exact held-out set.
    """
    rows = load_jsonl(path)
    result = dedupe_texts(rows)
    print(f"Loaded {len(rows)} rows; removed {result.removed} duplicate-text rows "
          f"({len(result.conflicts)} texts with conflicting labels) -> "
          f"{len(result.rows)} unique texts.")
    if result.conflicts:
        print(f"  WARNING: {len(result.conflicts)} texts still carry conflicting labels — "
              f"run `python scripts/check_conflicts.py` and resolve them "
              f"(data/LABELING_GUIDE.md). They put a ceiling on achievable accuracy.")

    texts = [row["text"] for row in result.rows]
    labels_raw = [row["label"] for row in result.rows]
    label_names, label2id, id2label = build_label_maps(labels_raw)
    labels = [label2id[lbl] for lbl in labels_raw]
    return texts, labels, labels_raw, label_names, label2id, id2label


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        logits = model(input_ids, attention_mask)
        preds = torch.argmax(logits, dim=-1)
        all_preds.extend(preds.cpu().numpy().tolist())
        all_labels.extend(batch["label"].numpy().tolist())
    f1_macro = f1_score(all_labels, all_preds, average="macro")
    f1_weighted = f1_score(all_labels, all_preds, average="weighted")
    return f1_macro, f1_weighted, all_preds, all_labels


@torch.no_grad()
def collect_logits_and_embeddings(model, loader, device):
    """Raw logits + CLS embeddings — the inputs to calibration and OOD detection.

    Gathered in one pass over a loader so neither step needs its own forward pass.
    """
    model.eval()
    logits_out, embeddings_out, labels_out = [], [], []
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        logits, embeddings = model(input_ids, attention_mask, return_embedding=True)
        logits_out.append(logits.cpu().numpy())
        embeddings_out.append(embeddings.cpu().numpy())
        labels_out.append(batch["label"].numpy())
    return (np.vstack(logits_out), np.vstack(embeddings_out), np.concatenate(labels_out))


def build_optimizer(model, base_lr, weight_decay, llrd_decay):
    """AdamW with Layer-wise LR Decay (LLRD).

    Lower AraBERT layers (closer to the embeddings) get exponentially smaller LRs
    than the top layers and the classifier head — a well-known technique for more
    stable, higher-quality BERT fine-tuning. `llrd_decay=1.0` reproduces a single
    uniform LR. Bias/LayerNorm params are excluded from weight decay as usual.
    """
    n_layers = model.bert.config.num_hidden_layers
    layer_re = re.compile(r"bert\.encoder\.layer\.(\d+)\.")
    no_decay = ("bias", "LayerNorm.weight")
    buckets = defaultdict(list)
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("bert.embeddings"):
            depth = n_layers + 1
        else:
            m = layer_re.search(name)
            depth = (n_layers - int(m.group(1))) if m else 0  # 0 == top (head / pooler)
        scale = llrd_decay ** depth
        wd = 0.0 if any(nd in name for nd in no_decay) else weight_decay
        buckets[(scale, wd)].append(param)
    groups = [
        {"params": ps, "lr": base_lr * scale, "weight_decay": wd}
        for (scale, wd), ps in buckets.items()
    ]
    return AdamW(groups)


def train(args) -> None:
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading data...")
    texts, labels, labels_raw, label_names, label2id, id2label = load_data(args.data_path)
    num_classes = len(label_names)
    distribution = class_distribution(labels_raw)
    print(f"Total samples: {len(texts)} | classes: {num_classes}")
    for lbl, count in distribution.items():
        print(f"  {lbl}: {count}")

    # inverse-frequency class weights (normalized to sum to num_classes)
    class_counts = np.array([Counter(labels)[i] for i in range(num_classes)], dtype=np.float64)
    class_weights = 1.0 / class_counts
    class_weights = class_weights / class_weights.sum() * num_classes
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

    print("\nSplitting data (70% train, 15% val, 15% test)...")
    # Shared, seed-reproducible split: eval/eval_classifier.py recreates the exact
    # same test set months later without it ever being stored.
    train_idx, val_idx, test_idx = stratified_split(labels_raw, seed=args.seed)
    pick = lambda idx, seq: [seq[i] for i in idx]  # noqa: E731
    train_texts, train_labels = pick(train_idx, texts), pick(train_idx, labels)
    val_texts, val_labels = pick(val_idx, texts), pick(val_idx, labels)
    test_texts, test_labels = pick(test_idx, texts), pick(test_idx, labels)
    print(f"Train: {len(train_texts)}, Val: {len(val_texts)}, Test: {len(test_texts)}")

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # --- real-world data from the correction loop (P7-08) ---------------------------
    # Appended to TRAIN only: validation and test must stay a clean, untouched measure.
    extra_texts, extra_raw_labels, extra_weights = load_extra_data(
        args.extra_data,
        weight_overrides={"human": args.extra_weight, "confirmed": args.extra_weight * 0.7,
                          "pseudo": args.pseudo_weight},
    )
    train_weights = [1.0] * len(train_texts)
    for text, raw_label, weight in zip(extra_texts, extra_raw_labels, extra_weights):
        if raw_label not in label2id:
            print(f"  WARNING: extra row has unknown label {raw_label!r} — skipped")
            continue
        train_texts.append(text)
        train_labels.append(label2id[raw_label])
        train_weights.append(weight)
    if extra_texts:
        print(f"Train set after extra data: {len(train_texts)} "
              f"(+{len(train_texts) - len(train_weights) + len(extra_texts)})")

    def loader(t, l, shuffle, weights=None):
        ds = MedicalDataset(t, l, tokenizer, args.max_len, weights)
        return DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle, num_workers=args.num_workers)

    train_loader = loader(train_texts, train_labels, True, train_weights)
    val_loader = loader(val_texts, val_labels, False)
    test_loader = loader(test_texts, test_labels, False)

    print("Initializing model...")
    model = AraBERTClassifier(args.model_name, num_classes, args.dropout).to(device)

    total_steps = len(train_loader) * args.epochs
    optimizer = build_optimizer(model, args.lr, weight_decay=0.01, llrd_decay=args.llrd_decay)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * args.warmup_ratio),
        num_training_steps=total_steps,
    )
    criterion = LabelSmoothingCrossEntropy(smoothing=args.label_smoothing, weight=class_weights)
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    best_path = os.path.join(args.output_dir, "best_model.pt")
    best_f1, best_epoch, patience_counter = 0.0, 0, 0
    history = []

    print("\nStarting training...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels_b = batch["label"].to(device)
            optimizer.zero_grad()

            weights_b = batch.get("weight")
            if weights_b is not None:
                weights_b = weights_b.to(device)

            if scaler:
                with torch.amp.autocast("cuda"):
                    logits = model(input_ids, attention_mask)
                    loss = criterion(logits, labels_b, weights_b)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(input_ids, attention_mask)
                loss = criterion(logits, labels_b, weights_b)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            scheduler.step()
            total_loss += loss.item()

        val_macro, val_weighted, _, _ = evaluate(model, val_loader, device)
        avg_loss = total_loss / len(train_loader)
        history.append(
            {"epoch": epoch, "train_loss": round(avg_loss, 4),
             "val_macro_f1": round(val_macro, 4), "val_weighted_f1": round(val_weighted, 4)}
        )
        print(f"Epoch {epoch:2d}/{args.epochs} | Loss: {avg_loss:.4f} | "
              f"LR: {scheduler.get_last_lr()[0]:.2e} | "
              f"Val Macro-F1: {val_macro:.4f} | Val Weighted-F1: {val_weighted:.4f}")

        if val_macro > best_f1:
            best_f1, best_epoch, patience_counter = val_macro, epoch, 0
            torch.save(model.state_dict(), best_path)
            print(f"  -> New best model saved (Macro-F1: {best_f1:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping triggered after {epoch} epochs")
                break

    print(f"\nBest validation Macro-F1: {best_f1:.4f} (epoch {best_epoch})")

    # --- test evaluation with the best checkpoint ---
    print("\nLoading best model for test evaluation...")
    model.load_state_dict(torch.load(best_path, map_location=device))
    test_macro, test_weighted, test_preds, test_true = evaluate(model, test_loader, device)
    print(f"\n=== TEST SET RESULTS ===")
    print(f"Macro-F1:    {test_macro:.4f}")
    print(f"Weighted-F1: {test_weighted:.4f}")

    # --- calibration + OOD statistics (P4-03 / P4-05) --------------------------------
    # Fitted on VALIDATION, never on test: a temperature fitted on the test set would
    # make the reported calibration meaningless.
    print("\nFitting confidence calibration on the validation set...")
    val_logits, val_embeddings, val_labels = collect_logits_and_embeddings(model, val_loader, device)
    calibration = calibrate(val_logits, val_labels)
    print(f"  temperature      : {calibration.temperature}")
    print(f"  ECE  {calibration.ece_before:.4f} -> {calibration.ece_after:.4f}")
    print(f"  90%-precision threshold: {calibration.threshold_90}")
    if not calibration.improved:
        print("  (calibration did not help; keeping raw confidences)")

    print("Collecting training-distribution statistics for OOD detection...")
    _, train_embeddings, train_stat_labels = collect_logits_and_embeddings(model, train_loader, device)
    train_stats = TrainStats.fit(train_embeddings, labels=train_stat_labels)
    train_stats = train_stats.calibrate_class_distances(val_embeddings, source="validation")
    print(f"  embedding dim: {train_stats.dim} over {len(train_embeddings)} samples")

    report_txt = classification_report(test_true, test_preds, target_names=label_names, digits=4)
    report_dict = classification_report(
        test_true, test_preds, target_names=label_names, output_dict=True, zero_division=0
    )
    cm = confusion_matrix(test_true, test_preds)
    print("\nClassification Report:\n" + report_txt)

    save_artifacts(
        args, model, tokenizer, label_names, label2id, id2label, num_classes,
        distribution, len(train_texts), len(val_texts), len(test_texts),
        best_epoch, best_f1, test_macro, test_weighted,
        report_txt, report_dict, cm, history, str(device),
        calibration=calibration, train_stats=train_stats,
    )
    print("\nDone! Model + metrics saved to", args.output_dir)


def save_artifacts(
    args, model, tokenizer, label_names, label2id, id2label, num_classes,
    distribution, n_train, n_val, n_test, best_epoch, best_val_f1,
    test_macro, test_weighted, report_txt, report_dict, cm, history, device,
    calibration=None, train_stats=None,
):
    out = args.output_dir
    print(f"\nSaving artifacts to {out}")

    # backbone + tokenizer (consumed by classifier.py's local-first loading)
    model.bert.save_pretrained(os.path.join(out, "bert"))
    tokenizer.save_pretrained(os.path.join(out, "tokenizer"))

    with open(os.path.join(out, "label_mapping.json"), "w", encoding="utf-8") as f:
        json.dump({"id2label": id2label, "label2id": label2id}, f, ensure_ascii=False, indent=2)

    # self-describing config so inference reads the exact max_len/model/preprocessing
    # it was trained with (classifier.py warns on a preprocessing mismatch).
    config = {
        "model_name": args.model_name, "max_len": args.max_len,
        "dropout_rate": args.dropout, "num_classes": num_classes,
        "preprocessing": get_preprocessing_mode(),
    }
    if calibration is not None:
        # Read back by classifier.py: the serving temperature and the confidence level
        # at which accepted predictions are actually right 90% of the time.
        config["calibration"] = calibration.to_dict()
    with open(os.path.join(out, "model_config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    if train_stats is not None:
        with open(os.path.join(out, "train_stats.json"), "w", encoding="utf-8") as f:
            json.dump(train_stats.to_payload(), f)

    # full metrics + reproducibility record
    metrics = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "device": device,
        "hyperparameters": {
            "model_name": args.model_name, "max_len": args.max_len,
            "batch_size": args.batch_size, "epochs": args.epochs, "lr": args.lr,
            "warmup_ratio": args.warmup_ratio, "dropout": args.dropout,
            "label_smoothing": args.label_smoothing, "patience": args.patience,
            "llrd_decay": args.llrd_decay, "preprocessing": get_preprocessing_mode(),
            "seed": args.seed,
        },
        "data": {
            "total": n_train + n_val + n_test, "num_classes": num_classes,
            "splits": {"train": n_train, "val": n_val, "test": n_test},
            "class_distribution": distribution,
        },
        "best_epoch": best_epoch,
        "best_val_macro_f1": round(best_val_f1, 4),
        "test": {
            "macro_f1": round(test_macro, 4),
            "weighted_f1": round(test_weighted, 4),
            "per_class": report_dict,
            "labels": label_names,
            "confusion_matrix": cm.tolist(),
        },
        "history": history,
    }
    with open(os.path.join(out, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    with open(os.path.join(out, "classification_report.txt"), "w", encoding="utf-8") as f:
        f.write(f"Test Macro-F1: {test_macro:.4f}\nTest Weighted-F1: {test_weighted:.4f}\n\n")
        f.write(report_txt)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Fine-tune AraBERT for medical sentence classification.")
    p.add_argument("--data-path", default=DEFAULT_DATA_PATH)
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    p.add_argument("--max-len", type=int, default=DEFAULT_MAX_LEN)
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    p.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    p.add_argument("--lr", type=float, default=DEFAULT_LR)
    p.add_argument("--warmup-ratio", type=float, default=DEFAULT_WARMUP_RATIO)
    p.add_argument("--dropout", type=float, default=DEFAULT_DROPOUT)
    p.add_argument("--label-smoothing", type=float, default=DEFAULT_LABEL_SMOOTHING)
    p.add_argument("--llrd-decay", type=float, default=DEFAULT_LLRD_DECAY,
                   help="layer-wise LR decay factor (1.0 = uniform LR)")
    p.add_argument("--extra-data", action="append", default=[],
                   help="exported corrections/pseudo-labels (repeatable); TRAIN only")
    p.add_argument("--extra-weight", type=float, default=3.0,
                   help="weight for doctor-corrected rows relative to synthetic ones")
    p.add_argument("--pseudo-weight", type=float, default=0.3,
                   help="weight for self-labelled rows — deliberately below 1.0")
    p.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return p


def main() -> None:
    train(build_arg_parser().parse_args())


if __name__ == "__main__":
    main()
