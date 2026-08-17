"""MedicalSentenceClassifier — load the fine-tuned AraBERT head and classify sentences.

Loading contract (all of it enforced, none of it assumed):

1. **Self-describing checkpoint** — `max_len`, `model_name`, `num_classes` and the
   normalization mode are read from `model_output/model_config.json` (written by
   `train_arabert.py`), so a retrain can never silently re-introduce a train/serve
   config mismatch. `DEFAULT_MAX_LEN` is only the fallback when that file is absent.
2. **Integrity check before use** — the checkpoint's head width, the label mapping and
   the config must agree, or loading fails loudly (`app/core/nlp/checkpoint.py`).
   A disagreement means every predicted label is wrong; with `strict=True`
   (`STRICT_MODEL_CHECKS=true`) trust-level problems block startup too.
3. **Strict by default** — any entry point that constructs the classifier refuses
   missing weights/mapping or preprocessing skew unless a developer explicitly opts
   out with `strict=False`.
4. **Local-first loading** — prefers `model_output/tokenizer/` + `model_output/bert/`
   over re-downloading the backbone from the HF Hub on every process start.
"""
from __future__ import annotations

import json
import logging
import os

import numpy as np
import torch

from .checkpoint import CheckpointFacts, validate_checkpoint
from .model import AraBERTClassifier
from .normalize import get_preprocessing_mode, normalize_arabic
from .uncertainty import (
    TrainStats,
    UncertaintyScores,
    mc_dropout_stats,
    ood_score,
    predictive_entropy,
    review_priority,
)

logger = logging.getLogger(__name__)

DEFAULT_MODEL_NAME = "aubmindlab/bert-base-arabertv02"
# Fallback only — the trained value in model_config.json always wins. Matches
# train_arabert.py's default (data max is ~135 chars, well under 128 tokens).
DEFAULT_MAX_LEN = 128
HEAD_WEIGHT_KEY = "classifier.weight"
# Last-resort class count: used only when neither label_mapping.json nor a checkpoint
# head is available (i.e. an untrained model_output/).
DEFAULT_NUM_CLASSES = 20


class MedicalSentenceClassifier:
    def __init__(
        self,
        model_dir: str = "model_output",
        model_name: str = DEFAULT_MODEL_NAME,
        default_max_len: int = DEFAULT_MAX_LEN,
        low_confidence_threshold: float = 0.5,
        device: torch.device | None = None,
        strict: bool = True,
    ):
        self.model_dir = model_dir
        self.low_confidence_threshold = low_confidence_threshold
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Loading classifier from %s (device=%s)", model_dir, self.device)

        # --- 1. self-describing config -------------------------------------------
        cfg = self._read_json(os.path.join(model_dir, "model_config.json")) or {}
        self.model_name = cfg.get("model_name", model_name)
        self.max_len = int(cfg.get("max_len", default_max_len))

        # Calibration fitted on the validation set at train time (P4-03). 1.0 = the
        # model was never calibrated, in which case confidences stay raw and the
        # low-confidence threshold keeps its configured default.
        calibration = cfg.get("calibration") or {}
        self.temperature = float(calibration.get("temperature", 1.0))
        self.calibrated_threshold = calibration.get("low_confidence_threshold")
        if self.temperature != 1.0:
            logger.info("Applying calibration temperature T=%.4f", self.temperature)

        # Training-distribution statistics for out-of-distribution detection (P4-05).
        stats_payload = self._read_json(os.path.join(model_dir, "train_stats.json"))
        self.train_stats = TrainStats.from_payload(stats_payload) if stats_payload else None
        if self.train_stats is None:
            logger.info("No train_stats.json — OOD scoring disabled for this checkpoint")
        elif self.train_stats.class_distance_quantiles:
            logger.info(
                "OOD detector ready (%s; calibrated on %s)",
                self.train_stats.ood_version,
                self.train_stats.calibration_source or "unknown",
            )
        else:
            logger.warning(
                "Legacy/global OOD stats detected. Run scripts/recalibrate_ood.py "
                "once to build class-conditional validation-calibrated OOD stats."
            )

        # --- 2. label mapping -----------------------------------------------------
        mapping = self._read_json(os.path.join(model_dir, "label_mapping.json"))
        if mapping:
            self.id2label = {int(k): v for k, v in mapping["id2label"].items()}
            self.label2id = mapping["label2id"]
            self.num_classes = len(self.id2label)
        else:
            self.id2label = None
            self.label2id = None
            self.num_classes = int(cfg.get("num_classes", DEFAULT_NUM_CLASSES))

        # --- 3. weights: read the head width BEFORE building anything -------------
        best_path = os.path.join(model_dir, "best_model.pt")
        state = None
        head_out_features = None
        if os.path.exists(best_path):
            state = torch.load(best_path, map_location="cpu", weights_only=True)
            head_weight = state.get(HEAD_WEIGHT_KEY)
            if head_weight is not None:
                head_out_features = int(head_weight.shape[0])

        # --- 4. integrity check: fail fast, never mislabel silently ---------------
        local_bert = os.path.join(model_dir, "bert")
        local_tok = os.path.join(model_dir, "tokenizer")
        warnings = validate_checkpoint(
            CheckpointFacts(
                config_num_classes=cfg.get("num_classes"),
                mapping_num_labels=len(self.id2label) if self.id2label else None,
                head_out_features=head_out_features,
                trained_preprocessing=cfg.get("preprocessing"),
                runtime_preprocessing=get_preprocessing_mode(),
                config_model_name=cfg.get("model_name"),
                runtime_model_name=model_name,
                has_config=bool(cfg),
                has_weights=state is not None,
                has_mapping=bool(mapping),
                has_local_backbone=os.path.isdir(local_bert),
                has_local_tokenizer=os.path.isdir(local_tok),
            ),
            strict=strict,
        )
        for message in warnings:
            logger.warning("checkpoint: %s", message)

        # A trustworthy head width overrides a stale mapping/config count.
        if head_out_features is not None:
            self.num_classes = head_out_features

        # --- 5. tokenizer + backbone (local-first, then Hub) ----------------------
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            local_tok if os.path.isdir(local_tok) else self.model_name
        )
        self.model = AraBERTClassifier(
            local_bert if os.path.isdir(local_bert) else self.model_name, self.num_classes
        ).to(self.device)

        # --- 6. fine-tuned weights ------------------------------------------------
        if state is not None:
            self.model.load_state_dict(state)
            logger.info("Loaded fine-tuned weights (max_len=%s, classes=%s)",
                        self.max_len, self.num_classes)
        self.model.eval()

        if self.id2label is None:
            self.id2label = {i: f"class_{i}" for i in range(self.num_classes)}

    @staticmethod
    def _read_json(path: str):
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        return None

    def predict(self, text, return_probs: bool = False):
        results = self.predict_batch([text], return_probs=return_probs)
        return results[0]

    def _encode(self, texts):
        normalized = [normalize_arabic(str(t)) for t in texts]
        enc = self.tokenizer(
            normalized,
            truncation=True,
            padding=True,
            max_length=self.max_len,
            return_tensors="pt",
        )
        return enc["input_ids"].to(self.device), enc["attention_mask"].to(self.device)

    def predict_batch(self, texts, return_probs: bool = False):
        """Classify a list of sentences in a single forward pass.

        Dynamic padding (to the longest sentence in the batch, capped at max_len)
        instead of always padding to max_len: same outputs, far less wasted compute.

        Confidence is the CALIBRATED probability when a temperature was fitted at train
        time — the raw softmax of an overconfident model is not a usable number.
        """
        if not texts:
            return []
        input_ids, attention_mask = self._encode(texts)

        with torch.no_grad():
            logits = self.model(input_ids, attention_mask)
            probs = torch.softmax(logits / self.temperature, dim=-1)

        results = []
        for row in probs:
            pred_id = int(torch.argmax(row).item())
            label = self.id2label.get(pred_id, f"unknown_{pred_id}")
            confidence = float(row[pred_id].item())
            if return_probs:
                all_probs = {
                    self.id2label.get(i, f"unknown_{i}"): round(float(row[i].item()), 4)
                    for i in range(self.num_classes)
                }
                results.append((label, confidence, all_probs))
            else:
                results.append((label, confidence))
        return results

    def predict_with_uncertainty(self, texts, *, mc_passes: int = 8,
                                 mc_only_below: float = 0.90):
        """Classify AND report how much the model should be trusted on each sentence.

        Three signals the plain softmax cannot give (see `uncertainty.py`):
          * entropy — ambiguity between known classes;
          * MC-dropout variance — whether the answer survives resampling the dropout
            mask, i.e. epistemic uncertainty. The head already has dropout=0.3, so this
            costs only extra forward passes;
          * Mahalanobis OOD — whether the sentence resembles the training data at all.

        **Screening.** MC-dropout is `mc_passes`x the cost of a normal prediction —
        measured at 5.7 s vs 0.48 s for 40 sentences on CPU, which is far too slow to
        run on everything. Confidently-correct sentences do not need it, so the passes
        run only on the sentences below `mc_only_below`. Typical visits spend the extra
        time on a handful of items instead of all of them.

        Returns one dict per sentence: label, confidence and `UncertaintyScores`.
        """
        if not texts:
            return []
        input_ids, attention_mask = self._encode(texts)

        with torch.no_grad():
            logits, embeddings = self.model(input_ids, attention_mask, return_embedding=True)
            probs = torch.softmax(logits / self.temperature, dim=-1).cpu().numpy()

            variance = np.zeros(len(texts))
            needs_mc = np.where(probs.max(axis=1) < mc_only_below)[0]
            if mc_passes > 1 and len(needs_mc):
                subset_ids = input_ids[needs_mc]
                subset_mask = attention_mask[needs_mc]
                # Dropout ON at inference — the whole point of MC-dropout. Restored to
                # eval mode immediately afterwards, even on error.
                self.model.train()
                try:
                    samples = np.stack([
                        torch.softmax(
                            self.model(subset_ids, subset_mask) / self.temperature, dim=-1
                        ).cpu().numpy()
                        for _ in range(mc_passes)
                    ])
                finally:
                    self.model.eval()
                variance[needs_mc] = mc_dropout_stats(samples)["variance"]
        entropy = predictive_entropy(probs)
        ood = (
            ood_score(embeddings.cpu().numpy(), self.train_stats)
            if self.train_stats is not None
            else np.zeros(len(texts))
        )

        results = []
        for i in range(len(texts)):
            pred_id = int(probs[i].argmax())
            confidence = float(probs[i][pred_id])
            scores = UncertaintyScores(
                confidence=confidence,
                entropy=float(entropy[i]),
                variance=float(variance[i]),
                ood=float(ood[i]),
                review_priority=review_priority(
                    confidence, float(entropy[i]), float(variance[i]), float(ood[i])
                ),
            )
            results.append({
                "label": self.id2label.get(pred_id, f"unknown_{pred_id}"),
                "confidence": confidence,
                "uncertainty": scores,
            })
        return results
