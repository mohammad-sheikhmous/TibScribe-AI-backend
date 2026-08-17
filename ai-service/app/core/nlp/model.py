"""AraBERT classifier architecture.

Single source of truth — previously duplicated verbatim in train_arabert.py
and inference.py. The architecture (AraBERT backbone -> dropout -> linear head
over the CLS token) MUST stay identical between training and inference or the
saved checkpoint will not load / will misbehave, so it lives in exactly one place.
"""
from __future__ import annotations

import torch.nn as nn
from transformers import AutoModel


class AraBERTClassifier(nn.Module):
    def __init__(self, model_name, num_classes, dropout_rate=0.3):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask, return_embedding: bool = False):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_token = outputs.last_hidden_state[:, 0, :]
        pooled = self.dropout(cls_token)
        logits = self.classifier(pooled)
        if return_embedding:
            # The pre-dropout CLS vector is what the out-of-distribution check compares
            # against the training distribution (uncertainty.mahalanobis_distance).
            return logits, cls_token
        return logits
