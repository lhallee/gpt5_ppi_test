from __future__ import annotations

import os
import random
from dataclasses import asdict
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
)


class AminoVocab:
    def __init__(self, letters: str, pad_token: str = "_", unknown_token: str = "X") -> None:
        # Ensure unique
        unique_letters = []
        for ch in letters:
            if ch not in unique_letters:
                unique_letters.append(ch)
        letters = "".join(unique_letters)

        if pad_token in letters:
            letters = letters.replace(pad_token, "")
        if unknown_token in letters:
            letters = letters.replace(unknown_token, "")

        self.pad_token = pad_token
        self.unknown_token = unknown_token
        self.itos: List[str] = [pad_token, unknown_token] + list(letters)
        self.stoi: Dict[str, int] = {ch: idx for idx, ch in enumerate(self.itos)}

    @property
    def pad_idx(self) -> int:
        return self.stoi[self.pad_token]

    @property
    def unknown_idx(self) -> int:
        return self.stoi[self.unknown_token]


def tokenize_sequence(seq: str, vocab: AminoVocab, max_len: int) -> List[int]:
    seq = (seq or "").strip().upper()
    ids = [vocab.stoi.get(ch, vocab.unknown_idx) for ch in seq]
    if len(ids) > max_len:
        ids = ids[:max_len]
    # Pad
    if len(ids) < max_len:
        ids = ids + [vocab.pad_idx] * (max_len - len(ids))
    return ids


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


@torch.no_grad()
def compute_classification_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= 0.5).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
    }
    # Some metrics can fail if only one class present
    try:
        metrics["auroc"] = float(roc_auc_score(y_true, y_prob))
    except Exception:
        metrics["auroc"] = float("nan")
    try:
        metrics["auprc"] = float(average_precision_score(y_true, y_prob))
    except Exception:
        metrics["auprc"] = float("nan")
    return metrics


def save_config(config_obj, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "config.json")
    import json

    with open(path, "w", encoding="utf-8") as f:
        json.dump(asdict(config_obj), f, indent=2)


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


