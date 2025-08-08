from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split

from .utils import AminoVocab, tokenize_sequence


REQUIRED_COLUMNS = ["seq_a", "seq_b", "label"]


class PPIDataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        vocab: AminoVocab,
        max_len: int,
    ) -> None:
        super().__init__()
        self.vocab = vocab
        self.max_len = max_len
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Dataset file not found: {csv_path}")
        df = pd.read_csv(csv_path)
        for col in REQUIRED_COLUMNS:
            if col not in df.columns:
                raise ValueError(f"Missing required column '{col}' in {csv_path}")
        self.seq_a = df["seq_a"].astype(str).tolist()
        self.seq_b = df["seq_b"].astype(str).tolist()
        self.labels = df["label"].astype(int).tolist()

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        sa = self.seq_a[idx]
        sb = self.seq_b[idx]
        la = tokenize_sequence(sa, self.vocab, self.max_len)
        lb = tokenize_sequence(sb, self.vocab, self.max_len)
        label = self.labels[idx]
        return (
            torch.tensor(la, dtype=torch.long),
            torch.tensor(lb, dtype=torch.long),
            torch.tensor(label, dtype=torch.float32),
        )


def create_dataloaders(
    csv_path: str,
    vocab: AminoVocab,
    max_len: int,
    batch_size: int,
    val_size: float,
    test_size: float,
    num_workers: int = 0,
    random_seed: int = 42,
    explicit_test_csv: Optional[str] = None,
):
    dataset = PPIDataset(csv_path, vocab=vocab, max_len=max_len)

    if explicit_test_csv and os.path.exists(explicit_test_csv):
        test_dataset = PPIDataset(explicit_test_csv, vocab=vocab, max_len=max_len)
        # Split train/val from the main dataset with minimum 1 sample in val if possible
        total = len(dataset)
        if total <= 1:
            n_val = 0
            n_train = total
        else:
            n_val = max(1, int(round(val_size * total))) if val_size > 0 else 1
            n_val = min(n_val, total - 1)
            n_train = total - n_val
        gen = torch.Generator().manual_seed(random_seed)
        train_dataset, val_dataset = random_split(dataset, [n_train, n_val], generator=gen)
    else:
        # Split dataset into train/val/test with minimum 1 sample in each non-zero split
        total = len(dataset)
        if total <= 2:
            # Keep all for training except optionally 1 for val
            n_test = 0
            if total == 2 and val_size > 0:
                n_val = 1
                n_train = 1
            else:
                n_val = 0
                n_train = total
        else:
            n_test = max(1, int(round(test_size * total))) if test_size > 0 else 1
            n_test = min(n_test, total - 2)  # leave at least 2 for train+val
            remaining = total - n_test
            if remaining <= 1:
                n_val = 0
                n_train = remaining
            else:
                n_val = max(1, int(round(val_size * remaining))) if val_size > 0 else 1
                n_val = min(n_val, remaining - 1)
                n_train = remaining - n_val
        gen = torch.Generator().manual_seed(random_seed)
        train_dataset, val_dataset, test_dataset = random_split(
            dataset, [n_train, n_val, n_test], generator=gen
        )

    def _loader(ds, shuffle: bool):
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
        )

    return _loader(train_dataset, True), _loader(val_dataset, False), _loader(test_dataset, False)


