from __future__ import annotations

import math
from typing import Iterable, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SequenceEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        pad_idx: int,
        embedding_dim: int = 64,
        cnn_channels: int = 128,
        kernel_sizes: tuple[int, ...] = (3, 5, 7),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        convs = []
        for k in kernel_sizes:
            convs.append(
                nn.Sequential(
                    nn.Conv1d(embedding_dim, cnn_channels, kernel_size=k, padding=k // 2),
                    nn.ReLU(inplace=True),
                )
            )
        self.convs = nn.ModuleList(convs)
        self.dropout = nn.Dropout(dropout)
        self.out_dim = cnn_channels * len(kernel_sizes)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # input_ids: [B, L]
        x = self.embedding(input_ids)  # [B, L, E]
        x = x.transpose(1, 2)  # [B, E, L]
        features = []
        for conv in self.convs:
            h = conv(x)  # [B, C, L]
            h = F.adaptive_max_pool1d(h, 1).squeeze(-1)  # [B, C]
            features.append(h)
        out = torch.cat(features, dim=1)  # [B, C*K]
        out = self.dropout(out)
        return out


class PPISiameseModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        pad_idx: int,
        embedding_dim: int = 64,
        cnn_channels: int = 128,
        kernel_sizes: tuple[int, ...] = (3, 5, 7),
        dropout: float = 0.1,
        projector_hidden: int = 256,
    ) -> None:
        super().__init__()
        self.encoder = SequenceEncoder(
            vocab_size=vocab_size,
            pad_idx=pad_idx,
            embedding_dim=embedding_dim,
            cnn_channels=cnn_channels,
            kernel_sizes=kernel_sizes,
            dropout=dropout,
        )
        in_dim = self.encoder.out_dim * 2
        self.classifier = nn.Sequential(
            nn.Linear(in_dim, projector_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(projector_hidden, 1),
        )

    def forward(self, seq_a: torch.Tensor, seq_b: torch.Tensor) -> torch.Tensor:
        fa = self.encoder(seq_a)
        fb = self.encoder(seq_b)
        fused = torch.cat([fa, fb], dim=1)
        logits = self.classifier(fused).squeeze(-1)
        return logits


