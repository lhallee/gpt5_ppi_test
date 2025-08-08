from __future__ import annotations

import os
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from rich.console import Console
from rich.table import Table

from .utils import compute_classification_metrics, count_parameters


console = Console()


def train_one_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: GradScaler | None,
) -> float:
    model.train()
    criterion = nn.BCEWithLogitsLoss()
    running_loss = 0.0
    n_samples = 0

    for seq_a, seq_b, labels in tqdm(loader, desc="train", leave=False):
        seq_a = seq_a.to(device)
        seq_b = seq_b.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            with autocast():
                logits = model(seq_a, seq_b)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(seq_a, seq_b)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

        bs = labels.size(0)
        running_loss += loss.item() * bs
        n_samples += bs

    return running_loss / max(1, n_samples)


@torch.no_grad()
def evaluate(model: nn.Module, loader, device: torch.device) -> Dict[str, float]:
    model.eval()
    y_true = []
    y_prob = []
    criterion = nn.BCEWithLogitsLoss()
    total_loss = 0.0
    n_samples = 0

    num_batches = 0
    for seq_a, seq_b, labels in tqdm(loader, desc="eval", leave=False):
        num_batches += 1
        seq_a = seq_a.to(device)
        seq_b = seq_b.to(device)
        labels = labels.to(device)
        logits = model(seq_a, seq_b)
        loss = criterion(logits, labels)
        probs = torch.sigmoid(logits)

        bs = labels.size(0)
        total_loss += loss.item() * bs
        n_samples += bs
        y_true.append(labels.detach().cpu().numpy())
        y_prob.append(probs.detach().cpu().numpy())

    if num_batches == 0:
        metrics = {"loss": float("nan"), "accuracy": float("nan"), "f1": float("nan"), "precision": float("nan"), "recall": float("nan"), "auroc": float("nan"), "auprc": float("nan")}
    else:
        y_true = np.concatenate(y_true, axis=0)
        y_prob = np.concatenate(y_prob, axis=0)
        metrics = compute_classification_metrics(y_true, y_prob)
        metrics["loss"] = float(total_loss / max(1, n_samples))
    return metrics


def format_metrics_table(title: str, metrics: Dict[str, float]) -> None:
    table = Table(title=title)
    table.add_column("metric")
    table.add_column("value")
    for k in sorted(metrics.keys()):
        table.add_row(k, f"{metrics[k]:.4f}")
    console.print(table)


def fit(
    model: nn.Module,
    train_loader,
    val_loader,
    test_loader,
    epochs: int,
    lr: float,
    weight_decay: float,
    device: torch.device,
    output_dir: str,
    amp: bool = True,
) -> Dict[str, float]:
    os.makedirs(output_dir, exist_ok=True)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scaler = GradScaler(enabled=amp)
    best_val = float("inf")
    best_path = os.path.join(output_dir, "best.ckpt")

    console.print(f"Model parameters: {count_parameters(model):,}")

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, scaler if amp else None)
        val_metrics = evaluate(model, val_loader, device)
        format_metrics_table(f"Epoch {epoch} - Validation", val_metrics)

        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            torch.save({"model": model.state_dict()}, best_path)
            console.print(f"Saved new best checkpoint to {best_path}")

    # Load best
    if os.path.exists(best_path):
        state = torch.load(best_path, map_location=device)
        model.load_state_dict(state["model"]) 

    test_metrics = evaluate(model, test_loader, device)
    format_metrics_table("Test", test_metrics)
    return test_metrics


