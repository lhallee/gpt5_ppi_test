from __future__ import annotations

import argparse
import os
import torch

from ppi.config import TrainConfig
from ppi.utils import AminoVocab, set_seed, save_config
from ppi.data import create_dataloaders
from ppi.model import PPISiameseModel
from ppi.train import fit


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train a PPI classifier from primary sequences")
    p.add_argument("--train_csv", type=str, required=True)
    p.add_argument("--test_csv", type=str, default=None)
    p.add_argument("--output_dir", type=str, default="runs/default")
    p.add_argument("--val_size", type=float, default=0.1)
    p.add_argument("--test_size", type=float, default=0.1)
    p.add_argument("--random_seed", type=int, default=42)

    # Data
    p.add_argument("--max_len", type=int, default=512)
    p.add_argument("--vocab", type=str, default="ACDEFGHIKLMNPQRSTVWY")
    p.add_argument("--unknown_token", type=str, default="X")
    p.add_argument("--pad_token", type=str, default="_")

    # Model
    p.add_argument("--embedding_dim", type=int, default=64)
    p.add_argument("--cnn_channels", type=int, default=128)
    p.add_argument("--cnn_kernel_sizes", type=str, default="3,5,7")
    p.add_argument("--cnn_dropout", type=float, default=0.1)
    p.add_argument("--projector_hidden", type=int, default=256)

    # Optimization
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--no_amp", action="store_true", help="Disable mixed-precision")
    return p


def parse_kernel_sizes(s: str) -> tuple[int, ...]:
    parts = [int(x.strip()) for x in s.split(",") if x.strip()]
    return tuple(parts) if parts else (3, 5, 7)


def main():
    args = build_argparser().parse_args()
    config = TrainConfig(
        train_csv=args.train_csv,
        test_csv=args.test_csv,
        output_dir=args.output_dir,
        val_size=args.val_size,
        test_size=args.test_size,
        random_seed=args.random_seed,
        max_len=args.max_len,
        vocab=args.vocab,
        unknown_token=args.unknown_token,
        pad_token=args.pad_token,
        embedding_dim=args.embedding_dim,
        cnn_channels=args.cnn_channels,
        cnn_kernel_sizes=parse_kernel_sizes(args.cnn_kernel_sizes),
        cnn_dropout=args.cnn_dropout,
        projector_hidden=args.projector_hidden,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        amp=not args.no_amp,
    )

    set_seed(config.random_seed)
    os.makedirs(config.output_dir, exist_ok=True)
    save_config(config, config.output_dir)

    vocab = AminoVocab(config.vocab, pad_token=config.pad_token, unknown_token=config.unknown_token)

    train_loader, val_loader, test_loader = create_dataloaders(
        csv_path=config.train_csv,
        vocab=vocab,
        max_len=config.max_len,
        batch_size=config.batch_size,
        val_size=config.val_size,
        test_size=config.test_size,
        num_workers=config.num_workers,
        random_seed=config.random_seed,
        explicit_test_csv=config.test_csv,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PPISiameseModel(
        vocab_size=len(vocab.itos),
        pad_idx=vocab.pad_idx,
        embedding_dim=config.embedding_dim,
        cnn_channels=config.cnn_channels,
        kernel_sizes=config.cnn_kernel_sizes,
        dropout=config.cnn_dropout,
        projector_hidden=config.projector_hidden,
    ).to(device)

    fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        epochs=config.epochs,
        lr=config.lr,
        weight_decay=config.weight_decay,
        device=device,
        output_dir=config.output_dir,
        amp=config.amp and device.type == "cuda",
    )


if __name__ == "__main__":
    main()


