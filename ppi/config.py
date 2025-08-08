from dataclasses import dataclass


@dataclass
class TrainConfig:
    train_csv: str
    output_dir: str = "runs/default"
    test_csv: str | None = None
    val_size: float = 0.1
    test_size: float = 0.1
    random_seed: int = 42

    # Data
    max_len: int = 512
    vocab: str = "ACDEFGHIKLMNPQRSTVWY"  # 20 canonical aas
    unknown_token: str = "X"
    pad_token: str = "_"

    # Model
    embedding_dim: int = 64
    cnn_channels: int = 128
    cnn_kernel_sizes: tuple[int, ...] = (3, 5, 7)
    cnn_dropout: float = 0.1
    projector_hidden: int = 256

    # Optimization
    batch_size: int = 128
    epochs: int = 20
    lr: float = 3e-4
    weight_decay: float = 0.0
    num_workers: int = 2
    amp: bool = True

    # Logging / Checkpointing
    log_every: int = 50
    ckpt_every: int = 1


