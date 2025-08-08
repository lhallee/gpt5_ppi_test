# PROMPT
Turn this repository into a fully functioning project that trains a deep learning model to predict the binary protein-protein interactions of proteins using only their primary sequences.

## Protein-Protein Interaction (PPI) Prediction from Primary Sequences

This repository trains a deep learning model to predict binary protein-protein interactions (PPIs) using only primary amino acid sequences.

### Features
- End-to-end training pipeline in PyTorch
- Siamese CNN encoder over amino acid sequences
- Flexible dataset input from CSV or FASTA + pairs
- Stratified train/val/test splits with reproducibility
- Metrics: AUROC, AUPRC, accuracy, F1, precision, recall
- Mixed-precision support (if CUDA available)
- Out-of-the-box toy dataset for a quick sanity check

### Project Structure
```
gpt5_ppi_test/
  main.py                 # CLI entry point
  requirements.txt        # Python dependencies
  README.md               # This file
  ppi/
    __init__.py
    config.py             # Default hyperparameters
    data.py               # Datasets, parsing, DataLoaders
    model.py              # Siamese CNN model
    train.py              # Training and evaluation loop
    utils.py              # Tokenization, metrics, seeding, helpers
  data/
    toy/
      pairs.csv           # Small toy dataset (seq_a, seq_b, label)
  scripts/
    prepare_dataset.py    # Helper to build a dataset from FASTA + pairs
```

### Installation
Create a virtual environment and install dependencies:

```bash
python -m venv .venv
. .venv/Scripts/activate  # on Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Dataset Formats
You can train from a simple CSV with the following columns:

- `seq_a`: primary amino acid sequence for protein A
- `seq_b`: primary amino acid sequence for protein B
- `label`: 1 for interacting, 0 for non-interacting

CSV example:

```csv
seq_a,seq_b,label
MKTAYIAKQRQISFVKSHFSRQDILDLWQ,MGDVEKGKKIFIMKCSQCHTVEKGGKHKTG,1
MADQLTEEQIAEFKEAFSLFDKDGDGTITTKELGTVMRSLGQ,MGSSHHHHHHSSGLVPRGSHMASMTGGQQMGR,0
```

Alternatively, you can provide FASTA files for proteins and a TSV/CSV with protein identifiers pairs. Use `scripts/prepare_dataset.py` to merge sequences with pairs and produce the `pairs.csv` file expected by the trainer.

### Quickstart (Toy Dataset)
Train on the included toy dataset to verify the pipeline works end-to-end:

```bash
python main.py \
  --train_csv data/toy/pairs.csv \
  --output_dir runs/toy \
  --epochs 5 \
  --batch_size 32 \
  --max_len 512
```

You should see training/validation metrics printed each epoch and a best model checkpoint saved to `runs/toy/best.ckpt`.

### Training on Your Dataset
If you already have a CSV:

```bash
python main.py \
  --train_csv path/to/your_pairs.csv \
  --output_dir runs/exp1 \
  --epochs 20 \
  --batch_size 128 \
  --lr 3e-4 \
  --max_len 1024
```

From FASTA + pairs (where pairs contain two protein IDs and a label):

```bash
python scripts/prepare_dataset.py \
  --fasta path/to/proteins.fasta \
  --pairs path/to/pairs.tsv \
  --id_col_a prot_a --id_col_b prot_b --label_col label \
  --out_csv data/my_dataset/pairs.csv

python main.py --train_csv data/my_dataset/pairs.csv --output_dir runs/my_dataset
```

### Reproducibility
We set seeds and perform stratified splits. For exact reproducibility across hardware and PyTorch versions, consider disabling nondeterministic CUDA kernels.

### Notes
- The included toy dataset is for functionality testing only and is not biologically validated.
- For realistic training, use curated PPI datasets and ensure appropriate negative sampling.

### License
MIT


