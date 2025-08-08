import argparse
import os
import pandas as pd
from Bio import SeqIO


def load_fasta_sequences(fasta_path: str) -> dict:
    id_to_seq = {}
    with open(fasta_path, "r", encoding="utf-8") as handle:
        for record in SeqIO.parse(handle, "fasta"):
            # Take first whitespace-separated token from record.id as ID
            prot_id = str(record.id).split()[0]
            id_to_seq[prot_id] = str(record.seq)
    return id_to_seq


def main():
    parser = argparse.ArgumentParser(description="Prepare PPI CSV from FASTA and pairs file")
    parser.add_argument("--fasta", required=True, help="Path to proteins FASTA")
    parser.add_argument("--pairs", required=True, help="TSV/CSV with id pairs and labels")
    parser.add_argument("--id_col_a", default="prot_a")
    parser.add_argument("--id_col_b", default="prot_b")
    parser.add_argument("--label_col", default="label")
    parser.add_argument("--sep", default=None, help="Delimiter for pairs file (auto by extension if None)")
    parser.add_argument("--out_csv", required=True)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    id_to_seq = load_fasta_sequences(args.fasta)

    # Infer separator by extension
    sep = args.sep
    if sep is None:
        if args.pairs.lower().endswith(".tsv"):
            sep = "\t"
        else:
            sep = ","

    pairs = pd.read_csv(args.pairs, sep=sep)
    if not {args.id_col_a, args.id_col_b, args.label_col}.issubset(pairs.columns):
        raise ValueError("Pairs file must contain id_col_a, id_col_b, label_col")

    rows = []
    missing = 0
    for _, row in pairs.iterrows():
        ida = str(row[args.id_col_a])
        idb = str(row[args.id_col_b])
        label = int(row[args.label_col])
        sa = id_to_seq.get(ida)
        sb = id_to_seq.get(idb)
        if sa is None or sb is None:
            missing += 1
            continue
        rows.append({"seq_a": sa, "seq_b": sb, "label": label})

    if missing:
        print(f"Warning: {missing} pairs skipped due to missing sequences")

    out_df = pd.DataFrame(rows)
    out_df.to_csv(args.out_csv, index=False)
    print(f"Wrote {len(out_df)} rows to {args.out_csv}")


if __name__ == "__main__":
    main()


