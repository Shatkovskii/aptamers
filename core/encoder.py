"""
ESM encoder for protein sequences from the aptamers dataset.
Encodes target_seq_ab, target_seq_apt, and Antibody Sequence columns
and saves embeddings as .npy files in data/esm_embeddings/.

Optimizations: batched forward passes, sequence deduplication, length-based
batching to minimize padding, truncation for very long sequences.
"""

import re
from pathlib import Path

import esm
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

COLUMNS_TO_ENCODE = ["target_seq_ab", "target_seq_apt", "Antibody Sequence"]
EMBEDDINGS_SUBDIR = "esm_embeddings"
ESM_MAX_LENGTH = 1024


def _prepare_sequence(seq: str) -> str:
    """Prepare sequence for ESM: remove invalid chars, concatenate chains."""
    if pd.isna(seq) or not isinstance(seq, str):
        return ""
    # ESM expects protein sequences; remove chain separator | and concatenate
    seq = str(seq).replace("|", "")
    return seq.strip()


def _encode_batch(
    model,
    batch_converter,
    alphabet,
    device: torch.device,
    batch: list[tuple[str, str]],
) -> list[np.ndarray]:
    """
    Encode a batch of (label, seq) pairs. Returns list of mean-pooled embeddings.
    """
    if not batch:
        return []
    batch_labels, batch_strs, batch_tokens = batch_converter(batch)
    batch_tokens = batch_tokens.to(device)
    batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)

    with torch.inference_mode():
        results = model(batch_tokens, repr_layers=[33], return_contacts=False)
    token_repr = results["representations"][33]  # (B, L, D)

    embeddings = []
    for i in range(len(batch)):
        start, end = 1, batch_lens[i].item() - 1
        if end <= start:
            # degenerate: no tokens between BOS/EOS
            seq_repr = token_repr[i, 1:2].mean(0)
        else:
            seq_repr = token_repr[i, start:end].mean(0)
        embeddings.append(seq_repr.detach().cpu().numpy())
    return embeddings


def encode_dataset_with_esm(
    csv_path: str | Path,
    output_dir: str | Path | None = None,
    esm_model_name: str = "esm2_t33_650M_UR50D",
    device: str | None = None,
    batch_size: int = 32,
) -> dict[str, list[dict]]:
    """
    Encode target_seq_ab, target_seq_apt, Antibody Sequence from the dataset
    using ESM and save .npy embeddings.

    Args:
        csv_path: Path to 3_checked_intersections_180t.csv
        output_dir: Base output directory (default: data/esm_embeddings)
        esm_model_name: ESM model to use
        device: torch device (cuda/cpu)
        batch_size: Number of sequences per forward pass (higher = faster on GPU,
            but more memory; use 8–16 for long antibodies on limited GPU)

    Returns:
        manifests: dict mapping column name -> list of {idx, filepath, ...}
    """
    csv_path = Path(csv_path)
    if output_dir is None:
        output_dir = csv_path.parent / EMBEDDINGS_SUBDIR
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    print(f"Using device: {device}")
    print(f"Loading ESM model: {esm_model_name}")

    model, alphabet = getattr(esm.pretrained, esm_model_name)()
    batch_converter = alphabet.get_batch_converter()
    model = model.to(device)
    model.eval()

    df = pd.read_csv(csv_path, index_col=0)

    # Create subfolders per column
    column_dirs = {}
    for col in COLUMNS_TO_ENCODE:
        if col not in df.columns:
            raise KeyError(
                f"Column '{col}' not found in dataset. Available: {list(df.columns)}"
            )
        safe_name = re.sub(r"[^\w\-]", "_", col)
        d = output_dir / safe_name
        d.mkdir(parents=True, exist_ok=True)
        column_dirs[col] = d

    # Collect (col, idx, seq); deduplicate by (col, seq) to avoid redundant encoding
    items: list[tuple[str, int, str]] = []
    seen: dict[tuple[str, str], None] = {}
    for idx, row in df.iterrows():
        for col in COLUMNS_TO_ENCODE:
            seq = _prepare_sequence(row[col])
            if not seq:
                continue
            seq = seq[:ESM_MAX_LENGTH]  # truncate if too long
            items.append((col, idx, seq))
            seen[(col, seq)] = None

    unique_seqs = list(seen.keys())
    print(f"Total items: {len(items)}, unique sequences: {len(unique_seqs)} "
          f"(~{100 * (1 - len(unique_seqs) / max(1, len(items))):.0f}% dedup)")

    # Sort unique seqs by length to reduce padding waste in batches
    unique_seqs.sort(key=lambda x: len(x[1]))

    # Encode unique sequences in batches
    seq_to_emb: dict[tuple[str, str], np.ndarray] = {}
    for i in tqdm(
        range(0, len(unique_seqs), batch_size),
        desc="Encoding",
        unit="batch",
    ):
        chunk = unique_seqs[i : i + batch_size]
        batch = [(f"{col}_{j}", seq[:ESM_MAX_LENGTH]) for j, (col, seq) in enumerate(chunk)]
        try:
            embs = _encode_batch(model, batch_converter, alphabet, device, batch)
            for (col, seq), emb in zip(chunk, embs):
                seq_to_emb[(col, seq)] = emb
        except Exception as e:
            tqdm.write(f"Batch failed: {e}")
            for j, (col, seq) in enumerate(chunk):
                try:
                    single = [(f"{col}_s", seq)]
                    emb = _encode_batch(model, batch_converter, alphabet, device, single)[0]
                    seq_to_emb[(col, seq)] = emb
                except Exception as e2:
                    tqdm.write(f"  Skip {col} len={len(seq)}: {e2}")

    # Save per (col, idx) and build manifest
    manifests: dict[str, list[dict]] = {col: [] for col in COLUMNS_TO_ENCODE}
    for col, idx, seq in tqdm(items, desc="Saving", unit="file"):
        key = (col, seq)
        if key not in seq_to_emb:
            continue
        out_path = column_dirs[col] / f"embed_{idx}.npy"
        np.save(out_path, seq_to_emb[key])
        manifests[col].append({"idx": idx, "filepath": str(out_path), "column": col})

    # Save manifests
    for col in COLUMNS_TO_ENCODE:
        if manifests[col]:
            mf = pd.DataFrame(manifests[col])
            mf.to_csv(column_dirs[col] / "manifest.csv", index=False)
            print(f"{col}: {len(manifests[col])} embeddings -> {column_dirs[col]}")

    return manifests


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Encode sequences with ESM")
    parser.add_argument(
        "--csv",
        type=str,
        default="data/3_checked_intersections_180t.csv",
        help="Path to dataset CSV",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory (default: data/esm_embeddings)",
    )
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Sequences per forward pass (default: 32)",
    )
    args = parser.parse_args()

    output = args.output
    if output is None:
        csv_path = Path(args.csv)
        output = csv_path.parent / EMBEDDINGS_SUBDIR

    encode_dataset_with_esm(
        csv_path=args.csv,
        output_dir=output,
        device=args.device,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
