#!/usr/bin/env python3
"""
Evaluate a saved checkpoint (e.g. data/checkpoints/last.pt) on train/val splits
and produce a metrics.csv file.

Usage:
    python -m aux.eval_checkpoint
    python -m aux.eval_checkpoint --checkpoint data/checkpoints/last.pt --output data/checkpoints/metrics.csv
"""

import argparse
import csv
import math
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from core.dataset import AptamerDecoderDataset, collate_fn, train_val_split
from core.decoder import PAD_ID, build_aptamer_decoder, compute_loss


class _MetricsAccumulator:
    """Accumulates loss, accuracy, RMSE, R2 over batches (on non-PAD tokens)."""

    def __init__(self):
        self.total_loss = 0.0
        self.n_batches = 0
        self.correct = 0
        self.total_tokens = 0
        self.ss_res = 0.0
        self.ss_tot = 0.0

    def update(self, logits: torch.Tensor, targets: torch.Tensor, loss_val: float):
        self.total_loss += loss_val
        self.n_batches += 1

        mask = targets != PAD_ID
        preds = logits.argmax(dim=-1)
        self.correct += (preds[mask] == targets[mask]).sum().item()
        n = mask.sum().item()
        self.total_tokens += n

        probs = F.softmax(logits, dim=-1)
        V = probs.size(-1)
        target_probs = probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        prob_sq_sum = (probs ** 2).sum(dim=-1)
        per_token_mse = 1.0 - 2.0 * target_probs + prob_sq_sum
        self.ss_res += per_token_mse[mask].sum().item()
        self.ss_tot += n * (V - 1) / V

    def compute(self) -> dict:
        loss = self.total_loss / max(1, self.n_batches)
        acc = self.correct / max(1, self.total_tokens)
        mse = self.ss_res / max(1, self.total_tokens)
        rmse = math.sqrt(max(0, mse))
        r2 = 1.0 - self.ss_res / max(self.ss_tot, 1e-8)
        return {"loss": loss, "accuracy": acc, "rmse": rmse, "r2": r2}


def evaluate(model, dataloader, device):
    model.eval()
    acc = _MetricsAccumulator()
    with torch.no_grad():
        for batch in dataloader:
            for k in batch:
                batch[k] = batch[k].to(device)
            logits = model(
                batch["E_ab"], batch["E_t_ab"], batch["E_t_apt"],
                batch["mask_ab"], batch["mask_t1"], batch["mask_t2"],
                batch["y_in"], batch["mask_y"],
            )
            loss = compute_loss(logits, batch["y_out"], label_smoothing=0.1)
            acc.update(logits, batch["y_out"], loss.item())
    return acc.compute()


def main():
    p = argparse.ArgumentParser(description="Evaluate checkpoint and save metrics.csv")
    p.add_argument("--checkpoint", default="data/checkpoints/last.pt", help="Path to .pt checkpoint")
    p.add_argument("--csv", default="data/3_checked_intersections_180t.csv", help="Dataset CSV")
    p.add_argument("--embeddings", default="data/esm_embeddings", help="ESM embeddings dir")
    p.add_argument("--output", default=None, help="Output metrics.csv path (default: next to checkpoint)")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-len", type=int, default=128)
    p.add_argument("--val-ratio", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--group-split", action="store_true")
    p.add_argument("--device", default=None)
    args = p.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}")

    # Load data
    df = pd.read_csv(args.csv, index_col=0)
    df = df.dropna(subset=["Aptamer Sequence", "target_seq_ab", "target_seq_apt", "Antibody Sequence"])
    emb_dir = Path(args.embeddings)

    def ids_in(col: str) -> set[int]:
        return {int(f.stem.replace("embed_", "")) for f in (emb_dir / col).glob("embed_*.npy")}

    existing = ids_in("target_seq_ab") & ids_in("target_seq_apt") & ids_in("Antibody_Sequence")
    df = df[df.index.isin(existing)]
    print(f"Samples with embeddings: {len(df)}")

    group_cols = ["target_seq_ab", "target_seq_apt"] if args.group_split else None
    train_df, val_df = train_val_split(df, val_ratio=args.val_ratio, seed=args.seed, group_cols=group_cols)
    print(f"Train: {len(train_df)}, Val: {len(val_df)}")

    train_ds = AptamerDecoderDataset(train_df, emb_dir, max_len=args.max_len)
    val_ds = AptamerDecoderDataset(val_df, emb_dir, max_len=args.max_len)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0)

    # Load model
    sample = train_ds[0]
    d_esm = sample["E_ab"].shape[-1]
    model = build_aptamer_decoder(d_esm=d_esm, max_len=args.max_len)

    ckpt_path = Path(args.checkpoint)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model"])
    model = model.to(device)
    epoch = ckpt.get("epoch", "?")
    print(f"Loaded checkpoint: {ckpt_path} (epoch {epoch})")

    # Evaluate
    print("Evaluating on train set...")
    train_stats = evaluate(model, train_dl, device)
    print("Evaluating on val set...")
    val_stats = evaluate(model, val_dl, device)

    print(
        f"Train: loss={train_stats['loss']:.4f} acc={train_stats['accuracy']:.4f} "
        f"rmse={train_stats['rmse']:.4f} r2={train_stats['r2']:.4f}"
    )
    print(
        f"Val:   loss={val_stats['loss']:.4f} acc={val_stats['accuracy']:.4f} "
        f"rmse={val_stats['rmse']:.4f} r2={val_stats['r2']:.4f}"
    )

    # Save metrics.csv
    out_path = Path(args.output) if args.output else ckpt_path.parent / "metrics.csv"
    fields = [
        "epoch",
        "train_loss", "train_accuracy", "train_rmse", "train_r2",
        "val_loss", "val_accuracy", "val_rmse", "val_r2",
    ]
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerow({
            "epoch": epoch,
            "train_loss": f"{train_stats['loss']:.6f}",
            "train_accuracy": f"{train_stats['accuracy']:.6f}",
            "train_rmse": f"{train_stats['rmse']:.6f}",
            "train_r2": f"{train_stats['r2']:.6f}",
            "val_loss": f"{val_stats['loss']:.6f}",
            "val_accuracy": f"{val_stats['accuracy']:.6f}",
            "val_rmse": f"{val_stats['rmse']:.6f}",
            "val_r2": f"{val_stats['r2']:.6f}",
        })
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
