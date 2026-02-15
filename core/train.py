#!/usr/bin/env python3
"""
Train aptamer decoder. Run after encoding: python -m core.encoder

Usage:
    python -m core.train
    python -m core.train --csv data/3_checked_intersections_180t.csv --embeddings data/esm_embeddings
    python -m core.train --epochs 50 --batch-size 32
"""

import argparse
import csv
import math
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from core.dataset import AptamerDecoderDataset, collate_fn, train_val_split
from core.decoder import PAD_ID, build_aptamer_decoder, compute_loss


def get_args():
    p = argparse.ArgumentParser(description="Train aptamer decoder")
    p.add_argument("--csv", default="data/3_checked_intersections_180t.csv", help="Dataset CSV")
    p.add_argument("--embeddings", default="data/esm_embeddings", help="ESM embeddings dir")
    p.add_argument("--output", default="checkpoints", help="Save checkpoints here")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-steps", type=int, default=500)
    p.add_argument("--max-len", type=int, default=128)
    p.add_argument("--val-ratio", type=float, default=0.15)
    p.add_argument("--early-stopping", type=int, default=5, help="Stop after N epochs without improvement")
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--group-split", action="store_true", help="Split by (target_seq_ab, target_seq_apt) groups")
    return p.parse_args()


class _MetricsAccumulator:
    """Accumulates loss, accuracy, RMSE, R2 over batches (on non-PAD tokens)."""

    def __init__(self):
        self.total_loss = 0.0
        self.n_batches = 0
        self.correct = 0
        self.total_tokens = 0
        self.ss_res = 0.0  # sum of (pred_prob - 1)^2 for correct class + pred_prob^2 for others
        self.ss_tot = 0.0  # sum of (one_hot - mean)^2

    def update(self, logits: torch.Tensor, targets: torch.Tensor, loss_val: float):
        """logits: [B, L, V], targets: [B, L]."""
        self.total_loss += loss_val
        self.n_batches += 1

        mask = targets != PAD_ID  # [B, L]
        preds = logits.argmax(dim=-1)  # [B, L]
        self.correct += (preds[mask] == targets[mask]).sum().item()
        n = mask.sum().item()
        self.total_tokens += n

        # RMSE & R2: compare softmax probabilities vs one-hot targets
        probs = F.softmax(logits, dim=-1)  # [B, L, V]
        V = probs.size(-1)
        # Gather prob of correct class at each position
        target_probs = probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)  # [B, L]
        # Per-token: MSE between prob vector and one-hot = sum_v (p_v - t_v)^2
        #   = (1 - p_correct)^2 + sum_{v != correct} p_v^2
        #   = 1 - 2*p_correct + sum_v p_v^2
        prob_sq_sum = (probs ** 2).sum(dim=-1)  # [B, L]
        per_token_mse = 1.0 - 2.0 * target_probs + prob_sq_sum  # [B, L]
        self.ss_res += per_token_mse[mask].sum().item()
        # SS_tot: variance of one-hot target = mean over V of (t_v - 1/V)^2
        # For one-hot: 1 position has value 1, rest 0. Mean = 1/V.
        # ss_tot_per_token = (1 - 1/V)^2 + (V-1)*(0 - 1/V)^2 = (V-1)/V
        self.ss_tot += n * (V - 1) / V

    def compute(self) -> dict:
        loss = self.total_loss / max(1, self.n_batches)
        acc = self.correct / max(1, self.total_tokens)
        mse = self.ss_res / max(1, self.total_tokens)
        rmse = math.sqrt(max(0, mse))
        r2 = 1.0 - self.ss_res / max(self.ss_tot, 1e-8)
        return {"loss": loss, "accuracy": acc, "rmse": rmse, "r2": r2}


def main():
    args = get_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}")

    df = pd.read_csv(args.csv, index_col=0)
    df = df.dropna(subset=["Aptamer Sequence", "target_seq_ab", "target_seq_apt", "Antibody Sequence"])
    emb_dir = Path(args.embeddings)
    for sub in ["Antibody_Sequence", "target_seq_ab", "target_seq_apt"]:
        d = emb_dir / sub
        if not d.exists():
            raise FileNotFoundError(f"Run encoder first. Missing: {d}")
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
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0)

    sample = train_ds[0]
    d_esm = sample["E_ab"].shape[-1]
    model = build_aptamer_decoder(d_esm=d_esm, max_len=args.max_len)
    model = model.to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    def lr_lambda(step):
        if step < args.warmup_steps:
            return step / args.warmup_steps
        return 1.0
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    # Create run directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.output) / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run directory: {run_dir}")

    # Prepare metrics CSV
    metrics_path = run_dir / "metrics.csv"
    metrics_fields = [
        "epoch",
        "train_loss", "train_accuracy", "train_rmse", "train_r2",
        "val_loss", "val_accuracy", "val_rmse", "val_r2",
    ]
    with open(metrics_path, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=metrics_fields).writeheader()

    best_val = float("inf")
    no_improve = 0

    for ep in range(args.epochs):
        # --- Train ---
        model.train()
        train_metrics = _MetricsAccumulator()
        for batch in train_dl:
            for k in batch:
                batch[k] = batch[k].to(device)
            logits = model(
                batch["E_ab"], batch["E_t_ab"], batch["E_t_apt"],
                batch["mask_ab"], batch["mask_t1"], batch["mask_t2"],
                batch["y_in"], batch["mask_y"],
            )
            loss = compute_loss(logits, batch["y_out"], label_smoothing=0.1)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            with torch.no_grad():
                train_metrics.update(logits, batch["y_out"], loss.item())
        train_stats = train_metrics.compute()

        # --- Val ---
        model.eval()
        val_metrics = _MetricsAccumulator()
        with torch.no_grad():
            for batch in val_dl:
                for k in batch:
                    batch[k] = batch[k].to(device)
                logits = model(
                    batch["E_ab"], batch["E_t_ab"], batch["E_t_apt"],
                    batch["mask_ab"], batch["mask_t1"], batch["mask_t2"],
                    batch["y_in"], batch["mask_y"],
                )
                loss = compute_loss(logits, batch["y_out"], label_smoothing=0.1)
                val_metrics.update(logits, batch["y_out"], loss.item())
        val_stats = val_metrics.compute()

        print(
            f"Epoch {ep+1}/{args.epochs}  "
            f"train[loss={train_stats['loss']:.4f} acc={train_stats['accuracy']:.4f} "
            f"rmse={train_stats['rmse']:.4f} r2={train_stats['r2']:.4f}]  "
            f"val[loss={val_stats['loss']:.4f} acc={val_stats['accuracy']:.4f} "
            f"rmse={val_stats['rmse']:.4f} r2={val_stats['r2']:.4f}]"
        )
        val_loss = val_stats["loss"]

        # Append metrics to CSV
        with open(metrics_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=metrics_fields).writerow({
                "epoch": ep + 1,
                "train_loss": f"{train_stats['loss']:.6f}",
                "train_accuracy": f"{train_stats['accuracy']:.6f}",
                "train_rmse": f"{train_stats['rmse']:.6f}",
                "train_r2": f"{train_stats['r2']:.6f}",
                "val_loss": f"{val_stats['loss']:.6f}",
                "val_accuracy": f"{val_stats['accuracy']:.6f}",
                "val_rmse": f"{val_stats['rmse']:.6f}",
                "val_r2": f"{val_stats['r2']:.6f}",
            })

        if val_loss < best_val:
            best_val = val_loss
            no_improve = 0
            torch.save({"model": model.state_dict(), "epoch": ep, "val_loss": val_loss}, run_dir / "best.pt")
        else:
            no_improve += 1
            if no_improve >= args.early_stopping:
                print(f"Early stopping after {ep+1} epochs")
                break
        torch.save({"model": model.state_dict(), "epoch": ep, "val_loss": val_loss}, run_dir / "last.pt")

    print(f"Best val_loss: {best_val:.4f}. Run directory: {run_dir}")
    return model


if __name__ == "__main__":
    main()
