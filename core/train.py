#!/usr/bin/env python3
"""
Train aptamer decoder. Run after encoding: python -m core.encoder

Usage:
    python -m core.train
    python -m core.train --csv data/preprocessed_data.csv --embeddings data/esm_embeddings
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
from core.decoder import (
    BOS_ID, EOS_ID, PAD_ID,
    build_aptamer_decoder, compute_loss, greedy_decode,
)
from aux.plot_metrics import plot_run


def _levenshtein(s: list[int], t: list[int]) -> int:
    """Levenshtein distance between two integer sequences."""
    n, m = len(s), len(t)
    if n == 0:
        return m
    if m == 0:
        return n
    prev = list(range(m + 1))
    curr = [0] * (m + 1)
    for i in range(1, n + 1):
        curr[0] = i
        for j in range(1, m + 1):
            cost = 0 if s[i - 1] == t[j - 1] else 1
            curr[j] = min(curr[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost)
        prev, curr = curr, prev
    return prev[m]


def _extract_seq(ids: torch.Tensor) -> list[int]:
    """Extract token ids from a single sequence, stripping BOS/PAD/EOS."""
    out = []
    for v in ids.tolist():
        if v in (PAD_ID, EOS_ID):
            break
        if v == BOS_ID:
            continue
        out.append(v)
    return out


def _norm_edit_distance(s: list[int], t: list[int]) -> float:
    """Levenshtein distance normalized by max(len(s), len(t)). Returns 0..1."""
    d = _levenshtein(s, t)
    maxlen = max(len(s), len(t))
    return d / maxlen if maxlen > 0 else 0.0


def get_args():
    p = argparse.ArgumentParser(description="Train aptamer decoder")
    p.add_argument("--csv", default="data/preprocessed_data.csv", help="Dataset CSV")
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
    p.add_argument(
        "--split-mode",
        default="random",
        choices=["random", "target", "antibody", "aptamer", "full"],
        help=(
            "Validation split strategy: "
            "random (default, no grouping), "
            "target (group by target_seq_ab + target_seq_apt), "
            "antibody (group by Antibody Sequence), "
            "aptamer (group by Aptamer Sequence), "
            "full (group by Antibody Sequence + target_seq_ab + target_seq_apt)"
        ),
    )
    return p.parse_args()


class _MetricsAccumulator:
    """Accumulates loss, perplexity, teacher-forcing & autoregressive seq metrics."""

    def __init__(self):
        self.total_loss = 0.0
        self.n_batches = 0
        self.total_nll = 0.0
        self.total_tokens = 0
        # Sequence-level metrics
        self.em = 0
        self.ned_tf = 0.0
        self.total_seqs = 0
        # Autoregressive edit distance (filled via update_ar)
        self.ned_ar = 0.0
        self.total_seqs_ar = 0

    def update(self, logits: torch.Tensor, targets: torch.Tensor, loss_val: float):
        """Update with teacher-forcing logits. logits: [B, L, V], targets: [B, L]."""
        self.total_loss += loss_val
        self.n_batches += 1

        mask = targets != PAD_ID
        n_tokens = mask.sum().item()
        self.total_tokens += n_tokens

        # NLL for perplexity (no label smoothing)
        log_probs = F.log_softmax(logits, dim=-1)
        token_nll = -log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        self.total_nll += token_nll[mask].sum().item()

        # Teacher-forcing: greedy next-token prediction vs target
        preds = logits.argmax(dim=-1)
        B = targets.size(0)
        self.total_seqs += B
        for b in range(B):
            pred_seq = _extract_seq(preds[b])
            tgt_seq = _extract_seq(targets[b])
            if pred_seq == tgt_seq:
                self.em += 1
            self.ned_tf += _norm_edit_distance(pred_seq, tgt_seq)

    def update_ar(self, generated: torch.Tensor, targets: torch.Tensor):
        """Update with autoregressive generated sequences. generated/targets: [B, L]."""
        B = targets.size(0)
        self.total_seqs_ar += B
        for b in range(B):
            gen_seq = _extract_seq(generated[b])
            tgt_seq = _extract_seq(targets[b])
            self.ned_ar += _norm_edit_distance(gen_seq, tgt_seq)

    def compute(self) -> dict:
        loss = self.total_loss / max(1, self.n_batches)
        avg_nll = self.total_nll / max(1, self.total_tokens)
        perplexity = math.exp(min(avg_nll, 100))
        em = self.em / max(1, self.total_seqs)
        ned_tf = self.ned_tf / max(1, self.total_seqs)
        ned_ar = self.ned_ar / max(1, self.total_seqs_ar) if self.total_seqs_ar > 0 else 0.0
        return {
            "loss": loss,
            "perplexity": perplexity,
            "em": em,
            "ed_tf": ned_tf,
            "ed_ar": ned_ar,
        }


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

    _SPLIT_MODE_COLS = {
        "random": None,
        "target": ["target_seq_ab", "target_seq_apt"],
        "antibody": ["Antibody Sequence"],
        "aptamer": ["Aptamer Sequence"],
        "full": ["Antibody Sequence", "target_seq_ab", "target_seq_apt"],
    }
    group_cols = _SPLIT_MODE_COLS[args.split_mode]
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

    # Save run description
    with open(run_dir / "description.txt", "w") as f:
        f.write(f"Dataset: {args.csv}\n")
        f.write(f"Embeddings: {args.embeddings}\n")
        f.write(f"Total samples with embeddings: {len(df)}\n")
        f.write(f"Train size: {len(train_df)}\n")
        f.write(f"Val size: {len(val_df)}\n")
        f.write(f"Val ratio: {args.val_ratio}\n")
        f.write(f"Split mode: {args.split_mode}\n")
        f.write(f"Group columns: {group_cols}\n")
        f.write(f"Seed: {args.seed}\n")
        f.write(f"\n")
        f.write(f"epochs: {args.epochs}\n")
        f.write(f"batch_size: {args.batch_size}\n")
        f.write(f"lr: {args.lr}\n")
        f.write(f"weight_decay: {args.weight_decay}\n")
        f.write(f"warmup_steps: {args.warmup_steps}\n")
        f.write(f"max_len: {args.max_len}\n")
        f.write(f"early_stopping: {args.early_stopping}\n")
        f.write(f"device: {device}\n")

    # Prepare metrics CSV
    metrics_path = run_dir / "metrics.csv"
    metrics_fields = [
        "epoch",
        "train_loss", "train_perplexity", "train_em", "train_ed_tf",
        "val_loss", "val_perplexity", "val_em", "val_ed_tf",
        "val_ed_ar",
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
                loss = compute_loss(logits, batch["y_out"], label_smoothing=0.0)
                val_metrics.update(logits, batch["y_out"], loss.item())
                # Autoregressive generation for AR metrics
                generated = greedy_decode(
                    model,
                    batch["E_ab"], batch["E_t_ab"], batch["E_t_apt"],
                    batch["mask_ab"], batch["mask_t1"], batch["mask_t2"],
                    max_len=args.max_len, device=device,
                )
                val_metrics.update_ar(generated, batch["y_out"])
        val_stats = val_metrics.compute()

        print(
            f"Epoch {ep+1}/{args.epochs}  "
            f"train[loss={train_stats['loss']:.4f} ppl={train_stats['perplexity']:.2f} "
            f"em={train_stats['em']:.4f} ed_tf={train_stats['ed_tf']:.4f}]  "
            f"val[loss={val_stats['loss']:.4f} ppl={val_stats['perplexity']:.2f} "
            f"em={val_stats['em']:.4f} ed_tf={val_stats['ed_tf']:.4f} "
            f"ed_ar={val_stats['ed_ar']:.4f}]"
        )
        val_loss = val_stats["loss"]

        # Append metrics to CSV
        with open(metrics_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=metrics_fields).writerow({
                "epoch": ep + 1,
                "train_loss": f"{train_stats['loss']:.6f}",
                "train_perplexity": f"{train_stats['perplexity']:.6f}",
                "train_em": f"{train_stats['em']:.6f}",
                "train_ed_tf": f"{train_stats['ed_tf']:.6f}",
                "val_loss": f"{val_stats['loss']:.6f}",
                "val_perplexity": f"{val_stats['perplexity']:.6f}",
                "val_em": f"{val_stats['em']:.6f}",
                "val_ed_tf": f"{val_stats['ed_tf']:.6f}",
                "val_ed_ar": f"{val_stats['ed_ar']:.6f}",
            })

        # Update metrics plot
        plot_run(run_dir, save=True)

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
