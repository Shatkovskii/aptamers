"""
Dataset for aptamer decoder training.
Loads ESM embeddings (target_seq_ab, target_seq_apt, Antibody Sequence) + Aptamer Sequence.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from core.decoder import PAD_ID, seq_to_ids


def _load_embed(path: Path) -> torch.Tensor:
    arr = np.load(path)
    return torch.from_numpy(arr).float()


class AptamerDecoderDataset(Dataset):
    """
    Returns E_ab, E_t_ab, E_t_apt (each [d_esm]), y_in, y_out, mask_y.
    Encoder saves mean-pooled embeddings → we use them as single-token conditioning:
    unsqueeze to [1, d_esm] for decoder compatibility.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        embeddings_dir: str | Path,
        aptamer_col: str = "Aptamer Sequence",
        max_len: int = 128,
    ):
        self.df = df.reset_index(drop=True)
        self.original_indices = df.index.tolist()
        self.emb_dir = Path(embeddings_dir)
        self.aptamer_col = aptamer_col
        self.max_len = max_len

        self.dir_ab = self.emb_dir / "Antibody_Sequence"
        self.dir_t_ab = self.emb_dir / "target_seq_ab"
        self.dir_t_apt = self.emb_dir / "target_seq_apt"

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int) -> dict:
        row = self.df.iloc[i]
        idx = int(self.original_indices[i])

        E_ab = _load_embed(self.dir_ab / f"embed_{idx}.npy")
        E_t_ab = _load_embed(self.dir_t_ab / f"embed_{idx}.npy")
        E_t_apt = _load_embed(self.dir_t_apt / f"embed_{idx}.npy")

        apt_seq = str(row[self.aptamer_col]).strip().upper()
        ids = seq_to_ids(apt_seq)  # [BOS, a1, ..., an, EOS]
        # Truncate if too long (keep BOS at start, EOS at end)
        if len(ids) > self.max_len + 1:
            ids = ids[: self.max_len] + [ids[-1]]

        # y_in:  [BOS, a1, ..., an]       (drop EOS)
        # y_out: [a1, ..., an, EOS]       (drop BOS)
        # Both have length len(ids)-1, pad to max_len
        seq_len = len(ids) - 1
        n_pad = self.max_len - seq_len

        y_in = ids[:-1] + [PAD_ID] * n_pad
        y_out = ids[1:] + [PAD_ID] * n_pad
        mask_y = [True] * seq_len + [False] * n_pad

        return {
            "E_ab": E_ab,
            "E_t_ab": E_t_ab,
            "E_t_apt": E_t_apt,
            "y_in": torch.tensor(y_in, dtype=torch.long),
            "y_out": torch.tensor(y_out, dtype=torch.long),
            "mask_y": torch.tensor(mask_y, dtype=torch.bool),
        }


def collate_fn(batch: list[dict]) -> dict:
    E_ab = torch.stack([b["E_ab"] for b in batch])
    E_t_ab = torch.stack([b["E_t_ab"] for b in batch])
    E_t_apt = torch.stack([b["E_t_apt"] for b in batch])
    y_in = torch.stack([b["y_in"] for b in batch])
    y_out = torch.stack([b["y_out"] for b in batch])
    mask_y = torch.stack([b["mask_y"] for b in batch])

    # Encoder saves mean-pooled [d_esm]. Decoder expects [B, L, d_esm]. Use L=1.
    E_ab = E_ab.unsqueeze(1)
    E_t_ab = E_t_ab.unsqueeze(1)
    E_t_apt = E_t_apt.unsqueeze(1)
    mask_ab = torch.ones(E_ab.size(0), 1, dtype=torch.bool)
    mask_t1 = torch.ones(E_t_ab.size(0), 1, dtype=torch.bool)
    mask_t2 = torch.ones(E_t_apt.size(0), 1, dtype=torch.bool)

    return {
        "E_ab": E_ab,
        "E_t_ab": E_t_ab,
        "E_t_apt": E_t_apt,
        "mask_ab": mask_ab,
        "mask_t1": mask_t1,
        "mask_t2": mask_t2,
        "y_in": y_in,
        "y_out": y_out,
        "mask_y": mask_y,
    }


def train_val_split(
    df: pd.DataFrame,
    val_ratio: float = 0.15,
    seed: int = 42,
    group_cols: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split by groups to avoid leakage. If group_cols given, split by unique groups.
    E.g. group_cols=["target_seq_ab", "Antibody Sequence"] avoids same protein in train/val.
    """
    rng = np.random.default_rng(seed)
    if group_cols:
        group_ids = df.groupby(group_cols).ngroup()
        uniq = group_ids.unique()
        rng.shuffle(uniq)
        n_val = max(1, int(len(uniq) * val_ratio))
        val_groups = set(uniq[:n_val])
        mask_val = group_ids.isin(val_groups)
        train_df = df[~mask_val]
        val_df = df[mask_val]
    else:
        idx = df.index.tolist()
        rng.shuffle(idx)
        n_val = max(1, int(len(idx) * val_ratio))
        val_idx = set(idx[:n_val])
        train_df = df[~df.index.isin(val_idx)]
        val_df = df[df.index.isin(val_idx)]
    return train_df, val_df
