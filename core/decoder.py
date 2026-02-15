"""
Transformer-based aptamer decoder conditioned on protein memory (Ab + target_seq_ab + target_seq_apt).
Input: ESM embeddings [B, L, d_esm] with padding masks.
Output: autoregressive RNA sequence generation.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# --- RNA vocabulary (character-level: A, C, G, U + special tokens) ---
RNA_VOCAB = ["<PAD>", "<BOS>", "<EOS>", "A", "C", "G", "U"]
PAD_ID = 0
BOS_ID = 1
EOS_ID = 2
VOCAB_SIZE = 7

# Map char -> id
_TOK_TO_ID = {t: i for i, t in enumerate(RNA_VOCAB)}
_ID_TO_TOK = {i: t for i, t in enumerate(RNA_VOCAB)}


def seq_to_ids(seq: str) -> list[int]:
    """Encode RNA sequence to token ids. Adds BOS/EOS. T->U for DNA inputs."""
    ids = [BOS_ID]
    for c in seq.upper().replace("T", "U"):
        if c in _TOK_TO_ID:
            ids.append(_TOK_TO_ID[c])
    ids.append(EOS_ID)
    return ids


def ids_to_seq(ids: list[int] | torch.Tensor, strip_special: bool = True) -> str:
    """Decode token ids to RNA string. Accepts 1D list or [L] tensor."""
    if isinstance(ids, torch.Tensor):
        ids = ids.squeeze().cpu().tolist()
    if not isinstance(ids, list):
        ids = list(ids)
    tokens = []
    for i in ids:
        if strip_special and i in (PAD_ID, BOS_ID, EOS_ID):
            continue
        if i in _ID_TO_TOK:
            t = _ID_TO_TOK[i]
            if strip_special and t.startswith("<"):
                continue
            tokens.append(t)
    return "".join(tokens)


def causal_mask(size: int, device: Optional[torch.device] = None) -> torch.Tensor:
    """Causal mask: True = attend, False = mask out. Shape [1, size, size]."""
    mask = torch.triu(torch.ones(size, size, device=device), diagonal=1).bool()
    return ~mask  # True where we attend


# =============================================================================
# Memory Builder: ESM embeddings -> conditioning memory
# =============================================================================


class MemoryBuilder(nn.Module):
    """
    Projects ESM embeddings to d_model, adds type embeddings, concatenates.
    Input: E_ab [B,L_ab,d_esm], E_t_ab [B,L_t1,d_esm], E_t_apt [B,L_t2,d_esm]
    Output: M [B, L_mem, d_model], mask_mem [B, L_mem]
    """

    def __init__(self, d_esm: int, d_model: int = 256, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Linear(d_esm, d_model)
        self.ln = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.type_ab = nn.Parameter(torch.zeros(1, 1, d_model))
        self.type_t1 = nn.Parameter(torch.zeros(1, 1, d_model))
        self.type_t2 = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.type_ab, std=0.02)
        nn.init.normal_(self.type_t1, std=0.02)
        nn.init.normal_(self.type_t2, std=0.02)

    def forward(
        self,
        E_ab: torch.Tensor,
        E_t_ab: torch.Tensor,
        E_t_apt: torch.Tensor,
        mask_ab: torch.Tensor,
        mask_t1: torch.Tensor,
        mask_t2: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        H_ab = self.ln(self.proj(E_ab)) + self.type_ab
        H_t1 = self.ln(self.proj(E_t_ab)) + self.type_t1
        H_t2 = self.ln(self.proj(E_t_apt)) + self.type_t2
        M = torch.cat([H_ab, H_t1, H_t2], dim=1)
        M = self.dropout(M)
        mask_mem = torch.cat([mask_ab, mask_t1, mask_t2], dim=1)
        return M, mask_mem


# =============================================================================
# Decoder Layer (pre-norm: self-attn -> cross-attn -> FFN)
# =============================================================================


class DecoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.ln2 = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.ln3 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        M: torch.Tensor,
        tgt_attn_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # 1. Masked self-attention (causal + padding in single attn_mask)
        x_norm = self.ln1(x)
        attn_out, _ = self.self_attn(
            x_norm,
            x_norm,
            x_norm,
            attn_mask=tgt_attn_mask,
            need_weights=False,
        )
        x = x + self.dropout(attn_out)

        # 2. Cross-attention to memory
        x_norm = self.ln2(x)
        attn_out, _ = self.cross_attn(
            x_norm,
            M,
            M,
            key_padding_mask=memory_key_padding_mask,
            need_weights=False,
        )
        x = x + self.dropout(attn_out)

        # 3. FFN
        x = x + self.ffn(self.ln3(x))
        return x


# =============================================================================
# Aptamer Decoder (full model)
# =============================================================================


class AptamerDecoder(nn.Module):
    """
    Transformer decoder for RNA aptamer generation conditioned on protein memory.
    Vocab: {PAD, BOS, EOS, A, C, G, U}.
    """

    def __init__(
        self,
        d_esm: int = 1280,
        d_model: int = 256,
        n_layers: int = 4,
        n_heads: int = 8,
        d_ff: int = 1024,
        max_len: int = 128,
        dropout: float = 0.1,
        vocab_size: int = VOCAB_SIZE,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.vocab_size = vocab_size
        self.max_len = max_len

        self.memory_builder = MemoryBuilder(d_esm, d_model, dropout)

        self.tok_emb = nn.Embedding(vocab_size, d_model, padding_idx=PAD_ID)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.emb_dropout = nn.Dropout(dropout)
        nn.init.normal_(self.tok_emb.weight, std=0.02)
        nn.init.normal_(self.pos_emb.weight, std=0.02)

        self.layers = nn.ModuleList([
            DecoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.ln_out = nn.LayerNorm(d_model)
        self.proj = nn.Linear(d_model, vocab_size)
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(
        self,
        E_ab: torch.Tensor,
        E_t_ab: torch.Tensor,
        E_t_apt: torch.Tensor,
        mask_ab: torch.Tensor,
        mask_t1: torch.Tensor,
        mask_t2: torch.Tensor,
        y_in: torch.Tensor,
        mask_y: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            E_ab, E_t_ab, E_t_apt: [B, L_*, d_esm]
            mask_ab, mask_t1, mask_t2: [B, L_*], True=valid token
            y_in: [B, L_y] decoder input (BOS + tokens[:-1])
            mask_y: [B, L_y], True=valid (optional, for padding)
        Returns:
            logits: [B, L_y, vocab_size]
        """
        B, L_y = y_in.shape
        M, mask_mem = self.memory_builder(E_ab, E_t_ab, E_t_apt, mask_ab, mask_t1, mask_t2)

        # Memory mask for cross-attn: key_padding_mask True = mask out
        memory_key_padding_mask = ~mask_mem

        # Decoder input embeddings
        positions = torch.arange(L_y, device=y_in.device).unsqueeze(0).expand(B, -1)
        x = self.tok_emb(y_in) + self.pos_emb(positions)
        x = self.emb_dropout(x)

        # Combined causal + padding mask for self-attention
        # MHA expects 3D mask of shape [B*n_heads, L_y, L_y]
        if mask_y is None:
            mask_y = y_in != PAD_ID
        causal = causal_mask(L_y, device=y_in.device)
        tgt_attn_mask = torch.where(causal, 0.0, float("-inf"))  # [L_y, L_y]
        tgt_attn_mask = tgt_attn_mask.unsqueeze(0).expand(B, -1, -1)  # [B, L_y, L_y]
        pad_mask = ~mask_y.unsqueeze(1)  # [B, 1, L_y]
        tgt_attn_mask = tgt_attn_mask.masked_fill(pad_mask, float("-inf"))
        # Repeat for each head: [B, L, L] -> [B*n_heads, L, L]
        tgt_attn_mask = tgt_attn_mask.repeat_interleave(self.n_heads, dim=0)

        for layer in self.layers:
            x = layer(
                x,
                M,
                tgt_attn_mask=tgt_attn_mask,
                memory_key_padding_mask=memory_key_padding_mask,
            )

        x = self.ln_out(x)
        logits = self.proj(x)
        return logits

    def build_memory(
        self,
        E_ab: torch.Tensor,
        E_t_ab: torch.Tensor,
        E_t_apt: torch.Tensor,
        mask_ab: torch.Tensor,
        mask_t1: torch.Tensor,
        mask_t2: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build memory only (for generation). Returns M, mask_mem."""
        return self.memory_builder(E_ab, E_t_ab, E_t_apt, mask_ab, mask_t1, mask_t2)


# =============================================================================
# Generation (greedy, top-p sampling)
# =============================================================================


def _decode_step(
    model: AptamerDecoder,
    y: torch.Tensor,
    M: torch.Tensor,
    memory_key_padding_mask: torch.Tensor,
) -> torch.Tensor:
    """Single decoder step. y: [B, L], returns logits [B, L, vocab]."""
    B, L = y.shape
    positions = torch.arange(L, device=y.device).unsqueeze(0).expand(B, -1)
    x = model.tok_emb(y) + model.pos_emb(positions)
    x = model.emb_dropout(x)

    causal = causal_mask(L, device=y.device)
    tgt_attn_mask = torch.where(causal, 0.0, float("-inf"))
    tgt_attn_mask = tgt_attn_mask.unsqueeze(0).expand(B, -1, -1)
    mask_y = y != PAD_ID
    pad_mask = ~mask_y.unsqueeze(1)
    tgt_attn_mask = tgt_attn_mask.masked_fill(pad_mask, float("-inf"))
    tgt_attn_mask = tgt_attn_mask.repeat_interleave(model.n_heads, dim=0)

    for layer in model.layers:
        x = layer(
            x,
            M,
            tgt_attn_mask=tgt_attn_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )
    x = model.ln_out(x)
    return model.proj(x)


def _top_p_sample(probs: torch.Tensor, p: float) -> torch.Tensor:
    """Nucleus sampling. probs: [B, V]. Returns [B]."""
    sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
    cumsum = torch.cumsum(sorted_probs, dim=-1)
    mask = cumsum - sorted_probs > p
    sorted_probs = torch.where(mask, torch.zeros_like(sorted_probs), sorted_probs)
    sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    sampled = torch.multinomial(sorted_probs, 1).squeeze(-1)
    return torch.gather(sorted_idx, -1, sampled.unsqueeze(-1)).squeeze(-1)


def greedy_decode(
    model: AptamerDecoder,
    E_ab: torch.Tensor,
    E_t_ab: torch.Tensor,
    E_t_apt: torch.Tensor,
    mask_ab: torch.Tensor,
    mask_t1: torch.Tensor,
    mask_t2: torch.Tensor,
    max_len: int,
    device: torch.device,
) -> torch.Tensor:
    """Greedy autoregressive decoding. Returns [B, L] token ids."""
    model.eval()
    B = E_ab.size(0)
    y = torch.full((B, 1), BOS_ID, dtype=torch.long, device=device)

    M, mask_mem = model.build_memory(E_ab, E_t_ab, E_t_apt, mask_ab, mask_t1, mask_t2)
    memory_key_padding_mask = ~mask_mem

    with torch.no_grad():
        for _ in range(max_len - 1):
            if y.size(1) >= max_len:
                break

            logits = _decode_step(model, y, M, memory_key_padding_mask)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            y = torch.cat([y, next_token], dim=1)

            if torch.all(next_token.squeeze(-1) == EOS_ID):
                break
    return y


def top_p_decode(
    model: AptamerDecoder,
    E_ab: torch.Tensor,
    E_t_ab: torch.Tensor,
    E_t_apt: torch.Tensor,
    mask_ab: torch.Tensor,
    mask_t1: torch.Tensor,
    mask_t2: torch.Tensor,
    max_len: int,
    device: torch.device,
    top_p: float = 0.9,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Top-p (nucleus) sampling. Returns [B, L] token ids."""
    model.eval()
    B = E_ab.size(0)
    y = torch.full((B, 1), BOS_ID, dtype=torch.long, device=device)

    M, mask_mem = model.build_memory(E_ab, E_t_ab, E_t_apt, mask_ab, mask_t1, mask_t2)
    memory_key_padding_mask = ~mask_mem

    with torch.no_grad():
        for _ in range(max_len - 1):
            if y.size(1) >= max_len:
                break

            logits = _decode_step(model, y, M, memory_key_padding_mask)
            next_logits = logits[:, -1, :] / max(temperature, 1e-8)
            probs = F.softmax(next_logits, dim=-1)
            next_token = _top_p_sample(probs, top_p)
            y = torch.cat([y, next_token.unsqueeze(-1)], dim=1)

            if torch.all(next_token == EOS_ID):
                break
    return y


# =============================================================================
# Training
# =============================================================================


def compute_loss(
    logits: torch.Tensor,
    y_out: torch.Tensor,
    label_smoothing: float = 0.1,
) -> torch.Tensor:
    """
    CrossEntropy loss for decoder output.
    logits: [B, L, vocab], y_out: [B, L] (targets, PAD for positions to ignore)
    """
    return F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        y_out.view(-1),
        ignore_index=PAD_ID,
        label_smoothing=label_smoothing,
    )


def build_aptamer_decoder(
    d_esm: int = 1280,
    d_model: int = 256,
    n_layers: int = 4,
    n_heads: int = 8,
    d_ff: int = 1024,
    max_len: int = 128,
    dropout: float = 0.1,
) -> AptamerDecoder:
    """Factory: build and init AptamerDecoder."""
    model = AptamerDecoder(
        d_esm=d_esm,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        d_ff=d_ff,
        max_len=max_len,
        dropout=dropout,
    )
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model
