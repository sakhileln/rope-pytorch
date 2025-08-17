"""
Minimal Transformer with RoPE attention.

Components
- MultiHeadAttentionRoPE: MHA that applies Rotary Position Embeddings to Q/K.
- TransformerBlock: pre-norm block (LN -> MHA -> residual -> LN -> MLP -> residual).
- SimpleTransformer: token embedding + N blocks + LM head.

This is intentionally small and clear, not optimized.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from rope import RotaryEmbedding, apply_rotary_emb_qk


def _shape_qkv(x: torch.Tensor, num_heads: int) -> torch.Tensor:
    """Reshape (B, S, D) -> (B, H, S, Dh)."""
    B, S, D = x.shape
    Dh = D // num_heads
    return x.view(B, S, num_heads, Dh).permute(0, 2, 1, 3)


def _merge_heads(x: torch.Tensor) -> torch.Tensor:
    """(B, H, S, Dh) -> (B, S, D)."""
    B, H, S, Dh = x.shape
    return x.permute(0, 2, 1, 3).contiguous().view(B, S, H * Dh)


class MultiHeadAttentionRoPE(nn.Module):
    def __init__(self, dim: int, num_heads: int, attn_dropout: float = 0.0, proj_dropout: float = 0.0):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.attn_drop = nn.Dropout(attn_dropout)
        self.proj_drop = nn.Dropout(proj_dropout)

        self.rope = RotaryEmbedding(self.head_dim)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,  # (B, 1, 1, S) with 0 for masked, 1 for keep
        causal: bool = True,
    ) -> torch.Tensor:
        B, S, D = x.shape
        qkv = self.qkv(x)  # (B, S, 3D)
        q, k, v = qkv.chunk(3, dim=-1)

        q = _shape_qkv(q, self.num_heads)  # (B, H, S, Dh)
        k = _shape_qkv(k, self.num_heads)
        v = _shape_qkv(v, self.num_heads)

        cache = self.rope.get_cos_sin(seq_len=S, device=x.device, dtype=x.dtype)
        q, k = apply_rotary_emb_qk(q, k, cache.cos, cache.sin)

        # Scaled dot-product attention
        attn = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)  # (B, H, S, S)

        if causal:
            causal_mask = torch.tril(torch.ones(S, S, device=x.device, dtype=torch.bool))
            attn = attn.masked_fill(~causal_mask, float("-inf"))

        if attention_mask is not None:
            # attention_mask expected broadcastable to (B, 1, S, S) as key mask on last dim
            # If provided as (B, 1, 1, S), we expand to (B, 1, S, S)
            if attention_mask.dim() == 4 and attention_mask.shape[-2] == 1:
                attention_mask = attention_mask.expand(-1, -1, S, -1)
            attn = attn.masked_fill(attention_mask == 0, float("-inf"))

        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        out = torch.matmul(attn, v)  # (B, H, S, Dh)
        out = _merge_heads(out)      # (B, S, D)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = MultiHeadAttentionRoPE(dim, num_heads, attn_dropout=dropout, proj_dropout=dropout)
        self.ln2 = nn.LayerNorm(dim)

        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), attention_mask=attention_mask, causal=True)
        x = x + self.mlp(self.ln2(x))
        return x


class SimpleTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        dim: int = 256,
        depth: int = 4,
        heads: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.embed = nn.Embedding(vocab_size, dim)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            TransformerBlock(dim, heads, mlp_ratio=mlp_ratio, dropout=dropout) for _ in range(depth)
        ])
        self.ln_f = nn.LayerNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)

    def forward(
        self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Args
        - input_ids: (B, S) Long
        - attention_mask: optional (B, 1, 1, S) with 1 keep, 0 mask

        Returns logits: (B, S, vocab_size)
        """
        x = self.embed(input_ids)  # (B, S, D)
        x = self.drop(x)
        for blk in self.blocks:
            x = blk(x, attention_mask=attention_mask)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits


__all__ = [
    "MultiHeadAttentionRoPE",
    "TransformerBlock",
    "SimpleTransformer",
]

