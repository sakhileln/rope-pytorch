"""
Simple Rotary Position Embedding (RoPE) implementation in PyTorch.

References
- RoFormer / RoPE: https://arxiv.org/abs/2104.09864

This module provides:
- `RotaryEmbedding`: precomputes cos/sin caches for a given head dimension.
- `apply_rotary_emb(x, cos, sin)`: applies RoPE to a tensor on its last dim.
- `apply_rotary_emb_qk(q, k, cos, sin)`: convenience for queries and keys.

Shapes (common usage)
- x, q, k: (..., seq_len, head_dim)
- cos, sin: (1, 1, seq_len, head_dim) broadcastable to x

Keep it simple: no fancy scaling or interpolation. Designed for clarity.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate last dimension by splitting it in half: (x1, x2) -> (-x2, x1).

    Expects last dimension to be even (head_dim divisible by 2).
    """
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


@dataclass
class RotaryCache:
    cos: torch.Tensor
    sin: torch.Tensor


class RotaryEmbedding(nn.Module):
    """Minimal RoPE cos/sin cache for a given head dimension.

    Args:
        head_dim: size of the per-head feature dimension (must be even).
        base: rotary base (default 10000.0).
    """

    def __init__(self, head_dim: int, base: float = 10000.0) -> None:
        super().__init__()
        assert head_dim % 2 == 0, "head_dim must be even for RoPE"
        self.head_dim = head_dim
        self.base = base

        # inv_freq shape: (head_dim/2,)
        half_dim = head_dim // 2
        idx = torch.arange(half_dim, dtype=torch.float32)
        inv_freq = 1.0 / (base ** (idx / half_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Cache
        self._seq_len_cached: int = 0
        self._cos_cached: torch.Tensor | None = None
        self._sin_cached: torch.Tensor | None = None

    @torch.no_grad()
    def get_cos_sin(
        self, seq_len: int, device: torch.device, dtype: torch.dtype
    ) -> RotaryCache:
        """Return cos/sin caches with shape (1, 1, seq_len, head_dim).

        Caches are recomputed only if seq_len increased or device/dtype changed.
        """
        need_new = (
            self._cos_cached is None
            or self._sin_cached is None
            or self._seq_len_cached < seq_len
            or self._cos_cached.device != device
            or self._cos_cached.dtype != dtype
        )

        if need_new:
            t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
            # Outer product: (seq_len, half_dim)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq.to(device))
            # Expand to full dim by interleaving cos/sin for pairs
            # Shape -> (seq_len, head_dim)
            emb = torch.cat([freqs, freqs], dim=-1)
            cos = emb.cos().to(dtype)
            sin = emb.sin().to(dtype)
            # Add broadcast dims: (1, 1, seq_len, head_dim)
            self._cos_cached = cos.unsqueeze(0).unsqueeze(0)
            self._sin_cached = sin.unsqueeze(0).unsqueeze(0)
            self._seq_len_cached = seq_len

        return RotaryCache(self._cos_cached, self._sin_cached)


def apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply RoPE to `x` given broadcastable cos/sin on the last dim.

    x: (..., seq_len, head_dim)
    cos/sin: (1, 1, seq_len, head_dim) or broadcastable
    """
    return (x * cos) + (rotate_half(x) * sin)


def apply_rotary_emb_qk(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply RoPE to queries and keys.
    Returns (q_rot, k_rot).
    """
    return apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)


if __name__ == "__main__":
    # Tiny sanity check
    b, h, s, d = 2, 4, 8, 64
    rope = RotaryEmbedding(d)
    q = torch.randn(b, h, s, d)
    k = torch.randn(b, h, s, d)
    cache = rope.get_cos_sin(seq_len=s, device=q.device, dtype=q.dtype)
    q_rot, k_rot = apply_rotary_emb_qk(q, k, cache.cos, cache.sin)
    print("q_rot shape:", q_rot.shape, "k_rot shape:", k_rot.shape)

