import pytest

# Skip all tests in this module if torch is not available
torch = pytest.importorskip("torch")

from rope import (
    RotaryEmbedding,
    apply_rotary_emb,
    apply_rotary_emb_qk,
    rotate_half,
)

def test_rotate_half():
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])  # last dim = 4
    y = rotate_half(x)
    # (x1, x2) = ([1,2], [3,4]) -> (-x2, x1) = ([-3,-4, 1, 2])
    expected = torch.tensor([[-3.0, -4.0, 1.0, 2.0]])
    assert torch.allclose(y, expected)


def test_constructor_requires_even_head_dim():
    with pytest.raises(AssertionError):
        RotaryEmbedding(head_dim=3)


def test_cos_sin_shapes_and_broadcast():
    d = 8
    s = 16
    rope = RotaryEmbedding(d)
    cache = rope.get_cos_sin(seq_len=s, device=torch.device("cpu"), dtype=torch.float32)
    assert cache.cos.shape == (1, 1, s, d)
    assert cache.sin.shape == (1, 1, s, d)


def test_apply_rotary_emb_shapes():
    b, h, s, d = 2, 3, 10, 8
    x = torch.randn(b, h, s, d)
    rope = RotaryEmbedding(d)
    cache = rope.get_cos_sin(seq_len=s, device=x.device, dtype=x.dtype)

    out = apply_rotary_emb(x, cache.cos, cache.sin)
    assert out.shape == x.shape


def test_apply_rotary_emb_qk():
    b, h, s, d = 1, 2, 12, 8
    q = torch.randn(b, h, s, d)
    k = torch.randn(b, h, s, d)
    rope = RotaryEmbedding(d)
    cache = rope.get_cos_sin(seq_len=s, device=q.device, dtype=q.dtype)

    q_rot, k_rot = apply_rotary_emb_qk(q, k, cache.cos, cache.sin)
    assert q_rot.shape == q.shape
    assert k_rot.shape == k.shape


def test_cache_updates_when_seq_len_increases():
    d = 8
    rope = RotaryEmbedding(d)
    cache1 = rope.get_cos_sin(seq_len=8, device=torch.device("cpu"), dtype=torch.float32)
    cache2 = rope.get_cos_sin(seq_len=16, device=torch.device("cpu"), dtype=torch.float32)

    assert cache1.cos.shape == (1, 1, 8, d)
    assert cache2.cos.shape == (1, 1, 16, d)


def test_cache_dtype_and_device_change_triggers_recompute():
    d = 8
    rope = RotaryEmbedding(d)
    cache_f32 = rope.get_cos_sin(seq_len=8, device=torch.device("cpu"), dtype=torch.float32)
    cache_f16 = rope.get_cos_sin(seq_len=8, device=torch.device("cpu"), dtype=torch.float16)

    assert cache_f32.cos.dtype == torch.float32
    assert cache_f16.cos.dtype == torch.float16
