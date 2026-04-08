import torch

from tempocache.config import RuntimeConfig
from tempocache.ops import TempoBMM, TempoMatMul


def test_tempo_matmul_full_equivalence():
    op = TempoMatMul(runtime_config=RuntimeConfig(mode="full", window_size=3))
    a = torch.randn(7, 3, 5, 6)
    b = torch.randn(7, 3, 6, 4)
    y = op(a, b)
    y_ref = torch.stack([torch.matmul(a[t], b[t]) for t in range(a.shape[0])], dim=0)
    assert torch.allclose(y, y_ref, atol=1e-6, rtol=1e-5)


def test_tempo_bmm_full_equivalence():
    op = TempoBMM(runtime_config=RuntimeConfig(mode="full", window_size=2))
    a = torch.randn(6, 4, 5, 6)
    b = torch.randn(6, 4, 6, 3)
    y = op(a, b)
    y_ref = torch.stack([torch.bmm(a[t], b[t]) for t in range(a.shape[0])], dim=0)
    assert torch.allclose(y, y_ref, atol=1e-6, rtol=1e-5)


def test_tempo_matmul_fixed_collapse_and_reuse():
    a = torch.randn(4, 2, 3, 4)
    b = torch.randn(4, 2, 4, 5)

    collapse_op = TempoMatMul(runtime_config=RuntimeConfig(mode="fixed_collapse", window_size=2))
    y_c = collapse_op(a, b)
    expected_c = []
    for start in (0, 2):
        a_col = a[start : start + 2].mean(dim=0)
        b_col = b[start : start + 2].mean(dim=0)
        y_col = torch.matmul(a_col, b_col)
        expected_c.append(y_col.unsqueeze(0).expand(2, *y_col.shape))
    y_c_ref = torch.cat(expected_c, dim=0)
    assert torch.allclose(y_c, y_c_ref, atol=1e-6, rtol=1e-5)

    reuse_op = TempoMatMul(runtime_config=RuntimeConfig(mode="fixed_reuse", window_size=2))
    y_r = reuse_op(a, b)
    first = torch.stack([torch.matmul(a[t], b[t]) for t in range(2)], dim=0)
    cache = first.mean(dim=0)
    second = cache.unsqueeze(0).expand(2, *cache.shape)
    assert torch.allclose(y_r[:2], first, atol=1e-6, rtol=1e-5)
    assert torch.allclose(y_r[2:], second, atol=1e-6, rtol=1e-5)

