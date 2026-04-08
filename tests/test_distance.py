import torch

from tempocache.core.distance import inter_signature_distance


def test_inter_signature_distance_zero_on_identical_inputs():
    x = torch.randn(3, 8)
    d = inter_signature_distance(x, x)
    assert torch.allclose(d, torch.zeros_like(d), atol=1e-7)


def test_inter_signature_distance_positive_for_different_inputs():
    a = torch.ones(2, 4)
    b = torch.zeros(2, 4)
    d = inter_signature_distance(a, b)
    assert torch.all(d > 0)

