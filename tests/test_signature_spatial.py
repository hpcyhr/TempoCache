import torch

from tempocache.core.signatures.spatial import SpatialTensorSignatureExtractor


def test_spatial_signature_constant_window_vtemp_zero():
    ext = SpatialTensorSignatureExtractor()
    x = torch.ones(4, 2, 3, 8, 8)
    sig, v = ext.extract(x)
    assert sig.shape[0] == 2
    assert torch.allclose(v, torch.zeros_like(v), atol=1e-6)


def test_spatial_signature_deterministic_and_distinguishable():
    ext = SpatialTensorSignatureExtractor()
    x = torch.randn(4, 2, 3, 8, 8)
    sig1, _ = ext.extract(x)
    sig2, _ = ext.extract(x.clone())
    assert torch.allclose(sig1, sig2, atol=1e-6)

    y = x + 0.5
    sig3, _ = ext.extract(y)
    assert not torch.allclose(sig1, sig3)

