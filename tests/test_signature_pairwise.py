import torch

from tempocache.core.signatures.pairwise import PairwiseTensorSignatureExtractor


def test_pairwise_signature_constant_window_vtemp_zero():
    ext = PairwiseTensorSignatureExtractor()
    a = torch.ones(4, 2, 5, 6)
    b = torch.ones(4, 2, 6, 7)
    sig, v = ext.extract(a, b)
    assert sig.shape[0] == 2
    assert torch.allclose(v, torch.zeros_like(v), atol=1e-6)


def test_pairwise_signature_deterministic_and_distinguishable():
    ext = PairwiseTensorSignatureExtractor()
    a = torch.randn(4, 2, 5, 6)
    b = torch.randn(4, 2, 6, 7)
    sig1, _ = ext.extract(a, b)
    sig2, _ = ext.extract(a.clone(), b.clone())
    assert torch.allclose(sig1, sig2, atol=1e-6)

    sig3, _ = ext.extract(a + 0.25, b)
    assert not torch.allclose(sig1, sig3)

