import torch

from tempocache.core.signatures.vector import VectorSignatureExtractor


def test_vector_signature_constant_window_vtemp_zero():
    ext = VectorSignatureExtractor()
    x = torch.ones(4, 2, 16)
    sig, v = ext.extract(x)
    assert sig.shape[0] == 2
    assert torch.allclose(v, torch.zeros_like(v), atol=1e-6)


def test_vector_signature_for_token_tensor():
    ext = VectorSignatureExtractor()
    x = torch.randn(4, 2, 6, 16)
    sig1, _ = ext.extract(x)
    sig2, _ = ext.extract(x.clone())
    assert torch.allclose(sig1, sig2, atol=1e-6)

    y = x * 1.2
    sig3, _ = ext.extract(y)
    assert not torch.allclose(sig1, sig3)

