import torch

from tempocache.core.signatures.attention import AttentionSignatureExtractor


def test_attention_signature_constant_window_vtemp_zero():
    ext = AttentionSignatureExtractor()
    q = torch.ones(4, 2, 3, 5, 8)
    k = torch.ones(4, 2, 3, 5, 8)
    v = torch.ones(4, 2, 3, 5, 8)
    sig, v_temp = ext.extract(q, k, v)
    assert sig.shape[0] == 2
    assert torch.allclose(v_temp, torch.zeros_like(v_temp), atol=1e-6)


def test_attention_signature_deterministic_and_distinguishable():
    ext = AttentionSignatureExtractor()
    q = torch.randn(4, 2, 3, 5, 8)
    k = torch.randn(4, 2, 3, 5, 8)
    v = torch.randn(4, 2, 3, 5, 8)
    sig1, _ = ext.extract(q, k, v)
    sig2, _ = ext.extract(q.clone(), k.clone(), v.clone())
    assert torch.allclose(sig1, sig2, atol=1e-6)

    sig3, _ = ext.extract(q + 0.1, k, v)
    assert not torch.allclose(sig1, sig3)

