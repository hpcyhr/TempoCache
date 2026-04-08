import torch

from tempocache.core.cache_state import CacheState


def test_cache_state_reset_resize_update_and_age():
    cache = CacheState()
    device = torch.device("cpu")
    cache.ensure_batch(batch_size=3, device=device)
    assert cache.valid is not None and cache.age is not None
    assert cache.valid.shape[0] == 3

    idx = torch.tensor([0, 2], dtype=torch.long)
    values = torch.randn(2, 4, 5)
    sig = torch.randn(2, 7)
    cache.update_drive(idx, values)
    cache.update_signature(idx, sig)
    assert cache.cache is not None and cache.signature is not None
    assert bool(cache.valid[0].item()) and bool(cache.valid[2].item())
    assert int(cache.age[0].item()) == 0

    mask = torch.tensor([True, False, True])
    cache.increment_age(mask=mask)
    assert int(cache.age[0].item()) == 1
    assert int(cache.age[1].item()) == 0
    assert int(cache.age[2].item()) == 1

    cache.resize(batch_size=2, device=device)
    assert cache.valid is not None and cache.valid.shape[0] == 2

    cache.reset()
    assert cache.cache is None
    assert cache.signature is None
    assert cache.valid is not None and not bool(cache.valid.any().item())

