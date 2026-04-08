from tempocache.config import PatchConfig, RuntimeConfig
from tempocache.integration.patcher import TempoPatcher
from tempocache.models import ToyCNNSNN
from tempocache.ops import TempoConv2d, TempoLinear


def test_module_patcher_replaces_hdo_and_preserves_stateful():
    model = ToyCNNSNN(use_tempo=False)
    runtime = RuntimeConfig(mode="adaptive", window_size=2)
    patcher = TempoPatcher(runtime_config=runtime, patch_config=PatchConfig())
    patched, report = patcher.patch(model)

    assert isinstance(patched.conv1, TempoConv2d)
    assert isinstance(patched.conv2, TempoConv2d)
    assert isinstance(patched.head, TempoLinear)
    assert len(report.hdo_replaced) >= 3
    assert any("neuron" in item["module_name"] for item in report.stateful_preserved)


def test_module_patcher_include_exclude():
    model = ToyCNNSNN(use_tempo=False)
    runtime = RuntimeConfig(mode="adaptive", window_size=2)
    cfg = PatchConfig(include_patterns=["*conv1"], exclude_patterns=["*head"])
    patcher = TempoPatcher(runtime_config=runtime, patch_config=cfg)
    patched, report = patcher.patch(model)

    assert isinstance(patched.conv1, TempoConv2d)
    assert not isinstance(patched.conv2, TempoConv2d)
    assert not isinstance(patched.head, TempoLinear)
    replaced_names = {item["module_name"] for item in report.hdo_replaced}
    assert "conv1" in replaced_names
    assert "head" not in replaced_names

