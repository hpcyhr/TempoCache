import torch
import torch.nn as nn

from tempocache.config import PatchConfig, RuntimeConfig
from tempocache.integration.fx_rewriter import TempoFXRewriter


class FXToy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 4, kernel_size=3, padding=1)
        self.linear = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        y = self.conv(x.flatten(0, 1))
        y = y.view(x.size(0), x.size(1), 4, 8, 8).mean(dim=(-1, -2))
        y = self.linear(y)
        m = torch.matmul(a, b).mean(dim=-1)
        z = y + m
        z = torch.cat([z, z], dim=-1)
        z = z.view(x.size(0), x.size(1), 2, 4)
        z = z.permute(0, 1, 3, 2)
        z = z.transpose(-1, -2)
        return z.reshape(x.size(0), x.size(1), -1)


def test_fx_rewriter_replaces_hdo_and_recognizes_non_hdo():
    model = FXToy()
    runtime = RuntimeConfig(mode="adaptive", window_size=2)
    patch = PatchConfig(use_fx_rewriter=True, replace_matmul=True)
    rewriter = TempoFXRewriter(runtime_config=runtime, patch_config=patch)
    gm, report = rewriter.rewrite(model)

    replaced_types = [item.get("old_type", "") for item in report.hdo_nodes_replaced]
    replaced_targets = [item.get("target", "") for item in report.hdo_nodes_replaced]
    recognized_kinds = [item.get("kind", "") for item in report.non_hdo_recognized]

    assert any("Conv2d" in t for t in replaced_types)
    assert any("Linear" in t for t in replaced_types)
    assert any("matmul" in t or "new_module" in item for item in report.hdo_nodes_replaced for t in [item.get("target", "")])
    assert "add" in recognized_kinds
    assert "cat" in recognized_kinds
    assert "reshape_family" in recognized_kinds

    x = torch.randn(4, 2, 3, 8, 8)
    a = torch.randn(4, 2, 4, 4)
    b = torch.randn(4, 2, 4, 4)
    y = gm(x, a, b)
    assert y.shape == (4, 2, 8)

