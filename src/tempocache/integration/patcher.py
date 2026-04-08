"""Module-based patcher for TempoCache."""

from __future__ import annotations

import torch.nn as nn

from ..config import AttentionConfig, PatchConfig, RuntimeConfig
from ..ops.tempo_attention import TempoSpikeSelfAttention
from ..ops.tempo_conv1d import TempoConv1d
from ..ops.tempo_conv2d import TempoConv2d
from ..ops.tempo_conv3d import TempoConv3d
from ..ops.tempo_linear import TempoLinear
from ..utils.module_utils import (
    is_attention_like_module,
    is_norm_module,
    is_pool_module,
    is_stateful_temporal_module,
    operator_family_of,
    should_patch_module,
)
from .reports import PatchReport


class TempoPatcher:
    """Recursive model patcher that replaces supported HDO modules."""

    def __init__(
        self,
        runtime_config: RuntimeConfig,
        patch_config: PatchConfig | None = None,
        attention_config: AttentionConfig | None = None,
    ) -> None:
        self.runtime_config = runtime_config
        self.patch_config = patch_config or PatchConfig()
        self.attention_config = attention_config or AttentionConfig()

    def _record_non_hdo(self, report: PatchReport, path: str, module: nn.Module) -> None:
        if is_norm_module(module) or is_pool_module(module) or isinstance(module, (nn.Flatten, nn.Identity, nn.Dropout)):
            report.recognized_non_hdo.append(
                {"module_name": path, "module_type": module.__class__.__name__, "family": operator_family_of(module).value}
            )

    def _replace(self, parent: nn.Module, name: str, new_module: nn.Module) -> None:
        setattr(parent, name, new_module)

    def _patch_recursive(self, module: nn.Module, prefix: str, report: PatchReport) -> None:
        for child_name, child in list(module.named_children()):
            path = f"{prefix}.{child_name}" if prefix else child_name

            if is_stateful_temporal_module(child):
                report.stateful_preserved.append({"module_name": path, "module_type": child.__class__.__name__})

            replaced = False
            if should_patch_module(path, self.patch_config.include_patterns, self.patch_config.exclude_patterns):
                if self.patch_config.replace_conv1d and isinstance(child, nn.Conv1d):
                    self._replace(module, child_name, TempoConv1d.from_conv(child, self.runtime_config, module_name=path))
                    replaced = True
                elif self.patch_config.replace_conv2d and isinstance(child, nn.Conv2d):
                    self._replace(module, child_name, TempoConv2d.from_conv(child, self.runtime_config, module_name=path))
                    replaced = True
                elif self.patch_config.replace_conv3d and isinstance(child, nn.Conv3d):
                    self._replace(module, child_name, TempoConv3d.from_conv(child, self.runtime_config, module_name=path))
                    replaced = True
                elif self.patch_config.replace_linear and isinstance(child, nn.Linear):
                    self._replace(module, child_name, TempoLinear.from_linear(child, self.runtime_config, module_name=path))
                    replaced = True
                elif (
                    self.patch_config.replace_attention
                    and is_attention_like_module(child)
                    and hasattr(child, "embed_dim")
                    and hasattr(child, "num_heads")
                ):
                    self._replace(
                        module,
                        child_name,
                        TempoSpikeSelfAttention.from_existing(
                            child,
                            runtime_config=self.runtime_config,
                            attention_config=self.attention_config,
                            module_name=path,
                        ),
                    )
                    replaced = True

            if replaced:
                new_mod = getattr(module, child_name)
                report.hdo_replaced.append(
                    {
                        "module_name": path,
                        "old_type": child.__class__.__name__,
                        "new_type": new_mod.__class__.__name__,
                    }
                )
                continue

            self._record_non_hdo(report, path, child)
            self._patch_recursive(child, path, report)

    def patch(self, model: nn.Module) -> tuple[nn.Module, PatchReport]:
        report = PatchReport()
        self._patch_recursive(model, prefix="", report=report)
        return model, report


def patch_model(
    model: nn.Module,
    runtime_config: RuntimeConfig,
    patch_config: PatchConfig | None = None,
    attention_config: AttentionConfig | None = None,
) -> tuple[nn.Module, PatchReport]:
    return TempoPatcher(
        runtime_config=runtime_config,
        patch_config=patch_config,
        attention_config=attention_config,
    ).patch(model)

