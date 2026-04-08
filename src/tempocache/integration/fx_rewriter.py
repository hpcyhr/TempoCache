"""FX graph-level rewriter and operator recognizer."""

from __future__ import annotations

from itertools import count

import torch
import torch.fx as fx
import torch.nn as nn

from ..config import AttentionConfig, PatchConfig, RuntimeConfig
from ..ops.tempo_bmm import TempoBMM
from ..ops.tempo_conv1d import TempoConv1d
from ..ops.tempo_conv2d import TempoConv2d
from ..ops.tempo_conv3d import TempoConv3d
from ..ops.tempo_linear import TempoLinear
from ..ops.tempo_matmul import TempoMatMul
from ..utils.fx_utils import is_add_target, is_cat_target, is_matmul_target, is_reshape_method, readable_target
from ..utils.module_utils import is_stateful_temporal_module
from .reports import FXRewriteReport


class TempoFXRewriter:
    """FX-based graph rewriter for HDO and recognized non-HDO operator nodes."""

    def __init__(
        self,
        runtime_config: RuntimeConfig,
        patch_config: PatchConfig | None = None,
        attention_config: AttentionConfig | None = None,
    ) -> None:
        self.runtime_config = runtime_config
        self.patch_config = patch_config or PatchConfig()
        self.attention_config = attention_config or AttentionConfig()

    def _replace_call_module_if_needed(self, gm: fx.GraphModule, node: fx.Node, report: FXRewriteReport) -> None:
        module = gm.get_submodule(str(node.target))
        target = str(node.target)
        old_type = module.__class__.__name__

        if is_stateful_temporal_module(module):
            report.stateful_nodes_preserved.append({"node": node.name, "target": target, "type": old_type})
            return

        replaced = False
        if self.patch_config.replace_conv1d and isinstance(module, nn.Conv1d):
            gm.add_submodule(target, TempoConv1d.from_conv(module, self.runtime_config, module_name=target))
            replaced = True
        elif self.patch_config.replace_conv2d and isinstance(module, nn.Conv2d):
            gm.add_submodule(target, TempoConv2d.from_conv(module, self.runtime_config, module_name=target))
            replaced = True
        elif self.patch_config.replace_conv3d and isinstance(module, nn.Conv3d):
            gm.add_submodule(target, TempoConv3d.from_conv(module, self.runtime_config, module_name=target))
            replaced = True
        elif self.patch_config.replace_linear and isinstance(module, nn.Linear):
            gm.add_submodule(target, TempoLinear.from_linear(module, self.runtime_config, module_name=target))
            replaced = True

        if replaced:
            report.hdo_nodes_replaced.append({"node": node.name, "target": target, "old_type": old_type})

    def _replace_function_node_if_needed(
        self,
        gm: fx.GraphModule,
        node: fx.Node,
        report: FXRewriteReport,
        counter: count,
    ) -> None:
        if node.op != "call_function":
            return
        target = node.target
        target_name = readable_target(target)
        if is_matmul_target(target):
            if target is torch.bmm and self.patch_config.replace_bmm:
                new_name = f"_tempo_bmm_{next(counter)}"
                gm.add_module(new_name, TempoBMM(runtime_config=self.runtime_config, module_name=new_name))
                with gm.graph.inserting_after(node):
                    new_node = gm.graph.call_module(new_name, args=node.args, kwargs=node.kwargs)
                node.replace_all_uses_with(new_node)
                gm.graph.erase_node(node)
                report.hdo_nodes_replaced.append({"node": node.name, "target": target_name, "new_module": new_name})
            elif target is torch.matmul and self.patch_config.replace_matmul:
                new_name = f"_tempo_matmul_{next(counter)}"
                gm.add_module(new_name, TempoMatMul(runtime_config=self.runtime_config, module_name=new_name))
                with gm.graph.inserting_after(node):
                    new_node = gm.graph.call_module(new_name, args=node.args, kwargs=node.kwargs)
                node.replace_all_uses_with(new_node)
                gm.graph.erase_node(node)
                report.hdo_nodes_replaced.append({"node": node.name, "target": target_name, "new_module": new_name})
            else:
                report.non_hdo_recognized.append({"node": node.name, "target": target_name, "kind": "pairwise"})
            return

        if is_add_target(target):
            report.non_hdo_recognized.append({"node": node.name, "target": target_name, "kind": "add"})
        elif is_cat_target(target):
            report.non_hdo_recognized.append({"node": node.name, "target": target_name, "kind": "cat"})

    def _mark_method_nodes(self, node: fx.Node, report: FXRewriteReport) -> None:
        if node.op == "call_method" and isinstance(node.target, str) and is_reshape_method(node.target):
            report.non_hdo_recognized.append({"node": node.name, "target": node.target, "kind": "reshape_family"})

    def rewrite(self, model: nn.Module) -> tuple[fx.GraphModule, FXRewriteReport]:
        gm = fx.symbolic_trace(model)
        report = FXRewriteReport()
        counter = count()

        for node in list(gm.graph.nodes):
            if node.op == "call_module":
                self._replace_call_module_if_needed(gm, node, report)
            elif node.op == "call_function":
                self._replace_function_node_if_needed(gm, node, report, counter)
            elif node.op == "call_method":
                self._mark_method_nodes(node, report)

        gm.graph.lint()
        gm.recompile()
        return gm, report

