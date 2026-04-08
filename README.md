# TempoCache

TempoCache is a production-style Python toolkit/runtime for **similarity-driven, operator-block temporal selective execution** in SNN inference.

It separates each temporal block into:

- `Heavy Drive Operator (HDO)`: expensive drive/pre-activation computation.
- `State Transition Operator`: stateful neuron/memory progression (`IF/LIF/PLIF/Delay/...`) that must stay step-wise correct.

TempoCache only routes HDO windows with `FULL / COLLAPSE / REUSE` while preserving temporal state semantics.

## 1. Project Overview

TempoCache provides:

- Unified HDO abstractions and runtime semantics.
- Per-window adaptive routing.
- Per-sample mode decisions in the same window.
- Cache management for drive/pre-activation only.
- Module patching and FX graph rewriting.
- Diagnostics export (JSON/CSV/model summary).
- Operator and model benchmarks.
- CPU-first tests with optional CUDA benchmarking.

## 2. System Abstractions

### HDO

Supported and Tempo-routed:

- `TempoConv1d`
- `TempoConv2d`
- `TempoConv3d`
- `TempoLinear`
- `TempoMatMul`
- `TempoBMM`
- `TempoAttentionCore`
- `TempoSpikeSelfAttention`

Unified interface is implemented across HDO modules:

- `forward_single_step(...)`
- `forward_window(...)`
- `collapse_window(...)`
- `update_cache(...)`
- `get_diagnostics()`

### StatelessPassThroughOps

Recognized by patch/report/FX (not cached):

- BatchNorm1d/2d/3d
- LayerNorm
- GroupNorm
- MaxPool/AvgPool/AdaptiveAvgPool families
- Flatten / Identity / Dropout(eval)
- Add / Cat / residual merge
- View / Reshape / Permute / Transpose

### StatefulTemporalOps

Preserved as temporal step modules (not cached):

- `IFNode`
- `LIFNode`
- `ParametricLIFNode`
- `Delay`
- Custom recurrent/memory-like modules

## 3. Supported Operator Families

- Conv family: 1D / 2D / 3D
- Linear family
- Pairwise family: MatMul / BMM
- Attention family: projection chain + qk + av + out projection

## 4. Why Not Cache Every Operator

TempoCache intentionally does **not** cache all operators because:

- Stateful neuron/memory operators require exact time-step progression.
- Pass-through ops are typically cheap and can be recognized without drive caching.
- Caching target is explicitly HDO drive/pre-activation, not spikes/membrane/recurrent states.

## 5. Installation

```bash
python -m venv .venv
. .venv/bin/activate  # Windows PowerShell: .\.venv\Scripts\Activate.ps1
pip install -e .[dev]
```

## 6. Directory Structure

```text
project_root/
  pyproject.toml
  README.md
  .gitignore
  src/
    tempocache/
      __init__.py
      version.py
      enums.py
      typing.py
      config.py
      registry.py
      utils/
        __init__.py
        seed.py
        tensor_ops.py
        logging_utils.py
        export.py
        module_utils.py
        fx_utils.py
      core/
        __init__.py
        base.py
        signatures/
          __init__.py
          spatial.py
          vector.py
          pairwise.py
          attention.py
        distance.py
        router.py
        cache_state.py
        executor_base.py
        diagnostics.py
      executors/
        __init__.py
        conv.py
        linear.py
        pairwise.py
        attention.py
      ops/
        __init__.py
        tempo_conv1d.py
        tempo_conv2d.py
        tempo_conv3d.py
        tempo_linear.py
        tempo_matmul.py
        tempo_bmm.py
        tempo_attention.py
      integration/
        __init__.py
        patcher.py
        fx_rewriter.py
        wrappers.py
        reports.py
      benchmark/
        __init__.py
        smoke.py
        bench_operator.py
        bench_model.py
        bench_attention.py
      models/
        __init__.py
        toy_cnn_snn.py
        toy_res_snn.py
        toy_spike_transformer.py
        neurons.py
      cli/
        __init__.py
        main.py
  tests/
    test_signature_spatial.py
    test_signature_vector.py
    test_signature_pairwise.py
    test_signature_attention.py
    test_distance.py
    test_router.py
    test_cache_state.py
    test_tempo_conv_family.py
    test_tempo_linear.py
    test_tempo_matmul_bmm.py
    test_tempo_attention.py
    test_patcher.py
    test_fx_rewriter.py
    test_toy_cnn_integration.py
    test_toy_res_integration.py
    test_toy_transformer_integration.py
```

## 7. Quick Start

```python
import torch
from tempocache.config import RuntimeConfig
from tempocache.ops import TempoConv2d

conv = torch.nn.Conv2d(16, 32, 3, padding=1)
runtime = RuntimeConfig(mode="adaptive", window_size=4)
tempo_conv = TempoConv2d.from_conv(conv, runtime, module_name="demo.conv")

x = torch.randn(12, 8, 16, 32, 32)  # [T,B,C,H,W]
y = tempo_conv(x)
print(y.shape)
print(tempo_conv.get_diagnostics())
```

## 8. Module Patcher Usage

```python
from tempocache.config import RuntimeConfig, PatchConfig
from tempocache.integration import patch_model
from tempocache.models import ToyCNNSNN

model = ToyCNNSNN(use_tempo=False)
runtime = RuntimeConfig(mode="adaptive", window_size=4)
patch_cfg = PatchConfig(include_patterns=["*"], exclude_patterns=["*.head"])

patched_model, report = patch_model(model, runtime, patch_cfg)
print(report.to_dict())
```

## 9. FX Graph Rewrite Usage

```python
from tempocache.config import RuntimeConfig, PatchConfig
from tempocache.integration import TempoFXRewriter

runtime = RuntimeConfig(mode="adaptive", window_size=4)
rewriter = TempoFXRewriter(runtime_config=runtime, patch_config=PatchConfig(use_fx_rewriter=True))

gm, report = rewriter.rewrite(your_model)
print(report.to_dict())
```

## 10. Conv / Linear / MatMul / Attention Support Notes

- Conv family supports grouped/depthwise/dilation/padding/stride/bias via wrapped native layers.
- Linear supports `[T,B,D]` and `[T,B,N,D]`.
- MatMul/BMM supports pairwise windows like:
  - `A_win [K,B,M,N]`
  - `B_win [K,B,N,P]`
  - output `[K,B,M,P]`
- Attention stack supports:
  - `q_proj/k_proj/v_proj/out_proj`
  - `qk^T`
  - `attn@v`
  - spike-friendly attention normalization
  - future variant hook through registry

## 11. Four Runtime Modes

- `full`: always full recompute
- `fixed_collapse`: always collapse each window
- `fixed_reuse`: reuse cache after first valid window
- `adaptive`: router-based per-sample decision

Adaptive routing checks (per sample):

1. cache invalid => FULL
2. hard drift (`d_inter >= tau_reuse_hard`) => invalidate + FULL
3. `cache_age >= max_reuse` => FULL
4. stable and close => REUSE
5. stable enough => COLLAPSE
6. otherwise FULL

## 12. Diagnostics Export

Each HDO tracks:

- `total_windows`, `total_samples`
- `full_count`, `collapse_count`, `reuse_count`
- `cache_hit_count`
- `hard_invalidation_count`, `forced_refresh_count`
- `mean_d_inter`, `mean_v_temp`
- optional `window_history`

Export APIs:

```python
from tempocache.integration import export_diagnostics

payload = export_diagnostics(model, export_dir="benchmark_outputs")
print(payload)
```

## 13. Benchmark Commands

```bash
python -m tempocache.cli.main bench-op --operator-family conv2d --mode adaptive --device cpu --dtype float32
python -m tempocache.cli.main bench-model --model-type cnn --mode adaptive --device cpu --dtype float32
python -m tempocache.cli.main bench-attn --mode adaptive --device cpu --dtype float32
```

## 14. Test Command

```bash
pytest
```

## 15. Known Limitations

- FX attention-pattern rewriting is currently focused on explicit op nodes (module + function) and lightweight annotation logic.
- Adaptive thresholds are numeric heuristics and need tuning per workload/hardware.
- Benchmarks are synthetic and toy-model centered; production model calibration still required.

## 16. Future Extensions

- Custom HDO plugin registration (specialized sparse/event kernels).
- Custom neuron/state module adapters and richer state-awareness policies.
- Custom attention variants (SSA/DSSA/QK variants) via registry hooks.
- More aggressive FX pattern fusion and graph-level schedule optimization.

## CLI Overview

```bash
python -m tempocache.cli.main smoke ...
python -m tempocache.cli.main bench-op ...
python -m tempocache.cli.main bench-model ...
python -m tempocache.cli.main bench-attn ...
python -m tempocache.cli.main patch-report ...
python -m tempocache.cli.main export-diag ...
```
