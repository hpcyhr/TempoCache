"""TempoCache CLI entrypoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from ..benchmark import benchmark_attention_configs, benchmark_model, benchmark_operator_family, run_smoke
from ..config import AttentionConfig, BenchmarkConfig, PatchConfig, RuntimeConfig
from ..integration import TempoFXRewriter, TempoPatcher, export_diagnostics
from ..models import ToyCNNSNN, ToyResSNN, ToySpikeTransformer
from ..utils.export import dump_json


def _build_model(model_type: str, runtime: RuntimeConfig, attention_config: AttentionConfig) -> torch.nn.Module:
    if model_type == "cnn":
        return ToyCNNSNN(use_tempo=False, runtime_config=runtime)
    if model_type == "res":
        return ToyResSNN(use_tempo=False, runtime_config=runtime)
    if model_type == "transformer":
        return ToySpikeTransformer(
            input_dim=32,
            embed_dim=64,
            num_heads=4,
            use_tempo=False,
            runtime_config=runtime,
            attention_config=attention_config,
        )
    raise ValueError(f"Unsupported model_type: {model_type}")


def _sample_input(model_type: str, t: int, batch: int, device: str, dtype: str) -> torch.Tensor:
    torch_dtype = getattr(torch, dtype)
    if model_type in {"cnn", "res"}:
        return torch.randn(t, batch, 3, 32, 32, device=device, dtype=torch_dtype)
    if model_type == "transformer":
        return torch.randn(t, batch, 32, 32, device=device, dtype=torch_dtype)
    raise ValueError(f"Unsupported model_type: {model_type}")


def _bench_cfg(args: argparse.Namespace) -> BenchmarkConfig:
    return BenchmarkConfig(
        device=args.device,
        dtype=args.dtype,
        T=args.T,
        batch_size=args.batch_size,
        warmup=args.warmup,
        iters=args.iters,
        seed=args.seed,
        export_dir=args.export_dir,
    )


def _attention_cfg_from_args(args: argparse.Namespace) -> AttentionConfig:
    return AttentionConfig(
        cache_q_proj=args.cache_q_proj,
        cache_k_proj=args.cache_k_proj,
        cache_v_proj=args.cache_v_proj,
        cache_qk_score=args.cache_qk_score,
        cache_av_product=args.cache_av_product,
        cache_out_proj=args.cache_out_proj,
    )


def _print_or_export(payload: dict, export_path: str | None) -> None:
    if export_path:
        dump_json(export_path, payload)
        print(json.dumps({"exported_to": export_path}, indent=2))
    else:
        print(json.dumps(payload, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="TempoCache CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--mode", type=str, default="adaptive", choices=["full", "fixed_collapse", "fixed_reuse", "adaptive"])
        p.add_argument("--T", type=int, default=16)
        p.add_argument("--K", type=int, default=4)
        p.add_argument("--batch-size", type=int, default=8)
        p.add_argument("--device", type=str, default="cpu")
        p.add_argument("--dtype", type=str, default="float32", choices=["float32", "float16", "bfloat16"])
        p.add_argument("--warmup", type=int, default=5)
        p.add_argument("--iters", type=int, default=20)
        p.add_argument("--seed", type=int, default=7)
        p.add_argument("--export-dir", type=str, default="benchmark_outputs")
        p.add_argument("--export-path", type=str, default="")

    smoke = sub.add_parser("smoke", help="Run smoke checks")
    smoke.add_argument("--device", type=str, default="cpu")
    smoke.add_argument("--dtype", type=str, default="float32")
    smoke.add_argument("--export-path", type=str, default="")

    bench_op = sub.add_parser("bench-op", help="Benchmark operator family")
    add_common(bench_op)
    bench_op.add_argument(
        "--operator-family",
        type=str,
        default="conv2d",
        choices=["conv1d", "conv2d", "conv3d", "linear", "matmul", "bmm", "attention"],
    )

    bench_model_cmd = sub.add_parser("bench-model", help="Benchmark toy model")
    add_common(bench_model_cmd)
    bench_model_cmd.add_argument("--model-type", type=str, default="cnn", choices=["cnn", "res", "transformer"])

    bench_attn = sub.add_parser("bench-attn", help="Benchmark attention caching breakdown")
    add_common(bench_attn)

    patch_report = sub.add_parser("patch-report", help="Generate patch and FX rewrite report")
    add_common(patch_report)
    patch_report.add_argument("--model-type", type=str, default="cnn", choices=["cnn", "res", "transformer"])
    patch_report.add_argument("--use-fx-rewriter", action="store_true")
    patch_report.add_argument("--cache-q-proj", action=argparse.BooleanOptionalAction, default=True)
    patch_report.add_argument("--cache-k-proj", action=argparse.BooleanOptionalAction, default=True)
    patch_report.add_argument("--cache-v-proj", action=argparse.BooleanOptionalAction, default=True)
    patch_report.add_argument("--cache-qk-score", action=argparse.BooleanOptionalAction, default=True)
    patch_report.add_argument("--cache-av-product", action=argparse.BooleanOptionalAction, default=True)
    patch_report.add_argument("--cache-out-proj", action=argparse.BooleanOptionalAction, default=True)

    export_diag = sub.add_parser("export-diag", help="Run model and export diagnostics")
    add_common(export_diag)
    export_diag.add_argument("--model-type", type=str, default="cnn", choices=["cnn", "res", "transformer"])
    export_diag.add_argument("--cache-q-proj", action=argparse.BooleanOptionalAction, default=True)
    export_diag.add_argument("--cache-k-proj", action=argparse.BooleanOptionalAction, default=True)
    export_diag.add_argument("--cache-v-proj", action=argparse.BooleanOptionalAction, default=True)
    export_diag.add_argument("--cache-qk-score", action=argparse.BooleanOptionalAction, default=True)
    export_diag.add_argument("--cache-av-product", action=argparse.BooleanOptionalAction, default=True)
    export_diag.add_argument("--cache-out-proj", action=argparse.BooleanOptionalAction, default=True)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "smoke":
        payload = run_smoke(device=args.device, dtype=args.dtype)
        _print_or_export(payload, args.export_path or None)
        return

    if args.command == "bench-op":
        cfg = _bench_cfg(args)
        payload = benchmark_operator_family(family=args.operator_family, mode=args.mode, bench_config=cfg)
        _print_or_export(payload, args.export_path or None)
        return

    if args.command == "bench-model":
        cfg = _bench_cfg(args)
        payload = benchmark_model(model_type=args.model_type, mode=args.mode, bench_config=cfg)
        _print_or_export(payload, args.export_path or None)
        return

    if args.command == "bench-attn":
        cfg = _bench_cfg(args)
        payload = benchmark_attention_configs(mode=args.mode, bench_config=cfg)
        _print_or_export(payload, args.export_path or None)
        return

    if args.command == "patch-report":
        runtime = RuntimeConfig(mode=args.mode, window_size=args.K)
        attention_cfg = _attention_cfg_from_args(args)
        model = _build_model(args.model_type, runtime, attention_cfg)
        patcher = TempoPatcher(runtime_config=runtime, patch_config=PatchConfig(), attention_config=attention_cfg)
        model, report = patcher.patch(model)
        payload = {"module_patch_report": report.to_dict()}
        if args.use_fx_rewriter:
            rewriter = TempoFXRewriter(runtime_config=runtime, patch_config=PatchConfig(use_fx_rewriter=True))
            _, fx_report = rewriter.rewrite(model)
            payload["fx_report"] = fx_report.to_dict()
        _print_or_export(payload, args.export_path or None)
        return

    if args.command == "export-diag":
        runtime = RuntimeConfig(mode=args.mode, window_size=args.K)
        attention_cfg = _attention_cfg_from_args(args)
        model = _build_model(args.model_type, runtime, attention_cfg)
        patcher = TempoPatcher(runtime_config=runtime, patch_config=PatchConfig(), attention_config=attention_cfg)
        model, patch_report = patcher.patch(model)
        x = _sample_input(args.model_type, args.T, args.batch_size, args.device, args.dtype)
        with torch.no_grad():
            _ = model(x)
        payload = export_diagnostics(model, export_dir=args.export_dir)
        payload["patch_report"] = patch_report.to_dict()
        _print_or_export(payload, args.export_path or None)
        return


if __name__ == "__main__":
    main()
