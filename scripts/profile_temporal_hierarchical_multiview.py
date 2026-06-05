#!/usr/bin/env python
"""Profile temporal-hierarchical attention with synthetic inputs.

This script is intentionally independent from datasets and trainers. It builds
extreme synthetic behavior-token sequences and compares relation-bias against
multi-view attention under the same model shape.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path
from types import MethodType
from typing import Callable

import torch
from torch.profiler import ProfilerActivity, profile, record_function
from transformers.models.qwen3_moe import Qwen3MoeConfig

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from SeqRec.models.generative.qwen3.temporal_hierarchical import Qwen3TemporalHierarchicalWithTemperature  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Synthetic profiler for Qwen3 temporal-hierarchical multi-view attention."
    )
    parser.add_argument("--config", default="./config/s2s-models/Qwen3TemporalHierarchicalMultiView")
    parser.add_argument(
        "--mode",
        choices=["relation_bias", "table_trainable", "table_fixed", "factorized", "multi_view", "both", "all"],
        default="both",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float32")
    parser.add_argument("--attn_implementation", choices=["eager", "sdpa"], default="sdpa")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--seq_len", type=int, default=1024)
    parser.add_argument("--num_positions", type=int, default=4)
    parser.add_argument("--num_behavior", type=int, default=4)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--warmup_steps", type=int, default=2)
    parser.add_argument("--backward", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--profile_steps", type=int, default=3)
    parser.add_argument("--trace_dir", default="./logs/profiler/temporal_hierarchical")
    parser.add_argument("--pattern", choices=["alternating", "same", "ascending", "random"], default="alternating")
    parser.add_argument("--relation_bias_type", choices=["table", "factorized"], default="table")
    parser.add_argument("--relation_bias_rank", type=int, default=4)
    parser.add_argument("--relation_bias_trainable", choices=["true", "false"], default="true")
    parser.add_argument("--skip_full_model", action="store_true")
    parser.add_argument("--bias_repeats", type=int, default=50)
    return parser.parse_args()


def get_dtype(name: str) -> torch.dtype:
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    return torch.float32


def synchronize(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def normalize_mode(mode: str, args: argparse.Namespace) -> tuple[str, str, bool]:
    if mode == "relation_bias":
        return "relation_bias", args.relation_bias_type, args.relation_bias_trainable == "true"
    if mode == "table_trainable":
        return "relation_bias", "table", True
    if mode == "table_fixed":
        return "relation_bias", "table", False
    if mode == "factorized":
        return "relation_bias", "factorized", True
    if mode == "multi_view":
        return "multi_view", args.relation_bias_type, args.relation_bias_trainable == "true"
    raise ValueError(f"Unsupported benchmark mode: {mode}")


def configure_synthetic_config(config: Qwen3MoeConfig, args: argparse.Namespace, mode: str) -> Qwen3MoeConfig:
    attention_mode, relation_bias_type, relation_bias_trainable = normalize_mode(mode, args)
    config.th_attention_mode = attention_mode
    config._attn_implementation = args.attn_implementation
    config.num_positions = args.num_positions
    config.num_behavior = args.num_behavior
    config.use_behavior_token = True
    config.use_user_token = False
    config.n_positions = math.ceil(args.seq_len / args.num_positions) + 2
    config.model_max_length = max(args.seq_len, getattr(config, "model_max_length", args.seq_len))
    config.num_experts = args.num_positions + 1
    config.behavior_maps = {i: i for i in range(args.num_behavior)}
    config.th_relation_bias_type = relation_bias_type
    config.th_relation_bias_rank = args.relation_bias_rank
    config.th_relation_bias_trainable = relation_bias_trainable
    if not hasattr(config, "Moe_behavior_only"):
        config.Moe_behavior_only = False
    if attention_mode == "multi_view":
        config.th_multi_view_types = ["temporal", "same", "up", "down"]
        if not hasattr(config, "th_multi_view_head_allocation"):
            base = config.num_attention_heads // 4
            allocation = [base] * 4
            for i in range(config.num_attention_heads - sum(allocation)):
                allocation[i] += 1
            config.th_multi_view_head_allocation = allocation
        config.th_multi_view_use_relation_bias = False
    return config


def build_synthetic_input(args: argparse.Namespace, device: torch.device) -> torch.LongTensor:
    batch_size = args.batch_size
    seq_len = args.seq_len
    num_positions = args.num_positions
    num_behavior = args.num_behavior
    num_items = math.ceil(seq_len / num_positions)

    input_ids = torch.full((batch_size, num_items * num_positions), 5, dtype=torch.long)
    item_tokens = torch.arange(5, 5 + max(1, num_positions - 1), dtype=torch.long)
    item_tokens = item_tokens.clamp(max=7)

    if args.pattern == "same":
        behavior_levels = torch.zeros((batch_size, num_items), dtype=torch.long)
    elif args.pattern == "ascending":
        behavior_levels = torch.arange(num_items, dtype=torch.long).remainder(num_behavior).repeat(batch_size, 1)
    elif args.pattern == "random":
        generator = torch.Generator()
        generator.manual_seed(2026)
        behavior_levels = torch.randint(0, num_behavior, (batch_size, num_items), generator=generator)
    else:
        base = torch.arange(num_items, dtype=torch.long).remainder(num_behavior)
        behavior_levels = base.unsqueeze(0).repeat(batch_size, 1)
        behavior_levels[1::2] = torch.flip(base, dims=[0])

    for item_idx in range(num_items):
        start = item_idx * num_positions
        input_ids[:, start] = behavior_levels[:, item_idx]
        if num_positions > 1:
            repeated_items = item_tokens[: num_positions - 1]
            input_ids[:, start + 1 : start + num_positions] = repeated_items

    input_ids = input_ids[:, :seq_len].contiguous()
    return input_ids.to(device)


def make_labels(input_ids: torch.LongTensor, pad_token_id: int) -> torch.LongTensor:
    labels = input_ids.clone()
    labels[labels == pad_token_id] = -100
    return labels


def patch_attention_record_functions(model: torch.nn.Module):
    for layer_idx, layer in enumerate(model.model.layers):
        attn = getattr(layer, "self_attn", None)
        if attn is None or not getattr(attn, "is_temporal_hierarchical", False):
            continue

        original_forward = attn.forward

        def forward_with_label(self, *forward_args, _original_forward=original_forward, _layer_idx=layer_idx, **kwargs):
            with record_function(f"th_layer_{_layer_idx}.attention_forward"):
                return _original_forward(*forward_args, **kwargs)

        attn.forward = MethodType(forward_with_label, attn)

        if hasattr(attn, "_compute_multi_view_bias"):
            original_multi_view = attn._compute_multi_view_bias

            def multi_view_with_label(self, *fn_args, _original_multi_view=original_multi_view, _layer_idx=layer_idx, **kwargs):
                with record_function(f"th_layer_{_layer_idx}.compute_multi_view_bias"):
                    return _original_multi_view(*fn_args, **kwargs)

            attn._compute_multi_view_bias = MethodType(multi_view_with_label, attn)

        if hasattr(attn, "_compute_level_pair_bias"):
            original_relation = attn._compute_level_pair_bias

            def relation_with_label(self, *fn_args, _original_relation=original_relation, _layer_idx=layer_idx, **kwargs):
                with record_function(f"th_layer_{_layer_idx}.compute_relation_bias"):
                    return _original_relation(*fn_args, **kwargs)

            attn._compute_level_pair_bias = MethodType(relation_with_label, attn)


def build_model(args: argparse.Namespace, mode: str, device: torch.device, dtype: torch.dtype):
    config = Qwen3MoeConfig.from_pretrained(args.config)
    configure_synthetic_config(config, args, mode)
    model = Qwen3TemporalHierarchicalWithTemperature(config)
    model.set_hyper(0.7)
    model.config.use_cache = False
    model.train()
    model.to(device=device, dtype=dtype)
    _, relation_bias_type, relation_bias_trainable = normalize_mode(mode, args)
    if relation_bias_type == "table" and not relation_bias_trainable:
        frozen_count = 0
        for module in model.modules():
            level_pair_bias = getattr(module, "level_pair_bias", None)
            if level_pair_bias is not None:
                level_pair_bias.requires_grad_(False)
                frozen_count += 1
        print(f"[setup] mode={mode} relation_bias_trainable=false frozen_level_pair_bias={frozen_count}")
    patch_attention_record_functions(model)
    return model


def run_step(model: torch.nn.Module, input_ids: torch.LongTensor, labels: torch.LongTensor, backward: bool):
    outputs = model(input_ids=input_ids, labels=labels, use_cache=False)
    loss = outputs.loss if outputs.loss is not None else outputs.logits.float().mean()
    if backward:
        loss.backward()
        model.zero_grad(set_to_none=True)
    return float(loss.detach().cpu())


def benchmark_full_model(args: argparse.Namespace, mode: str):
    device = torch.device(args.device)
    dtype = get_dtype(args.dtype)
    model = build_model(args, mode, device, dtype)
    input_ids = build_synthetic_input(args, device)
    labels = make_labels(input_ids, model.config.pad_token_id)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for _ in range(args.warmup_steps):
        run_step(model, input_ids, labels, args.backward)
    synchronize(device)

    elapsed = []
    losses = []
    for _ in range(args.steps):
        start = time.perf_counter()
        losses.append(run_step(model, input_ids, labels, args.backward))
        synchronize(device)
        elapsed.append(time.perf_counter() - start)

    peak_memory_mb = (
        torch.cuda.max_memory_allocated(device) / 1024 / 1024
        if device.type == "cuda"
        else 0.0
    )
    print(
        f"[full_model] mode={mode} batch={args.batch_size} seq={args.seq_len} "
        f"backward={args.backward} mean_step_ms={sum(elapsed) / len(elapsed) * 1000:.3f} "
        f"last_loss={losses[-1]:.6f} peak_mem_mb={peak_memory_mb:.1f}"
    )

    if args.profile:
        activities = [ProfilerActivity.CPU]
        if device.type == "cuda":
            activities.append(ProfilerActivity.CUDA)
        trace_path = Path(args.trace_dir) / mode
        trace_path.mkdir(parents=True, exist_ok=True)
        with profile(
            activities=activities,
            record_shapes=True,
            profile_memory=True,
            with_stack=False,
            on_trace_ready=torch.profiler.tensorboard_trace_handler(str(trace_path)),
        ) as prof:
            for _ in range(args.profile_steps):
                run_step(model, input_ids, labels, args.backward)
                prof.step()
        print(prof.key_averages().table(sort_by="cuda_time_total" if device.type == "cuda" else "cpu_time_total", row_limit=30))
        print(f"[profile] TensorBoard trace written to {trace_path}")


def get_first_temporal_attention(model: torch.nn.Module):
    for layer in model.model.layers:
        attn = getattr(layer, "self_attn", None)
        if attn is not None and getattr(attn, "is_temporal_hierarchical", False):
            return attn
    raise RuntimeError("No temporal-hierarchical attention layer found.")


def build_action_indices(args: argparse.Namespace, device: torch.device) -> torch.LongTensor:
    input_ids = build_synthetic_input(args, device)
    config = Qwen3MoeConfig.from_pretrained(args.config)
    configure_synthetic_config(config, args, "multi_view")
    model = Qwen3TemporalHierarchicalWithTemperature(config).to(device)
    with torch.no_grad():
        _, _, action_indices = model.model.router(input_ids)
    return action_indices


def time_cuda_callable(device: torch.device, fn: Callable[[], torch.Tensor], repeats: int) -> tuple[float, torch.Tensor]:
    synchronize(device)
    if device.type == "cuda":
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        result = None
        for _ in range(repeats):
            result = fn()
        end_event.record()
        synchronize(device)
        assert result is not None
        return start_event.elapsed_time(end_event) / repeats, result

    start = time.perf_counter()
    result = None
    for _ in range(repeats):
        result = fn()
    assert result is not None
    return (time.perf_counter() - start) * 1000 / repeats, result


def benchmark_bias_build(args: argparse.Namespace, mode: str):
    device = torch.device(args.device)
    dtype = get_dtype(args.dtype)
    model = build_model(args, mode, device, dtype)
    attn = get_first_temporal_attention(model)
    action_indices = build_action_indices(args, device)
    key_action_indices = action_indices

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    def compute_multi_view_bias():
        return attn._compute_multi_view_bias(action_indices, key_action_indices, dtype)

    def compute_relation_bias():
        return attn._compute_level_pair_bias(action_indices, key_action_indices, dtype)

    fn = compute_multi_view_bias if normalize_mode(mode, args)[0] == "multi_view" else compute_relation_bias

    with torch.no_grad():
        ms, result = time_cuda_callable(device, fn, args.bias_repeats)
    peak_memory_mb = (
        torch.cuda.max_memory_allocated(device) / 1024 / 1024
        if device.type == "cuda"
        else 0.0
    )
    print(
        f"[bias_build] mode={mode} repeats={args.bias_repeats} shape={tuple(result.shape)} "
        f"dtype={result.dtype} mean_ms={ms:.3f} peak_mem_mb={peak_memory_mb:.1f}"
    )


def iter_modes(mode: str) -> list[str]:
    if mode == "both":
        return ["relation_bias", "multi_view"]
    if mode == "all":
        return ["table_trainable", "table_fixed", "factorized", "multi_view"]
    return [mode]


def main():
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")

    print(
        f"Profiling with device={device}, dtype={args.dtype}, attn={args.attn_implementation}, "
        f"batch={args.batch_size}, seq={args.seq_len}, pattern={args.pattern}, "
        f"relation_bias_type={args.relation_bias_type}, relation_bias_rank={args.relation_bias_rank}, "
        f"relation_bias_trainable={args.relation_bias_trainable}."
    )
    for mode in iter_modes(args.mode):
        benchmark_bias_build(args, mode)
        if not args.skip_full_model:
            benchmark_full_model(args, mode)


if __name__ == "__main__":
    main()
