"""Tiny two-step CUDA/DDP checks; run only on explicitly selected free GPUs."""

from datetime import timedelta
import os
from pathlib import Path
import sys

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from SeqRec.models.generative.qwen3.temporal_hierarchical import Qwen3TemporalHierarchicalWithTemperature  # noqa: E402
from tests.models.generative.qwen3.test_rebuttal_ablation import tiny_config  # noqa: E402


def main():
    rank = int(os.environ.get('LOCAL_RANK', 0))
    world = int(os.environ.get('WORLD_SIZE', 1))
    torch.cuda.set_device(rank)
    device = torch.device('cuda', rank)
    if world > 1:
        dist.init_process_group('nccl', timeout=timedelta(seconds=45))
    variants = {
        'full': {}, 'no_qkv': {'th_use_behavior_qkv': False}, 'no_gate': {'th_use_output_gate': False},
        'shared_ffn': {'sparse_layers_decoder': []}, 'zero_prior': {'th_relation_bias_soft_scale': 0.0},
        'plain': {'temporal_hierarchical_attention_decoder': [], 'sparse_layers_decoder': [], 'behavior_injection_decoder': []},
    }
    try:
        for name, updates in variants.items():
            torch.manual_seed(42)
            model = Qwen3TemporalHierarchicalWithTemperature(tiny_config(**updates)).to(device)
            wrapped = DistributedDataParallel(model, device_ids=[rank], find_unused_parameters=False) if world > 1 else model
            inputs = torch.tensor([[11, 12, 13, 10, 14, 15, 0]], device=device)
            labels = inputs.clone()
            labels[:, [0, 3, 6]] = -100
            for _ in range(2):
                wrapped.zero_grad(set_to_none=True)
                output = wrapped(input_ids=inputs, attention_mask=(inputs != 0).long(), labels=labels, use_cache=False)
                if not torch.isfinite(output.loss):
                    raise AssertionError(f'{name}: nonfinite loss')
                output.loss.backward()
                if not torch.isfinite(model.model.layers[0].self_attn.q_proj.weight.grad).all():
                    raise AssertionError(f'{name}: nonfinite projection gradient')
            model.eval()
            with torch.no_grad():
                prompt = inputs[:, :4]
                cached = model.generate(input_ids=prompt, attention_mask=torch.ones_like(prompt), max_new_tokens=2, num_beams=2, use_cache=True)
                uncached = model.generate(input_ids=prompt, attention_mask=torch.ones_like(prompt), max_new_tokens=2, num_beams=2, use_cache=False)
                if not torch.equal(cached, uncached):
                    raise AssertionError(f'{name}: cached/uncached beam generation differs')
            if world > 1:
                dist.barrier()
            if rank == 0:
                print(f'PASS {name}: two forward/backward steps and beam cache, world_size={world}', flush=True)
            del wrapped, model
        torch.cuda.synchronize()
    finally:
        if world > 1:
            dist.destroy_process_group()


if __name__ == '__main__':
    main()
