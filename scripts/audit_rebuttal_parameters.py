#!/usr/bin/env python
"""Probe a separate TH model instance; no optimizer, checkpoint writes or training data cache access."""

import argparse
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402
from transformers.models.qwen3_moe import Qwen3MoeConfig  # noqa: E402
from SeqRec.models.generative.qwen3.temporal_hierarchical import Qwen3TemporalHierarchicalWithTemperature  # noqa: E402
from SeqRec.utils.parameter_audit import audit_parameters, synthetic_probe_batch  # noqa: E402
from SeqRec.utils.rebuttal import file_sha256, git_state, write_json  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, help='Resolved config.json with the final expanded vocabulary.')
    parser.add_argument('--checkpoint', help='Optional completed checkpoint; otherwise use an independent random initialization.')
    parser.add_argument('--batch_file', help='JSON list of actual collated tensor dictionaries; default is an explicitly synthetic probe.')
    parser.add_argument('--output_file', required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--temperature', type=float, default=0.7, help='LM loss temperature; align with the audited training recipe.')
    args = parser.parse_args()
    if Path(args.output_file).exists():
        raise FileExistsError(args.output_file)
    if not math.isfinite(args.temperature) or args.temperature <= 0:
        parser.error('--temperature must be finite and positive.')
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    config = Qwen3MoeConfig(**json.loads(Path(args.config).read_text()))
    model = Qwen3TemporalHierarchicalWithTemperature.from_pretrained(args.checkpoint) if args.checkpoint else Qwen3TemporalHierarchicalWithTemperature(config)
    if args.checkpoint:
        structural_keys = ('vocab_size', 'hidden_size', 'intermediate_size', 'num_hidden_layers', 'num_attention_heads', 'num_key_value_heads', 'head_dim',
                           'num_positions', 'num_behavior', 'behavior_maps', 'n_positions', 'model_max_length', 'mlp_type', 'Moe_behavior_only',
                           'behavior_injection_decoder', 'sparse_layers_decoder', 'temporal_hierarchical_attention_decoder', 'th_attention_mode',
                           'th_relation_bias_type', 'th_relation_bias_trainable', 'th_relation_bias_init', 'th_relation_bias_soft_scale',
                           'th_relation_bias_scale', 'th_relation_bias_learnable_scale', 'th_prior_behavior_ranks', 'th_use_behavior_qkv',
                           'th_use_output_gate', 'dropout_rate', 'attention_dropout', 'tie_word_embeddings')
        for key in structural_keys:
            default = True if key.startswith('th_use_') else None
            actual = json.loads(json.dumps(getattr(model.config, key, default)))
            expected = json.loads(json.dumps(getattr(config, key, default)))
            if actual != expected:
                raise ValueError(f'Checkpoint differs from requested config: {key}')
    model = model.to(args.device)
    model.set_hyper(args.temperature)
    if args.batch_file:
        batches = []
        for batch in json.loads(Path(args.batch_file).read_text()):
            converted = {}
            for key, value in batch.items():
                try:
                    converted[key] = torch.tensor(value, device=args.device)
                except (TypeError, ValueError):
                    converted[key] = value
            batches.append(converted)
    else:
        batches = [synthetic_probe_batch(config, args.device)]
    report = audit_parameters(model, batches)
    report.update(**git_state(), config=str(Path(args.config).resolve()), config_sha256=file_sha256(args.config),
                  checkpoint=args.checkpoint, probe='provided_collated_batches' if args.batch_file else 'synthetic_all_behaviors',
                  batch_sha256=file_sha256(args.batch_file) if args.batch_file else None, seed=args.seed,
                  temperature=args.temperature,
                  attention_backend=model.config._attn_implementation, dtype=str(next(model.parameters()).dtype), device=args.device)
    write_json(args.output_file, report)
    print(f'Saved parameter audit to {args.output_file}; total={report["total"]}, observed active={report["active_participating"]}.')


if __name__ == '__main__':
    main()
