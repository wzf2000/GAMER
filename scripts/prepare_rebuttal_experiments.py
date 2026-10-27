#!/usr/bin/env python
"""Write isolated TH configs, a run manifest, and shell commands; never launch jobs."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from SeqRec.utils.rebuttal import prepare_plan  # noqa: E402


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', choices=['components', 'seeds', 'hierarchy', 'lambda', 'handoff'], required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--base_model', default='./config/s2s-models/Qwen3TemporalHierarchicalFixedSoft')
    parser.add_argument('--dataset', default='ShortVideoAD')
    parser.add_argument('--data_path', default='./data')
    parser.add_argument('--index_file', default='.index.json')
    parser.add_argument('--tasks', default='smb_explicit_decoder_4')
    parser.add_argument('--max_his_len', type=int, default=100)
    parser.add_argument('--model_max_length', type=int, default=1024)
    parser.add_argument('--seeds', type=int, nargs='+')
    parser.add_argument('--key_seeds_only', action='store_true', help='For components: use all seeds for full/plain_matched and only the first seed for the four remaining ablations.')
    parser.add_argument('--gpu', '--gpus', dest='gpu', default='0,1,2,3', help='Comma-separated GPU IDs, e.g. 2,5.')
    parser.add_argument('--port', type=int, default=2314)
    parser.add_argument('--num_train_epochs', type=int, default=200)
    parser.add_argument('--per_device_train_batch_size', type=int, default=128)
    parser.add_argument('--per_device_eval_batch_size', type=int, default=128)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4)
    parser.add_argument('--learning_rate', type=float, default=5e-4)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--warmup_ratio', type=float, default=0.04)
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--precision', choices=['fp32', 'fp16', 'bf16'], default='fp32')
    parser.add_argument('--test_batch_size', type=int, default=64, help='Per-device ranking evaluation batch size.')
    parser.add_argument('--num_beams', type=int, default=20)
    parser.add_argument('--parameter_tolerance', type=float, default=0.001)
    parser.add_argument('--lambdas', type=float, nargs='+', default=[0.0, 0.01, 0.05, 0.1])
    parser.add_argument('--hierarchy_maps', help='JSON object mapping experiment names to behavior-name/rank dictionaries.')
    parser.add_argument('--validation_behavior', help='Optional behavior filter for training-time validation loss.')
    parser.add_argument('--eval_behavior', help='Optional behavior filter for validation/test ranking; default evaluates all behaviors.')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--reuse_plans', nargs='+', default=[], help='Handoff only: verified compatible source plans whose paths should be reused.')
    parser.add_argument('--export_user_targets', action='store_true', help='Enable original target-ID export in generated ranking commands (automatic for handoff).')
    args = parser.parse_args(argv)
    if args.seeds is None:
        args.seeds = [42, 43, 44] if args.suite == 'seeds' else [42]
    if args.key_seeds_only and args.suite != 'components':
        parser.error('--key_seeds_only applies only to the components suite.')
    if args.reuse_plans and args.suite != 'handoff':
        parser.error('--reuse_plans applies only to the handoff suite.')
    if args.parameter_tolerance < 0:
        parser.error('--parameter_tolerance must be nonnegative.')
    return args


if __name__ == '__main__':
    arguments = parse_args()
    plan = prepare_plan(arguments)
    print(f'Prepared {len(plan["runs"])} runs in {Path(arguments.output_dir).resolve()}; no jobs launched.')
