#!/usr/bin/env python
"""Generate machine-local remaining-experiment plans; optionally warm CPU caches, never train."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.prepare_rebuttal_experiments import parse_args as experiment_args  # noqa: E402
from SeqRec.utils.rebuttal import prepare_plan, render_command  # noqa: E402


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu', '--gpus', dest='gpu', default='0,1')
    parser.add_argument('--port', type=int, default=2314)
    parser.add_argument('--groups', nargs='+', choices=['p0', 'seeds', 'tmall'], default=['p0', 'seeds'])
    parser.add_argument('--output_root', default='./results/rebuttal')
    parser.add_argument('--data_path', default='./data')
    parser.add_argument('--dataset', default='ShortVideoAD', help='Dataset for P0 and repeat seeds.')
    parser.add_argument('--index_file', default='.index.json')
    parser.add_argument('--base_model', default='./config/s2s-models/Qwen3TemporalHierarchicalFixedSoft')
    parser.add_argument('--effective_batch_size', type=int, default=2048)
    parser.add_argument('--per_device_train_batch_size', type=int, default=128)
    parser.add_argument('--per_device_eval_batch_size', type=int, default=128)
    parser.add_argument('--test_batch_size', type=int, default=8)
    parser.add_argument('--num_train_epochs', type=int, default=200)
    parser.add_argument('--max_his_len', type=int, default=100)
    parser.add_argument('--model_max_length', type=int, default=1024)
    parser.add_argument('--learning_rate', type=float, default=5e-4)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--warmup_ratio', type=float, default=0.04)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--num_beams', type=int, default=20)
    parser.add_argument('--precision', choices=['fp32', 'fp16', 'bf16'], default='fp32')
    parser.add_argument('--prepare_cache', action='store_true', help='Also build/load training and ranking caches sequentially on CPU; may use several GB of RAM/disk.')
    return parser.parse_args(argv)


def check_input(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f'Missing input: {path}. Obtain the dataset/LFS object or copy the required index first.')
    with path.open('rb') as file:
        header = file.read(256).lstrip()
    if header.startswith(b'version https://git-lfs.github.com/spec/v1'):
        raise ValueError(f'Git LFS pointer instead of data: {path}. Run git lfs pull for the required dataset.')
    if not header:
        raise ValueError(f'Empty input file: {path}')


def prepare_remaining(args):
    render_command([], args.gpu, args.port)
    if not 1 <= args.port <= 65535 or len(set(args.groups)) != len(args.groups):
        raise ValueError('Use a valid TCP port and distinct groups.')
    denominator = args.per_device_train_batch_size * len(args.gpu.split(','))
    if denominator <= 0 or args.effective_batch_size <= 0 or args.effective_batch_size % denominator:
        raise ValueError('effective_batch_size must be a positive multiple of per-device train batch × GPU count.')
    accumulation = args.effective_batch_size // denominator
    root = Path(args.output_root).resolve()
    descriptions = []
    for group in args.groups:
        dataset = 'Tmall-24-0.25-V2' if group == 'tmall' else args.dataset
        index_file = '.index.cid.shuffle.chunk64.json' if group == 'tmall' else args.index_file
        destination = root / f'remaining-{group}'
        if destination.exists():
            raise FileExistsError(f'Refusing to overwrite existing plan directory: {destination}')
        data = Path(args.data_path).resolve() / dataset
        for suffix in (index_file, '.behavior_level.json', '.SMB.inter.json', '.SMB.behavior.json', '.SMB.session.json', '.SMB.time.json'):
            check_input(data / f'{dataset}{suffix}')
        levels = json.loads((data / f'{dataset}.behavior_level.json').read_text())
        if not levels:
            raise ValueError('Behavior metadata is empty.')
        core = [behavior for behavior, level in levels.items() if level == max(levels.values())][-1]
        descriptions.append((group, dataset, index_file, core, destination))
    for name in ('config.json', 'tokenizer_config.json', 'vocab.json', 'merges.txt'):
        check_input(Path(args.base_model) / name)
    common = ['--base_model', str(Path(args.base_model).resolve()), '--data_path', str(Path(args.data_path).resolve()),
              '--gpu', args.gpu, '--port', str(args.port), '--tasks', 'smb_explicit_decoder_4', '--export_user_targets',
              '--gradient_accumulation_steps', str(accumulation), '--precision', args.precision]
    for key in ('per_device_train_batch_size', 'per_device_eval_batch_size', 'test_batch_size', 'num_train_epochs', 'max_his_len', 'model_max_length',
                'learning_rate', 'weight_decay', 'warmup_ratio', 'patience', 'temperature', 'num_beams'):
        common.extend(['--' + key, str(getattr(args, key))])
    plans = []
    for group, dataset, index_file, core, destination in descriptions:
        specific = ['--suite', {'p0': 'handoff', 'seeds': 'components', 'tmall': 'hierarchy'}[group],
                    '--dataset', dataset, '--index_file', index_file, '--eval_behavior', core, '--output_dir', str(destination),
                    '--seeds', *(['43', '44'] if group == 'seeds' else ['42'])]
        if group == 'tmall':
            specific.extend(['--hierarchy_maps', str(Path(__file__).resolve().parents[1] / 'config/rebuttal/tmall-hierarchy-screening.json')])
        plans.append(prepare_plan(experiment_args(common + specific)))
    if args.prepare_cache:
        from SeqRec.datasets.loaders.session_behavior import load_SMB_datasets, load_SMB_test_dataset

        seen = set()
        for group, dataset, index_file, core, destination in descriptions:
            if (dataset, index_file) in seen:
                continue
            seen.add((dataset, index_file))
            settings = dict(dataset=dataset, data_path=args.data_path, max_his_len=args.max_his_len, index_file=index_file)
            load_SMB_datasets(**settings, tasks='smb_explicit_decoder_4', train_session=True)
            for task in ('smb_explicit_valid', 'smb_explicit'):
                load_SMB_test_dataset(**settings, test_task=task)
    print(f'Prepared {len(plans)} local plans; effective batch={args.effective_batch_size}, accumulation={accumulation}. No training launched.')
    for plan in plans:
        print(Path(plan['settings']['output_dir']) / 'plan.json')
    return plans


def main():
    prepare_remaining(parse_args())


if __name__ == '__main__':
    main()
