#!/usr/bin/env python
"""Run explicitly selected plan entries through training and ranking; dry-run by default."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from SeqRec.utils.rebuttal import dataset_fingerprints, file_sha256, write_json  # noqa: E402
from SeqRec.utils.rebuttal_queue import _saved_weights_present, rewrite_command  # noqa: E402


def run_experiments(plan_path, variants, seeds, phases=('train', 'validation', 'test'), gpu=None, port=None, execute=False, command_runner=None):
    plan_path = Path(plan_path).resolve()
    plan = json.loads(plan_path.read_text())
    repo = Path(__file__).resolve().parents[1]
    phases = list(phases)
    if not phases or len(set(phases)) != len(phases) or phases != [phase for phase in ('train', 'validation', 'test') if phase in phases]:
        raise ValueError('Use distinct phases in train/validation/test order.')
    if not variants or not seeds or len(set(variants)) != len(variants) or len(set(seeds)) != len(seeds):
        raise ValueError('Select distinct variants and seeds.')
    settings = plan['settings']
    gpu = gpu or settings['gpu']
    port = settings['port'] if port is None else port
    if not 1 <= port <= 65535:
        raise ValueError('Use a TCP port between 1 and 65535.')
    if 'train' in phases and len(gpu.split(',')) != len(settings['gpu'].split(',')):
        raise ValueError('Changing the training GPU count requires a new plan to preserve the effective batch.')
    if plan.get('python_executable') and Path(plan['python_executable']).resolve() != Path(sys.executable).resolve():
        raise ValueError('Plan belongs to a different Python environment; regenerate it on this machine.')
    selected = []
    for variant in variants:
        for seed in seeds:
            matches = [run for run in plan['runs'] if run['variant'] == variant and run['seed'] == seed]
            if len(matches) != 1:
                raise ValueError(f'Expected one plan entry for {variant}/seed_{seed}, found {len(matches)}.')
            selected.append(matches[0])
    outputs = []
    for run in selected:
        if file_sha256(run['config']) != run['config_sha256']:
            raise ValueError(f'Configuration changed: {run["config"]}')
        for phase in phases:
            target = Path(run['checkpoint'] if phase == 'train' else run['results'][phase])
            if str(target.resolve()) in outputs:
                raise ValueError('Selected entries share an output path.')
            outputs.append(str(target.resolve()))
            if target.exists():
                raise FileExistsError(f'Refusing to overwrite {target}; for completed weights select --phases validation test only.')
        if execute and 'train' not in phases and not _saved_weights_present(Path(run['checkpoint'])):
            raise FileNotFoundError(f'Final root weights are unavailable: {run["checkpoint"]}')
    command_runner = command_runner or subprocess.run
    receipts = []
    for run in selected:
        for phase in phases:
            if plan.get('data_fingerprints'):
                current = dataset_fingerprints(settings['data_path'], settings['dataset'], settings['index_file'])
                if current != plan['data_fingerprints']:
                    raise ValueError('Data fingerprints differ; regenerate a local plan after checking the inputs.')
            command = rewrite_command(run['commands'][phase], gpu, port)
            print(f'[{run["variant"]}/seed_{run["seed"]}] {phase}: {command}', flush=True)
            if not execute:
                continue
            target = Path(run['checkpoint'] if phase == 'train' else run['results'][phase])
            if target.exists():
                raise FileExistsError(f'Output appeared after preflight: {target}')
            if phase != 'train' and not _saved_weights_present(Path(run['checkpoint'])):
                raise FileNotFoundError(f'Training did not save final root weights: {run["checkpoint"]}')
            directory = Path(run['checkpoint']).parent
            directory.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
            record = {'schema_version': 1, 'plan': str(plan_path), 'plan_sha256': file_sha256(plan_path),
                      'variant': run['variant'], 'seed': run['seed'], 'phase': phase, 'command': command,
                      'started_at': datetime.now(timezone.utc).isoformat(), 'output': str(target)}
            write_json(directory / f'pipeline-{stamp}-{phase}.start.json', record)
            result = command_runner(command, shell=True, executable='/bin/bash', cwd=repo)
            valid_output = _saved_weights_present(target) if phase == 'train' else target.is_file()
            record.update(returncode=result.returncode, output_present=valid_output, finished_at=datetime.now(timezone.utc).isoformat())
            receipt = directory / f'pipeline-{stamp}-{phase}.finish.json'
            write_json(receipt, record)
            receipts.append(str(receipt))
            if result.returncode or not valid_output:
                raise RuntimeError(f'{run["variant"]}/seed_{run["seed"]} {phase} failed; subsequent phases/jobs were not started. See {receipt}')
            if phase != 'train':
                rows = json.loads(target.read_text())
                if not isinstance(rows, list) or not rows:
                    raise ValueError(f'Ranking output is empty or has an unexpected format: {target}')
    return receipts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--variants', nargs='+', required=True, help='Exact variant names, executed in the given order.')
    parser.add_argument('--seeds', nargs='+', type=int, default=[42])
    parser.add_argument('--phases', nargs='+', choices=['train', 'validation', 'test'], default=['train', 'validation', 'test'])
    parser.add_argument('--gpu', '--gpus', dest='gpu', help='Comma-separated GPU IDs, e.g. 2,5. Applied to training and ranking; training device count must match the plan.')
    parser.add_argument('--port', type=int)
    parser.add_argument('--execute', action='store_true', help='Actually run the selected jobs; otherwise only validate and print commands.')
    args = parser.parse_args()
    run_experiments(args.plan, args.variants, args.seeds, args.phases, args.gpu, args.port, args.execute)


if __name__ == '__main__':
    main()
