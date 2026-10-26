"""Handoff-plan reuse and read-only queue status; no job execution."""

import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex

from SeqRec.utils.rebuttal import dataset_fingerprints, file_sha256, render_command


RECIPE_KEYS = ('dataset', 'index_file', 'tasks', 'max_his_len', 'model_max_length', 'num_train_epochs',
               'per_device_train_batch_size', 'per_device_eval_batch_size', 'gradient_accumulation_steps',
               'learning_rate', 'warmup_ratio', 'patience', 'temperature', 'precision', 'validation_behavior')


def rewrite_command(command, gpu, port, replacements=None):
    tokens = shlex.split(command)
    task = next(i for i, token in enumerate(tokens) if token in ('train_SMB_decoder', 'test_SMB_decoder'))
    arguments = tokens[task:]
    for key, value in (replacements or {}).items():
        index = arguments.index(key)
        arguments[index + 1] = str(value)
    return render_command(arguments, gpu, port)


def reuse_handoff_runs(plan, source_paths):
    """Reuse only explicitly supplied plans with matching configuration/data/recipe."""
    candidates = []
    for path in source_paths:
        source = json.loads(Path(path).read_text())
        if source.get('code_commit') != plan.get('code_commit'):
            raise ValueError(f'Reuse code revision differs: {path}')
        if source['data_fingerprints'] != plan['data_fingerprints']:
            raise ValueError(f'Reuse data fingerprints differ: {path}')
        if any(source['settings'].get(key) != plan['settings'].get(key) for key in RECIPE_KEYS):
            raise ValueError(f'Reuse training recipe differs: {path}')
        if source['settings'].get('weight_decay', 0.01) != plan['settings'].get('weight_decay', 0.01):
            raise ValueError(f'Reuse weight decay differs: {path}')
        if len(source['settings']['gpu'].split(',')) != len(plan['settings']['gpu'].split(',')):
            raise ValueError(f'Reuse world size differs: {path}')
        for run in source['runs']:
            if file_sha256(run['config']) != run['config_sha256']:
                raise ValueError(f'Reuse configuration changed: {run["config"]}')
            candidates.append((str(Path(path).resolve()), run, json.loads(Path(run['config']).read_text())))
    cache_reference = None
    for run in plan['runs']:
        expected = json.loads(Path(run['config']).read_text())
        matches = [(path, other) for path, other, config in candidates if other['seed'] == run['seed'] and config == expected]
        # Prefer an already-started run; ambiguity between independent started runs needs a narrower source plan.
        started = [(path, other) for path, other in matches if Path(other['checkpoint']).exists()]
        chosen = started or matches
        checkpoints = {other['checkpoint'] for _, other in chosen}
        if len(checkpoints) > 1:
            raise ValueError(f'Ambiguous reusable checkpoints for {run["variant"]}.')
        if not chosen:
            continue
        source_path, other = chosen[0]
        start_path = Path(other['checkpoint']) / 'run_manifest.start.json'
        if start_path.exists():
            start = json.loads(start_path.read_text())
            if start['source_config_sha256'] != other['config_sha256'] or start['arguments']['model']['seed'] != run['seed']:
                raise ValueError('Reuse executed configuration/seed differs.')
            if start.get('code_commit') != plan.get('code_commit'):
                raise ValueError('Reuse executed code revision differs.')
            if start['data_fingerprints'] != plan['data_fingerprints'] or start['trainable_parameters'] != run['trainable_parameters']:
                raise ValueError('Reuse executed data/parameter count differs.')
            for key in ('dataset', 'index_file', 'tasks', 'max_his_len', 'validation_behavior'):
                if start['arguments']['dataset'][key] != plan['settings'][key]:
                    raise ValueError(f'Reuse executed dataset setting differs: {key}')
            for key in ('model_max_length', 'patience', 'temperature'):
                if start['arguments']['training'][key] != plan['settings'][key]:
                    raise ValueError(f'Reuse executed script setting differs: {key}')
            hf = start['effective_training_args']
            for key in ('num_train_epochs', 'per_device_train_batch_size', 'per_device_eval_batch_size', 'gradient_accumulation_steps', 'learning_rate', 'warmup_ratio'):
                if hf[key] != plan['settings'][key]:
                    raise ValueError(f'Reuse effective training argument differs: {key}')
            if hf['weight_decay'] != plan['settings'].get('weight_decay', 0.01) or start['world_size'] != len(plan['settings']['gpu'].split(',')):
                raise ValueError('Reuse effective weight decay/world size differs.')
            if bool(hf['fp16']) != (plan['settings']['precision'] == 'fp16') or bool(hf['bf16']) != (plan['settings']['precision'] == 'bf16'):
                raise ValueError('Reuse precision differs.')
            if cache_reference is None:
                cache_reference = start['cache_fingerprints']
            elif start['cache_fingerprints'] != cache_reference:
                raise ValueError('Reuse executed train/validation caches differ.')
        run['prepared_config'] = run['config']
        for key in ('config', 'config_sha256', 'checkpoint', 'results'):
            run[key] = copy.deepcopy(other[key])
        run['reuse'] = {'source_plan': source_path, 'source_variant': other['variant'], 'start_manifest': str(start_path) if start_path.exists() else None,
                        'basis': 'Matching resolved configuration, seed, data snapshots and training recipe; effective start arguments checked when present.'}
        gpu, port = plan['settings']['gpu'], plan['settings']['port']
        run['commands']['train'] = rewrite_command(run['commands']['train'], gpu, port, {'--base_model': Path(run['config']).parent, '--output_dir': run['checkpoint']})
        for phase in ('validation', 'test'):
            run['commands'][phase] = rewrite_command(run['commands'][phase], gpu, port, {'--ckpt_path': run['checkpoint'], '--results_file': run['results'][phase]})


def training_processes():
    """Read process arguments only; an inaccessible PID namespace yields no liveness claim."""
    found = {}
    for path in Path('/proc').glob('[0-9]*/cmdline'):
        try:
            tokens = path.read_bytes().decode().rstrip('\0').split('\0')
            if 'train_SMB_decoder' not in tokens or '--output_dir' not in tokens:
                continue
            value = Path(tokens[tokens.index('--output_dir') + 1])
            if not value.is_absolute():
                value = (path.parent / 'cwd').resolve() / value
            found.setdefault(str(value.resolve()), []).append(int(path.parent.name))
        except (OSError, UnicodeError, ValueError):
            continue
    return found


def _result_status(path, split, run):
    path = Path(path)
    if not path.exists():
        return 'missing'
    receipt = path.with_suffix('.run.json')
    if not receipt.exists():
        return 'unverified_result'
    record = json.loads(receipt.read_text())
    start = Path(run['checkpoint']) / 'run_manifest.start.json'
    if record.get('split') != ('test' if split == 'test' else 'valid_test') or record.get('result_sha256') != file_sha256(path):
        return 'invalid_result'
    if not start.exists() or record.get('training_manifest_sha256') != file_sha256(start):
        return 'unverified_result'
    training = json.loads(start.read_text())
    if record.get('index_sha256') != training['index_sha256']:
        return 'invalid_result'
    return 'complete'


def refresh_status(plan, processes=None):
    processes = training_processes() if processes is None else processes
    data_changed = False
    if plan.get('data_fingerprints'):
        settings = plan['settings']
        data_changed = dataset_fingerprints(settings['data_path'], settings['dataset'], settings['index_file']) != plan['data_fingerprints']
    rows = []
    for run in plan['runs']:
        checkpoint = Path(run['checkpoint'])
        row = {'variant': run['variant'], 'seed': run['seed'], 'stage': run.get('stage'), 'checkpoint': str(checkpoint),
               'training': 'not_started', 'pids': processes.get(str(checkpoint.resolve()), []), 'progress': None}
        try:
            if data_changed:
                row['training'] = 'incompatible_data'
            elif file_sha256(run['config']) != run['config_sha256']:
                row['training'] = 'incompatible_config'
            elif row['pids']:
                row['training'] = 'running'
            elif checkpoint.exists():
                row['training'] = 'started_liveness_unknown'
                state = checkpoint / 'trainer_state.json'
                weights = _saved_weights_present(checkpoint)
                if state.exists() and weights:
                    content = json.loads(state.read_text())
                    if any('train_runtime' in entry for entry in content.get('log_history', [])):
                        row['training'] = 'completed'
                        row['completion_evidence'] = 'Final root weights and root trainer_state with train_runtime; no observed training PID.'
                elif weights:
                    row['training'] = 'saved_weights_unconfirmed'
            states = [checkpoint / 'trainer_state.json', *sorted(checkpoint.glob('checkpoint-*/trainer_state.json'), key=lambda path: int(path.parent.name.split('-')[-1]), reverse=True)]
            contents = []
            for path in states:
                if path.exists():
                    contents.append(json.loads(path.read_text()))
            if contents:
                state = max(contents, key=lambda content: content.get('global_step', 0))
                row['progress'] = {key: state.get(key) for key in ('epoch', 'global_step', 'best_metric', 'best_model_checkpoint')}
                if row['training'] == 'completed' and state['global_step'] > json.loads((checkpoint / 'trainer_state.json').read_text())['global_step']:
                    row['training'] = 'started_liveness_unknown'
                    row.pop('completion_evidence', None)
            for split in ('validation', 'test'):
                row[split] = _result_status(run['results'][split], split, run)
        except (OSError, ValueError, KeyError) as error:
            row.update(error=str(error), training='unverified_artifacts')
        rows.append(row)
    return {'schema_version': 1, 'captured_at': datetime.now(timezone.utc).isoformat(), 'runs': rows,
            'note': 'Read-only snapshot. No observed PID does not establish that a started job stopped; GPU availability is not inferred.'}


def _saved_weights_present(checkpoint):
    for name in ('model.safetensors', 'pytorch_model.bin'):
        path = checkpoint / name
        if path.is_file() and path.stat().st_size > 0:
            return True
    for name in ('model.safetensors.index.json', 'pytorch_model.bin.index.json'):
        path = checkpoint / name
        if path.is_file():
            shards = set(json.loads(path.read_text()).get('weight_map', {}).values())
            if shards and all((checkpoint / shard).is_file() and (checkpoint / shard).stat().st_size > 0 for shard in shards):
                return True
    return False


def attach_parameter_audits(status, plan, paths):
    audits = {}
    for path in paths:
        report = json.loads(Path(path).read_text())
        audits[report['config_sha256']] = (str(Path(path).resolve()), report)
    for row, run in zip(status['runs'], plan['runs']):
        if run['config_sha256'] in audits:
            path, report = audits[run['config_sha256']]
            if report['trainable'] != run['trainable_parameters']:
                raise ValueError('Parameter audit differs from planned trainable count.')
            row['parameter_audit'] = {'path': path, 'sha256': file_sha256(path),
                                      **{key: report[key] for key in ('total', 'trainable', 'active_participating', 'buffers', 'nonzero_gradient_tensors', 'probe', 'scope', 'seed', 'checkpoint')}}


def next_pending(plan, status, phase='train'):
    rows = {(row['variant'], row['seed']): row for row in status['runs']}
    for stage in ('P0-A', 'P0-B'):
        runs = [run for run in plan['runs'] if run.get('stage') == stage]
        if not runs:
            continue
        if any(rows[run['variant'], run['seed']]['training'] in ('incompatible_config', 'incompatible_data', 'unverified_artifacts') for run in runs):
            return None
        for run in runs:
            row = rows[run['variant'], run['seed']]
            if phase == 'train' and row['training'] == 'not_started':
                return run
            if phase != 'train' and row['training'] == 'completed' and row[phase] == 'missing':
                return run
        # Do not advance to P0-B while P0-A training (or requested ranking) is incomplete.
        if any(rows[run['variant'], run['seed']]['training'] != 'completed'
               or (phase != 'train' and rows[run['variant'], run['seed']][phase] != 'complete') for run in runs):
            return None
    return None


def write_stage_scripts(root, plan):
    """Stage launchers obtain a fresh snapshot before printing/executing pending work."""
    root = Path(root)
    for stage in ('P0-A', 'P0-B'):
        for phase in ('train', 'validation', 'test'):
            path = root / f'{stage}-{phase}.sh'
            with path.open('x') as file:
                file.write('#!/bin/bash\nset -euo pipefail\ncd ' + shlex.quote(str(Path(__file__).resolve().parents[2])) + '\n')
                # The manager prints commands by default; the user opts into executing the produced command.
                command = [plan['python_executable'], 'scripts/manage_rebuttal_queue.py', '--plan', str(root / 'plan.json'), '--phase', phase, '--stage', stage]
                file.write(shlex.join(command) + '\n')
