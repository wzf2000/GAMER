"""Prepare explicit, isolated TH experiment configurations without launching jobs."""

import copy
import hashlib
import json
import math
from pathlib import Path
import shlex
import subprocess
import sys


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as file:
        for block in iter(lambda: file.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def git_state():
    try:
        return {'code_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                'dirty_workspace': bool(subprocess.check_output(['git', 'status', '--porcelain'], text=True).strip())}
    except (OSError, subprocess.CalledProcessError):
        return {'code_commit': None, 'dirty_workspace': None}


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as file:
        json.dump(value, file, ensure_ascii=False, indent=2, allow_nan=False)
        file.write('\n')


def dataset_fingerprints(data_path, dataset, index_file):
    root = Path(data_path).resolve() / dataset
    fingerprints = {}
    for path in [root / f'{dataset}{index_file}', root / f'{dataset}.behavior_level.json']:
        fingerprints[path.name] = {'path': str(path.resolve()), 'sha256': file_sha256(path)}
    for suffix in ('inter', 'behavior', 'session', 'time'):
        path = root / f'{dataset}.SMB.{suffix}.json'
        stat = path.stat()
        fingerprints[path.name] = {'path': str(path.resolve()), 'bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns}
    return fingerprints


def count_parameters(values):
    import torch
    from transformers.models.qwen3_moe import Qwen3MoeConfig
    from SeqRec.models.generative.qwen3.temporal_hierarchical import Qwen3TemporalHierarchicalWithTemperature

    with torch.device('meta'):
        model = Qwen3TemporalHierarchicalWithTemperature(Qwen3MoeConfig(**values))
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def component_configs(full):
    variants = {'full': copy.deepcopy(full)}
    plain = copy.deepcopy(full)
    plain.update(temporal_hierarchical_attention_decoder=[], sparse_layers_decoder=[], behavior_injection_decoder=[], behavior_injection=False)
    first = count_parameters({**plain, 'intermediate_size': 1})
    slope = count_parameters({**plain, 'intermediate_size': 2}) - first
    if slope <= 0:
        raise ValueError('Cannot match parameters by changing the plain FFN width.')
    width = max(1, math.floor((count_parameters(full) - first + slope) / slope + 0.5))
    variants['plain_matched'] = {**plain, 'intermediate_size': width}
    variants['no_qkv'] = {**copy.deepcopy(full), 'th_use_behavior_qkv': False}
    variants['no_gate'] = {**copy.deepcopy(full), 'th_use_output_gate': False}
    variants['shared_ffn'] = {**copy.deepcopy(full), 'sparse_layers_decoder': []}
    variants['zero_prior'] = {**copy.deepcopy(full), 'th_relation_bias_soft_scale': 0.0}
    return variants


def effective_config(base_model, data_path, dataset, index_file, max_his_len, model_max_length):
    from SeqRec.models.generative.registry import load_config_and_tokenizer

    root = Path(data_path) / dataset
    levels = json.loads((root / f'{dataset}.behavior_level.json').read_text())
    indices = json.loads((root / f'{dataset}{index_file}').read_text())
    if not levels or not indices:
        raise ValueError('Behavior metadata and item indices must be nonempty.')
    if len({len(tokens) for tokens in indices.values()}) != 1:
        raise ValueError('Item index lengths must agree.')
    config, tokenizer = load_config_and_tokenizer('Qwen3TemporalHierarchical', str(base_model), model_max_length)
    tokens = {token for item in indices.values() for token in item}
    tokens.update(f'<behavior_{behavior}>' for behavior in levels)
    tokenizer.add_tokens(sorted(tokens))
    behavior_ids = [tokenizer.encode(f'<behavior_{behavior}>', add_special_tokens=False) for behavior in levels]
    if any(len(ids) != 1 for ids in behavior_ids):
        raise ValueError('Each behavior token must encode to exactly one token.')
    item = ''.join(next(iter(indices.values())))
    num_positions = len(tokenizer.encode(f'<behavior_{next(iter(levels))}>' + item, add_special_tokens=False))
    values = config.to_dict()
    values.update(vocab_size=len(tokenizer), num_behavior=len(levels), behavior_maps={ids[0]: index for index, ids in enumerate(behavior_ids)},
                  num_positions=num_positions, num_experts=2 if config.Moe_behavior_only else num_positions + 1,
                  n_positions=max_his_len + 1, model_max_length=model_max_length, use_user_token=False, use_behavior_token=True)
    if values.get('th_relation_bias_type') != 'table' or values.get('th_relation_bias_trainable', True):
        raise ValueError('This experiment planner requires a frozen relation table.')
    if values.get('th_relation_bias_init') != 'soft':
        raise ValueError('Use the submission FixedSoft configuration as the full reference.')
    if float(values.get('th_relation_bias_scale', 1.0)) != 1.0 or values.get('th_relation_bias_learnable_scale', False):
        raise ValueError('The fixed-prior suite requires runtime multiplier 1 without a learnable scale.')
    if values.get('th_level_auxiliary_loss_weight', 0.0) or values.get('th_relation_regularization_weight', 0.0):
        raise ValueError('Auxiliary objectives must be disabled for the FixedSoft component suite.')
    return values, levels


def render_command(arguments, gpu, port):
    devices = gpu.split(',')
    if not devices or any(not device.isdigit() for device in devices) or len(set(devices)) != len(devices):
        raise ValueError('GPU IDs must be a comma-separated list of distinct nonnegative integers.')
    launcher = [sys.executable] if len(devices) == 1 else [sys.executable, '-m', 'torch.distributed.run', f'--nproc_per_node={len(devices)}', f'--master_port={port}']
    return f'CUDA_VISIBLE_DEVICES={shlex.quote(gpu)} OMP_NUM_THREADS=1 ' + shlex.join(launcher + ['main.py'] + arguments)


def prepare_plan(args):
    root = Path(args.output_dir).resolve()
    if root.exists():
        raise FileExistsError(f'Refusing to overwrite an existing plan directory: {root}')
    if args.max_his_len <= 0 or args.model_max_length <= 0 or not args.seeds or len(set(args.seeds)) != len(args.seeds):
        raise ValueError('Use positive lengths and distinct seeds.')
    if not args.dataset or args.dataset in ('.', '..') or Path(args.dataset).name != args.dataset:
        raise ValueError('Use a simple dataset directory name.')
    if not math.isfinite(args.parameter_tolerance) or args.parameter_tolerance < 0:
        raise ValueError('Parameter tolerance must be finite and nonnegative.')
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0 or not 0 <= args.warmup_ratio <= 1:
        raise ValueError('Use a positive learning rate and a warmup ratio between zero and one.')
    for value in (args.num_train_epochs, args.per_device_train_batch_size, args.per_device_eval_batch_size,
                  args.gradient_accumulation_steps, args.test_batch_size, args.num_beams):
        if value <= 0:
            raise ValueError('Epoch, batch, accumulation, and beam settings must be positive.')
    base_model = Path(args.base_model).resolve()
    full, levels = effective_config(base_model, args.data_path, args.dataset, args.index_file, args.max_his_len, args.model_max_length)
    if args.suite == 'handoff':
        if len(args.seeds) != 1 or float(full.get('th_relation_bias_soft_scale', 0.0)) != 0.05:
            raise ValueError('The first-round handoff requires one seed and the Full prior lambda=0.05.')
        required = {'num_hidden_layers': 8, 'temporal_hierarchical_attention_decoder': [2, 3, 4, 5],
                    'sparse_layers_decoder': list(range(8)), 'behavior_injection_decoder': list(range(6)), 'mlp_type': 'Qwen3'}
        if args.tasks != 'smb_explicit_decoder_4' or any(full.get(key) != value for key, value in required.items()):
            raise ValueError('Handoff requires the complete submission FixedSoft architecture and smb_explicit_decoder_4 task.')
        if not full.get('th_use_behavior_qkv', True) or not full.get('th_use_output_gate', True):
            raise ValueError('Handoff Full must retain Q/K/V conditioning and the output gate.')
        components = component_configs(full)
        variants = {key: components[key] for key in ('full', 'no_qkv', 'no_gate', 'shared_ffn', 'zero_prior')}
        variants.update({f'lambda_{value:g}': {**copy.deepcopy(full), 'th_relation_bias_soft_scale': value} for value in (0.01, 0.1)})
    elif args.suite in ('components', 'seeds'):
        variants = component_configs(full)
        if args.suite == 'seeds':
            variants = {key: variants[key] for key in ('full', 'plain_matched')}
        target = count_parameters(full)
        if abs(count_parameters(variants['plain_matched']) - target) / target > args.parameter_tolerance:
            raise ValueError('The closest plain decoder exceeds the parameter matching tolerance.')
    elif args.suite == 'lambda':
        if not args.lambdas or any(not math.isfinite(value) or value < 0 for value in args.lambdas) or len(set(args.lambdas)) != len(args.lambdas):
            raise ValueError('Prior scales must be distinct, finite, and nonnegative.')
        variants = {f'lambda_{value:g}'.replace('-', 'm').replace('+', 'p'): {**copy.deepcopy(full), 'th_relation_bias_soft_scale': value} for value in args.lambdas}
    else:
        if not args.hierarchy_maps:
            raise ValueError('--hierarchy_maps is required for hierarchy interventions.')
        maps = json.loads(Path(args.hierarchy_maps).read_text())
        variants = {'full': copy.deepcopy(full)}
        for name, mapping in maps.items():
            if name == 'full' or not name.replace('_', '').isalnum() or set(mapping) != set(levels):
                raise ValueError('Each named map must specify exactly the dataset behavior identities; full is reserved.')
            ranks = [mapping[behavior] for behavior in levels]
            if any(isinstance(rank, bool) or not isinstance(rank, (int, float)) or not math.isfinite(rank) or rank < 0 for rank in ranks):
                raise ValueError('Prior ranks must be finite nonnegative numbers.')
            variants[name] = {**copy.deepcopy(full), 'th_prior_behavior_ranks': ranks}

    for name in variants:
        if not name.replace('_', '').replace('.', '').isalnum():
            raise ValueError(f'Invalid variant name: {name}')
    render_command([], args.gpu, args.port)
    fingerprints = dataset_fingerprints(args.data_path, args.dataset, args.index_file)
    counts = {name: count_parameters(config) for name, config in variants.items()}
    revision = git_state()

    root.mkdir(parents=True)
    runs = []
    for name, config in variants.items():
        config_dir = root / 'configs' / name
        config_dir.mkdir(parents=True)
        write_json(config_dir / 'config.json', config)
        for resource in base_model.iterdir():
            if resource.name != 'config.json' and resource.is_file() and resource.suffix in ('.json', '.txt', '.model'):
                (config_dir / resource.name).symlink_to(resource.resolve())
        seeds = args.seeds[:1] if args.key_seeds_only and name not in ('full', 'plain_matched') else args.seeds
        for seed in seeds:
            run_dir = root / name / f'seed_{seed}'
            checkpoint = run_dir / 'checkpoint'
            shared = ['--backbone', 'Qwen3TemporalHierarchical', '--dataset', args.dataset, '--data_path', str(Path(args.data_path).resolve()),
                      '--index_file', args.index_file, '--max_his_len', str(args.max_his_len), '--seed', str(seed)]
            train = ['train_SMB_decoder', *shared, '--base_model', str(config_dir), '--output_dir', str(checkpoint), '--tasks', args.tasks,
                     '--model_max_length', str(args.model_max_length), '--num_train_epochs', str(args.num_train_epochs),
                     '--per_device_train_batch_size', str(args.per_device_train_batch_size), '--per_device_eval_batch_size', str(args.per_device_eval_batch_size),
                     '--gradient_accumulation_steps', str(args.gradient_accumulation_steps), '--learning_rate', str(args.learning_rate),
                     '--weight_decay', str(getattr(args, 'weight_decay', 0.01)),
                     '--warmup_ratio', str(args.warmup_ratio), '--patience', str(args.patience), '--temperature', str(args.temperature),
                     '--wandb_run_name', f'{root.name}/{name}/seed_{seed}', '--record_run_manifest']
            if args.precision != 'fp32':
                train.append('--' + args.precision)
            if args.validation_behavior:
                train.extend(['--validation_behavior', args.validation_behavior])
            if args.debug:
                train.append('--debug')
            commands = {'train': render_command(train, args.gpu, args.port)}
            files = {}
            for split, task in (('validation', 'smb_explicit_valid'), ('test', 'smb_explicit')):
                result = run_dir / f'{split}-ranking.json'
                evaluation = ['test_SMB_decoder', *shared, '--ckpt_path', str(checkpoint), '--results_file', str(result),
                              '--test_task', task, '--test_batch_size', str(args.test_batch_size), '--num_beams', str(args.num_beams)]
                if args.eval_behavior:
                    evaluation.extend(['--behaviors', args.eval_behavior])
                if args.suite == 'handoff' or getattr(args, 'export_user_targets', False):
                    evaluation.append('--export_user_targets')
                commands[split] = render_command(evaluation, args.gpu, args.port)
                files[split] = str(result)
            runs.append({'variant': name, 'seed': seed, 'config': str(config_dir / 'config.json'), 'config_sha256': file_sha256(config_dir / 'config.json'),
                         'trainable_parameters': counts[name], 'checkpoint': str(checkpoint), 'results': files, 'commands': commands})

    manifest = {'schema_version': 1, 'suite': args.suite, **revision, 'settings': vars(args), 'python_executable': sys.executable,
                'requires_run_manifest': True,
                'behavior_order': list(levels), 'data_fingerprints': fingerprints, 'runs': runs,
                'notes': ['Prepared commands only; no training or evaluation was launched.',
                          'Settings are a new experiment plan, not recovered submission settings.',
                          'Training includes periodic validation; standalone ranking evaluation is separate.']}
    if args.suite == 'handoff':
        from SeqRec.utils.rebuttal_queue import reuse_handoff_runs
        for run in runs:
            run['stage'] = 'P0-A' if run['variant'] in ('full', 'no_qkv', 'no_gate', 'shared_ffn') else 'P0-B'
            run['prior_lambda'] = variants[run['variant']]['th_relation_bias_soft_scale']
            run['data_seed'] = 42
            run['resolved_config'] = variants[run['variant']]
            run['config_diff_from_full'] = {key: {'full': full.get(key), 'variant': variants[run['variant']].get(key)} for key in full.keys() | variants[run['variant']].keys()
                                            if full.get(key) != variants[run['variant']].get(key)}
        reuse_handoff_runs(manifest, getattr(args, 'reuse_plans', []))
        manifest['stage_order'] = ['P0-A', 'P0-B']
        manifest['aliases'] = {'lambda_0.05': 'full', 'lambda_0': 'zero_prior'}
    write_json(root / 'plan.json', manifest)
    if args.suite == 'handoff':
        from SeqRec.utils.rebuttal_queue import write_stage_scripts
        write_stage_scripts(root, manifest)
        return manifest
    for phase in ('train', 'validation', 'test'):
        with (root / f'run_{phase}.sh').open('x') as file:
            file.write('#!/bin/bash\nset -euo pipefail\n')
            file.write('cd ' + shlex.quote(str(Path(__file__).resolve().parents[2])) + '\n')
            for run in runs:
                # A fresh checkpoint directory is required unless the caller explicitly resumes manually.
                if phase == 'train':
                    file.write(f'test ! -e {shlex.quote(run["checkpoint"])}\n')
                else:
                    file.write(f'test ! -e {shlex.quote(run["results"][phase])}\n')
                file.write(run['commands'][phase] + '\n')
    if args.suite in ('components', 'seeds'):
        for group, names in [('key', {'full', 'plain_matched'}), ('remaining', set(variants) - {'full', 'plain_matched'})]:
            if not names:
                continue
            with (root / f'run_train_{group}.sh').open('x') as file:
                file.write('#!/bin/bash\nset -euo pipefail\ncd ' + shlex.quote(str(Path(__file__).resolve().parents[2])) + '\n')
                for run in runs:
                    if run['variant'] in names:
                        file.write(f'test ! -e {shlex.quote(run["checkpoint"])}\n' + run['commands']['train'] + '\n')
    return manifest


def record_training_manifest(parsed_args, model, training_args, world_size, datasets=()):
    import torch
    from datetime import datetime, timezone

    model_args, data_args, script_args = parsed_args.model, parsed_args.dataset, parsed_args.training
    root = Path(model_args.output_dir)
    source_config = Path(model_args.base_model) / 'config.json'
    index = Path(data_args.data_path) / data_args.dataset / f'{data_args.dataset}{data_args.index_file}'
    caches = {}
    for name, dataset in datasets:
        cache_name = getattr(dataset, 'cached_file_name', None)
        caches[name] = {'samples': len(dataset)}
        if cache_name is not None:
            cache = Path(cache_name)
            stat = cache.stat()
            caches[name].update(path=str(cache.resolve()), bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)
    record = {'schema_version': 1, **git_state(), 'arguments': parsed_args.as_log_dict(),
              'source_config_sha256': file_sha256(source_config), 'index_sha256': file_sha256(index),
              'data_fingerprints': dataset_fingerprints(data_args.data_path, data_args.dataset, data_args.index_file), 'cache_fingerprints': caches,
              'effective_model_config': model.config.to_dict(), 'effective_training_args': training_args.to_dict(),
              'attention_backend': model.config._attn_implementation, 'parameter_dtype': str(next(model.parameters()).dtype),
              'trainable_parameters': sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
              'world_size': world_size, 'gpu_names': [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())] if torch.cuda.is_available() else [],
              'python_executable': sys.executable, 'note': 'Recorded before training; this file alone does not establish successful completion.'}
    name = 'run_manifest.start.json'
    if (root / name).exists():
        name = 'run_manifest.resume.' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ') + '.json'
    write_json(root / name, record)
