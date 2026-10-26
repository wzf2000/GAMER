"""Aggregate explicit experiment plans without rerunning model inference."""

import json
import csv
import math
from pathlib import Path
import statistics

from SeqRec.utils.rebuttal import file_sha256


def metric_row(path, behavior):
    rows = json.loads(Path(path).read_text())
    label = 'Merged Behavior' if behavior == 'merged' else f'Behavior {behavior}'
    matches = [row for row in rows if row.get('eval_type') == label]
    if len(matches) != 1:
        raise ValueError(f'{path}: expected exactly one {label} result.')
    metrics = {name: value for name, value in matches[0].items() if '@' in name and isinstance(value, (int, float))}
    if not metrics or any(not math.isfinite(value) for value in metrics.values()):
        raise ValueError(f'{path}: missing or nonfinite ranking metrics.')
    return metrics


def mean_std(values):
    return {'mean': statistics.mean(values), 'std': statistics.stdev(values) if len(values) > 1 else None, 'n': len(values)}


def summarize_plan(plan_path, split='test', behavior='cvr', reference='full', allow_partial=False):
    if split not in ('validation', 'test'):
        raise ValueError('Choose validation or test explicitly.')
    plan = json.loads(Path(plan_path).read_text())
    groups, missing, run_results = {}, [], []
    cache_reference = None
    for run in plan['runs']:
        if file_sha256(run['config']) != run['config_sha256']:
            raise ValueError(f'Configuration changed after plan creation: {run["config"]}')
        result_path = Path(run['results'][split])
        record = {'variant': run['variant'], 'seed': run['seed'], 'result': str(result_path), 'status': 'missing', 'metrics': {}}
        run_results.append(record)
        if not result_path.exists():
            missing.append(str(result_path))
            continue
        if plan.get('requires_run_manifest'):
            execution_path = Path(run['checkpoint']) / 'run_manifest.start.json'
            execution = json.loads(execution_path.read_text())
            if execution.get('data_fingerprints') != plan.get('data_fingerprints'):
                raise ValueError('Executed data fingerprints differ from the prepared snapshot.')
            if cache_reference is None:
                cache_reference = execution.get('cache_fingerprints')
            elif execution.get('cache_fingerprints') != cache_reference:
                raise ValueError('Training/validation cache fingerprints differ between planned runs.')
            if execution['source_config_sha256'] != run['config_sha256'] or execution['arguments']['model']['seed'] != run['seed']:
                raise ValueError('Executed configuration or seed differs from the prepared plan.')
            if execution['trainable_parameters'] != run['trainable_parameters']:
                raise ValueError('Executed parameter count differs from the prepared plan.')
            for key in ('max_his_len', 'dataset', 'index_file', 'validation_behavior'):
                if execution['arguments']['dataset'][key] != plan['settings'][key]:
                    raise ValueError(f'Executed dataset argument differs from plan: {key}')
            evaluation = json.loads(result_path.with_suffix('.run.json').read_text())
            expected_split = 'test' if split == 'test' else 'valid_test'
            if evaluation['split'] != expected_split or evaluation['result_sha256'] != file_sha256(result_path):
                raise ValueError('Evaluation split or result fingerprint differs from the requested summary.')
            if evaluation['training_manifest_sha256'] != file_sha256(execution_path) or evaluation['index_sha256'] != execution['index_sha256']:
                raise ValueError('Evaluation is not bound to the training run/candidate index.')
            if evaluation['max_his_len'] != plan['settings']['max_his_len']:
                raise ValueError('Evaluation history length differs from the prepared plan.')
        group = groups.setdefault(run['variant'], {})
        if run['seed'] in group:
            raise ValueError('Duplicate variant/seed in the plan.')
        group[run['seed']] = metric_row(result_path, behavior)
        record.update(status='complete', metrics=group[run['seed']])
    if missing and not allow_partial:
        raise FileNotFoundError(f'{len(missing)} runs are missing; use --allow_partial only for an explicitly partial report.')
    if not groups and not allow_partial:
        raise ValueError('No completed results to summarize.')
    metric_names = set(next(iter(next(iter(groups.values())).values()))) if groups else set()
    if any(set(row) != metric_names for seeds in groups.values() for row in seeds.values()):
        raise ValueError('Metric sets differ between runs.')
    summary = {name: {'seeds': sorted(seeds), 'metrics': {metric: mean_std([row[metric] for row in seeds.values()]) for metric in sorted(metric_names)}}
               for name, seeds in groups.items()}
    paired = {}
    if reference in groups:
        for name, seeds in groups.items():
            if name == reference:
                continue
            common = sorted(set(seeds) & set(groups[reference]))
            if common:
                paired[name] = {'reference': reference, 'seeds': common,
                                'metrics': {metric: mean_std([seeds[seed][metric] - groups[reference][seed][metric] for seed in common]) for metric in sorted(metric_names)}}
    return {'plan': str(Path(plan_path).resolve()), 'split': split, 'behavior': behavior, 'partial': bool(missing),
            'missing': missing, 'variants': summary, 'paired_differences': paired, 'reference': reference, 'run_results': run_results,
            'note': 'Standard deviation uses ddof=1. Paired differences are variant minus reference for matching seeds.'}


def write_summary_csv(path, summary):
    """Export actual single-run values and matching-seed differences; leave missing values blank."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    reference = {run['seed']: run['metrics'] for run in summary['run_results'] if run['variant'] == summary['reference'] and run['status'] == 'complete'}
    fields = ['variant', 'seed', 'split', 'behavior', 'status', 'metric', 'value', 'reference', 'reference_value', 'difference_from_reference', 'relative_difference_percent', 'result']
    with path.open('x', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for run in summary['run_results']:
            base = {key: run[key] for key in ('variant', 'seed', 'status', 'result')}
            base.update(split=summary['split'], behavior=summary['behavior'], reference=summary['reference'])
            if run['status'] != 'complete':
                writer.writerow(base)
                continue
            for metric, value in sorted(run['metrics'].items()):
                ref = reference.get(run['seed'], {}).get(metric)
                writer.writerow({**base, 'metric': metric, 'value': value, 'reference_value': ref,
                                 'difference_from_reference': value - ref if ref is not None else None,
                                 'relative_difference_percent': (value - ref) / ref * 100 if ref else None})
