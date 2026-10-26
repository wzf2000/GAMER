"""Paired offline buckets using stored multi-target metrics and explicit UID alignment."""

import json
import math
from pathlib import Path

import numpy as np

from SeqRec.utils.rebuttal import file_sha256


def history_counts(behavior_file, session_file, behavior, max_his_len, split='test'):
    if split not in ('validation', 'test') or max_his_len == 0 or max_his_len < -1:
        raise ValueError('Use an explicit split and a positive window, or -1 for full history.')
    behaviors = json.loads(Path(behavior_file).read_text())
    sessions = json.loads(Path(session_file).read_text())
    if set(behaviors) != set(sessions):
        raise ValueError('Behavior/session user populations differ.')
    counts = {}
    for uid, events in behaviors.items():
        user_sessions = sessions[uid]
        if len(user_sessions) != len(events) or not user_sessions:
            raise ValueError(f'{uid}: behavior/session lengths differ or are empty.')
        if user_sessions != sorted(user_sessions):
            raise ValueError(f'{uid}: session IDs must increase chronologically.')
        ordered = sorted(set(user_sessions))
        if split == 'validation' and len(ordered) < 2:
            continue
        boundary = user_sessions.index(ordered[-1 if split == 'test' else -2])
        end = len(events) if split == 'test' else user_sessions.index(ordered[-1])
        if behavior not in events[boundary:end]:
            continue
        history = events[:boundary]
        visible = history[-max_his_len:] if max_his_len > 0 else history
        counts[uid] = {'visible_core': visible.count(behavior), 'history_core': history.count(behavior),
                       'visible_events': len(visible), 'history_events': len(history), 'target_events': events[boundary:end].count(behavior)}
    if not counts:
        raise ValueError('No target-behavior users in the requested split.')
    return counts


def aligned_metrics(metric_file, expected_uids, uid_file=None, legacy_lexicographic_order=False, split=None, max_his_len=None):
    path = Path(metric_file)
    data = json.loads(path.read_text())
    uid_path = Path(uid_file) if uid_file else path.with_suffix('.uids.json')
    if uid_path.exists():
        record = json.loads(uid_path.read_text())
        if not isinstance(record, dict) or record.get('metric_sha256') != file_sha256(path):
            raise ValueError(f'{uid_path}: UID sidecar must bind to this metric file by SHA256.')
        if split is not None and record.get('split') != ('valid_test' if split == 'validation' else 'test'):
            raise ValueError(f'{uid_path}: UID sidecar split is missing or differs from the requested split.')
        if max_his_len is not None and record.get('max_his_len') != max_his_len:
            raise ValueError(f'{uid_path}: UID sidecar history window is missing or differs from the requested window.')
        uids = record['uids']
        alignment = 'explicit_uid_sidecar'
    elif legacy_lexicographic_order and uid_file is None:
        uids = sorted(expected_uids)
        alignment = 'legacy_order_assumption_from_current_writer_and_reconstructed_population'
    else:
        raise ValueError(f'{path}: no matching UID sidecar; legacy ordering requires explicit opt-in.')
    if any(not isinstance(uid, str) for uid in uids) or len(uids) != len(set(uids)) or set(uids) != set(expected_uids):
        raise ValueError(f'{path}: UID population differs from the requested split.')
    if not data or any(not isinstance(values, list) or len(values) != len(uids) for values in data.values()):
        raise ValueError(f'{path}: metric lengths disagree with the UID population.')
    if any(not isinstance(value, (int, float)) or not math.isfinite(value) for values in data.values() for value in values):
        raise ValueError(f'{path}: metrics must be finite numeric values.')
    positions = {uid: index for index, uid in enumerate(uids)}
    return {metric: {uid: values[positions[uid]] for uid in expected_uids} for metric, values in data.items()}, alignment


def paired_interval(differences, samples, rng):
    if samples <= 0:
        return None
    values = np.asarray(differences, dtype=float)
    estimates = []
    for start in range(0, samples, 64):
        indices = rng.integers(0, len(values), size=(min(64, samples - start), len(values)))
        estimates.extend(values[indices].mean(axis=1).tolist())
    return np.quantile(estimates, [0.025, 0.975]).tolist()


def bucket_metrics(counts, ours, baseline, thresholds=(1, 3, 6), count_scope='visible', bootstrap_samples=1000, seed=42):
    if not thresholds or any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in thresholds) or sorted(set(thresholds)) != list(thresholds):
        raise ValueError('Bucket thresholds must be distinct increasing positive integers.')
    if count_scope not in ('visible', 'full') or bootstrap_samples < 0:
        raise ValueError('Invalid count scope or bootstrap count.')
    if set(ours) != set(baseline) or not ours:
        raise ValueError('Both models must provide identical nonempty metric sets.')
    uids = sorted(counts)
    if any(set(values) != set(uids) for values in list(ours.values()) + list(baseline.values())):
        raise ValueError('All metric UID populations must match exactly.')
    groups = [[] for _ in range(len(thresholds) + 1)]
    key = 'visible_core' if count_scope == 'visible' else 'history_core'
    for uid in uids:
        index = sum(counts[uid][key] >= threshold for threshold in thresholds)
        groups[index].append(uid)
    rng = np.random.default_rng(seed)
    rows = []
    lower = 0
    for index, group in enumerate(groups):
        upper = thresholds[index] - 1 if index < len(thresholds) else None
        label = f'{lower}+' if upper is None else str(lower) if lower == upper else f'{lower}-{upper}'
        metrics = {}
        for metric in sorted(ours):
            differences = [ours[metric][uid] - baseline[metric][uid] for uid in group]
            metrics[metric] = {'ours': float(np.mean([ours[metric][uid] for uid in group])) if group else None,
                               'baseline': float(np.mean([baseline[metric][uid] for uid in group])) if group else None,
                               'delta': float(np.mean(differences)) if group else None,
                               'paired_95_interval': paired_interval(differences, bootstrap_samples, rng) if group else None}
        rows.append({'bucket': label, 'n_users': len(group), 'metrics': metrics})
        if upper is not None:
            lower = upper + 1
    return rows


def offline_buckets(args):
    counts = history_counts(args.behavior_file, args.session_file, args.behavior, args.max_his_len, args.split)
    ours, ours_alignment = aligned_metrics(args.ours, counts, args.ours_uid_file, args.legacy_lexicographic_order, args.split, args.max_his_len)
    baseline_window = args.baseline_max_his_len if args.baseline_max_his_len is not None else args.max_his_len
    baseline, baseline_alignment = aligned_metrics(args.baseline, counts, args.baseline_uid_file, args.legacy_lexicographic_order, args.split, baseline_window)
    if args.metrics:
        ours = {metric: ours[metric] for metric in args.metrics}
        baseline = {metric: baseline[metric] for metric in args.metrics}
    return {'schema_version': 1, 'split': args.split, 'behavior': args.behavior, 'max_his_len': args.max_his_len,
            'baseline_max_his_len': baseline_window, 'count_scope': args.count_scope, 'thresholds': args.thresholds, 'n_users': len(counts),
            'alignment': {'ours': ours_alignment, 'baseline': baseline_alignment},
            'bootstrap_samples': args.bootstrap_samples, 'bootstrap_seed': args.seed,
            'sources': {name: {'path': str(Path(path).resolve()), 'sha256': file_sha256(path)} for name, path in
                        [('ours', args.ours), ('baseline', args.baseline), ('behavior_file', args.behavior_file), ('session_file', args.session_file)]},
            'buckets': bucket_metrics(counts, ours, baseline, args.thresholds, args.count_scope, args.bootstrap_samples, args.seed),
            'note': 'Uses saved multi-target metrics directly. Legacy alignment is an explicit historical-order assumption, not proof of original UID provenance.'}
