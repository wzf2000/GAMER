"""Optional SMB evaluation provenance, without rewriting dataset caches."""

import copy
import json
from pathlib import Path

from SeqRec.utils.rebuttal import file_sha256, write_json


def with_verified_eval_uids(dataset):
    """Attach missing legacy UIDs only after checking the complete cached target order."""
    if dataset.mode not in ('test', 'valid_test'):
        raise ValueError('Target export requires SMB test or validation ranking.')
    users = list(dataset.inters)
    if len(users) != len(dataset.inter_data):
        raise ValueError('Cannot align legacy cache: user/sample counts differ.')
    copied = copy.copy(dataset)
    copied.inter_data = []
    seen = set()
    for position, sample in enumerate(dataset.inter_data):
        uid = sample.get('uid', users[position])
        if uid not in dataset.inters or uid in seen:
            raise ValueError('Cached evaluation UIDs are unknown or repeated.')
        seen.add(uid)
        targets = held_out_targets(dataset, uid)
        expected = [target['representation'] for target in targets]
        if sample['item'] != expected or sample['behavior'] != [target['behavior'] for target in targets]:
            raise ValueError(f'Cached target order differs from held-out source events for UID {uid}.')
        start = int(dataset.test_pos[uid] if dataset.mode == 'test' else dataset.valid_pos[uid])
        history = [''.join(dataset.indices[str(item)]) for item in dataset.inters[uid][:start]]
        expected_history = dataset._get_inters(history, dataset.history_behaviors[uid][:start])
        if sample.get('inters') != expected_history or sample.get('inters_item_list') != dataset._get_inters_with_only_items(history):
            raise ValueError(f'Cached history differs from source events for UID {uid}.')
        copied.inter_data.append({**sample, 'uid': uid})
    return copied


def held_out_targets(dataset, uid):
    start = dataset.test_pos[uid] if dataset.mode == 'test' else dataset.valid_pos[uid]
    end = len(dataset.inters[uid]) if dataset.mode == 'test' else dataset.test_pos[uid]
    if start < 0 or start >= end:
        raise ValueError(f'No held-out session for UID {uid}.')
    targets = []
    for event_index in range(int(start), int(end)):
        item_id = dataset.inters[uid][event_index]
        behavior = dataset.history_behaviors[uid][event_index]
        sid = ''.join(dataset.indices[str(item_id)])
        targets.append({'item_id': item_id, 'behavior': behavior, 'sid': sid, 'event_index': event_index,
                        'session_id': int(dataset.session[uid][event_index]),
                        'representation': dataset.get_behavior_item(sid, behavior)})
    return targets


def export_user_targets(dataset, behavior, metrics_path):
    """Bind source item IDs and duplicate events to the actual UID-aligned metrics."""
    metrics_path = Path(metrics_path)
    sidecar = metrics_path.with_suffix('.uids.json')
    alignment = json.loads(sidecar.read_text())
    if alignment['metric_sha256'] != file_sha256(metrics_path):
        raise ValueError('Metric checksum differs from its UID sidecar.')
    for key, value in [('split', dataset.mode), ('dataset', dataset.dataset), ('max_his_len', dataset.max_his_len), ('index_file', dataset.index_file)]:
        if alignment.get(key) != value:
            raise ValueError(f'Metric target context differs: {key}.')
    by_uid = {sample['uid']: sample for sample in dataset.inter_data}
    uids = alignment['uids']
    if len(set(uids)) != len(uids) or set(by_uid) != set(uids):
        raise ValueError('Metric and target UID sets differ.')
    metrics = json.loads(metrics_path.read_text())
    if any(len(values) != len(uids) for values in metrics.values()):
        raise ValueError('Metric array lengths differ from the UID sidecar.')
    candidates = {}
    for item_id, tokens in dataset.indices.items():
        candidates.setdefault(''.join(tokens), []).append(item_id)
    rows = []
    for position, uid in enumerate(uids):
        targets = [target for target in held_out_targets(dataset, uid) if target['behavior'] == behavior]
        if [target['representation'] for target in targets] != by_uid[uid]['item']:
            raise ValueError(f'Filtered cached targets differ for UID {uid}.')
        rows.append({'uid': uid, 'ground_truth_item_ids': [target['item_id'] for target in targets],
                     'targets': [{**target, 'candidate_item_ids_with_same_sid': candidates[target['sid']]} for target in targets],
                     'metrics': {metric: values[position] for metric, values in metrics.items()}})
    path = metrics_path.with_suffix('.targets.json')
    sources = {}
    for suffix in ('inter', 'behavior', 'session'):
        source = Path(dataset.data_path) / f'{dataset.dataset}.SMB.{suffix}.json'
        if source.is_file():
            stat = source.stat()
            sources[suffix] = {'path': str(source.resolve()), 'bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns}
    write_json(path, {'schema_version': 1, 'split': dataset.mode, 'dataset': dataset.dataset, 'behavior': behavior,
                      'max_his_len': dataset.max_his_len, 'index_file': dataset.index_file,
                      'metric_sha256': file_sha256(metrics_path), 'uid_sidecar_sha256': file_sha256(sidecar),
                      'index_sha256': file_sha256(Path(dataset.data_path) / f'{dataset.dataset}{dataset.index_file}'),
                      'source_snapshots': sources,
                      'target_source': 'Held-out source events; repeated item IDs are retained in event order.',
                      'collision_semantics': 'Ranking uses SID representations; original item IDs are never inferred by inverse SID lookup.',
                      'users': rows})
    return path
