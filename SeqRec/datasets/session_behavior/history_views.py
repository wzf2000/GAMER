"""Export comparable SMB views from explicitly supplied raw events and provenance.

This adapter does not reconstruct raw events or infer attribution/session rules.
It rejects temporal/provenance cases the static SMB representation cannot express.
"""

from collections import defaultdict
from datetime import datetime
import json
from pathlib import Path
import shutil

from SeqRec.utils.rebuttal import file_sha256, write_json


TIME_FORMAT = '%Y-%m-%d %H:%M:%S'


def read_events(path, levels, indices, consolidated=False):
    users = defaultdict(list)
    seen = set()
    with Path(path).open() as file:
        for number, line in enumerate(file, 1):
            event = json.loads(line)
            required = {'uid', 'event_id', 'item_id', 'behavior', 'session_id', 'timestamp', 'available_at'}
            if not required <= event.keys():
                raise ValueError(f'{path}:{number}: missing event fields {required - event.keys()}')
            if not isinstance(event['uid'], str) or not isinstance(event['event_id'], str) or not event['event_id']:
                raise ValueError('UIDs/event IDs must be strings and event IDs must be nonempty.')
            key = (event['uid'], event['event_id'])
            if key in seen:
                raise ValueError(f'Duplicate event ID: {key}')
            seen.add(key)
            if isinstance(event['item_id'], bool) or not isinstance(event['item_id'], int) or str(event['item_id']) not in indices:
                raise ValueError('Every event item must be an integer in the shared candidate index.')
            if event['behavior'] not in levels or isinstance(event['session_id'], bool) or not isinstance(event['session_id'], int):
                raise ValueError('Unknown behavior or noninteger original session ID.')
            timestamp = datetime.strptime(event['timestamp'], TIME_FORMAT)
            available = datetime.strptime(event['available_at'], TIME_FORMAT)
            if timestamp.strftime(TIME_FORMAT) != event['timestamp'] or available.strftime(TIME_FORMAT) != event['available_at']:
                raise ValueError('Use canonical zero-padded timestamps with second precision.')
            if available < timestamp:
                raise ValueError('available_at cannot precede event timestamp in this input contract.')
            if consolidated and (not isinstance(event.get('source_event_ids'), list) or not event['source_event_ids']):
                raise ValueError('Each consolidated event requires real source_event_ids.')
            users[event['uid']].append(event)
    for uid, events in users.items():
        if [event['timestamp'] for event in events] != sorted(event['timestamp'] for event in events):
            raise ValueError(f'{uid}: events must already be ordered by timestamp.')
        if [event['session_id'] for event in events] != sorted(event['session_id'] for event in events):
            raise ValueError(f'{uid}: original session IDs must increase chronologically.')
    return dict(users)


def validate_history_inputs(raw, consolidated, manifest, levels):
    if not manifest.get('raw_provenance'):
        raise ValueError('The split manifest must identify the real raw source in raw_provenance.')
    core = manifest['core_behavior']
    if core not in levels or core != [name for name, level in levels.items() if level == max(levels.values())][-1]:
        raise ValueError('core_behavior must match the dataset loader target behavior.')
    if not raw or set(raw) != set(consolidated) or set(raw) != set(manifest['users']):
        raise ValueError('Both views and the split manifest must contain exactly the same nonempty user set.')
    for uid, events in raw.items():
        split = manifest['users'][uid]
        valid, test = split['valid_session'], split['test_session']
        if not isinstance(valid, int) or not isinstance(test, int) or valid >= test:
            raise ValueError(f'{uid}: invalid validation/test sessions.')
        raw_by_id = {event['event_id']: event for event in events}
        consumed = set()
        for event in consolidated[uid]:
            sources = event['source_event_ids']
            if len(sources) != len(set(sources)) or any(source not in raw_by_id for source in sources):
                raise ValueError(f'{uid}: invalid consolidated provenance IDs.')
            if consumed.intersection(sources):
                raise ValueError(f'{uid}: a raw event was assigned to multiple consolidated events.')
            consumed.update(sources)
            parents = [raw_by_id[source] for source in sources]
            if any(parent['item_id'] != event['item_id'] or parent['session_id'] != event['session_id'] for parent in parents):
                raise ValueError(f'{uid}: consolidation must retain raw item/session identity.')
            if max(parent['timestamp'] for parent in parents) > event['timestamp'] or max(parent['available_at'] for parent in parents) > event['available_at']:
                raise ValueError(f'{uid}: consolidated state is backdated before its source evidence.')
        if consumed != set(raw_by_id):
            raise ValueError(f'{uid}: consolidated provenance must cover every supplied raw event exactly once.')
        starts = {}
        for event in events:
            starts.setdefault(event['session_id'], event['timestamp'])
        if sorted(starts)[-2:] != [valid, test] or not any(session < valid for session in starts):
            raise ValueError(f'{uid}: reserve the final two original sessions and retain a training prefix.')
        for view, rows in [('raw', events), ('consolidated', consolidated[uid])]:
            if sorted({event['session_id'] for event in rows})[-2:] != [valid, test]:
                raise ValueError(f'{uid}/{view}: held-out sessions disappeared.')
            for event in rows:
                cutoff = starts[valid] if event['session_id'] < valid else starts[test] if event['session_id'] < test else None
                if cutoff is not None and event['available_at'] > cutoff:
                    raise ValueError(f'{uid}/{view}: late evidence requires query-time views; static SMB export would leak.')
            for label, session in [('valid', valid), ('test', test)]:
                requested = split[f'{label}_target_event_ids']
                observed = []
                for event in rows:
                    if event['session_id'] != session or event['behavior'] != core:
                        continue
                    sources = event.get('source_event_ids', [event['event_id']])
                    core_sources = [source for source in sources if raw_by_id[source]['behavior'] == core]
                    if len(core_sources) != 1:
                        raise ValueError(f'{uid}/{view}: each held-out core target must preserve one original core event.')
                    observed.extend(core_sources)
                if not requested or observed != requested:
                    raise ValueError(f'{uid}/{view}: held-out core targets differ from the common manifest.')
    return core


def export_history_views(raw_file, consolidated_file, split_manifest, behavior_levels, index_file, output_dir,
                         dataset='HistoryViews', max_his_len=100, index_kind='SID'):
    root = Path(output_dir)
    if root.exists():
        raise FileExistsError(f'Refusing to overwrite {root}')
    if max_his_len <= 0 or not dataset or dataset in ('.', '..') or Path(dataset).name != dataset or index_kind not in ('SID', 'CID'):
        raise ValueError('Use a positive history budget, a simple dataset name, and explicit SID/CID index kind.')
    levels = json.loads(Path(behavior_levels).read_text())
    indices = json.loads(Path(index_file).read_text())
    lengths = {len(tokens) for tokens in indices.values()}
    if len(lengths) != 1 or not next(iter(lengths)):
        raise ValueError('The shared item index must have a nonzero uniform token length.')
    manifest = json.loads(Path(split_manifest).read_text())
    raw = read_events(raw_file, levels, indices)
    consolidated = read_events(consolidated_file, levels, indices, consolidated=True)
    core = validate_history_inputs(raw, consolidated, manifest, levels)
    views = {'core_only': {uid: [event for event in events if event['behavior'] == core] for uid, events in raw.items()},
             'raw': raw, 'consolidated': consolidated}
    token_count = next(iter(lengths)) + 1
    coverage = []
    root.mkdir(parents=True)
    for name, users in views.items():
        directory = root / name / dataset
        directory.mkdir(parents=True)
        for suffix, key in [('inter', 'item_id'), ('behavior', 'behavior'), ('session', 'session_id'), ('time', 'timestamp')]:
            write_json(directory / f'{dataset}.SMB.{suffix}.json', {uid: [event[key] for event in events] for uid, events in users.items()})
        shutil.copyfile(index_file, directory / f'{dataset}.index.json')
        shutil.copyfile(behavior_levels, directory / f'{dataset}.behavior_level.json')
        for uid, events in users.items():
            split = manifest['users'][uid]
            for label, boundary in [('validation', split['valid_session']), ('test', split['test_session'])]:
                history = [event for event in events if event['session_id'] < boundary]
                visible = history[-max_his_len:]
                source_ids = {source for event in visible for source in event.get('source_event_ids', [event['event_id']])}
                coverage.append({'uid': uid, 'view': name, 'split': label, 'history_events': len(history), 'visible_events': len(visible),
                                 'history_tokens': len(visible) * token_count, 'visible_raw_events': len(source_ids),
                                 'visible_core_events': sum(event['behavior'] == core for event in visible),
                                 'history_core_events': sum(event['behavior'] == core for event in history),
                                 'visible_start': visible[0]['timestamp'] if visible else None,
                                 'visible_end': visible[-1]['timestamp'] if visible else None})
    write_json(root / 'coverage.json', coverage)
    record = {'schema_version': 1, 'dataset': dataset, 'core_behavior': core, 'index_kind': index_kind, 'index_suffix': '.index.json',
              'max_his_len': max_his_len, 'token_count_per_event': token_count, 'users': sorted(raw),
              'inputs': {name: {'path': str(Path(path).resolve()), 'sha256': file_sha256(path)} for name, path in
                         [('raw', raw_file), ('consolidated', consolidated_file), ('split_manifest', split_manifest), ('behavior_levels', behavior_levels), ('index', index_file)]},
              'data_paths': {name: str((root / name).resolve()) for name in views},
              'note': 'Raw was supplied explicitly, not expanded from consolidated states. Use the same core behavior for validation loss and ranking.'}
    write_json(root / 'view_manifest.json', record)
    return record
