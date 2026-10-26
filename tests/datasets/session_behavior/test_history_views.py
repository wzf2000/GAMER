import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import PreTrainedTokenizerFast

from SeqRec.datasets.collators.generative import DecoderOnlyTestCollator
from SeqRec.datasets.loaders.session_behavior import load_SMB_datasets, load_SMB_test_dataset
from SeqRec.datasets.session_behavior.history_views import export_history_views


class HistoryViewTest(unittest.TestCase):
    def inputs(self, root):
        raw, consolidated = [], []
        for uid in ('u1', 'u2'):
            specification = [(0, 0, 'click'), (0, 0, 'cvr' if uid == 'u1' else 'click'),
                             (1, 1, 'click'), (2, 2, 'click'), (2, 2, 'cvr'), (3, 3, 'click'), (3, 3, 'cvr')]
            user = []
            for index, (session, item, behavior) in enumerate(specification):
                time = f'2026-01-0{session + 1} 00:0{index % 2}:00'
                user.append(dict(uid=uid, event_id=f'e{index}', item_id=item, behavior=behavior,
                                 session_id=session, timestamp=time, available_at=time))
            # Preserve chronological order within each original session.
            for session in range(4):
                rows = [event for event in user if event['session_id'] == session]
                for minute, event in enumerate(rows):
                    event['timestamp'] = event['available_at'] = f'2026-01-0{session + 1} 00:0{minute}:00'
            raw.extend(user)
            for indices in ([0, 1], [2], [3, 4], [5, 6]):
                row = copy.deepcopy(user[indices[-1]])
                row['event_id'] = 'c' + str(indices[-1])
                row['source_event_ids'] = [f'e{index}' for index in indices]
                consolidated.append(row)
        manifest = {'raw_provenance': 'Synthetic event fixture, not production raw evidence.', 'core_behavior': 'cvr',
                    'users': {uid: {'valid_session': 2, 'test_session': 3, 'valid_target_event_ids': ['e4'], 'test_target_event_ids': ['e6']} for uid in ('u1', 'u2')}}
        paths = {}
        for name, value in [('raw_file', raw), ('consolidated_file', consolidated)]:
            paths[name] = root / f'{name}.jsonl'
            paths[name].write_text(''.join(json.dumps(row) + '\n' for row in value))
        for name, value in [('split_manifest', manifest), ('behavior_levels', {'click': 0, 'cvr': 1}),
                            ('index_file', {str(index): [f'<i_{index}>', '<s_0>'] for index in range(4)})]:
            paths[name] = root / f'{name}.json'
            paths[name].write_text(json.dumps(value))
        return paths

    def test_exports_common_targets_candidates_and_reports_coverage(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            paths = self.inputs(root)
            output = root / 'views'
            result = export_history_views(**paths, output_dir=output, max_his_len=2)
            targets = []
            for name, data_path in result['data_paths'].items():
                train, valid = load_SMB_datasets('HistoryViews', data_path, 2, '.index.json', 'smb_explicit_decoder', validation_behavior='cvr')
                self.assertTrue(all(sample['behavior'] == 'cvr' for sample in valid))
                test = load_SMB_test_dataset('HistoryViews', data_path, 2, '.index.json', 'smb_explicit').filter_by_behavior('cvr')
                targets.append([(sample['uid'], sample['labels']) for sample in test])
                self.assertEqual((Path(data_path) / 'HistoryViews/HistoryViews.index.json').read_bytes(), paths['index_file'].read_bytes())
                if name == 'core_only':
                    ranking = load_SMB_test_dataset('HistoryViews', data_path, 2, '.index.json', 'smb_explicit_valid')
                    empty = next(sample for sample in ranking if not sample['input_ids'])
                    tokenizer = PreTrainedTokenizerFast(
                        tokenizer_object=Tokenizer(WordLevel({'[PAD]': 0, '[UNK]': 1}, unk_token='[UNK]')),
                        pad_token='[PAD]', unk_token='[UNK]', model_max_length=24,
                    )
                    inputs, _ = DecoderOnlyTestCollator(tokenizer)([empty])
                    self.assertEqual(inputs.session_ids.tolist(), [[0]])
                    self.assertEqual(inputs.extended_session_ids.tolist(), [[0]])
            self.assertEqual(targets[0], targets[1])
            self.assertEqual(targets[1], targets[2])
            coverage = json.loads((output / 'coverage.json').read_text())
            self.assertTrue(any(row['view'] == 'core_only' and row['visible_events'] == 0 for row in coverage))
            self.assertTrue(any(row['view'] == 'consolidated' and row['visible_raw_events'] > row['visible_events'] for row in coverage))
            with self.assertRaises(FileExistsError):
                export_history_views(**paths, output_dir=output)

    def test_rejects_backfilled_or_invented_raw_provenance(self):
        for kind in ('late', 'missing_provenance', 'target_change', 'backdated'):
            with self.subTest(kind=kind), TemporaryDirectory() as directory:
                root = Path(directory)
                paths = self.inputs(root)
                rows = [json.loads(line) for line in paths['consolidated_file'].read_text().splitlines()]
                if kind == 'late':
                    rows[0]['available_at'] = '2026-01-05 00:00:00'
                elif kind == 'missing_provenance':
                    rows[0]['source_event_ids'] = ['invented_event']
                elif kind == 'target_change':
                    rows[2]['behavior'] = 'click'
                else:
                    rows[0]['timestamp'] = '2026-01-01 00:00:00'
                paths['consolidated_file'].write_text(''.join(json.dumps(row) + '\n' for row in rows))
                with self.assertRaises(ValueError):
                    export_history_views(**paths, output_dir=root / 'views')
                self.assertFalse((root / 'views').exists())


if __name__ == '__main__':
    unittest.main()
