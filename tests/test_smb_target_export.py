import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from SeqRec.datasets.session_behavior.explicit import SMBExplicitDataset
from SeqRec.evaluation.smb_targets import export_user_targets, with_verified_eval_uids
from SeqRec.tasks.evaluation.base import _BaseDecoderTestTask


class Writer(_BaseDecoderTestTask):
    def info(self, message):
        pass


def fixture(root, mode='valid_test'):
    dataset = SMBExplicitDataset.__new__(SMBExplicitDataset)
    dataset.mode, dataset.dataset, dataset.data_path = mode, 'Toy', str(root)
    dataset.max_his_len, dataset.index_file, dataset.behavior_first = 100, '.index.json', True
    dataset.indices = {'0': ['<i0>'], '1': ['<i1>'], '2': ['<i1>']}
    dataset.inters = {'2': [0, 1, 2, 1, 2], '10': [0, 1, 2, 1, 2]}
    dataset.history_behaviors = {uid: ['click', 'cvr', 'cvr', 'cvr', 'cvr'] for uid in dataset.inters}
    dataset.session = {uid: [0, 1, 1, 2, 2] for uid in dataset.inters}
    dataset.valid_pos = {uid: 1 for uid in dataset.inters}
    dataset.test_pos = {uid: 3 for uid in dataset.inters}
    dataset.inter_data = []
    for uid in dataset.inters:
        pos = dataset.test_pos[uid] if mode == 'test' else dataset.valid_pos[uid]
        history = [''.join(dataset.indices[str(item)]) for item in dataset.inters[uid][:pos]]
        dataset.inter_data.append({'item': ['<behavior_cvr><i1>', '<behavior_cvr><i1>'], 'behavior': ['cvr', 'cvr'],
                                   'inters': dataset._get_inters(history, dataset.history_behaviors[uid][:pos]), 'inters_item_list': history})
    (root / 'Toy.index.json').write_text(json.dumps(dataset.indices))
    return dataset


class SMBTargetExportTest(unittest.TestCase):
    def test_legacy_validation_uids_collisions_and_duplicate_targets_are_preserved(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = fixture(root)
            original = json.dumps(dataset.inter_data)
            aligned = with_verified_eval_uids(dataset)
            self.assertEqual(json.dumps(dataset.inter_data), original)
            self.assertEqual([row['uid'] for row in aligned.inter_data], ['2', '10'])
            path = root / 'metrics.json'
            writer = Writer()
            writer.metric_list, writer.datasets = ['hit@10'], [aligned]
            writer._save_user_metrics({'hit@10': {'2': 0.0, '10': 1.0}}, 2, str(path), {})
            exported = json.loads(export_user_targets(aligned, 'cvr', path).read_text())
            self.assertEqual([row['uid'] for row in exported['users']], ['10', '2'])
            self.assertEqual(exported['users'][0]['ground_truth_item_ids'], [1, 2])
            self.assertEqual(exported['users'][0]['targets'][0]['candidate_item_ids_with_same_sid'], ['1', '2'])
            self.assertEqual(exported['users'][0]['metrics'], {'hit@10': 1.0})
            self.assertEqual(exported['split'], 'valid_test')

    def test_export_rejects_changed_cached_targets_history_and_metric_hash(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = fixture(root, 'test')
            dataset.inter_data[0]['inters'] = 'stale history'
            with self.assertRaisesRegex(ValueError, 'history'):
                with_verified_eval_uids(dataset)
            dataset = fixture(root, 'test')
            dataset.inter_data[0]['item'][0] = 'wrong target'
            with self.assertRaisesRegex(ValueError, 'target order'):
                with_verified_eval_uids(dataset)
            dataset = with_verified_eval_uids(fixture(root, 'test'))
            path = root / 'metrics.json'
            path.write_text('{"hit@10":[0,1]}')
            path.with_suffix('.uids.json').write_text(json.dumps({'uids': ['10', '2'], 'metric_sha256': 'changed'}))
            with self.assertRaisesRegex(ValueError, 'checksum'):
                export_user_targets(dataset, 'cvr', path)


if __name__ == '__main__':
    unittest.main()
