import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

from SeqRec.evaluation.sparse_buckets import aligned_metrics, bucket_metrics, history_counts
from SeqRec.tasks.evaluation.base import _BaseDecoderTestTask


class MetricWriter(_BaseDecoderTestTask):
    def info(self, message):
        pass


class SparseMetricsTest(unittest.TestCase):
    def test_writer_preserves_legacy_arrays_and_adds_bound_uid_sidecar(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'metrics.json'
            task = MetricWriter()
            task.metric_list = ['ndcg@10']
            task.datasets = [SimpleNamespace(mode='test', dataset='Toy', max_his_len=100, index_file='.index.json')]
            task._save_user_metrics({'ndcg@10': {'2': 0.18154, '10': 0.7}}, 2, str(path), {})
            self.assertEqual(json.loads(path.read_text()), {'ndcg@10': [0.7, 0.18154]})
            mapped, alignment = aligned_metrics(path, {'10': {}, '2': {}})
            self.assertEqual(alignment, 'explicit_uid_sidecar')
            self.assertEqual(mapped['ndcg@10']['2'], 0.18154)
            with self.assertRaisesRegex(ValueError, 'split'):
                aligned_metrics(path, {'10': {}, '2': {}}, split='validation', max_his_len=100)
            with self.assertRaisesRegex(ValueError, 'window'):
                aligned_metrics(path, {'10': {}, '2': {}}, split='test', max_his_len=20)
            path.write_text('{"ndcg@10":[0,0]}')
            with self.assertRaisesRegex(ValueError, 'SHA256'):
                aligned_metrics(path, {'10': {}, '2': {}})

    def test_missing_uids_require_explicit_legacy_assumption(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'old.json'
            path.write_text('{"hit@5":[0,1]}')
            with self.assertRaises(ValueError):
                aligned_metrics(path, ['2', '10'])
            mapped, alignment = aligned_metrics(path, ['2', '10'], legacy_lexicographic_order=True)
            self.assertEqual(mapped['hit@5'], {'2': 1, '10': 0})
            self.assertIn('assumption', alignment)

    def test_visible_history_counts_and_paired_metrics_preserve_multi_target_values(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            behaviors = root / 'behaviors.json'
            sessions = root / 'sessions.json'
            behaviors.write_text(json.dumps({'u1': ['cvr', 'click', 'click', 'cvr'], 'u2': ['click', 'cvr', 'cvr']}))
            sessions.write_text(json.dumps({'u1': [0, 1, 1, 2], 'u2': [0, 1, 2]}))
            counts = history_counts(behaviors, sessions, 'cvr', 2, 'test')
            self.assertEqual(counts['u1']['visible_core'], 0)
            self.assertEqual(counts['u1']['history_core'], 1)
            ours = {'ndcg@10': {'u1': 0.18154, 'u2': 0.7}}
            baseline = {'ndcg@10': {'u1': 0.0, 'u2': 0.2}}
            rows = bucket_metrics(counts, ours, baseline, [1, 2], 'visible', 50, 7)
            self.assertEqual([row['n_users'] for row in rows], [1, 1, 0])
            self.assertEqual(rows[0]['metrics']['ndcg@10']['delta'], 0.18154)
            self.assertEqual(rows[0]['metrics']['ndcg@10']['paired_95_interval'], [0.18154, 0.18154])
            self.assertIsNone(rows[-1]['metrics']['ndcg@10']['ours'])
            self.assertEqual(rows, bucket_metrics(counts, ours, baseline, [1, 2], 'visible', 50, 7))
            validation = history_counts(behaviors, sessions, 'cvr', 2, 'validation')
            self.assertEqual(set(validation), {'u2'})


if __name__ == '__main__':
    unittest.main()
