import copy
import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from scripts.prepare_rebuttal_experiments import parse_args
from SeqRec.utils.experiment_results import summarize_plan, write_summary_csv
from SeqRec.utils.rebuttal import file_sha256, prepare_plan
from SeqRec.utils.rebuttal_queue import next_pending, refresh_status, reuse_handoff_runs


def make_plan(root):
    arguments = ['prepare', '--suite', 'handoff', '--dataset', 'Toy', '--output_dir', str(root), '--gpu', '4,7', '--gradient_accumulation_steps', '8']
    with patch('sys.argv', arguments):
        return prepare_plan(parse_args())


class RebuttalQueueTest(unittest.TestCase):
    def test_handoff_rejects_a_different_training_task(self):
        with TemporaryDirectory() as directory:
            with patch('sys.argv', ['prepare', '--suite', 'handoff', '--dataset', 'Toy', '--tasks', 'smb_explicit', '--output_dir', directory + '/plan']):
                args = parse_args()
            with self.assertRaisesRegex(ValueError, 'complete submission'):
                prepare_plan(args)

    def test_handoff_has_seven_configs_aliases_safe_stage_scripts_and_reuse(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            first = make_plan(root / 'first')
            self.assertEqual([run['variant'] for run in first['runs']], ['full', 'no_qkv', 'no_gate', 'shared_ffn', 'zero_prior', 'lambda_0.01', 'lambda_0.1'])
            self.assertEqual(first['aliases']['lambda_0.05'], 'full')
            self.assertTrue(all('--export_user_targets' in run['commands']['test'] for run in first['runs']))
            self.assertFalse((root / 'first/run_train.sh').exists())
            self.assertIn('manage_rebuttal_queue.py', (root / 'first/P0-A-train.sh').read_text())
            second = make_plan(root / 'second')
            reuse_handoff_runs(second, [root / 'first/plan.json'])
            self.assertEqual([run['checkpoint'] for run in first['runs']], [run['checkpoint'] for run in second['runs']])
            self.assertEqual(len(set(run['checkpoint'] for run in second['runs'])), 7)
            bad = copy.deepcopy(second)
            bad['settings']['gradient_accumulation_steps'] = 4
            with self.assertRaisesRegex(ValueError, 'recipe'):
                reuse_handoff_runs(bad, [root / 'first/plan.json'])

    def test_status_never_treats_start_config_or_epoch_checkpoint_as_completion(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            plan = make_plan(root / 'plan')
            first = plan['runs'][0]
            checkpoint = Path(first['checkpoint'])
            checkpoint.mkdir(parents=True)
            state = checkpoint / 'checkpoint-5'
            state.mkdir()
            (state / 'trainer_state.json').write_text(json.dumps({'global_step': 5, 'epoch': 1.0}))
            status = refresh_status(plan, processes={})
            self.assertEqual(status['runs'][0]['training'], 'started_liveness_unknown')
            self.assertEqual(next_pending(plan, status)['variant'], 'no_qkv')
            status = refresh_status(plan, processes={str(checkpoint.resolve()): [123]})
            self.assertEqual(status['runs'][0]['training'], 'running')
            for run in plan['runs'][1:4]:
                Path(run['checkpoint']).mkdir(parents=True)
            self.assertIsNone(next_pending(plan, refresh_status(plan, processes={})))
            for run in plan['runs'][:4]:
                path = Path(run['checkpoint'])
                (path / 'model.safetensors').write_bytes(b'fixture weights')
                (path / 'trainer_state.json').write_text(json.dumps({'global_step': 10, 'log_history': [{'train_runtime': 1.0}]}))
            status = refresh_status(plan, processes={})
            self.assertEqual(next_pending(plan, status)['variant'], 'zero_prior')
            self.assertEqual(next_pending(plan, status, 'test')['variant'], 'full')
            config = Path(plan['runs'][4]['config'])
            config.write_text('{}')
            status = refresh_status(plan, processes={})
            self.assertEqual(status['runs'][4]['training'], 'incompatible_config')

    def test_csv_reports_actual_values_paired_differences_and_missing_rows(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / 'config.json'
            config.write_text('{}')
            runs = []
            for name, value in [('full', 0.2), ('no_gate', 0.1), ('shared_ffn', None)]:
                result = root / f'{name}.json'
                if value is not None:
                    result.write_text(json.dumps([{'eval_type': 'Behavior cvr', 'hit@10': value}]))
                runs.append({'variant': name, 'seed': 42, 'config': str(config), 'config_sha256': file_sha256(config), 'results': {'test': str(result)}})
            plan = root / 'plan.json'
            plan.write_text(json.dumps({'runs': runs}))
            summary = summarize_plan(plan, allow_partial=True)
            path = root / 'summary.csv'
            write_summary_csv(path, summary)
            with path.open() as file:
                rows = list(csv.DictReader(file))
            self.assertEqual(float(rows[1]['difference_from_reference']), -0.1)
            self.assertEqual(float(rows[1]['relative_difference_percent']), -50.0)
            self.assertEqual(rows[2]['status'], 'missing')
            self.assertEqual(rows[2]['value'], '')
            self.assertIsNone(summary['variants']['full']['metrics']['hit@10']['std'])
            for run in runs:
                Path(run['results']['test']).unlink(missing_ok=True)
            summary = summarize_plan(plan, allow_partial=True)
            self.assertEqual(summary['variants'], {})
            self.assertEqual(len(summary['missing']), 3)


if __name__ == '__main__':
    unittest.main()
