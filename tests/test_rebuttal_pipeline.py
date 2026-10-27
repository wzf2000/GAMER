import io
import json
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from scripts.run_rebuttal_experiment import main, run_experiments
from SeqRec.utils.rebuttal import file_sha256


def make_plan(root):
    config = root / 'config.json'
    config.write_text('{}')
    runs = []
    for variant in ('no_gate', 'shared_ffn'):
        run_dir = root / variant / 'seed_42'
        checkpoint = run_dir / 'checkpoint'
        runs.append({'variant': variant, 'seed': 42, 'config': str(config), 'config_sha256': file_sha256(config), 'checkpoint': str(checkpoint),
                     'results': {phase: str(run_dir / f'{phase}.json') for phase in ('validation', 'test')},
                     'commands': {'train': f'CUDA_VISIBLE_DEVICES=0,1 python main.py train_SMB_decoder --output_dir {checkpoint}',
                                  **{phase: f'CUDA_VISIBLE_DEVICES=0,1 python main.py test_SMB_decoder --results_file {run_dir / (phase + ".json")}' for phase in ('validation', 'test')}}})
    plan = root / 'plan.json'
    plan.write_text(json.dumps({'settings': {'gpu': '0,1', 'port': 2314}, 'runs': runs}))
    return plan, runs


class RebuttalPipelineTest(unittest.TestCase):
    def test_gpu_override_remaps_all_phases_and_preserves_saved_plan(self):
        with TemporaryDirectory() as directory:
            plan, runs = make_plan(Path(directory))
            original = plan.read_bytes()
            output = io.StringIO()
            with patch('sys.stdout', output):
                run_experiments(plan, ['no_gate'], [42], gpu='2,5', port=24567)
            commands = output.getvalue().splitlines()
            self.assertEqual(len(commands), 3)
            for command in commands:
                self.assertIn('CUDA_VISIBLE_DEVICES=2,5', command)
                self.assertIn('--nproc_per_node=2', command)
                self.assertIn('--master_port=24567', command)
            self.assertEqual(plan.read_bytes(), original)

    def test_gpu_cli_aliases_forward_selected_devices_and_port(self):
        for flag in ('--gpu', '--gpus'):
            with self.subTest(flag=flag), patch('sys.argv', ['runner', '--plan', 'plan.json', '--variants', 'no_gate', flag, '2,5', '--port', '24567']):
                with patch('scripts.run_rebuttal_experiment.run_experiments') as runner:
                    main()
                self.assertEqual(runner.call_args.args[4:6], ('2,5', 24567))

    def test_dry_run_does_not_create_job_outputs_and_missing_selection_fails(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            plan, runs = make_plan(root)
            self.assertEqual(run_experiments(plan, ['no_gate'], [42]), [])
            self.assertFalse((root / 'no_gate').exists())
            with self.assertRaisesRegex(ValueError, 'found 0'):
                run_experiments(plan, ['no_gate'], [43])
            with self.assertRaisesRegex(ValueError, 'effective batch'):
                run_experiments(plan, ['no_gate'], [42], gpu='0')

    def test_one_invocation_runs_training_validation_test_in_order(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            plan, runs = make_plan(root)
            calls = []

            def execute(command, **kwargs):
                calls.append(command)
                if 'train_SMB_decoder' in command:
                    output = Path(command.split('--output_dir ')[1])
                    output.mkdir(parents=True)
                    (output / 'model.safetensors').write_bytes(b'fixture weights')
                else:
                    output = Path(command.split('--results_file ')[1])
                    output.write_text('[{"eval_type":"Behavior cvr","hit@10":0.1}]')
                return subprocess.CompletedProcess(command, 0)

            receipts = run_experiments(plan, ['no_gate', 'shared_ffn'], [42], execute=True, command_runner=execute)
            self.assertEqual(len(receipts), 6)
            self.assertIn('train_SMB_decoder', calls[0])
            self.assertIn('validation.json', calls[1])
            self.assertIn('test.json', calls[2])
            self.assertTrue(all(Path(run['results']['test']).is_file() for run in runs))
            with self.assertRaises(FileExistsError):
                run_experiments(plan, ['no_gate'], [42], execute=True, command_runner=execute)

    def test_failed_training_stops_before_evaluation_and_records_returncode(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            plan, runs = make_plan(root)
            calls = []

            def execute(command, **kwargs):
                calls.append(command)
                return subprocess.CompletedProcess(command, 3)

            with self.assertRaisesRegex(RuntimeError, 'subsequent phases'):
                run_experiments(plan, ['no_gate', 'shared_ffn'], [42], execute=True, command_runner=execute)
            self.assertEqual(len(calls), 1)
            receipt = next((root / 'no_gate/seed_42').glob('*.finish.json'))
            self.assertEqual(json.loads(receipt.read_text())['returncode'], 3)
            self.assertFalse((root / 'shared_ffn').exists())


if __name__ == '__main__':
    unittest.main()
