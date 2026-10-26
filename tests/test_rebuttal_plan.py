import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from SeqRec.utils.experiment_results import summarize_plan
from SeqRec.utils.rebuttal import component_configs, count_parameters, file_sha256
from tests.models.generative.qwen3.test_rebuttal_ablation import tiny_config


class RebuttalPlanTest(unittest.TestCase):
    def test_matched_decoder_and_components_change_only_intended_options(self):
        full = tiny_config().to_dict()
        variants = component_configs(full)
        target = count_parameters(full)
        matched = count_parameters(variants['plain_matched'])
        slope = count_parameters({**variants['plain_matched'], 'intermediate_size': 2}) - count_parameters({**variants['plain_matched'], 'intermediate_size': 1})
        self.assertLessEqual(abs(matched - target), slope / 2)
        self.assertEqual(count_parameters(variants['zero_prior']), target)
        self.assertEqual(variants['shared_ffn']['behavior_injection_decoder'], full['behavior_injection_decoder'])
        self.assertFalse(variants['no_qkv']['th_use_behavior_qkv'])
        self.assertFalse(variants['no_gate']['th_use_output_gate'])

    def test_summary_uses_matching_seeds_and_rejects_missing_or_changed_inputs(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / 'config.json'
            config.write_text('{}')
            runs = []
            for variant, values in [('full', [0.1, 0.3, 0.5]), ('plain_matched', [0.0, 0.2, 0.4])]:
                for seed, value in enumerate(values):
                    result = root / f'{variant}-{seed}.json'
                    result.write_text(json.dumps([{'eval_type': 'Behavior cvr', 'hit@5': value}]))
                    runs.append({'variant': variant, 'seed': seed, 'config': str(config), 'config_sha256': file_sha256(config),
                                 'results': {'test': str(result)}})
            plan = root / 'plan.json'
            plan.write_text(json.dumps({'runs': runs}))
            result = summarize_plan(plan)
            self.assertAlmostEqual(result['variants']['full']['metrics']['hit@5']['std'], 0.2)
            self.assertAlmostEqual(result['paired_differences']['plain_matched']['metrics']['hit@5']['mean'], -0.1)
            runs[-1]['results']['test'] = str(root / 'not-yet-run.json')
            plan.write_text(json.dumps({'runs': runs}))
            with self.assertRaises(FileNotFoundError):
                summarize_plan(plan)
            self.assertTrue(summarize_plan(plan, allow_partial=True)['partial'])
            config.write_text('{"modified":true}')
            with self.assertRaisesRegex(ValueError, 'changed'):
                summarize_plan(plan, allow_partial=True)

    def test_execution_records_reject_validation_results_as_test(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / 'config.json'
            config.write_text('{}')
            result = root / 'test.json'
            result.write_text(json.dumps([{'eval_type': 'Behavior cvr', 'hit@5': 0.2}]))
            settings = dict(max_his_len=100, dataset='Toy', index_file='.index.json', validation_behavior=None)
            execution = root / 'run_manifest.start.json'
            execution.write_text(json.dumps({'source_config_sha256': file_sha256(config), 'arguments': {'model': {'seed': 7}, 'dataset': settings},
                                             'trainable_parameters': 3, 'index_sha256': 'shared-index'}))
            evaluation = {'split': 'valid_test', 'result_sha256': file_sha256(result), 'training_manifest_sha256': file_sha256(execution),
                          'index_sha256': 'shared-index', 'max_his_len': 100}
            sidecar = result.with_suffix('.run.json')
            sidecar.write_text(json.dumps(evaluation))
            plan = root / 'plan.json'
            plan.write_text(json.dumps({'requires_run_manifest': True, 'settings': settings, 'runs': [
                {'variant': 'full', 'seed': 7, 'config': str(config), 'config_sha256': file_sha256(config),
                 'checkpoint': str(root), 'trainable_parameters': 3, 'results': {'test': str(result)}}]}))
            with self.assertRaisesRegex(ValueError, 'split'):
                summarize_plan(plan)
            evaluation['split'] = 'test'
            sidecar.write_text(json.dumps(evaluation))
            self.assertEqual(summarize_plan(plan)['variants']['full']['seeds'], [7])


if __name__ == '__main__':
    unittest.main()
