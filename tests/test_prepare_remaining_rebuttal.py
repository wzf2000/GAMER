import json
from pathlib import Path
import shutil
import sys
from tempfile import TemporaryDirectory
import unittest

from scripts.prepare_remaining_rebuttal import parse_args, prepare_remaining


class PrepareRemainingTest(unittest.TestCase):
    def fixture(self, root, *extra):
        shutil.copytree('data/Toy', root / 'data/Toy', ignore=shutil.ignore_patterns('*.pkl'))
        return parse_args(['--dataset', 'Toy', '--data_path', str(root / 'data'), '--output_root', str(root / 'output'), '--gpu', '2,5', *extra])

    def test_fresh_machine_generates_both_local_plans_and_optional_caches(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            args = self.fixture(root, '--prepare_cache')
            plans = prepare_remaining(args)
            self.assertEqual([len(plan['runs']) for plan in plans], [7, 12])
            for group, plan in zip(('p0', 'seeds'), plans):
                path = root / f'output/remaining-{group}/plan.json'
                self.assertTrue(path.is_file())
                self.assertEqual(plan['python_executable'], sys.executable)
                self.assertEqual(plan['settings']['gradient_accumulation_steps'], 8)
                self.assertEqual(plan['settings']['gpu'], '2,5')
                self.assertEqual(plan['settings']['eval_behavior'], 'like')
                self.assertTrue(all(str(root) in run['config'] for run in plan['runs']))
                self.assertFalse(any(Path(run['checkpoint']).exists() for run in plan['runs']))
            self.assertTrue(list((root / 'data/Toy').glob('*.aug4.train.pkl')))
            self.assertTrue(list((root / 'data/Toy').glob('*.valid_test.pkl')))

    def test_lfs_pointer_and_existing_output_fail_before_generating_plans(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            args = self.fixture(root)
            index = root / 'data/Toy/Toy.index.json'
            original = index.read_bytes()
            index.write_text('version https://git-lfs.github.com/spec/v1\noid sha256:placeholder\nsize 100\n')
            with self.assertRaisesRegex(ValueError, 'Git LFS pointer'):
                prepare_remaining(args)
            self.assertFalse((root / 'output').exists())
            index.write_bytes(original)
            (root / 'output/remaining-seeds').mkdir(parents=True)
            with self.assertRaises(FileExistsError):
                prepare_remaining(args)
            self.assertFalse((root / 'output/remaining-p0').exists())

    def test_four_gpus_adjust_accumulation_and_nondivisible_count_is_rejected(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            args = self.fixture(root, '--groups', 'p0', '--gpus', '0,2,4,6')
            plan = prepare_remaining(args)[0]
            self.assertEqual(plan['settings']['gradient_accumulation_steps'], 4)
            self.assertEqual(plan['settings']['per_device_train_batch_size'], 128)
            args.gpu = '0,2,4'
            with self.assertRaisesRegex(ValueError, 'positive multiple'):
                prepare_remaining(args)

    def test_optional_tmall_missing_inputs_are_not_silently_skipped(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            args = self.fixture(root, '--groups', 'p0', 'tmall')
            with self.assertRaisesRegex(FileNotFoundError, 'Tmall-24-0.25-V2'):
                prepare_remaining(args)
            self.assertFalse((root / 'output').exists())


if __name__ == '__main__':
    unittest.main()
