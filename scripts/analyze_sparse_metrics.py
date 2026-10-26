#!/usr/bin/env python
"""Bucket existing per-user ranking metrics on CPU, with explicit split and UID checks."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from SeqRec.evaluation.sparse_buckets import offline_buckets  # noqa: E402
from SeqRec.utils.rebuttal import write_json  # noqa: E402


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ours', required=True)
    parser.add_argument('--baseline', required=True)
    parser.add_argument('--ours_uid_file')
    parser.add_argument('--baseline_uid_file')
    parser.add_argument('--behavior_file', required=True)
    parser.add_argument('--session_file', required=True)
    parser.add_argument('--behavior', default='cvr')
    parser.add_argument('--split', choices=['validation', 'test'], required=True)
    parser.add_argument('--max_his_len', type=int, required=True)
    parser.add_argument('--baseline_max_his_len', type=int, help='Only if the baseline actually used a different window; buckets still use the model window above.')
    parser.add_argument('--count_scope', choices=['visible', 'full'], default='visible')
    parser.add_argument('--thresholds', type=int, nargs='+', default=[1, 3, 6])
    parser.add_argument('--metrics', nargs='+')
    parser.add_argument('--bootstrap_samples', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--legacy_lexicographic_order', action='store_true', help='Explicitly assume the old writer order after recovering the split population; records this limitation.')
    parser.add_argument('--output_file', required=True)
    args = parser.parse_args()
    result = offline_buckets(args)
    write_json(args.output_file, result)
    print(f'Saved {len(result["buckets"])} buckets for {result["n_users"]} paired users; no model inference.')
