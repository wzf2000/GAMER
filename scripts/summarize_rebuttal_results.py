#!/usr/bin/env python
"""Summarize ranking results from a prepared experiment plan, including paired seeds."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from SeqRec.utils.experiment_results import summarize_plan, write_summary_csv  # noqa: E402
from SeqRec.utils.rebuttal import write_json  # noqa: E402


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--split', choices=['validation', 'test'], required=True)
    parser.add_argument('--behavior', default='cvr')
    parser.add_argument('--reference', default='full')
    parser.add_argument('--output_file', required=True)
    parser.add_argument('--csv_file', help='Optional long-form per-run CSV with matching-seed reference differences.')
    parser.add_argument('--allow_partial', action='store_true')
    args = parser.parse_args()
    for path in (args.output_file, args.csv_file):
        if path and Path(path).exists():
            raise FileExistsError(path)
    if args.csv_file and Path(args.csv_file).resolve() == Path(args.output_file).resolve():
        parser.error('--csv_file and --output_file must differ.')
    summary = summarize_plan(args.plan, args.split, args.behavior, args.reference, args.allow_partial)
    write_json(args.output_file, summary)
    if args.csv_file:
        write_summary_csv(args.csv_file, summary)
    print(f'Saved {len(summary["variants"])} variants to {args.output_file}; partial={summary["partial"]}.')
