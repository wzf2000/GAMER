#!/usr/bin/env python
"""Validate supplied raw/consolidated provenance and export three comparable SMB inputs."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from SeqRec.datasets.session_behavior.history_views import export_history_views  # noqa: E402


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw_file', required=True, help='Real raw event JSONL; never inferred from consolidated states.')
    parser.add_argument('--consolidated_file', required=True, help='Consolidated event JSONL with source_event_ids.')
    parser.add_argument('--split_manifest', required=True)
    parser.add_argument('--behavior_levels', required=True)
    parser.add_argument('--index_file', required=True)
    parser.add_argument('--index_kind', choices=['SID', 'CID'], required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--dataset', default='HistoryViews')
    parser.add_argument('--max_his_len', type=int, default=100)
    args = parser.parse_args()
    result = export_history_views(**vars(args))
    print(f'Exported {len(result["users"])} users into three history views; no training launched.')
