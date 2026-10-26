#!/usr/bin/env python
"""Refresh experiment status and print the next command; never start a job."""

import argparse
import json
from pathlib import Path
import shlex
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from SeqRec.utils.rebuttal import write_json  # noqa: E402
from SeqRec.utils.rebuttal_queue import attach_parameter_audits, next_pending, refresh_status, rewrite_command  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--phase', choices=['train', 'validation', 'test'], default='train')
    parser.add_argument('--stage', choices=['P0-A', 'P0-B'])
    parser.add_argument('--gpu', help='GPU IDs, with the same device count as the prepared training recipe.')
    parser.add_argument('--port', type=int)
    parser.add_argument('--output_file', help='Optional new JSON status snapshot.')
    parser.add_argument('--output_script', help='Optional new executable shell text for the single next job; not executed.')
    parser.add_argument('--parameter_audits', nargs='+', default=[], help='Bind independent parameter audit JSONs by configuration SHA256.')
    args = parser.parse_args()
    for value in (args.output_file, args.output_script):
        if value and Path(value).exists():
            raise FileExistsError(value)
    plan = json.loads(Path(args.plan).read_text())
    if not all('stage' in run for run in plan['runs']):
        parser.error('Use a handoff suite plan with explicit stages.')
    gpu = args.gpu or plan['settings']['gpu']
    if len(gpu.split(',')) != len(plan['settings']['gpu'].split(',')):
        parser.error('Changing GPU count requires a new recipe to preserve the effective batch.')
    status = refresh_status(plan)
    attach_parameter_audits(status, plan, args.parameter_audits)
    run = next_pending(plan, status, args.phase)
    if run and args.stage and run['stage'] != args.stage:
        run = None
    status['next'] = {'variant': run['variant'], 'seed': run['seed'], 'stage': run['stage'], 'phase': args.phase} if run else None
    if args.output_file:
        write_json(args.output_file, status)
    if not run:
        if args.output_script:
            parser.error('No safely launchable next job; check the status snapshot for running/unverified/incomplete jobs.')
        print(json.dumps(status, indent=2))
        return
    command = rewrite_command(run['commands'][args.phase], gpu, args.port or plan['settings']['port'])
    target = run['checkpoint'] if args.phase == 'train' else run['results'][args.phase]
    script = '#!/bin/bash\nset -euo pipefail\ncd ' + shlex.quote(str(Path(__file__).resolve().parents[1])) + '\n'
    script += 'test ! -e ' + shlex.quote(target) + '\n' + command + '\n'
    if args.output_script:
        Path(args.output_script).parent.mkdir(parents=True, exist_ok=True)
        with Path(args.output_script).open('x') as file:
            file.write(script)
    print(script, end='')


if __name__ == '__main__':
    main()
