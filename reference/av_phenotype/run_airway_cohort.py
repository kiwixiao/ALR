#!/usr/bin/env python
"""Run the pure-airway pipeline across an entire cohort.

Usage:
    python run_airway_cohort.py NL          # all cases under final_analysis/NL*/
    python run_airway_cohort.py CF          # all cases under final_analysis/CF*/
    python run_airway_cohort.py NL --force  # re-run even if outputs exist

Iterates one case at a time (no parallel TS — would blow up GPU). Per-case
failures are caught and logged but do NOT abort the cohort sweep.

A summary file is written at:
    final_analysis/<COHORT>_airway_cohort_<YYYYMMDD_HHMMSS>.csv
with columns: case_id, status, runtime_s, error
"""

import os, re, sys, csv, time, subprocess, argparse, datetime

ROOT = '/path/to/av_phenotype/final_analysis'
HERE = os.path.dirname(os.path.abspath(__file__))


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('cohort', type=str,
                    help='cohort prefix (NL, CF, CFR, CFN, etc.)')
    ap.add_argument('--force', action='store_true',
                    help='pass --force to each per-case run')
    ap.add_argument('--limit', type=int, default=None,
                    help='only run first N cases (for testing)')
    return ap.parse_args()


def main():
    args = parse_args()

    # Discover cases — accept NL, NL_t2, CFR123_t2 etc.
    pattern = re.compile(rf'^{re.escape(args.cohort)}[0-9]+(_[A-Za-z0-9]+)?$')
    cases = sorted([d for d in os.listdir(ROOT)
                    if os.path.isdir(os.path.join(ROOT, d))
                    and pattern.match(d)])
    if args.limit:
        cases = cases[:args.limit]
    print(f'\nCohort {args.cohort}: {len(cases)} cases')
    if not cases:
        print('  no matching cases; nothing to do.'); return

    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    summary_path = f'{ROOT}/{args.cohort}_airway_cohort_{ts}.csv'

    rows = []
    t_total = time.time()
    for i, case in enumerate(cases, 1):
        case_dir = f'{ROOT}/{case}/{case}'
        ct_path = f'{case_dir}/ct.nii.gz'
        if not os.path.exists(ct_path):
            print(f'\n[{i}/{len(cases)}] {case} — SKIP: no ct.nii.gz')
            rows.append(dict(case_id=case, status='skip_no_ct',
                              runtime_s=0.0, error=''))
            continue

        print(f'\n══════════════════════════════════════════════════════════════')
        print(f'  [{i}/{len(cases)}]  {case}')
        print(f'══════════════════════════════════════════════════════════════')
        t0 = time.time()
        cmd = [sys.executable, f'{HERE}/run_airway_pipeline.py', '--case', case]
        if args.force:
            cmd.append('--force')
        try:
            r = subprocess.run(cmd, capture_output=False)
            ok = (r.returncode == 0)
            err = '' if ok else f'exit={r.returncode}'
        except Exception as e:
            ok = False
            err = repr(e)[:200]
        dt = time.time() - t0
        rows.append(dict(case_id=case,
                         status='ok' if ok else 'failed',
                         runtime_s=round(dt, 1),
                         error=err))
        print(f'  → {case}: {"OK" if ok else "FAILED"} in {dt:.1f}s')

        # Append to summary file as we go (so partial results survive ctrl-C)
        with open(summary_path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=['case_id','status','runtime_s','error'])
            w.writeheader()
            for row in rows: w.writerow(row)

    n_ok   = sum(1 for r in rows if r['status'] == 'ok')
    n_fail = sum(1 for r in rows if r['status'] == 'failed')
    n_skip = sum(1 for r in rows if r['status'].startswith('skip'))
    elapsed = time.time() - t_total

    print(f'\n══════════════════════════════════════════════════════════════')
    print(f'  COHORT {args.cohort} COMPLETE')
    print(f'══════════════════════════════════════════════════════════════')
    print(f'  total cases : {len(cases)}')
    print(f'  ok          : {n_ok}')
    print(f'  failed      : {n_fail}')
    print(f'  skip        : {n_skip}')
    print(f'  elapsed     : {elapsed/60:.1f} min')
    print(f'  summary CSV : {summary_path}')


if __name__ == '__main__':
    main()
