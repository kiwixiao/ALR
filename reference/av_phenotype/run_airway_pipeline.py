#!/usr/bin/env python
"""Pure-airway pipeline.

Given just `ct.nii.gz` in the case directory, produce the full airway
output set (centerline, generations, Strahler order, lobar topology
labels, Smith dysanapsis index, interactive 3-D HTML) without doing any
of the AV particle / vessel processing.

Stages:
   1.  compute_totalseg_airway.py       → totalseg_vessels/lung_airways.nii.gz
                                          totalseg_merged/lung_lobes_multilabel.nii.gz
                                          (auto-invokes TotalSegmentator in
                                           conda env totalseg213; skips if the
                                           outputs are already on disk)
   2.  compute_airway_mask_isotropic.py → lung_airways_iso.nii.gz
   3.  compute_airway_skeleton_teasar.py → lung_airways_teasar.{nii.gz,vtk}
   4.  compute_airway_generations.py    → in-place (Weibel Generation, Is_Root)
   5.  compute_airway_strahler.py       → in-place (Strahler_Order)
   6.  compute_airway_lobe_labels.py    → in-place (Lobe — sampled from lobe mask)
   7.  compute_airway_lobe_topology.py  → in-place (Lobe_Topology — rule-based)
   8.  compute_smith_dysanapsis.py      → {CASE}_smith_dysanapsis.{csv,json,png}
   9.  export_airway_csv.py             → {CASE}_airway_centerline.csv
  10.  export_airway_3d.py              → {CASE}_airway_3d.html

Usage:
    python run_airway_pipeline.py --case NL010
    python run_airway_pipeline.py --case NL010 --force
    CASE_ID=NL010 python run_airway_pipeline.py
"""

import os, sys, subprocess, time, argparse


def _parse_args():
    ap = argparse.ArgumentParser(
        description='Pure-airway pipeline. Two ways to invoke:\n'
                    '  --case <CASE>                 cohort-style under final_analysis/\n'
                    '  --ct <PATH/TO/scan.nii.gz>    ad-hoc CT anywhere on disk',
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--case', type=str, default=None,
                    help='case id under final_analysis/<CASE>/<CASE>/ '
                         '(mutually exclusive with --ct)')
    ap.add_argument('--ct', type=str, default=None,
                    help='path to a CT NIfTI file outside the standard layout')
    ap.add_argument('--out', type=str, default=None,
                    help='output directory (default: dir containing the CT)')
    ap.add_argument('--name', type=str, default=None,
                    help='case-id used as filename prefix '
                         '(default: CT filename stem)')
    ap.add_argument('--force', action='store_true',
                    help='re-run all stages (ignore skip-if-output-exists)')
    args, _ = ap.parse_known_args()
    if args.case and args.ct:
        ap.error('use either --case OR --ct, not both')
    if not args.case and not args.ct:
        # fall back to env CASE_ID, then CF001 default
        args.case = os.environ.get('CASE_ID', 'CF001')
    return args


_args = _parse_args()
FORCE = _args.force
HERE = os.path.dirname(os.path.abspath(__file__))

if _args.ct:
    CT_PATH = os.path.abspath(_args.ct)
    if not os.path.exists(CT_PATH):
        sys.exit(f'ERROR: CT not found at {CT_PATH}')
    CASE_DIR = os.path.abspath(_args.out) if _args.out else os.path.dirname(CT_PATH)
    os.makedirs(CASE_DIR, exist_ok=True)
    CASE = _args.name or os.path.splitext(os.path.basename(CT_PATH))[0].replace('.nii', '')
    # Ensure ct.nii.gz exists at CASE_DIR (symlink if elsewhere)
    canon_ct = f'{CASE_DIR}/ct.nii.gz'
    if os.path.abspath(CT_PATH) != os.path.abspath(canon_ct):
        if os.path.lexists(canon_ct):
            os.remove(canon_ct)
        os.symlink(CT_PATH, canon_ct)
        print(f'  symlinked ct.nii.gz → {CT_PATH}')
else:
    CASE = _args.case
    CASE_DIR = f'/path/to/av_phenotype/final_analysis/{CASE}/{CASE}'


AIRWAY_DIR = f'{CASE_DIR}/airway_results'

STAGES = [
    ('compute_totalseg_airway.py',
     f'{CASE_DIR}/totalseg_vessels/lung_airways.nii.gz'),
    ('compute_airway_mask_isotropic.py',
     f'{CASE_DIR}/lung_airways_iso.nii.gz'),
    ('compute_airway_skeleton_teasar.py',
     f'{AIRWAY_DIR}/lung_airways_teasar.vtk'),
    ('compute_airway_generations.py',     None),
    ('compute_airway_strahler.py',        None),
    ('compute_airway_lobe_labels.py',     None),
    ('compute_airway_lobe_topology.py',   None),
    ('compute_smith_dysanapsis.py',
     f'{AIRWAY_DIR}/{CASE}_smith_dysanapsis.csv'),
    ('export_airway_csv.py',
     f'{AIRWAY_DIR}/{CASE}_airway_centerline.csv'),
    ('export_airway_3d.py',
     f'{AIRWAY_DIR}/{CASE}_airway_3d.html'),
]


def run_stage(idx, total, script, expected_output):
    if not FORCE and expected_output and os.path.exists(expected_output):
        print(f'\n⤷ STAGE {idx}/{total} {script} — '
              f'skip (exists: {os.path.basename(expected_output)})')
        return True
    print(f'\n══════════════ STAGE {idx}/{total}  {script} ══════════════')
    env = os.environ.copy()
    env['CASE_ID'] = CASE
    env['CASE_DIR'] = CASE_DIR
    r = subprocess.run([sys.executable, os.path.join(HERE, script)], env=env)
    return r.returncode == 0


def main():
    t0 = time.time()
    print(f'\n▶ Pure-airway pipeline for {CASE}')
    print(f'  case directory: {CASE_DIR}')
    print(f'  force re-run  : {FORCE}')

    if not os.path.exists(f'{CASE_DIR}/ct.nii.gz'):
        print(f'\nERROR: {CASE_DIR}/ct.nii.gz not found.')
        print('       drop the CT into the case directory and re-run.')
        sys.exit(1)

    n = len(STAGES)
    for i, (script, exp) in enumerate(STAGES, start=1):
        if not run_stage(i, n, script, exp):
            print(f'\n  STAGE {i} ({script}) FAILED — aborting.')
            sys.exit(1)

    dt = time.time() - t0
    print(f'\n══════════════════════════════════════════════════════════════')
    print(f'  AIRWAY PIPELINE COMPLETE for {CASE}  in {dt:.1f}s')
    print(f'══════════════════════════════════════════════════════════════')
    print(f'  Centerline VTK : {AIRWAY_DIR}/lung_airways_teasar.vtk')
    print(f'  Dysanapsis     : {AIRWAY_DIR}/{CASE}_smith_dysanapsis.csv (+ .png, .json)')
    print(f'  Interactive 3D : {AIRWAY_DIR}/{CASE}_airway_3d.html')


if __name__ == '__main__':
    main()
