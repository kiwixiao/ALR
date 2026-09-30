#!/usr/bin/env python
"""Run the ORIGINAL av_phenotype airway scripts (reference/av_phenotype/) on the
stored TotalSegmentator masks and compare the result with the stored outputs.

It answers one question before anything is ported: does the original code, on
this machine, reproduce the stored outputs? The steps are 2 to 10 of the spec.
Step 1 (TotalSegmentator) is not run; its stored masks are the input.

    python tools/reproduce_reference.py --data DATA/final_analysis
    python tools/reproduce_reference.py --data DATA/final_analysis --cases CF121 --scratch /tmp/alr_ref

Exit code 0 means every artifact matched. The VTK, the CSV files and the Smith
JSON are compared field by field. The HTML viewers are only checked for
existence. A byte for byte match is reported as information.
"""
import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np
import pyvista as pv

REPO = Path(__file__).resolve().parents[1]
REF = REPO / 'reference' / 'av_phenotype'
GOLDEN = REPO / 'tests' / 'regression' / 'golden.json'
DEFAULT_CASES = ['CF121', 'CF008', 'NL001', 'CF005']
STEPS = ['compute_airway_mask_isotropic.py', 'compute_airway_skeleton_teasar.py',
         'compute_airway_generations.py', 'compute_airway_strahler.py',
         'compute_airway_lobe_labels.py', 'compute_airway_lobe_topology.py',
         'compute_smith_dysanapsis.py', 'export_airway_csv.py', 'export_airway_3d.py']
INPUTS = ['totalseg_vessels/lung_airways.nii.gz', 'lobes_ml.nii.gz',
          'totalseg_merged/lung_lobes_multilabel.nii.gz']
RTOL, ATOL = 1e-5, 1e-6


def md5(path):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def check_integrity(data, cases):
    """The copied data must be the original files."""
    golden = json.loads(GOLDEN.read_text())
    bad = []
    for case in cases:
        for rel, meta in golden[case]['files'].items():
            p = data / case / case / rel
            if not p.exists() or md5(p) != meta['md5']:
                bad.append(f'{case}/{rel}')
    return bad


def compare_array(name, a, b):
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape:
        return f'{name}: shape {a.shape} vs {b.shape}'
    if a.dtype.kind == 'f':
        if not np.allclose(a, b, rtol=RTOL, atol=ATOL, equal_nan=True):
            return f'{name}: max abs difference {np.nanmax(np.abs(a - b)):.3g}'
    elif not np.array_equal(a, b):
        return f'{name}: {int((a != b).sum())} of {a.size} values differ'
    return None


def compare_vtk(new, old):
    a, b = pv.read(new), pv.read(old)
    problems = []
    for what, x, y in [('points', a.points, b.points), ('lines', a.lines, b.lines)]:
        problems.append(compare_array(what, x, y))
    if sorted(a.point_data.keys()) != sorted(b.point_data.keys()):
        problems.append(f'arrays differ: {sorted(a.point_data.keys())} vs {sorted(b.point_data.keys())}')
    for key in set(a.point_data.keys()) & set(b.point_data.keys()):
        problems.append(compare_array(key, a.point_data[key], b.point_data[key]))
    return [p for p in problems if p]


def close(x, y):
    try:
        return math.isclose(float(x), float(y), rel_tol=RTOL, abs_tol=ATOL)
    except ValueError:
        return x == y


def compare_csv(new, old):
    with open(new, newline='') as f:
        a = list(csv.reader(f))
    with open(old, newline='') as f:
        b = list(csv.reader(f))
    if a[0] != b[0]:
        return [f'header differs: {a[0]} vs {b[0]}']
    if len(a) != len(b):
        return [f'rows {len(a) - 1} vs {len(b) - 1}']
    bad = sum(1 for ra, rb in zip(a[1:], b[1:]) if not all(close(x, y) for x, y in zip(ra, rb)))
    return [f'{bad} of {len(a) - 1} rows differ'] if bad else []


def json_close(a, b):
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(json_close(a[k], b[k]) for k in a)
    if isinstance(a, list):
        return len(a) == len(b) and all(json_close(x, y) for x, y in zip(a, b))
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return math.isclose(a, b, rel_tol=RTOL, abs_tol=ATOL)
    return a == b


def compare_json(new, old):
    return [] if json_close(json.loads(Path(new).read_text()), json.loads(Path(old).read_text())) else ['values differ']


def compare_nifti(new, old):
    a, b = nib.load(new), nib.load(old)
    problems = []
    if not np.allclose(a.affine, b.affine, atol=1e-4):
        problems.append('affine differs')
    p = compare_array('data', np.asarray(a.dataobj), np.asarray(b.dataobj))
    return problems + ([p] if p else [])


def reproduce(case, data, scratch):
    src, dst = data / case / case, scratch / case / case
    (dst / 'airway_results').mkdir(parents=True, exist_ok=True)
    for rel in INPUTS:
        if (src / rel).exists():
            (dst / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src / rel, dst / rel)
    env = dict(os.environ, CASE_ID=case, CASE_DIR=str(dst))
    for step in STEPS:
        r = subprocess.run([sys.executable, str(REF / step)], env=env, cwd=REF,
                           capture_output=True, text=True)
        if r.returncode != 0:
            return [(step, [f'exit code {r.returncode}: ' + r.stderr.strip().splitlines()[-1] if r.stderr.strip() else f'exit code {r.returncode}'])], src, dst
    results = []
    ar = 'airway_results'
    pairs = [('lung_airways_iso.nii.gz', compare_nifti),
             (f'{ar}/lung_airways_teasar.nii.gz', compare_nifti),
             (f'{ar}/lung_airways_teasar.vtk', compare_vtk),
             (f'{ar}/{case}_airway_centerline.csv', compare_csv),
             (f'{ar}/{case}_smith_dysanapsis.csv', compare_csv),
             (f'{ar}/{case}_smith_dysanapsis.json', compare_json)]
    for rel, fn in pairs:
        results.append((rel, fn(dst / rel, src / rel)))
    for rel in [f'{ar}/{case}_airway_3d.html', f'{ar}/{case}_airway_3d_strahler.html']:
        results.append((rel, [] if (dst / rel).exists() else ['not written']))
    return results, src, dst


def main():
    ap = argparse.ArgumentParser(description='Reproduce the stored airway outputs with the original scripts.')
    ap.add_argument('--data', required=True, help='folder that holds CASE/CASE/ (the extracted regression data)')
    ap.add_argument('--cases', nargs='*', default=DEFAULT_CASES)
    ap.add_argument('--scratch', help='working folder (default: a temporary folder, removed at the end)')
    args = ap.parse_args()
    data = Path(args.data)

    bad = check_integrity(data, args.cases)
    if bad:
        sys.exit(f'data integrity check failed for {len(bad)} files, for example {bad[:3]}')
    print('data integrity: every stored file matches tests/regression/golden.json')

    scratch = Path(args.scratch) if args.scratch else Path(tempfile.mkdtemp(prefix='alr_ref_'))
    failed = False
    for case in args.cases:
        results, src, dst = reproduce(case, data, scratch)
        print(f'\n{case}')
        for rel, problems in results:
            same_bytes = (dst / rel).exists() and (src / rel).exists() and md5(dst / rel) == md5(src / rel)
            status = 'PASS' if not problems else 'FAIL'
            failed |= bool(problems)
            note = ' (byte identical)' if same_bytes and not problems else ''
            print(f'  {status}  {rel}{note}' + ''.join(f'\n        {p}' for p in problems))
    if not args.scratch:
        shutil.rmtree(scratch, ignore_errors=True)
    print('\nRESULT:', 'FAIL' if failed else 'all artifacts reproduced')
    sys.exit(1 if failed else 0)


if __name__ == '__main__':
    main()
