#!/usr/bin/env python
"""Build regression reference data from CT scans on THIS machine.

For every CT it runs TotalSegmentator (airway and lobe masks), then the ORIGINAL
scripts in reference/av_phenotype/ (steps 2 to 10 and the surface script), in place.
The outputs act as the stored original outputs for the regression tests, so the
port is compared with the original code on the same machine and the same masks.

    python tools/build_reference_data.py --root ~/alr_ref --ct scan_a.nii.gz scan_b.nii.gz
    export ALR_DATA=~/alr_ref
    export ALR_CASES=scan_a,scan_b

The case name is the CT file name without .nii.gz. The folder layout is
ROOT/NAME/NAME/, the same as the cohort layout. Use 3 or 4 CTs that differ from
each other (for example different slice thickness and different disease severity).
"""
import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
REF = REPO / 'reference' / 'av_phenotype'
LOBE_CLASSES = ['lung_upper_lobe_left', 'lung_lower_lobe_left', 'lung_upper_lobe_right',
                'lung_middle_lobe_right', 'lung_lower_lobe_right']
STEPS = ['compute_airway_mask_isotropic.py', 'compute_airway_skeleton_teasar.py',
         'compute_airway_generations.py', 'compute_airway_strahler.py',
         'compute_airway_lobe_labels.py', 'compute_airway_lobe_topology.py',
         'compute_smith_dysanapsis.py', 'export_airway_csv.py', 'export_airway_3d.py',
         'compute_airway_surface.py']


def pick_device(requested):
    if requested != 'auto':
        return requested
    import torch
    if torch.cuda.is_available():
        return 'gpu'
    mps = getattr(torch.backends, 'mps', None)
    return 'mps' if mps and mps.is_available() else 'cpu'


def find_ts(explicit):
    if explicit:
        return str(explicit)
    beside = Path(sys.executable).parent / 'TotalSegmentator'
    if beside.exists():
        return str(beside)
    found = shutil.which('TotalSegmentator')
    if not found:
        sys.exit('TotalSegmentator not found; activate the ALR environment or pass --ts-exe')
    return found


def run(cmd, **kw):
    print('  $', ' '.join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], check=True, **kw)


def build_case(ct, root, ts_exe, device):
    name = ct.name[:-len('.nii.gz')]
    case_dir = root / name / name
    case_dir.mkdir(parents=True, exist_ok=True)
    link = case_dir / 'ct.nii.gz'
    if link.is_symlink() or link.exists():
        link.unlink()
    os.symlink(ct, link)

    vessels, merged = case_dir / 'totalseg_vessels', case_dir / 'totalseg_merged'
    if not (vessels / 'lung_airways.nii.gz').exists():
        run([ts_exe, '-i', link, '-o', vessels, '-ta', 'lung_vessels', '-d', device])
    lobes = merged / 'lung_lobes_multilabel.nii.gz'
    if not lobes.exists():
        merged.mkdir(parents=True, exist_ok=True)
        run([ts_exe, '-i', link, '-o', lobes, '-ta', 'total', '-ml', '-rmb', '-d', device,
             '-rs', *LOBE_CLASSES])

    shutil.rmtree(case_dir / 'airway_results', ignore_errors=True)
    env = dict(os.environ, CASE_ID=name, CASE_DIR=str(case_dir))
    for step in STEPS:
        run([sys.executable, REF / step], env=env, cwd=REF)
    return name


def main():
    ap = argparse.ArgumentParser(description='Build regression reference data from CTs with the original scripts.')
    ap.add_argument('--root', required=True, help='output folder (created)')
    ap.add_argument('--ct', nargs='+', required=True, help='CT files, each a .nii.gz')
    ap.add_argument('--device', default='auto', help='auto, gpu, mps or cpu')
    ap.add_argument('--ts-exe', help='TotalSegmentator executable (default: the one of this environment)')
    args = ap.parse_args()

    root = Path(args.root).expanduser().resolve()
    cts = [Path(c).expanduser().resolve() for c in args.ct]
    for ct in cts:
        if not ct.exists() or not ct.name.endswith('.nii.gz'):
            sys.exit(f'{ct}: expected an existing .nii.gz file')
    device, ts_exe = pick_device(args.device), find_ts(args.ts_exe)
    names = []
    for ct in cts:
        print(f'\n=== {ct.name}', flush=True)
        names.append(build_case(ct, root, ts_exe, device))
    print('\nDone. Now run:')
    print(f'  export ALR_DATA={root}')
    print(f'  export ALR_CASES={",".join(names)}')
    print('  python tools/reproduce_reference.py --data "$ALR_DATA" --cases ' + ' '.join(names))


if __name__ == '__main__':
    main()
