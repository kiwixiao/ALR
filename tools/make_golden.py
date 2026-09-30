#!/usr/bin/env python
"""Write tests/regression/golden.json: md5 and size of every stored file of the
regression cases, plus a few counts read from the stored VTK and CSV files.
The file holds aggregate numbers only, no image data.

    python tools/make_golden.py --data /path/to/final_analysis
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import pyvista as pv

CASES = ['CF121', 'CF008', 'NL001', 'CF005']
INPUTS = ['totalseg_vessels/lung_airways.nii.gz', 'lung_airways_iso.nii.gz',
          'totalseg_merged/lung_lobes_multilabel.nii.gz', 'lobes_ml.nii.gz']


def md5(path):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True, help='folder that holds CASE/CASE/')
    ap.add_argument('--out', default=str(Path(__file__).resolve().parents[1] / 'tests/regression/golden.json'))
    args = ap.parse_args()
    root = Path(args.data)

    golden = {}
    for case in CASES:
        d = root / case / case
        files = [d / f for f in INPUTS if (d / f).exists()]
        files += sorted(p for p in (d / 'airway_results').iterdir() if p.is_file())
        entry: dict = {'files': {str(p.relative_to(d)): {'bytes': p.stat().st_size, 'md5': md5(p)} for p in files}}

        cl = pv.read(d / 'airway_results' / 'lung_airways_teasar.vtk')
        gen = cl.point_data['Generation']
        entry['vtk'] = {'n_points': int(cl.n_points), 'n_cells': int(cl.n_cells),
                        'arrays': sorted(cl.point_data.keys()),
                        'fragment_vertices': int((gen < 0).sum()),
                        'max_generation': int(gen.max()),
                        'max_strahler': int(cl.point_data['Strahler_Order'].max())}
        with open(d / 'airway_results' / f'{case}_airway_centerline.csv', newline='') as f:
            rows = list(csv.reader(f))
        entry['centerline_csv'] = {'header': rows[0], 'rows': len(rows) - 1}
        golden[case] = entry

    Path(args.out).write_text(json.dumps(golden, indent=1, sort_keys=True) + '\n')
    print('wrote', args.out)


if __name__ == '__main__':
    main()
