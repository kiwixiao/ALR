# Tag each AIRWAY centerline vertex with its lobe by sampling the
# TotalSegmentator 5-lobe multilabel mask at the vertex's voxel position.
#
# TotalSegmentator lobe codes (after our merge step):
#    10 = LUL  (left  upper lobe)
#    11 = LLL  (left  lower lobe)
#    12 = RUL  (right upper lobe)
#    13 = RML  (right middle lobe)
#    14 = RLL  (right lower lobe)
#     0 = extra-lobar (trachea + main bronchi run through the mediastinum,
#        which is correctly NOT inside any lobe mask)
#
# Adds to lung_airways_teasar.vtk:
#    Lobe        int    label as above
#    Lobe_Name   str    'extra_lobar' / 'LUL' / 'LLL' / 'RUL' / 'RML' / 'RLL'

import os
import numpy as np
import nibabel as nib
import pyvista as pv

CASE = os.environ.get('CASE_ID', 'NL001')
ROOT = os.environ.get(
    'CASE_DIR',
    f'/path/to/av_phenotype/final_analysis/{CASE}/{CASE}')
AIR_VTK = f'{ROOT}/airway_results/lung_airways_teasar.vtk'

LOBE_CANDIDATES = [
    f'{ROOT}/lobes_ml.nii.gz',
    f'{ROOT}/totalseg_merged/lung_lobes_multilabel.nii.gz',
]
LOBE_NAMES = {0: 'extra_lobar',
              10: 'LUL', 11: 'LLL', 12: 'RUL', 13: 'RML', 14: 'RLL'}


def main():
    print(f'\n══════════════ airway lobe labels  ({CASE}) ══════════════')
    if not os.path.exists(AIR_VTK):
        print(f'  airway VTK not found at {AIR_VTK}; skipping airway lobe labels.')
        return
    lobe_path = next((p for p in LOBE_CANDIDATES if os.path.exists(p)), None)
    if lobe_path is None:
        print('  no lobe mask found — skipping (Lobe field will not be set).')
        return
    print(f'  lobe mask: {lobe_path}')

    img = nib.load(lobe_path)
    arr = img.get_fdata().astype(np.int16)
    inv = np.linalg.inv(img.affine)         # world → voxel

    cl = pv.read(AIR_VTK)
    pts = np.asarray(cl.points)
    n = len(pts)

    # World coords (mm) → voxel indices (round to nearest)
    homog = np.c_[pts, np.ones(n)]
    vox = (inv @ homog.T).T[:, :3]
    vox = np.round(vox).astype(np.int64)
    sx, sy, sz = arr.shape
    vox[:, 0] = np.clip(vox[:, 0], 0, sx - 1)
    vox[:, 1] = np.clip(vox[:, 1], 0, sy - 1)
    vox[:, 2] = np.clip(vox[:, 2], 0, sz - 1)
    lobes = arr[vox[:, 0], vox[:, 1], vox[:, 2]].astype(np.int32)

    # Reporting
    uniq, cnt = np.unique(lobes, return_counts=True)
    print('  per-vertex lobe distribution:')
    for L, c in zip(uniq, cnt):
        name = LOBE_NAMES.get(int(L), f'unknown_{int(L)}')
        print(f'    {int(L):>3}  {name:>11s}: {int(c):>4d}  ({100*c/n:5.1f}%)')

    # Save
    cl.point_data['Lobe'] = lobes
    cl.point_data['Lobe_Name'] = np.array(
        [LOBE_NAMES.get(int(L), f'unknown_{int(L)}') for L in lobes],
        dtype='U12')
    cl.save(AIR_VTK, binary=False)
    print(f'  updated {AIR_VTK}  (added Lobe + Lobe_Name)')


if __name__ == '__main__':
    main()
    print('\n[Done]')
