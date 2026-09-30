# Resample the airway lumen mask to 0.625 mm isotropic (matches the CIP iso
# grid used by all downstream centerline / radius / dysanapsis steps).
#
# Sources tried in order (first one found wins):
#   {CASE_DIR}/lung_airways.nii.gz                       (top-level symlink or copy)
#   {CASE_DIR}/totalseg_vessels/lung_airways.nii.gz      (canonical TS output)
#
# Output: {CASE_DIR}/lung_airways_iso.nii.gz
#
# Idempotent: skips if output is newer than the source.

import os
import numpy as np
import nibabel as nib
from nibabel.processing import resample_to_output

CASE = os.environ.get('CASE_ID', 'CF001')
CASE_DIR = os.environ.get(
    'CASE_DIR',
    f'/path/to/av_phenotype/final_analysis/{CASE}/{CASE}')
TARGET_VOXEL_MM = 0.625

SRC_CANDIDATES = [
    f'{CASE_DIR}/lung_airways.nii.gz',
    f'{CASE_DIR}/totalseg_vessels/lung_airways.nii.gz',
]
OUT = f'{CASE_DIR}/lung_airways_iso.nii.gz'


def needs_rerun(src, dst):
    if not os.path.exists(dst):
        return True
    return os.path.getmtime(dst) < os.path.getmtime(src)


def main():
    print(f'\n══════════════ airway mask iso-resample  ({CASE}) ══════════════')
    src = next((p for p in SRC_CANDIDATES
                if os.path.exists(p) and not os.path.islink(p)), None)
    if src is None:
        # also accept symlinks (e.g., {CASE_DIR}/lung_airways.nii.gz → totalseg_vessels/...)
        src = next((p for p in SRC_CANDIDATES if os.path.exists(p)), None)
    if src is None:
        print('  no airway mask found — run compute_totalseg_airway.py first. abort.')
        return

    if not needs_rerun(src, OUT):
        print(f'  ⤷ skip: {OUT} is up to date'); return

    print(f'  source : {src}')
    img = nib.load(src)
    print(f'  shape  : {img.shape}, voxel size: {img.header.get_zooms()[:3]}')

    iso = resample_to_output(img, voxel_sizes=[TARGET_VOXEL_MM]*3, order=0)
    iso = nib.Nifti1Image(np.asarray(iso.get_fdata()).astype(np.uint8), iso.affine)
    nib.save(iso, OUT)

    print(f'  out shape: {iso.shape}, voxel size: {TARGET_VOXEL_MM} mm iso')
    print(f'  wrote {OUT}')


if __name__ == '__main__':
    main()
    print('\n[Done]')
