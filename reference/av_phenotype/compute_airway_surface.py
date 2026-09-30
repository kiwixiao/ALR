#!/usr/bin/env python
"""Airway lumen surface (binary STL) from the isotropic airway mask.

Input : {CASE_DIR}/lung_airways_iso.nii.gz            (0.625 mm isotropic, binary)
Output: {CASE_DIR}/airway_results/{CASE}_airway_surface.stl

Method, identical to mask_to_stl() in compute_mask_stl.py (copied rather than
imported, because importing that script runs its artery/vein loop):
  1. marching cubes (vtkContourFilter) at iso = 0.5
  2. Taubin smoothing (vtkWindowedSincPolyDataFilter), 30 iterations,
     pass band 0.05; non shrinking, so lumen calibre is preserved
  3. binary STL, vertices in world mm

The image is placed in world space from the nibabel affine (translation and
voxel size). vtkNIFTIImageReader is avoided on purpose: it drops the NIfTI
translation (see PROGRESS.md, 2026-08-31). The origin plus spacing placement is
exact only for an axis aligned, unflipped affine, so any other affine is
rejected instead of producing a silently mirrored surface.

The surface uses the same mask and the same world frame as the TEASAR
centerline (airway_results/lung_airways_teasar.vtk), so the two overlay with no
transform.

Usage:
    python compute_airway_surface.py --case CF001
    CASE_ID=CF001 python compute_airway_surface.py
"""
import os
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import vtk
from vtk.util import numpy_support as nps

TAUBIN_ITER = 30
TAUBIN_PASS_BAND = 0.05


def _is_axis_aligned(affine):
    m = affine[:3, :3]
    off_diagonal = m - np.diag(np.diag(m))
    return np.abs(off_diagonal).max() < 1e-6 and np.all(np.diag(m) > 0)


def mask_to_surface(src_path, dst_path):
    """Write the smoothed iso = 0.5 surface of a binary mask as binary STL.

    Returns the triangle count, or None when dst is newer than src (skipped).
    """
    src_path, dst_path = Path(src_path), Path(dst_path)
    if dst_path.exists() and dst_path.stat().st_mtime >= src_path.stat().st_mtime:
        return None

    nii = nib.load(src_path)
    if not _is_axis_aligned(nii.affine):
        raise ValueError(f'{src_path}: affine is not axis aligned with positive '
                         f'voxel sizes, surface would be misplaced:\n{nii.affine}')
    arr = np.asarray(nii.dataobj).astype(np.uint8)

    img = vtk.vtkImageData()
    img.SetDimensions(*arr.shape)
    img.SetSpacing(*[float(s) for s in nii.header.get_zooms()[:3]])
    img.SetOrigin(*[float(o) for o in nii.affine[:3, 3]])
    # numpy (i, j, k) with i fastest in Fortran order matches VTK point order
    scalars = nps.numpy_to_vtk(arr.flatten(order='F'), deep=True,
                               array_type=vtk.VTK_UNSIGNED_CHAR)
    scalars.SetName('mask')
    img.GetPointData().SetScalars(scalars)

    contour = vtk.vtkContourFilter()
    contour.SetInputData(img)
    contour.SetValue(0, 0.5)
    contour.ComputeNormalsOff()

    smoother = vtk.vtkWindowedSincPolyDataFilter()
    smoother.SetInputConnection(contour.GetOutputPort())
    smoother.SetNumberOfIterations(TAUBIN_ITER)
    smoother.SetPassBand(TAUBIN_PASS_BAND)
    smoother.NormalizeCoordinatesOn()
    smoother.NonManifoldSmoothingOff()
    smoother.BoundarySmoothingOff()
    smoother.Update()

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    writer = vtk.vtkSTLWriter()
    writer.SetInputConnection(smoother.GetOutputPort())
    writer.SetFileName(str(dst_path))
    writer.SetFileTypeToBinary()
    writer.Write()
    return smoother.GetOutput().GetNumberOfCells()


def _parse_case():
    if '--case' in sys.argv:
        i = sys.argv.index('--case')
        if i + 1 >= len(sys.argv):
            sys.exit('error: --case requires a value (e.g. --case CF003)')
        return sys.argv[i + 1]
    return os.environ.get('CASE_ID', 'CF001')


def main():
    case = _parse_case()
    root = Path(__file__).resolve().parent
    case_dir = Path(os.environ.get('CASE_DIR', root / 'final_analysis' / case / case))
    src = case_dir / 'lung_airways_iso.nii.gz'
    dst = case_dir / 'airway_results' / f'{case}_airway_surface.stl'
    if not src.exists():
        sys.exit(f'{case}: missing input {src}')
    n_tri = mask_to_surface(src, dst)
    if n_tri is None:
        print(f'{case}: skip, {dst.name} is up to date')
    else:
        print(f'{case}: {dst.name}  {n_tri} triangles  {dst.stat().st_size / 1e6:.1f} MB')


if __name__ == '__main__':
    main()
