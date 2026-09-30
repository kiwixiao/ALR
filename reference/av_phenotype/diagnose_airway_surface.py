#!/usr/bin/env python
"""QA for compute_airway_surface.py: does each airway STL sit on its mask and
around its TEASAR centerline?

Per case (console, one row each):
  bounds_dmax_mm  largest |surface bound - centerline bound| over the 6 bounds
  d_over_r_med    median over centerline vertices of (distance to surface / Radius);
                  about 1 when the surface is the lumen wall the radius was measured to
  d_over_r_p05/p95
  open_edges      boundary edges (0 = closed; the trachea cut at the scan edge opens it)
  vol_ratio       mesh volume / mask voxel volume (only meaningful when closed)
  triangles, MB

--png DIR writes one render per case: surface translucent, centerline by Generation.

Usage:
    python diagnose_airway_surface.py --cases CF001 CF008 --png /tmp/qc
    python diagnose_airway_surface.py --share CF_airway_for_share       # every case in the bundle
"""
import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
import pyvista as pv

ROOT = Path(__file__).resolve().parent


def case_paths(case):
    d = ROOT / 'final_analysis' / case / case
    return (d / 'airway_results' / f'{case}_airway_surface.stl',
            d / 'airway_results' / 'lung_airways_teasar.vtk',
            d / 'lung_airways_iso.nii.gz')


def measure(case):
    stl_p, cl_p, mask_p = case_paths(case)
    surf, cl = pv.read(stl_p), pv.read(cl_p)
    dmax = float(np.abs(np.array(surf.bounds) - np.array(cl.bounds)).max())

    _, closest = surf.find_closest_cell(cl.points, return_closest_point=True)
    d_over_r = np.linalg.norm(closest - cl.points, axis=1) / np.asarray(cl.point_data['Radius'])

    clean = surf.clean()
    open_edges = clean.extract_feature_edges(boundary_edges=True, feature_edges=False,
                                             manifold_edges=False, non_manifold_edges=False).n_cells
    mask = nib.load(mask_p)
    voxel_volume = float(np.count_nonzero(np.asarray(mask.dataobj))) * float(np.prod(mask.header.get_zooms()[:3]))
    return dict(case=case, bounds_dmax_mm=dmax,
                d_over_r_med=float(np.median(d_over_r)),
                d_over_r_p05=float(np.percentile(d_over_r, 5)),
                d_over_r_p95=float(np.percentile(d_over_r, 95)),
                open_edges=open_edges, vol_ratio=clean.volume / voxel_volume,
                triangles=surf.n_cells, MB=stl_p.stat().st_size / 1e6)


def render(case, out_dir):
    stl_p, cl_p, _ = case_paths(case)
    surf, cl = pv.read(stl_p), pv.read(cl_p)
    p = pv.Plotter(off_screen=True, window_size=(1200, 1200))
    p.set_background('white')
    p.add_mesh(surf, color='lightsteelblue', opacity=0.35)
    p.add_mesh(cl, scalars='Generation', cmap='turbo', line_width=3,
               render_lines_as_tubes=True, scalar_bar_args={'title': 'Generation'})
    p.view_xz()                     # coronal
    p.reset_camera()
    p.camera.zoom(0.8)              # keep a margin on every side
    out = Path(out_dir) / f'{case}_airway_surface_qc.png'
    p.screenshot(str(out))
    p.close()
    return out


def main():
    ap = argparse.ArgumentParser(description='QA for airway surfaces.')
    ap.add_argument('--cases', nargs='*', default=[])
    ap.add_argument('--share', help='use every case folder in this bundle')
    ap.add_argument('--png', help='directory for renders')
    args = ap.parse_args()
    cases = list(args.cases)
    if args.share:
        cases += sorted(p.name for p in (ROOT / args.share).iterdir() if p.is_dir())

    cols = ['case', 'bounds_dmax_mm', 'd_over_r_med', 'd_over_r_p05', 'd_over_r_p95',
            'open_edges', 'vol_ratio', 'triangles', 'MB']
    print(','.join(cols))
    for case in cases:
        m = measure(case)
        print(','.join(f'{m[c]:.3f}' if isinstance(m[c], float) else str(m[c]) for c in cols))
        if args.png:
            Path(args.png).mkdir(parents=True, exist_ok=True)
            render(case, args.png)


if __name__ == '__main__':
    main()
