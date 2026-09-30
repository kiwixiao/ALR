# TEASAR-based AIRWAY skeletonization via kimimaro.
#
# Mirror of compute_skeleton_teasar.py (which handles arteries + veins) but
# applied to the airway lumen mask. Produces:
#   *_teasar.nii.gz  — voxelized airway skeleton
#   *_teasar.vtk     — VTK PolyData with lines + per-vertex Radius scalar (DT-derived)
#
# DT-derived radius on the airway lumen = inner-bronchus radius. That radius,
# paired with the matched accompanying artery radius, is the input to the
# A:B (artery-to-bronchus) dysanapsis ratio.
#
# Inputs and outputs follow the same conventions as compute_skeleton_teasar.py
# so that downstream generation / Strahler / pairing scripts can re-use the
# vessel utilities with only a path swap.

import os, time
import numpy as np
import nibabel as nib
import kimimaro
from scipy.ndimage import label as nd_label, generate_binary_structure
import networkx as nx
import vtk
from vtk.util import numpy_support as nps

CASE = os.environ.get('CASE_ID', 'NL001')

CASE_DIR = os.environ.get(
    'CASE_DIR',
    f'/path/to/av_phenotype/final_analysis/{CASE}/{CASE}')
AIRWAY_DIR  = f'{CASE_DIR}/airway_results'
os.makedirs(AIRWAY_DIR, exist_ok=True)

AIR_NII     = f'{CASE_DIR}/lung_airways_iso.nii.gz'    # shared input (also used by visual_effect)
AIR_NII_OUT = f'{AIRWAY_DIR}/lung_airways_teasar.nii.gz'
AIR_VTK_OUT = f'{AIRWAY_DIR}/lung_airways_teasar.vtk'

# TEASAR parameters — same starting point as the vessel script.
# Airway radii span a wider range (trachea ~10 mm down to sub-mm in distal),
# but kimimaro's distance-field-Dijkstra adapts automatically.
TEASAR_PARAMS = {
    'scale':          1.5,
    'const':          10,
    'pdrf_exponent':  4,
    'pdrf_scale':     100000,
}
DUST_THRESHOLD = 200          # ignore connected components < 200 voxels
ANISOTROPY     = (0.625, 0.625, 0.625)   # iso-resampled mask is 0.625 mm isotropic

CONN26 = generate_binary_structure(3, 3)


def run():
    print(f'\n══════════════ AIRWAY ({CASE}) ══════════════')
    if not os.path.exists(AIR_NII):
        print(f'  iso airway mask not found at {AIR_NII}')
        print('  → skipping airway pipeline for this case '
              '(no error; downstream airway stages will also no-op).')
        return
    img = nib.load(AIR_NII)
    mask = img.get_fdata().astype(np.uint8)
    n_mask = int(mask.sum())
    print(f'  input mask voxels:  {n_mask}')
    if n_mask == 0:
        print('  empty mask — abort.'); return

    # Drop tiny disconnected components (not the trachea-bronchial tree)
    cc, ncc = nd_label(mask, CONN26)
    sizes = np.bincount(cc.ravel())
    if len(sizes) > 1:
        sizes[0] = 0
        keep = np.where(sizes >= DUST_THRESHOLD)[0]
        cleaned = np.isin(cc, keep).astype(np.uint8)
        print(f'  components: {ncc};  kept {len(keep)} (>= {DUST_THRESHOLD} vox);  '
              f'voxels after dust removal: {int(cleaned.sum())}')
    else:
        cleaned = mask

    # ── TEASAR ─────────────────────────────────────────────────────────────
    t0 = time.time()
    skels = kimimaro.skeletonize(
        cleaned,
        teasar_params=TEASAR_PARAMS,
        anisotropy=ANISOTROPY,
        dust_threshold=DUST_THRESHOLD,
        progress=False,
        parallel=1,
    )
    print(f'  kimimaro returned {len(skels)} skeleton object(s)  '
          f'in {time.time()-t0:.1f}s')

    if len(skels) == 0:
        print('  no skeletons produced — abort.'); return

    skel = kimimaro.join_close_components(list(skels.values()), radius=2.0)
    print(f'  joined skeleton: {len(skel.vertices)} vertices, '
          f'{len(skel.edges)} edges')

    radii = getattr(skel, 'radii', None)
    if radii is not None:
        print(f'  radii (mm) — min/median/max: '
              f'{radii.min():.2f}/{np.median(radii):.2f}/{radii.max():.2f}')

    # ── Per-vertex degree (endpoint=1, body=2, bifurcation>=3) ─────────────
    G = nx.Graph()
    G.add_nodes_from(range(len(skel.vertices)))
    G.add_edges_from(skel.edges.tolist())
    degs = np.array([G.degree(i) for i in range(len(skel.vertices))], dtype=np.int32)

    # ── Voxelize skeleton back to NIfTI for QC ─────────────────────────────
    aniso = np.array(ANISOTROPY)
    vox_idx = np.round(skel.vertices / aniso).astype(np.int64)
    sx, sy, sz = mask.shape
    vox_idx[:, 0] = np.clip(vox_idx[:, 0], 0, sx - 1)
    vox_idx[:, 1] = np.clip(vox_idx[:, 1], 0, sy - 1)
    vox_idx[:, 2] = np.clip(vox_idx[:, 2], 0, sz - 1)
    skel_vol = np.zeros_like(mask, dtype=np.uint8)
    skel_vol[vox_idx[:, 0], vox_idx[:, 1], vox_idx[:, 2]] = 1
    skel_img = nib.Nifti1Image(skel_vol, img.affine)
    skel_img.set_qform(img.affine, code=1)
    skel_img.set_sform(img.affine, code=1)
    nib.save(skel_img, AIR_NII_OUT)
    print(f'  wrote {AIR_NII_OUT}')

    # ── VTK PolyData (world coords; per-vertex Radius + Degree) ────────────
    vox_continuous = skel.vertices / aniso
    homog = np.c_[vox_continuous, np.ones(len(vox_continuous))]
    world = (img.affine @ homog.T).T[:, :3]

    points = vtk.vtkPoints()
    for v in world:
        points.InsertNextPoint(float(v[0]), float(v[1]), float(v[2]))

    lines = vtk.vtkCellArray()
    for e in skel.edges:
        ln = vtk.vtkLine()
        ln.GetPointIds().SetId(0, int(e[0]))
        ln.GetPointIds().SetId(1, int(e[1]))
        lines.InsertNextCell(ln)

    pd = vtk.vtkPolyData()
    pd.SetPoints(points)
    pd.SetLines(lines)

    if radii is not None:
        rad_arr = nps.numpy_to_vtk(radii.astype(np.float32), deep=True)
        rad_arr.SetName('Radius')
        pd.GetPointData().AddArray(rad_arr)
        pd.GetPointData().SetActiveScalars('Radius')

    deg_arr = nps.numpy_to_vtk(degs.astype(np.int32), deep=True)
    deg_arr.SetName('Degree')
    pd.GetPointData().AddArray(deg_arr)

    writer = vtk.vtkPolyDataWriter()
    writer.SetFileName(AIR_VTK_OUT)
    writer.SetInputData(pd)
    writer.SetFileTypeToASCII()
    writer.Write()
    print(f'  wrote {AIR_VTK_OUT}')


if __name__ == '__main__':
    run()
    print('\nDone.')
