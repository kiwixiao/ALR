# Dump the airway TEASAR centerline to a per-vertex CSV — the "raw data"
# table for any downstream analysis that doesn't want to parse VTK.
#
# Per-vertex columns:
#   vertex_id, x_mm, y_mm, z_mm,
#   Radius_mm, Diameter_mm, Degree,
#   Generation, Generation_Type, Strahler_Order,
#   Is_Root, Distance_From_Root_mm,
#   Lobe, Lobe_Name,
#   Lobe_Topology, Lobe_Topology_Name
#
# Case-level columns (repeated on every row for easy pandas/SQL grouping):
#   case_id,
#   V_lung_mL, V_LUL_mL, V_LLL_mL, V_RUL_mL, V_RML_mL, V_RLL_mL,
#   smith_dysanapsis_index, simple_ratio_D_over_V13
#
# Output: {AIRWAY_DIR}/{CASE}_airway_centerline.csv

import os
import json
import numpy as np
import pyvista as pv

CASE = os.environ.get('CASE_ID', 'CF001')
ROOT = os.environ.get(
    'CASE_DIR',
    f'/path/to/av_phenotype/final_analysis/{CASE}/{CASE}')
AIRWAY_DIR = f'{ROOT}/airway_results'
os.makedirs(AIRWAY_DIR, exist_ok=True)

AIR_VTK = f'{AIRWAY_DIR}/lung_airways_teasar.vtk'
DYSANAPSIS_JSON = f'{AIRWAY_DIR}/{CASE}_smith_dysanapsis.json'
OUT_CSV = f'{AIRWAY_DIR}/{CASE}_airway_centerline.csv'

LOBE_NAMES = {0: 'extra_lobar',
              10: 'LUL', 11: 'LLL',
              12: 'RUL', 13: 'RML', 14: 'RLL'}


def main():
    print(f'\n══════════════ airway centerline → CSV  ({CASE}) ══════════════')
    if not os.path.exists(AIR_VTK):
        print(f'  airway VTK not found at {AIR_VTK}; skipping CSV export.')
        return

    cl = pv.read(AIR_VTK)
    pts = np.asarray(cl.points)
    n = len(pts)
    pd_obj = cl.point_data

    # ── pull per-vertex arrays (with safe fallback if a field is missing) ──
    def get_arr(field, dtype=np.float32, default=np.nan):
        if field in pd_obj:
            return np.asarray(pd_obj[field])
        return np.full(n, default, dtype=dtype)

    radius = get_arr('Radius', np.float32, default=np.nan)
    degree = get_arr('Degree', np.int32, default=-1).astype(np.int32)
    generation = get_arr('Generation', np.int32, default=-1).astype(np.int32)
    gen_type = (np.asarray(pd_obj['Generation_Type'])
                if 'Generation_Type' in pd_obj
                else np.array([''] * n, dtype='U18'))
    strahler = get_arr('Strahler_Order', np.int32, default=-1).astype(np.int32)
    is_root = get_arr('Is_Root', np.int32, default=0).astype(np.int32)
    dist = get_arr('Distance_From_Root_mm', np.float32, default=np.nan)
    lobe = get_arr('Lobe', np.int32, default=-1).astype(np.int32)
    lobe_name = (np.asarray(pd_obj['Lobe_Name'])
                 if 'Lobe_Name' in pd_obj
                 else np.array([''] * n, dtype='U12'))
    lobe_topo = get_arr('Lobe_Topology', np.int32, default=-1).astype(np.int32)
    lobe_topo_name = (np.asarray(pd_obj['Lobe_Topology_Name'])
                      if 'Lobe_Topology_Name' in pd_obj
                      else np.array([''] * n, dtype='U12'))
    is_trachea = get_arr('Is_Trachea', np.int32, default=0).astype(np.int32)

    # ── case-level metadata from dysanapsis JSON if present ────────────────
    V_lung = np.nan
    per_lobe = {n_: np.nan for n_ in ['LUL', 'LLL', 'RUL', 'RML', 'RLL']}
    smith_idx = np.nan
    simple_ratio = np.nan
    if os.path.exists(DYSANAPSIS_JSON):
        with open(DYSANAPSIS_JSON) as f:
            j = json.load(f)
        V_lung = float(j.get('V_lung_mL', np.nan))
        smith_idx = float(j.get('smith_dysanapsis_index', np.nan))
        simple_ratio = float(j.get('simple_ratio_D_over_V13', np.nan))
        for k, v in (j.get('V_lung_per_lobe_mL') or {}).items():
            per_lobe[k] = float(v)
    print(f'  per-vertex rows: {n}')
    print(f'  V_lung_mL = {V_lung:.1f}  (per-lobe: '
          f"LUL={per_lobe['LUL']:.0f}, LLL={per_lobe['LLL']:.0f}, "
          f"RUL={per_lobe['RUL']:.0f}, RML={per_lobe['RML']:.0f}, "
          f"RLL={per_lobe['RLL']:.0f} mL)")
    print(f'  smith_dysanapsis_index = {smith_idx:+.4f}')

    # ── write CSV (no pandas dependency to keep it light) ──────────────────
    cols = [
        'case_id', 'vertex_id', 'x_mm', 'y_mm', 'z_mm',
        'Radius_mm', 'Diameter_mm', 'Degree',
        'Generation', 'Generation_Type', 'Strahler_Order',
        'Is_Root', 'Is_Trachea', 'Distance_From_Root_mm',
        'Lobe', 'Lobe_Name',
        'Lobe_Topology', 'Lobe_Topology_Name',
        'V_lung_mL', 'V_LUL_mL', 'V_LLL_mL', 'V_RUL_mL', 'V_RML_mL', 'V_RLL_mL',
        'smith_dysanapsis_index', 'simple_ratio_D_over_V13',
    ]

    with open(OUT_CSV, 'w') as f:
        f.write(','.join(cols) + '\n')
        for i in range(n):
            row = [
                CASE,
                i,
                f'{pts[i, 0]:.4f}',
                f'{pts[i, 1]:.4f}',
                f'{pts[i, 2]:.4f}',
                f'{radius[i]:.4f}' if np.isfinite(radius[i]) else '',
                f'{2.0 * radius[i]:.4f}' if np.isfinite(radius[i]) else '',
                int(degree[i]),
                int(generation[i]),
                str(gen_type[i]),
                int(strahler[i]),
                int(is_root[i]),
                int(is_trachea[i]),
                f'{dist[i]:.4f}' if np.isfinite(dist[i]) else '',
                int(lobe[i]),
                str(lobe_name[i]),
                int(lobe_topo[i]),
                str(lobe_topo_name[i]),
                f'{V_lung:.4f}' if np.isfinite(V_lung) else '',
                f'{per_lobe["LUL"]:.4f}' if np.isfinite(per_lobe["LUL"]) else '',
                f'{per_lobe["LLL"]:.4f}' if np.isfinite(per_lobe["LLL"]) else '',
                f'{per_lobe["RUL"]:.4f}' if np.isfinite(per_lobe["RUL"]) else '',
                f'{per_lobe["RML"]:.4f}' if np.isfinite(per_lobe["RML"]) else '',
                f'{per_lobe["RLL"]:.4f}' if np.isfinite(per_lobe["RLL"]) else '',
                f'{smith_idx:.6f}' if np.isfinite(smith_idx) else '',
                f'{simple_ratio:.6f}' if np.isfinite(simple_ratio) else '',
            ]
            f.write(','.join(str(x) for x in row) + '\n')
    print(f'  wrote {OUT_CSV}  ({n} rows × {len(cols)} cols)')


if __name__ == '__main__':
    main()
    print('\n[Done]')
