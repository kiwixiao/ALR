# Smith 2020 dysanapsis index (CT-based).
#
# Reference: Smith BM et al. JAMA 2020;323(22):2268–2280.
# "Association of Dysanapsis With Chronic Obstructive Pulmonary Disease Among Older Adults."
#
# Definition (Smith 2020):
#   dysanapsis_index = ln(D_geom_mm) − 0.341 · ln(V_lung_mL)
#
# where
#   D_geom_mm = geometric mean of central airway luminal diameters (Smith
#               sampled 10 standardized anatomic landmarks: trachea, R/L main,
#               R/L upper lobar, bronchus intermedius, R middle lobar, R/L
#               lower lobar, lingular bronchus). We approximate with BFS-
#               generation 0–3 vertices on the airway centerline (trachea +
#               main + lobar + segmental bronchi).
#   V_lung_mL = total lung volume at the scan's inspiratory state, computed
#               from the TotalSegmentator lobe multilabel (labels 10–14:
#               LUL, LLL, RUL, RML, RLL).
#   −0.341 is the slope coefficient Smith fit in healthy controls; we use it
#   as a fixed constant so our index is comparable to the published one.
#
# Lower index ⇒ smaller airways relative to lung volume ⇒ structural
# dysanapsis ⇒ associated with incident airflow obstruction in adults.

import os
import json
import numpy as np
import nibabel as nib
import pyvista as pv

CASE = os.environ.get('CASE_ID', 'NL001')
ROOT = os.environ.get(
    'CASE_DIR',
    f'/path/to/av_phenotype/final_analysis/{CASE}/{CASE}')
AIRWAY_DIR = f'{ROOT}/airway_results'
os.makedirs(AIRWAY_DIR, exist_ok=True)
AIR_VTK = f'{AIRWAY_DIR}/lung_airways_teasar.vtk'

# Try both lobe-mask filename conventions used in this project.
LOBE_CANDIDATES = [
    f'{ROOT}/lobes_ml.nii.gz',
    f'{ROOT}/totalseg_merged/lung_lobes_multilabel.nii.gz',
]

# Smith 2020 fixed slope coefficient
SMITH_COEFF = 0.341

# Generations included as "central airway" (main + lobar + segmental).
# We deliberately exclude Weibel gen 0: the BFS root sits at the carina
# (highest-radius bifurcation), and gen 0 contains the trachea-trunk vertices
# walked UPWARD from the carina; on cropped CT those upper-trachea verts
# narrow toward the field-of-view edge and bias the geometric mean down.
# Using gens 1–3 corresponds to main + lobar + segmental — the body of
# Smith 2020's standard 10 anatomic landmarks.
CENTRAL_GENS = (1, 2, 3)

OUT_CSV  = f'{AIRWAY_DIR}/{CASE}_smith_dysanapsis.csv'
OUT_JSON = f'{AIRWAY_DIR}/{CASE}_smith_dysanapsis.json'
OUT_PNG  = f'{AIRWAY_DIR}/{CASE}_smith_dysanapsis.png'

# Smith 2020 healthy-reference distribution (older-adult cohort, n>2,500).
# Median of the published healthy distribution sits near 0; the central 50% is
# roughly within ±0.13; lower values correspond to more structural dysanapsis
# and higher COPD risk. We use these as visual reference bands only.
SMITH_REF_MEDIAN = 0.0
SMITH_REF_IQR    = (-0.13, +0.13)
SMITH_REF_2SD    = (-0.40, +0.40)


def find_lobe_path():
    for p in LOBE_CANDIDATES:
        if os.path.exists(p):
            return p
    return None


def main():
    print(f'\n══════════════ Smith 2020 dysanapsis  ({CASE}) ══════════════')
    if not os.path.exists(AIR_VTK):
        print(f'  airway VTK not found at {AIR_VTK}; skipping dysanapsis.')
        return

    # ── Lung volume from lobe mask ────────────────────────────────────────
    lobe_path = find_lobe_path()
    if lobe_path is None:
        print('  no lobe mask found — abort.')
        return
    img = nib.load(lobe_path)
    arr = img.get_fdata().astype(np.int16)
    voxvol_mm3 = abs(np.linalg.det(img.affine[:3, :3]))
    nvox_lung = int((arr > 0).sum())
    V_lung_mL = nvox_lung * voxvol_mm3 / 1000.0
    print(f'  lobe mask: {lobe_path}')
    print(f'  voxel volume: {voxvol_mm3:.4f} mm³')
    print(f'  lung volume (Σ label>0): {V_lung_mL:.1f} mL')

    # Per-lobe breakdown (5-lobe TotalSegmentator labels)
    lobe_labels = {10: 'LUL', 11: 'LLL', 12: 'RUL', 13: 'RML', 14: 'RLL'}
    per_lobe_mL = {}
    for L, name in lobe_labels.items():
        v = int((arr == L).sum()) * voxvol_mm3 / 1000.0
        per_lobe_mL[name] = v
        print(f'    {name} ({L}): {v:6.1f} mL')

    # ── Airway central-generation diameters ──────────────────────────────
    cl = pv.read(AIR_VTK)
    if 'Generation' not in cl.point_data:
        print('  airway VTK has no Generation field. '
              'Run compute_airway_generations.py first. abort.')
        return
    radii = np.asarray(cl.point_data['Radius'])
    gens = np.asarray(cl.point_data['Generation'])

    # Restrict to central generations and require finite, positive radius.
    # Also EXCLUDE the trachea trunk (carina's superior child + its deg-2 chain),
    # which is BFS-tagged gen 1 but anatomically not a "main bronchus".
    is_trachea = (np.asarray(cl.point_data['Is_Trachea']).astype(bool)
                  if 'Is_Trachea' in cl.point_data
                  else np.zeros(len(radii), dtype=bool))
    mask_central = (np.isin(gens, CENTRAL_GENS)
                    & np.isfinite(radii) & (radii > 0)
                    & ~is_trachea)
    if mask_central.sum() == 0:
        print('  no central-generation vertices found. abort.')
        return

    diameters_mm = 2.0 * radii[mask_central]
    log_d = np.log(diameters_mm)
    D_geom_mm = float(np.exp(log_d.mean()))     # geometric mean = exp(mean of log)
    n_central = int(mask_central.sum())

    print(f'  central airway vertices (gen ∈ {CENTRAL_GENS}): {n_central}')
    print(f'  geometric-mean luminal diameter: {D_geom_mm:.3f} mm')

    # Per-generation breakdown for transparency
    print('  per-generation diameter (geom mean):')
    for g in CENTRAL_GENS:
        m = (gens == g) & np.isfinite(radii) & (radii > 0)
        if m.sum() == 0: continue
        d_g = float(np.exp(np.log(2.0 * radii[m]).mean()))
        print(f'    gen {g}: n={int(m.sum()):>4d}  geom-mean D = {d_g:.3f} mm')

    # ── Smith 2020 dysanapsis index ──────────────────────────────────────
    dysanapsis = float(np.log(D_geom_mm) - SMITH_COEFF * np.log(V_lung_mL))
    # Also report a lung-size-adjusted ratio for intuition:
    ratio_simple = D_geom_mm / (V_lung_mL ** (1.0 / 3.0))

    print()
    print(f'  Smith 2020 dysanapsis index = ln(D) − 0.341·ln(V) = '
          f'{np.log(D_geom_mm):.3f} − 0.341·{np.log(V_lung_mL):.3f} = {dysanapsis:+.4f}')
    print(f'  simple ratio D / V^(1/3): {ratio_simple:.4f}  '
          f'(D in mm, V in mL → V^(1/3) in mL^(1/3))')

    # Save
    rec = {
        'case_id': CASE,
        'V_lung_mL': V_lung_mL,
        'V_lung_per_lobe_mL': per_lobe_mL,
        'D_geom_mm_central': D_geom_mm,
        'n_central_vertices': n_central,
        'central_generations': list(CENTRAL_GENS),
        'smith_dysanapsis_index': dysanapsis,
        'simple_ratio_D_over_V13': ratio_simple,
        'smith_coeff': SMITH_COEFF,
    }
    with open(OUT_JSON, 'w') as f:
        json.dump(rec, f, indent=2)

    # CSV (single-row, easy to concat across cohort)
    import csv
    keys = ['case_id', 'V_lung_mL', 'D_geom_mm_central', 'n_central_vertices',
            'smith_dysanapsis_index', 'simple_ratio_D_over_V13']
    with open(OUT_CSV, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(keys)
        w.writerow([rec[k] for k in keys])
    print(f'\n  wrote {OUT_CSV}')
    print(f'  wrote {OUT_JSON}')

    # ── Plot: this case vs Smith 2020 healthy reference ────────────────────
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    fig.suptitle(f'{CASE} — Smith 2020 dysanapsis index', fontsize=13, fontweight='bold')

    # Panel A: the index on a horizontal axis with reference bands
    ax = axes[0]
    ax.axhspan(SMITH_REF_2SD[0], SMITH_REF_2SD[1], color='#bdc3c7', alpha=0.35,
               label='Smith healthy ~95% range')
    ax.axhspan(SMITH_REF_IQR[0], SMITH_REF_IQR[1], color='#7f8c8d', alpha=0.45,
               label='Smith healthy IQR')
    ax.axhline(SMITH_REF_MEDIAN, color='black', lw=1.0, ls='--', alpha=0.7,
               label='Smith healthy median')
    ax.scatter([0], [dysanapsis], s=180, c='#c0392b', edgecolor='white',
               zorder=5, label=f'{CASE}: {dysanapsis:+.3f}')
    ax.set_xlim(-0.5, 0.5); ax.set_xticks([])
    ax.set_ylim(min(-0.6, dysanapsis - 0.1), 0.6)
    ax.set_ylabel('Smith dysanapsis index  (ln D − 0.341 · ln V)')
    ax.set_title('A. Where this case sits vs healthy reference')
    ax.grid(axis='y', alpha=0.3)
    ax.legend(loc='lower right', fontsize=9)

    # Panel B: per-generation diameter bars
    ax = axes[1]
    bar_g = []; bar_d = []; bar_n = []
    for g in CENTRAL_GENS:
        m = (gens == g) & np.isfinite(radii) & (radii > 0)
        if m.sum() == 0: continue
        bar_g.append(f'gen {g}\nn={int(m.sum())}')
        bar_d.append(float(np.exp(np.log(2.0 * radii[m]).mean())))
        bar_n.append(int(m.sum()))
    bars = ax.bar(bar_g, bar_d, color='#2980b9', alpha=0.75,
                  edgecolor='black', linewidth=0.6)
    for b, d in zip(bars, bar_d):
        ax.text(b.get_x() + b.get_width() / 2, d + 0.15, f'{d:.2f} mm',
                ha='center', va='bottom', fontsize=10)
    ax.axhline(D_geom_mm, color='#c0392b', lw=1.5, ls='--',
               label=f'geom-mean = {D_geom_mm:.2f} mm')
    ax.set_ylabel('luminal diameter (mm, geom-mean)')
    ax.set_title(f'B. Central airway diameters used\n'
                 f'lung volume = {V_lung_mL:.0f} mL,  index = {dysanapsis:+.3f}')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout(); plt.subplots_adjust(top=0.86)
    plt.savefig(OUT_PNG, dpi=150, bbox_inches='tight')
    print(f'  wrote {OUT_PNG}')


if __name__ == '__main__':
    main()
    print('\n[Done]')
