# TotalSegmentator multi-label lobe pipeline

The cleanest single-command way to get a 5-lobe multi-label NIfTI from a chest CT, using the `totalseg213` conda env (TotalSegmentator v2.13.0, MPS-enabled).

## Command

```bash
TotalSegmentator -i ct.nii.gz -o lobes_ml.nii.gz \
  -ta total -ml -rmb -d mps \
  -rs lung_upper_lobe_left lung_lower_lobe_left lung_upper_lobe_right \
      lung_middle_lobe_right lung_lower_lobe_right
```

Run inside the env: `conda activate totalseg213` first, or prefix with `conda run -n totalseg213`.

## What each flag does

| Flag | Effect |
|---|---|
| `-ta total` | use the default v2 `total` task (the only task that contains all 5 lobe classes) |
| `-ml` | output ONE multi-label NIfTI instead of 5 separate binary files |
| `-rmb` | remove per-class connected components <0.2 mL (~819 vox at 0.625 mm); cleans noise |
| `-d mps` | use Apple Metal GPU (5–10× faster than CPU on M-series) |
| `-rs <5 lobes>` | filter output to only the 5 lobe classes; underlying nnUNet still runs all 117 in one forward pass |

## Output label legend (TotalSegmentator native IDs)

| Value | Lobe |
|---|---|
| 10 | lung_upper_lobe_left  (LUL) |
| 11 | lung_lower_lobe_left  (LLL) |
| 12 | lung_upper_lobe_right (RUL) |
| 13 | lung_middle_lobe_right (RML) |
| 14 | lung_lower_lobe_right (RLL) |

## Companion command (vessels + airways, full main → distal)

```bash
TotalSegmentator -i ct.nii.gz -o vessels/ -ta lung_vessels -d mps
```

Outputs 4 binary NIfTIs in `vessels/`:
- `lung_arteries.nii.gz` — pulmonary arteries (main + distal, per AirRC training)
- `lung_veins.nii.gz` — pulmonary veins (main + distal)
- `lung_airways.nii.gz` — airway lumen (trachea + bronchi + distal)
- `lung_airways_wall.nii.gz` — airway wall

## Pitfalls to avoid

- `-ta total_v1` is **not** a valid CLI task — it's only a Python dict key. CLI rejects it.
- `--roi_subset` works **only** for `total` and `total_mr` tasks. Other tasks reject it.
- v2 `total` task does NOT have `pulmonary_artery` (only `pulmonary_vein`). Don't try to add it to `-rs`.
- `heartchambers_highres` (the only other task with `pulmonary_artery`) requires a non-commercial license key — and you don't need it: Task 117's `lung_arteries` already covers main + distal per the AirRC paper (Nature Sci Data 2025).
- Bronchial arteries are NOT segmented by any TotalSegmentator task.

## Validated on (Apr 2026)

| Case | CT shape | Total lung | Runtime on MPS |
|---|---|---|---|
| CF001 | 445³ | ~3.0 L | ~30 s |
| CF005 | 545×545×443 | 5.1 L | ~42 s |
| NL001 | 640×640×545 | 6.1 L | ~51 s |

## Known limitation

The default v2 `total` task and lungmask both show ~0.80 Dice on diseased lungs (Eur Radiol Exp 2025, PMC12411369), with the **right oblique fissure as the dominant failure mode** for both. RUL/RLL split disagrees by up to ~1 L of voxels between TS and lungmask on CF005. For CF cohorts, validate the right fissure visually on a few cases before trusting downstream lobar metrics.

## Source files (in installed package)

- `~/anaconda3/envs/totalseg213/lib/python3.10/site-packages/totalsegmentator/python_api.py` — task configs
- `~/anaconda3/envs/totalseg213/lib/python3.10/site-packages/totalsegmentator/map_to_binary.py` — class label maps
- Task 117 weights: `~/.totalsegmentator/nnunet/results/Dataset117_lung_airways_arteries_veins_282subj/`
