# PROGRESS

## NEXT STEP
Linux session: read `docs/HANDOFF.md`, get the regression data pack from the author, then start Task 1 of `docs/plans/2026-09-30-alr-implementation-plan.md` (environment file and package skeleton) and Task 2 (baseline on Linux).

## LAST SESSION
2026-09-30, on the author's Mac. Wrote the design spec, the 20 task implementation plan, the handoff note and the tools to reproduce the stored outputs. No package code exists yet.

## STATUS
| Work item | Stage |
|---|---|
| Design specification | written, decisions recorded, one question open (author email) |
| Implementation plan | written, 20 tasks |
| Handoff note, original scripts in `reference/`, regression tools and golden file | done |
| Baseline of the original scripts on the Mac | done, 24 of 24 comparisons byte identical |
| Baseline on Linux (Task 2) | not started |
| Package code and tests (Tasks 1, 3 to 19) | not started |
| Clean machine acceptance (Task 20) | not started |
| Publication | repository is public at https://github.com/kiwixiao/ALR, branch `feature/port-airway-chain` |

## SESSION LOG
- 2026-09-30 (handoff): added `reference/av_phenotype/` (the ten stage scripts, two runners, the surface script with its test, the QA script, the TotalSegmentator notes), `tools/reproduce_reference.py`, `tools/pack_regression_data.sh`, `tools/make_golden.py`, `tests/regression/golden.json`, `docs/baseline.md`, `docs/HANDOFF.md` and `docs/plans/2026-09-30-alr-implementation-plan.md`. Ran `tools/reproduce_reference.py` on the Mac for CF121, CF008, NL001 and CF005: the integrity check passed and all six compared artifacts of every case are byte identical. Found that GLB and PDF export are commented out in `export_airway_3d.py`, so `pymeshlab` and `trimesh` are dropped and `kimimaro` is the only GPL dependency. Found that the original runner skips step 1 whenever the airway mask exists, so a missing lobe mask is never made; ALR changes that one rule. The synthetic tree values in the unit tests come from running the original generations and Strahler scripts on a 14 vertex tree.
- 2026-09-30: created the repository and wrote the spec. Facts checked for it: the airway scripts read only the CT and two TotalSegmentator outputs; the four models ALR needs (117, 291, 297, 298) are not licensed models in TotalSegmentator 2.13.0; `kimimaro` is GPL 3 or later; the isotropic mask equals a resample of the TotalSegmentator mask for CF121, CF008, NL001 and CF005; CF001 does not (Dice 0.851) and is excluded. Docker Desktop is installed on the Mac for a container test.
- 2026-09-30 (later): the author decided: MIT, no CCHMC approval needed, copyright line as written, usage statistics off, keep the artery, vein and wall files, no demo CT, ALR means airway lung ratio.
