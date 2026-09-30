# PROGRESS

## NEXT STEP
The author reviews `docs/specs/2026-09-30-alr-design.md` and answers its seven open questions (section 13). After approval, write the implementation plan, whose first task is to show that the current `av_phenotype` scripts reproduce the stored outputs of CF121, CF008, NL001 and CF005 from the stored masks.

## LAST SESSION
2026-09-30. Wrote the design specification for ALR and created this repository on the branch `feature/port-airway-chain`. Nothing is pushed.

## STATUS
| Work item | Stage |
|---|---|
| Design specification | written, awaiting author review |
| Implementation plan | not started, waits for spec approval |
| Package code and tests | not started |
| Linux container test | not started |
| GitHub repository `kiwixiao/ALR` | not created, name not yet checked |

## SESSION LOG
- 2026-09-30: created the repository and wrote the spec. Facts checked for it: the airway scripts read only the CT and two TotalSegmentator outputs; the four models ALR needs (117, 291, 297, 298) are not licensed models in TotalSegmentator 2.13.0; `kimimaro` is GPL 3 or later and `pymeshlab` is GPL 3, imported only in the 3D PDF export; the isotropic mask equals a resample of the TotalSegmentator mask for CF121, CF008, NL001 and CF005; CF001 does not (Dice 0.851) and is excluded from regression. Docker Desktop is installed for the Linux container test.
