# Baseline: do the original scripts reproduce the stored outputs?

The question is answered per machine with `python tools/reproduce_reference.py --data DATA`. The tool runs steps 2 to 10 of the original `av_phenotype` scripts (copied in `reference/av_phenotype/`) on the stored TotalSegmentator masks and compares the result with the stored outputs.

## macOS, arm64 (the author's Mac), 2026-09-30

Platform macOS 26.5.2 arm64, Python 3.10.19, conda base environment. Versions: kimimaro 5.8.1, vtk 9.6.0, pyvista 0.47.1, nibabel 5.4.0, numpy 2.2.6, scipy 1.15.3, networkx 3.4.2, matplotlib 3.10.8, plotly 6.5.2. These are the starting pins of `environment.yml`.

Result: the integrity check of the data pack passed, and for CF121, CF008, NL001 and CF005 every artifact was reproduced, 24 of 24 comparisons byte identical:

| Artifact | Result |
|---|---|
| `lung_airways_iso.nii.gz` | byte identical in 4 of 4 cases |
| `airway_results/lung_airways_teasar.nii.gz` | byte identical in 4 of 4 cases |
| `airway_results/lung_airways_teasar.vtk` | byte identical in 4 of 4 cases |
| `airway_results/<CASE>_airway_centerline.csv` | byte identical in 4 of 4 cases |
| `airway_results/<CASE>_smith_dysanapsis.csv` | byte identical in 4 of 4 cases |
| `airway_results/<CASE>_smith_dysanapsis.json` | byte identical in 4 of 4 cases |
| the two HTML viewers | written, not compared |

Consequence: on the Mac the stored outputs are fully reproducible from the stored masks, so a regression test may demand exact equality there.

## Linux

Not yet measured. Task 2 of the implementation plan adds the result here. Until then it is unknown whether `kimimaro`, `vtk` and `numpy` give identical skeletons on Linux.
