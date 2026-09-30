# ALR design specification

Date: 2026-09-30. Status: draft for author review. Branch: `feature/port-airway-chain`.

## 1. Purpose and success criteria

ALR turns a chest CT into an airway lumen mask, an airway surface mesh and a labeled TEASAR centerline. The centerline carries Weibel generation, Horton Strahler order, lobe labels and the Smith dysanapsis index. ALR is a packaged port of the airway chain in the `av_phenotype` pipeline.

The author's requirement is a public repository on the author's GitHub that anyone can clone and run on Linux. One environment file installs our code and the pinned TotalSegmentator.

Success criteria:

1. On a clean Linux machine, these commands finish and write the files listed in section 6: `git clone`, `conda env create -f environment.yml`, `conda activate ALR`, `alr setup`, `alr run --ct scan.nii.gz --case-dir out/`.
2. On the author's Mac, steps 2 to 10 reproduce the stored `av_phenotype` outputs for the regression cases in section 10. Integer and text fields match exactly. Real numbers match within float32 tolerance.
3. No path, device name or conda environment name specific to the author's machine appears in the code.
4. The repository contains no patient data and no file derived from a patient scan.

## 2. Decisions already made

| Topic | Decision |
|---|---|
| Scope | The airway chain only, plus the airway surface as step 11. pythoncip is not part of ALR. The airway scripts read only the CT and two TotalSegmentator outputs (checked in the source on 2026-09-24). |
| Approach | Faithful port. The algorithms are copied unchanged. Only mechanical refactoring is allowed: functions instead of scripts, and one shared graph module. |
| Name | ALR for the repository, the conda environment and the distribution. The Python import name and the command are lowercase `alr`. |
| Environment | One conda environment. TotalSegmentator 2.13.0 is pinned. Its weights live inside the environment. |
| Publication | Public repository on the author's GitHub, MIT license. The push happens only when the author says so. |
| Source of truth | ALR becomes the source of truth after the regression tests pass. The hand over in `av_phenotype` is a separate step that needs the author's approval (section 11). |
| Input | A chest CT as a NIfTI file. |

## 3. Non goals

- DICOM conversion. The README points to `dcm2niix`.
- Vessel, artery and vein processing, particles and CFD.
- Any change to an algorithm or to the output file names and columns.
- Model training or changes to TotalSegmentator.
- A GPU is not required. CPU works and is slow.

## 4. Interface

```
alr run --ct FILE [--case-dir DIR] [--name NAME] [--force] [--device auto|gpu|mps|cpu]
alr run --case ID --root DIR [--force]
alr cohort PREFIX --root DIR [--force] [--limit N]
alr setup
alr check
alr --version
```

- `alr run --ct` is the ad hoc mode of today's `run_airway_pipeline.py`. The case directory defaults to the folder that contains the CT. The case name defaults to the CT file stem. If the CT is not already named `ct.nii.gz` in the case directory, ALR links it there under that name.
- `alr run --case` uses the cohort layout `ROOT/ID/ID/`, which is the layout under `final_analysis/` in `av_phenotype`.
- `alr cohort` ports `run_airway_cohort.py`. It runs one case at a time, records failures without stopping, and writes `PREFIX_airway_cohort_<timestamp>.csv` with the columns `case_id, status, runtime_s, error`.
- `alr setup` downloads the four model folders (section 7). `alr check` prints versions, the device, the weights folder and which model folders are present. It exits with a nonzero code if a model folder is missing.
- The flag `--case-dir` replaces the flag `--out` of the current runner.

## 5. Repository layout

```
ALR/
  LICENSE                 MIT
  README.md               four command quick start, requirements, outputs, citations
  pyproject.toml          package metadata, console command alr
  environment.yml         one environment, pinned versions
  CLAUDE.md  PROGRESS.md  .claude/memory/
  docs/specs/             this file
  src/alr/
    cli.py                argument parsing, the commands in section 4
    pipeline.py           ordered steps, skip rule, --force
    layout.py             case directory layout and every file name
    device.py             choice of cuda, mps or cpu
    weights.py            TotalSegmentator home folder, setup, check
    segment.py            step 1
    resample.py           step 2
    skeleton.py           step 3
    graph.py              VTK lines to graph, shared code
    generations.py        step 4
    strahler.py           step 5
    lobe_labels.py        step 6
    lobe_topology.py      step 7
    dysanapsis.py         step 8
    export_csv.py         step 9
    export_html.py        step 10
    surface.py            step 11
  tests/unit/  tests/regression/  tests/conftest.py
```

Each step module exposes one function, `run(case)`, where `case` holds the case name, the case directory, `force` and the device. A module does nothing when imported. The current scripts run code at import time and read `CASE_ID` and `CASE_DIR` from the environment, which makes them hard to test.

## 6. Steps and file contract

The source is the ten stage scripts in `av_phenotype`, 1,963 lines in total, plus `compute_airway_surface.py`.

| Step | Module | Source script | Reads | Writes | Skipped when |
|---|---|---|---|---|---|
| 1 | segment | `compute_totalseg_airway.py` | `ct.nii.gz` | `totalseg_vessels/lung_airways.nii.gz` and the other three files that task writes, `totalseg_merged/lung_lobes_multilabel.nii.gz`, link `lung_airways.nii.gz` | both mask files exist |
| 2 | resample | `compute_airway_mask_isotropic.py` | the airway mask | `lung_airways_iso.nii.gz` | it is newer than its source |
| 3 | skeleton | `compute_airway_skeleton_teasar.py` | `lung_airways_iso.nii.gz` | `airway_results/lung_airways_teasar.{nii.gz,vtk}` | the VTK exists |
| 4 | generations | `compute_airway_generations.py` | the VTK | adds `Generation`, `Generation_Type`, `Is_Root`, `Distance_From_Root_mm` | runs every time |
| 5 | strahler | `compute_airway_strahler.py` | the VTK | adds `Strahler_Order` | runs every time |
| 6 | lobe_labels | `compute_airway_lobe_labels.py` | the VTK, a lobe mask | adds `Lobe`, `Lobe_Name` | runs every time |
| 7 | lobe_topology | `compute_airway_lobe_topology.py` | the VTK, a lobe mask | adds `Lobe_Topology`, `Lobe_Topology_Name` | runs every time |
| 8 | dysanapsis | `compute_smith_dysanapsis.py` | the VTK, a lobe mask | `airway_results/<CASE>_smith_dysanapsis.{csv,json,png}` | the CSV exists |
| 9 | export_csv | `export_airway_csv.py` | the VTK, the Smith JSON | `airway_results/<CASE>_airway_centerline.csv` | the CSV exists |
| 10 | export_html | `export_airway_3d.py` | the VTK | `airway_results/<CASE>_airway_3d.html` and `<CASE>_airway_3d_strahler.html`. A 3D PDF is written only if `pdflatex` and `pymeshlab` are available. | the first HTML exists |
| 11 | surface | `compute_airway_surface.py` | `lung_airways_iso.nii.gz` | `airway_results/<CASE>_airway_surface.stl` | it is newer than the mask |

The skip rules and the "runs every time" behavior are copied from the current runner. The lobe mask lookup order is copied too: `lobes_ml.nii.gz` first, then `totalseg_merged/lung_lobes_multilabel.nii.gz`.

Two behaviors are kept exactly as they are today:

- Vertices that are not connected to the largest piece of the centerline keep `Generation = -1`, `Strahler_Order = -1` and `Distance_From_Root_mm = -1`. This is 2.49% of the vertices of the 74 shared CF cases. The header comment of `compute_airway_generations.py` says "never -1", which is wrong. The port corrects the comment and does not change the behavior.
- The mask keeps every connected piece, including small pieces outside the lungs.

Step 11 is new relative to the ten scripts. It has passed its own tests in `av_phenotype` (five tests) and a check on 74 cases.

## 7. Environment and weights

`environment.yml` creates an environment named `ALR` with Python 3.10 and installs pinned packages with pip. The starting pins are the versions that produced the stored cohort outputs on the author's Mac. The container build in section 10 settles the final set.

| Group | Packages and starting pins |
|---|---|
| Segmentation | `TotalSegmentator==2.13.0`, `torch==2.11.0`, `nnunetv2==2.7.0` |
| Centerline and geometry | `kimimaro==5.8.1`, `networkx==3.4.2`, `vtk==9.6.0`, `pyvista==0.47.1`, `nibabel==5.4.0`, `numpy==2.2.6`, `scipy==1.15.3` |
| Outputs | `matplotlib==3.10.8`, `plotly==6.5.2`, `trimesh==4.12.1` |
| Optional extra for the 3D PDF | `pymeshlab==2025.7.post1` |

The Mac has two environments with different versions of `vtk`, `pyvista` and `nibabel`. The runners start each stage with the interpreter that launched them (`sys.executable`), which is the conda base environment, so the pins above are the base versions and not the newer ones in `totalseg213`. This is an inference from the code. The first task of Phase A confirms it by reproducing a stored output.

Weights:

- TotalSegmentator reads its home folder from `TOTALSEG_HOME_DIR` (`config.py:16`) and looks for weights under `nnunet/results` in it.
- Before any TotalSegmentator call, ALR sets `TOTALSEG_HOME_DIR` to `<sys.prefix>/share/totalsegmentator` unless the user has set it. The weights therefore live inside the conda environment and never in the home folder.
- `alr setup` downloads the four models with `totalsegmentator.libs.download_pretrained_weights` (`libs.py:162`): 117 (`lung_vessels`), 291 (the organs part of `total`, which holds the five lobes), and 297 and 298 (the 3 mm and 6 mm cropping models). They take about 0.8 GB on disk. None of the four is in TotalSegmentator's list of licensed models, so no registration is needed.
- TotalSegmentator sends a usage record to `stats.totalsegmentator.com` after each run when `send_usage_stats` is true in its `config.json` (`config.py:218`). The record holds the task, flags, platform, versions, whether CUDA is available and an anonymous id. `alr setup` sets `send_usage_stats` to false with TotalSegmentator's own `set_config_key` (`config.py:204`) inside the environment's home folder. The README states this and how to turn it back on.
- `alr setup` also writes a conda `activate.d` and `deactivate.d` script that sets and unsets the variable, so a manual `TotalSegmentator` call in the activated environment uses the same folder.
- TotalSegmentator is called as the executable next to the running interpreter. This replaces `conda run -n totalseg213`. The flags are the current ones: `-ta lung_vessels` for the airway mask, and `-ta total -ml -rmb -rs <five lobe classes>` for the lobes.
- The device is chosen by `device.py`: CUDA if available, then MPS, then CPU. This replaces the fixed `mps` in `compute_totalseg_airway.py:30`.

## 8. Portability changes from the source scripts

| Source | ALR |
|---|---|
| The path `/path/to/av_phenotype/final_analysis/{CASE}/{CASE}` in 12 scripts | The `--case-dir` flag, or `--root` with `--case` |
| `DEVICE = 'mps'` | Automatic choice, with `--device` to override |
| `conda run -n totalseg213 TotalSegmentator` | The executable in the active environment |
| An absolute symlink for `lung_airways.nii.gz` | A relative link. If linking fails, a copy. |
| `CASE_ID` and `CASE_DIR` environment variables | Function arguments |
| Code that runs at import | Functions, so every module can be imported by a test |
| The VTK to graph code repeated in four scripts | `graph.py` (to be confirmed line by line during the port) |

## 9. License and dependencies

ALR's own code is MIT, as the author chose. Two dependencies are GPL:

- `kimimaro` is GPL 3 or later. It performs the TEASAR skeletonization and is required.
- `pymeshlab` is GPL 3. It is imported only inside the 3D PDF export (`export_airway_3d.py:490`), so the port makes it an optional extra.

Installing a GPL library as a separate dependency is common practice for MIT projects. The position is less clear for anyone who redistributes ALR together with `kimimaro`, for example in a container image, because that bundle carries the GPL terms of `kimimaro`. This is a summary of the facts and not legal advice. The README will list the license of every dependency. See open question 1.

The other dependencies are permissive: `trimesh`, `pyvista`, `nibabel` and `plotly` are MIT, `vtk`, `scipy`, `numpy` and `networkx` are BSD, and `matplotlib` uses the Python Software Foundation license. TotalSegmentator is Apache 2.0. The README asks users to cite TotalSegmentator, and takes the citation text from its repository when the README is written.

## 10. Testing

Each module is written test first, one at a time. The full test folder runs after each module.

| Layer | Purpose | Data | Runs |
|---|---|---|---|
| Unit | The graph, generations, Strahler order and lobe topology on small trees with answers known by hand: a single Y, a trachea with two bronchi, a cut off fragment that must get `-1`. | Synthetic | Always |
| Regression | Steps 2 to 10 from the stored TotalSegmentator masks reproduce the stored outputs. Step 11 is compared where a stored surface exists, which is the CF cases and not NL001. | Real cases, in a folder named by `ALR_DATA`. Nothing is copied into git. | When the folder exists |
| Segmentation | TotalSegmentator on this machine gives an airway mask close to the stored one. | One CT | Only when asked |

Regression cases: CF121 and CF008 (many disconnected fragments), NL001 and CF005 (both use the older `lobes_ml.nii.gz` lobe file). For all four, the stored isotropic mask is voxel identical to a resample of the stored TotalSegmentator mask. CF001 is excluded because its isotropic mask is older than its current TotalSegmentator output (Dice 0.851).

Comparisons:

- Exact: the number of points and lines, every integer and text array in the VTK, every integer and text column in the CSV.
- Float32 tolerance: `Radius`, coordinates, distances, and the numbers in the Smith files.
- The HTML viewers: the test checks that they build and contain the expected number of vertices. It does not compare bytes, because I have not checked whether Plotly writes random ids.
- The STL: bounds, closed or open status, and volume within the tolerance seen in the `av_phenotype` check.

The segmentation test sets its Dice threshold from measurement on the Mac and later on Linux. The threshold is not chosen in advance.

Before any push, a Linux container built from `environment.yml` runs the unit tests and the full command sequence of success criterion 1 on one CT from the author's cohort, mounted into the container and never committed or copied into the image. The repository ships no demo scan. Docker Desktop is installed on the Mac. The container is Linux on ARM by default, and an x86 container is possible with emulation but slow. Neither replaces a run on a real Linux machine.

## 11. Rollout

- **Phase A, ALR alone.** The author's `av_phenotype` code and its 249 finished cases stay untouched. The first task is to show, on this Mac, that the current scripts reproduce the stored outputs of the regression cases when run from the stored masks. Nothing is ported before that is shown. Then the ten steps are ported one at a time, each with its tests, and step 11 is added.
- **Phase B, side by side.** ALR runs on the regression cases into a scratch folder, and the outputs are compared with the stored ones.
- **Phase C, hand over. This needs the author's explicit approval.** `run_pipeline.py` stages 22 to 29 and `run_airway_pipeline.py` call ALR, and the ten old scripts move to `archive/`. Five other scripts mention the airway stages or their outputs and must be checked first: `update_airway_csv_cohort.py`, `run_dysanapsis_cohort.py`, `compute_vascular_dysanapsis_volume_ratios.py`, `export_vessel_csv.py` and `export_vessel_3d.py`. ALR keeps every file name and column, so they should need no change.
- **Linux acceptance.** Clone, create the environment, run `alr setup` and `alr check`. Copy CF008's CT and its TotalSegmentator masks and run `alr run`. The output must equal the Mac output. Then run TotalSegmentator on Linux from the CT alone and record the Dice against the Mac mask.
- **Publication.** The repository is created as `kiwixiao/ALR`, public, after checking that the name is free. It is pushed only when the author says so. The author merges to `main`. Commits carry no Claude attribution line, following the author's standing rule.

## 12. Risks and unverified items

- Whether `kimimaro` returns identical skeletons on Linux and on macOS. The Linux acceptance run answers this.
- TotalSegmentator masks will differ slightly between CUDA and MPS because of floating point. This is why the regression starts from stored masks.
- Not every pin has been checked for a Linux wheel. The container build will show this.
- Whether `- .` inside the pip section of `environment.yml` installs the package from the repository root. The container build will show this.
- Whether calling `download_pretrained_weights` directly works without the command line setup steps.
- Writing into `<sys.prefix>/share` fails if the conda installation is read only. The fallback is to set `TOTALSEG_HOME_DIR` by hand, and the README says so.
- On CPU only machines the `torch` download is large.
- `.nii` input is not yet specified. The first version accepts `.nii.gz`.

## 13. Decisions on the open questions

1. **License.** Resolved 2026-09-30: ALR stays MIT, and the README documents the license of every dependency.
2. **Copyright holder.** Resolved 2026-09-30: the code is the author's own work, no CCHMC approval is needed, and the line reads "Copyright (c) 2026 Qiwei Xiao".
3. **Author email.** Open. The package metadata is public: no email, a GitHub noreply address, or the work address. Until decided, the metadata carries the name only.
4. **TotalSegmentator usage statistics.** Resolved 2026-09-30: `alr setup` turns them off (section 7).
5. **Extra TotalSegmentator files.** Resolved 2026-09-30: keep the artery, vein and airway wall files that the `lung_vessels` task writes. Step 1 deletes nothing.
6. **Demo CT.** Resolved 2026-09-30: none. The repository ships no scan (section 10).
7. **Name.** Resolved 2026-09-30: ALR means airway lung ratio. The README spells it out.
