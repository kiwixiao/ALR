# Handoff: build ALR from this repository

You are a Claude Code session on a Linux machine. The author wants you to build the whole of ALR, task by task, from `docs/plans/2026-09-30-alr-implementation-plan.md`. This note tells you where things are, what is verified, and how to work.

## What ALR is

ALR (airway lung ratio) is a public, MIT licensed Python package and one conda environment. From a chest CT it produces the airway mask (TotalSegmentator), an airway surface mesh, and a TEASAR centerline labeled with Weibel generation, Horton Strahler order and lobe labels, plus the Smith dysanapsis index and two interactive HTML viewers. It is a faithful port of ten scripts from the author's `av_phenotype` pipeline.

## Read in this order

1. `CLAUDE.md` (standing rules)
2. `docs/specs/2026-09-30-alr-design.md` (what to build and why)
3. `docs/plans/2026-09-30-alr-implementation-plan.md` (20 tasks, each with tests and code)
4. `docs/baseline.md` (what is measured)

## What is in the repository

| Path | Content |
|---|---|
| `reference/av_phenotype/` | The original scripts. They are the source of truth for behavior. Never edit them. It holds the ten stage scripts, the two runners, `compute_airway_surface.py` with its test, `diagnose_airway_surface.py` and the TotalSegmentator notes. |
| `tools/reproduce_reference.py` | Runs the original scripts on stored masks and compares with stored outputs. |
| `tools/pack_regression_data.sh` | Run by the author on the Mac to pack the regression data (74 MB, no CTs). |
| `tools/make_golden.py` | Wrote `tests/regression/golden.json` (md5 and counts of the stored files). |
| `tests/regression/golden.json` | Integrity reference for the data pack. Aggregate numbers only. |

Everything under `src/alr/`, `environment.yml` and `pyproject.toml` does not exist yet. You create it.

## Verified on the author's Mac (2026-09-30)

- The original scripts reproduce all stored outputs of CF121, CF008, NL001 and CF005 byte for byte (see `docs/baseline.md`).
- GLB and PDF export in `export_airway_3d.py` are commented out in `main()`. They are not ported, so `pymeshlab` and `trimesh` are not dependencies.
- The four TotalSegmentator models ALR needs (117, 291, 297, 298) are not licensed models in TotalSegmentator 2.13.0, so no registration is needed.
- `download_pretrained_weights` (`totalsegmentator/libs.py:162`), `setup_totalseg` (`config.py:54`) and `set_config_key` (`config.py:204`) exist in TotalSegmentator 2.13.0. `set_config_key` only edits an existing `config.json`.
- The original scripts read `CASE_ID` and `CASE_DIR` from the environment, which is how `tools/reproduce_reference.py` runs them on a scratch copy.

## Not verified yet (you will find out)

- Whether every pin in `environment.yml` resolves on Linux, and whether `- -e .` works in the pip section.
- Whether `kimimaro`, `vtk` and `numpy` reproduce the stored skeleton on Linux. Task 2 measures this.
- Whether calling `download_pretrained_weights` directly works without the TotalSegmentator command line.
- How much TotalSegmentator output differs between this machine and the Mac. Task 20 measures it. Do not invent a threshold.

## What you need from the author before you start

1. The regression data pack. The author creates it on the Mac: `bash tools/pack_regression_data.sh final_analysis alr_regression_data.tar` inside the `av_phenotype` folder, then copies the tar here. Extract it and set `ALR_DATA=<extract folder>/final_analysis`.
2. For Task 20 only: the CT of CF121 (about 280 MB). It is never committed.
3. Tell the author whether this machine has a GPU (`nvidia-smi`).

## How to work

- Use the test driven development skill if you have it: write the failing test, watch it fail, write the code, watch it pass. Run the whole test folder before every commit.
- One commit per task, on `feature/port-airway-chain`. Never merge to `main`, never force push. The author merges.
- Commit messages carry no Claude attribution line.
- Git identity: commit as `kiwixiao <41949675+kiwixiao@users.noreply.github.com>` (`git config user.email 41949675+kiwixiao@users.noreply.github.com`). Never commit a personal or work email address, and never write a path under a home folder into a file in this repository.
- After every task, update `PROGRESS.md` (NEXT STEP, STATUS, SESSION LOG) in the same commit.
- Port by the recipe in Task 8. Do not change an algorithm, a constant, a file name or a column. If a test shows the original behavior looks wrong, keep it and tell the author.
- Report facts only. If a test fails, say which and show the output. Do not weaken a test to make it pass.
- Documents you write (README and everything under `docs/`) follow the author's style: no em dashes or en dashes, no hyphens between words, short complete sentences, ranges with "to". The plan lists the words to avoid.
- Use `python`, not `python3`.

## Stop and ask the author

- Task 2 fails on this machine.
- A pin cannot be resolved and loosening it changes a scientific library version.
- The spec and the original script disagree on behavior.
- Anything would change an output file name, a column, or a number.
- You want to add a dependency, a feature or a file type the spec does not list.

## Behaviors of the source that are kept on purpose

- Vertices not connected to the largest piece of the centerline keep `Generation = -1`, `Strahler_Order = -1`, `Distance_From_Root_mm = -1` and `Generation_Type = Unknown`. Between 2 and 3 percent of vertices in the CF cohort.
- Steps 4 to 7 run on every invocation, because they edit the VTK in place.
- Step 2 skips itself when the mask is newer than its source, even with `--force`.
- The mask keeps every connected piece, including small false positive pieces outside the lungs.

One deliberate change: step 1 is also not considered done when no lobe mask exists (the original runner skipped it whenever the airway mask existed). This makes a missing lobe mask heal on the next run and does not change any output.

## Open question still with the author

The email address in the public package metadata (spec question 3). Until the author answers, `pyproject.toml` carries the name only.
