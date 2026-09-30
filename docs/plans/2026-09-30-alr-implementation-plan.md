# ALR Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task by task. Steps use checkbox (`- [ ]`) syntax for tracking. Read `docs/HANDOFF.md` first.

**Goal:** Build the `alr` package, a conda environment file and a command line tool that turn a chest CT into an airway mask, an airway surface and a labeled TEASAR centerline, reproducing the stored `av_phenotype` outputs.

**Architecture:** A faithful port of ten scripts in `reference/av_phenotype/` into one module per step, each exposing `run(case)`, plus new thin modules for layout, device choice, weights, pipeline order, the command line and the cohort loop. Steps pass data through files on disk with the original names. Tests start from stored TotalSegmentator masks so our code is isolated from the segmentation model.

**Tech Stack:** Python 3.10, TotalSegmentator 2.13.0 with torch and nnunetv2, kimimaro (TEASAR), vtk and pyvista, nibabel, numpy, scipy, matplotlib, plotly, pytest.

**Spec:** `docs/specs/2026-09-30-alr-design.md`. Read it together with this plan. Where they disagree, stop and ask the author.

## Global Constraints

- Python 3.10. The target platform is Linux. The code must also run on macOS.
- Import name and command are lowercase `alr`. The distribution, the repository and the conda environment are `ALR`. ALR means airway lung ratio.
- License MIT. The copyright line reads `Copyright (c) 2026 Qiwei Xiao`.
- Starting pins: `TotalSegmentator==2.13.0`, `torch==2.11.0`, `nnunetv2==2.7.0`, `kimimaro==5.8.1`, `networkx==3.4.2`, `vtk==9.6.0`, `pyvista==0.47.1`, `nibabel==5.4.0`, `numpy==2.2.6`, `scipy==1.15.3`, `matplotlib==3.10.8`, `plotly==6.5.2`, `pytest`. If the Linux resolver rejects a pin, change it in Task 1 and record the reason in `PROGRESS.md`.
- The algorithms are copied unchanged. Output file names and columns never change. Fragment vertices keep `Generation = -1`, `Strahler_Order = -1`, `Distance_From_Root_mm = -1`.
- No absolute home path, no environment name and no device name specific to the author's machine in `src/` or `tests/`.
- TotalSegmentator weights live under `<sys.prefix>/share/totalsegmentator` unless `TOTALSEG_HOME_DIR` is set. `alr setup` downloads exactly models 117, 291, 297 and 298 and turns usage statistics off.
- The repository holds no patient data and no file derived from a patient scan. `.gitignore` blocks image, mesh and archive files.
- Commands in documents use `python`, never `python3`.
- Documents (README, docs) follow the author's writing style: no em dashes or en dashes, no hyphens between words (write "cross sectional", not "cross-sectional"), short complete sentences, ranges with "to", none of the words comprehensive, robust, leverage, seamless, crucial or furthermore.
- Commit messages carry no Claude attribution line. Work on `feature/port-airway-chain`. Never merge to `main` and never force push.

## Review Focus

1. A CT given as uncompressed `.nii` must fail with a message that says how to gzip it, not a traceback (Task 18).
2. A CT path with spaces or non ASCII characters must work (Task 18).
3. A rerun after a crash leaves partial outputs. Finished steps are skipped and later steps run (Task 18).
4. A conda prefix that is not writable must give an error that names `TOTALSEG_HOME_DIR` (Task 5).
5. A machine without a GPU must select the CPU without touching a GPU only code path (Task 4).

---

## File Structure

```
src/alr/__init__.py        version
src/alr/__main__.py        python -m alr
src/alr/cli.py             argument parsing, commands run, cohort, setup, check
src/alr/layout.py          Case: every path and file name
src/alr/device.py          choose_device
src/alr/weights.py         TotalSegmentator home folder, setup, check, conda hooks
src/alr/segment.py         step 1
src/alr/resample.py        step 2
src/alr/skeleton.py        step 3
src/alr/graph.py           edges, adjacency, connected components
src/alr/generations.py     step 4
src/alr/strahler.py        step 5
src/alr/lobe_labels.py     step 6
src/alr/lobe_topology.py   step 7
src/alr/dysanapsis.py      step 8
src/alr/export_csv.py      step 9
src/alr/export_html.py     step 10
src/alr/surface.py         step 11
src/alr/pipeline.py        ordered steps, skip rule, force
src/alr/cohort.py          case discovery and the cohort loop
tests/conftest.py          ALR_DATA fixture, stored cases
tests/unit/                no patient data, always run
tests/regression/          real cases, need ALR_DATA
tools/                     reproduce_reference.py, make_golden.py, pack_regression_data.sh (already present)
reference/av_phenotype/    the original scripts (read only, never edit)
```

The interface every step module offers is `run(case: Case) -> None`, where `Case` comes from `alr.layout`.

---

### Task 1: Environment file and package skeleton

**Files:**
- Create: `environment.yml`, `pyproject.toml`, `LICENSE`, `.gitignore`, `README.md` (stub), `src/alr/__init__.py`, `src/alr/__main__.py`, `src/alr/cli.py`, `tests/unit/test_version.py`

**Interfaces:**
- Produces: `alr.__version__: str`, `alr.cli.main(argv: list[str] | None = None) -> int`

- [ ] **Step 1: Write the failing test** in `tests/unit/test_version.py`

```python
import subprocess
import sys

import alr


def test_version_string():
    assert alr.__version__ == '0.1.0'


def test_module_prints_version():
    result = subprocess.run([sys.executable, '-m', 'alr', '--version'], capture_output=True, text=True)
    assert result.returncode == 0
    assert '0.1.0' in result.stdout
```

- [ ] **Step 2: Create the environment and confirm the test fails**

First create the files of Step 3 (`pyproject.toml`, `src/alr/__init__.py`, `src/alr/__main__.py`, `src/alr/cli.py`, `README.md`) but leave `src/alr/__main__.py` and `src/alr/cli.py` empty, so that `-e .` can install the package. Then create `environment.yml`:

```yaml
name: ALR
channels:
  - conda-forge
dependencies:
  - python=3.10
  - pip
  - pip:
      - TotalSegmentator==2.13.0
      - torch==2.11.0
      - nnunetv2==2.7.0
      - kimimaro==5.8.1
      - networkx==3.4.2
      - vtk==9.6.0
      - pyvista==0.47.1
      - nibabel==5.4.0
      - numpy==2.2.6
      - scipy==1.15.3
      - matplotlib==3.10.8
      - plotly==6.5.2
      - pytest
      - -e .
```

Then run:

```bash
conda env create -f environment.yml
conda activate ALR
python -m pytest tests/unit/test_version.py -v
```

Expected: the environment builds; `test_version_string` passes and `test_module_prints_version` fails because `alr.__main__` does nothing yet. If `test_version_string` also fails, the install is wrong: fix that first. If `conda env create` rejects `- -e .`, remove that line, run `pip install -e .` as a separate command, and note in `PROGRESS.md` that the README must show that extra command. If a pin fails to resolve on Linux, loosen only that pin and record it.

- [ ] **Step 3: Write the package skeleton** (the content of every file)

`pyproject.toml`:

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "ALR"
version = "0.1.0"
description = "ALR (airway lung ratio): airway mask, surface and labeled TEASAR centerline from a chest CT"
readme = "README.md"
requires-python = ">=3.10"
license = { text = "MIT" }
authors = [{ name = "Qiwei Xiao" }]
dependencies = [
    "numpy", "scipy", "nibabel", "kimimaro", "networkx", "vtk", "pyvista",
    "matplotlib", "plotly", "TotalSegmentator",
]

[project.optional-dependencies]
dev = ["pytest"]

[project.scripts]
alr = "alr.cli:main"

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-q"
markers = ["network: downloads model weights or runs TotalSegmentator"]
```

`src/alr/__init__.py`:

```python
__version__ = '0.1.0'
```

`src/alr/__main__.py`:

```python
from alr.cli import main

raise SystemExit(main())
```

`src/alr/cli.py`:

```python
import argparse

from alr import __version__


def build_parser():
    parser = argparse.ArgumentParser(prog='alr', description='ALR: airway lung ratio pipeline')
    parser.add_argument('--version', action='version', version=f'alr {__version__}')
    return parser


def main(argv=None):
    build_parser().parse_args(argv)
    return 0
```

`LICENSE` is the standard MIT text with the line `Copyright (c) 2026 Qiwei Xiao`. `README.md` for now holds one line: `# ALR`. `.gitignore`:

```
__pycache__/
*.py[cod]
*.egg-info/
.pytest_cache/
build/
dist/
.DS_Store
# No patient data or derived images, meshes or archives in this repository
*.nii
*.nii.gz
*.nrrd
*.vtk
*.vtp
*.vti
*.stl
*.tar
*.tar.gz
*.zip
```

- [ ] **Step 4: Run the tests**

Run: `python -m pytest tests/unit/test_version.py -v` (after `pip install -e .` if the environment did not).
Expected: 2 passed.

- [ ] **Step 5: Check the environment itself**

Run:

```bash
python -c "import torch, nnunetv2, kimimaro, vtk, pyvista, nibabel; print(torch.__version__, torch.cuda.is_available())"
TotalSegmentator --help | head -3
```

Expected: versions print without error; `TotalSegmentator --help` prints usage. Record the output in `PROGRESS.md`.

- [ ] **Step 6: Commit**

```bash
git add environment.yml pyproject.toml LICENSE .gitignore README.md src tests
git commit -m "feat: package skeleton, environment file and MIT license"
```

---

### Task 2: Baseline with the original scripts (no production code)

This task decides whether the regression approach works on this machine. Do not skip it.

**Files:**
- Modify: `docs/baseline.md` (the macOS result is already there; add a Linux section)
- Use: `tools/reproduce_reference.py`, `tools/pack_regression_data.sh`, `tests/regression/golden.json`

- [ ] **Step 1: Get the regression data.** The author creates the data pack on the machine that holds the cohort, from the `av_phenotype` folder:

```bash
bash /path/to/ALR/tools/pack_regression_data.sh final_analysis alr_regression_data.tar
```

It holds CF121, CF008, NL001 and CF005 (about 74 MB, no CTs). Ask the author to copy it to this machine. Then:

```bash
mkdir -p ~/alr_data && tar -xf alr_regression_data.tar -C ~/alr_data
export ALR_DATA=~/alr_data/final_analysis
```

- [ ] **Step 2: Run the original chain and compare**

```bash
python tools/reproduce_reference.py --data "$ALR_DATA"
```

Expected on the author's Mac (already verified 2026-09-30): the integrity line, then PASS with "byte identical" for the isotropic mask, both TEASAR files, both centerline and Smith CSV files and the Smith JSON, for all four cases, and the final line `RESULT: all artifacts reproduced`.

- [ ] **Step 3: Decide**

| Result on this machine | Action |
|---|---|
| All PASS, byte identical | Continue to Task 3. |
| All PASS, some not byte identical (float tolerance) | Continue. Regression tests keep the tolerance in `tools/reproduce_reference.py`. Record which artifacts differ in `docs/baseline.md`. |
| Any FAIL, or the integrity check fails | STOP. Do not port anything. Write the failing artifact, the size of the difference and the package versions into `docs/baseline.md`, commit it, and tell the author. Likely causes: a different `kimimaro`, `numpy` or `vtk` build on this platform. The author decides whether to compare by statistics instead. |

- [ ] **Step 4: Replace the "Linux" section of `docs/baseline.md`** with the date, the machine (`uname -a`, CPU or GPU), the output of `pip list | grep -iE "kimimaro|vtk|pyvista|nibabel|numpy|scipy|torch"`, and the result table for the four cases.

- [ ] **Step 5: Commit**

```bash
git add docs/baseline.md
git commit -m "docs: baseline reproduction of stored outputs with the original scripts"
```

---

### Task 3: layout.py

**Files:**
- Create: `src/alr/layout.py`
- Test: `tests/unit/test_layout.py`

**Interfaces:**
- Produces: `alr.layout.Case(name: str, dir: Path, force: bool = False, device: str = 'auto')`, a frozen dataclass with `Case.from_root(root, name, **kw)` and these properties, all `Path`: `ct`, `vessels_dir`, `merged_dir`, `airway_mask`, `airway_link`, `lobe_mask_ts`, `airway_iso`, `results`, `teasar_nii`, `teasar_vtk`, `smith_csv`, `smith_json`, `smith_png`, `centerline_csv`, `html`, `html_strahler`, `surface_stl`; and `lobe_candidates: list[Path]` (`lobes_ml.nii.gz` first, then `lobe_mask_ts`).

- [ ] **Step 1: Write the failing test**

```python
from pathlib import Path

from alr.layout import Case


def test_paths_follow_the_stored_layout(tmp_path):
    c = Case('CF121', tmp_path)
    assert c.ct == tmp_path / 'ct.nii.gz'
    assert c.airway_mask == tmp_path / 'totalseg_vessels' / 'lung_airways.nii.gz'
    assert c.airway_link == tmp_path / 'lung_airways.nii.gz'
    assert c.lobe_mask_ts == tmp_path / 'totalseg_merged' / 'lung_lobes_multilabel.nii.gz'
    assert c.airway_iso == tmp_path / 'lung_airways_iso.nii.gz'
    assert c.teasar_vtk == tmp_path / 'airway_results' / 'lung_airways_teasar.vtk'
    assert c.teasar_nii == tmp_path / 'airway_results' / 'lung_airways_teasar.nii.gz'


def test_result_file_names_carry_the_case_name(tmp_path):
    c = Case('NL001', tmp_path)
    r = tmp_path / 'airway_results'
    assert c.smith_csv == r / 'NL001_smith_dysanapsis.csv'
    assert c.smith_json == r / 'NL001_smith_dysanapsis.json'
    assert c.smith_png == r / 'NL001_smith_dysanapsis.png'
    assert c.centerline_csv == r / 'NL001_airway_centerline.csv'
    assert c.html == r / 'NL001_airway_3d.html'
    assert c.html_strahler == r / 'NL001_airway_3d_strahler.html'
    assert c.surface_stl == r / 'NL001_airway_surface.stl'


def test_lobe_candidates_prefer_the_older_file(tmp_path):
    c = Case('CF005', tmp_path)
    assert c.lobe_candidates == [tmp_path / 'lobes_ml.nii.gz', c.lobe_mask_ts]


def test_from_root_uses_the_doubled_folder_name(tmp_path):
    assert Case.from_root(tmp_path, 'CF001').dir == tmp_path / 'CF001' / 'CF001'
```

- [ ] **Step 2: Run it.** `python -m pytest tests/unit/test_layout.py -v`. Expected: FAIL, `ModuleNotFoundError: alr.layout`.

- [ ] **Step 3: Implement**

```python
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Case:
    name: str
    dir: Path
    force: bool = False
    device: str = 'auto'

    @classmethod
    def from_root(cls, root, name, **kw):
        return cls(name, Path(root) / name / name, **kw)

    @property
    def ct(self): return self.dir / 'ct.nii.gz'
    @property
    def vessels_dir(self): return self.dir / 'totalseg_vessels'
    @property
    def merged_dir(self): return self.dir / 'totalseg_merged'
    @property
    def airway_mask(self): return self.vessels_dir / 'lung_airways.nii.gz'
    @property
    def airway_link(self): return self.dir / 'lung_airways.nii.gz'
    @property
    def lobe_mask_ts(self): return self.merged_dir / 'lung_lobes_multilabel.nii.gz'
    @property
    def lobe_candidates(self): return [self.dir / 'lobes_ml.nii.gz', self.lobe_mask_ts]
    @property
    def airway_iso(self): return self.dir / 'lung_airways_iso.nii.gz'
    @property
    def results(self): return self.dir / 'airway_results'
    @property
    def teasar_nii(self): return self.results / 'lung_airways_teasar.nii.gz'
    @property
    def teasar_vtk(self): return self.results / 'lung_airways_teasar.vtk'
    @property
    def smith_csv(self): return self.results / f'{self.name}_smith_dysanapsis.csv'
    @property
    def smith_json(self): return self.results / f'{self.name}_smith_dysanapsis.json'
    @property
    def smith_png(self): return self.results / f'{self.name}_smith_dysanapsis.png'
    @property
    def centerline_csv(self): return self.results / f'{self.name}_airway_centerline.csv'
    @property
    def html(self): return self.results / f'{self.name}_airway_3d.html'
    @property
    def html_strahler(self): return self.results / f'{self.name}_airway_3d_strahler.html'
    @property
    def surface_stl(self): return self.results / f'{self.name}_airway_surface.stl'
```

- [ ] **Step 4: Run it.** Expected: 4 passed.
- [ ] **Step 5: Commit.** `git add src/alr/layout.py tests/unit/test_layout.py && git commit -m "feat: Case layout with every file name"`

---

### Task 4: device.py

**Files:**
- Create: `src/alr/device.py`
- Test: `tests/unit/test_device.py`

**Interfaces:**
- Produces: `choose_device(requested: str = 'auto', cuda: bool | None = None, mps: bool | None = None) -> str`, returning one of `'gpu'`, `'mps'`, `'cpu'` or a string `'gpu:N'`. The values are the ones TotalSegmentator's `-d` flag accepts.

- [ ] **Step 1: Write the failing test**

```python
import pytest

from alr.device import choose_device


def test_explicit_choice_wins():
    assert choose_device('cpu', cuda=True, mps=True) == 'cpu'
    assert choose_device('gpu:1', cuda=False, mps=False) == 'gpu:1'


def test_auto_prefers_cuda_then_mps_then_cpu():
    assert choose_device('auto', cuda=True, mps=True) == 'gpu'
    assert choose_device('auto', cuda=False, mps=True) == 'mps'
    assert choose_device('auto', cuda=False, mps=False) == 'cpu'


def test_unknown_device_is_rejected():
    with pytest.raises(ValueError, match='device'):
        choose_device('tpu')
```

- [ ] **Step 2: Run it.** Expected: FAIL, module missing.
- [ ] **Step 3: Implement**

```python
import re


def _torch_flags():
    import torch
    mps_backend = getattr(torch.backends, 'mps', None)
    return torch.cuda.is_available(), bool(mps_backend and mps_backend.is_available())


def choose_device(requested='auto', cuda=None, mps=None):
    """Return the value for TotalSegmentator's -d flag."""
    if requested != 'auto':
        if requested in ('gpu', 'mps', 'cpu') or re.fullmatch(r'gpu:\d+', requested):
            return requested
        raise ValueError(f'unknown device {requested!r}; use auto, gpu, gpu:N, mps or cpu')
    if cuda is None or mps is None:
        found_cuda, found_mps = _torch_flags()
        cuda = found_cuda if cuda is None else cuda
        mps = found_mps if mps is None else mps
    return 'gpu' if cuda else 'mps' if mps else 'cpu'
```

- [ ] **Step 4: Run it.** Expected: 3 passed.
- [ ] **Step 5: Commit.** `git commit -am "feat: device choice for CUDA, MPS and CPU"` (after `git add` of the two new files).

---

### Task 5: weights.py

**Files:**
- Create: `src/alr/weights.py`
- Test: `tests/unit/test_weights.py`

**Interfaces:**
- Consumes: TotalSegmentator functions `totalsegmentator.config.get_weights_dir`, `setup_totalseg`, `set_config_key`; `totalsegmentator.libs.download_pretrained_weights(task_id)`.
- Produces: `REQUIRED_MODELS: dict[int, str]`; `default_home(prefix=None) -> Path`; `ensure_env(prefix=None) -> Path`; `missing_models() -> list[int]`; `write_conda_hooks(prefix) -> list[Path]`; `setup(prefix=None, download=True) -> Path`; `check() -> tuple[list[str], bool]`.

- [ ] **Step 1: Write the failing test**

```python
import os
import subprocess
from pathlib import Path

import pytest

from alr import weights


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv('TOTALSEG_HOME_DIR', str(tmp_path / 'ts_home'))
    monkeypatch.delenv('TOTALSEG_WEIGHTS_PATH', raising=False)
    return tmp_path / 'ts_home'


def test_default_home_is_inside_the_prefix(tmp_path):
    assert weights.default_home(tmp_path) == tmp_path / 'share' / 'totalsegmentator'


def test_ensure_env_keeps_the_users_choice(home):
    assert weights.ensure_env() == home
    assert home.is_dir()


def test_ensure_env_sets_the_default_when_unset(tmp_path, monkeypatch):
    monkeypatch.delenv('TOTALSEG_HOME_DIR', raising=False)
    assert weights.ensure_env(tmp_path) == tmp_path / 'share' / 'totalsegmentator'
    assert os.environ['TOTALSEG_HOME_DIR'] == str(tmp_path / 'share' / 'totalsegmentator')


@pytest.mark.skipif(os.geteuid() == 0, reason='root ignores folder permissions')
def test_unwritable_home_names_the_variable(tmp_path, monkeypatch):
    locked = tmp_path / 'locked'
    locked.mkdir()
    locked.chmod(0o500)
    monkeypatch.setenv('TOTALSEG_HOME_DIR', str(locked / 'home'))
    try:
        with pytest.raises(PermissionError, match='TOTALSEG_HOME_DIR'):
            weights.ensure_env()
    finally:
        locked.chmod(0o700)


def test_missing_models_lists_absent_folders(home):
    weights.ensure_env()
    results = home / 'nnunet' / 'results'
    for task_id in (117, 291):
        (results / weights.REQUIRED_MODELS[task_id]).mkdir(parents=True)
    assert weights.missing_models() == [297, 298]


def test_check_reports_ok_only_when_all_four_are_present(home):
    weights.ensure_env()
    results = home / 'nnunet' / 'results'
    lines, ok = weights.check()
    assert not ok and any('297' in line for line in lines)
    for name in weights.REQUIRED_MODELS.values():
        (results / name).mkdir(parents=True)
    assert weights.check()[1] is True


def test_conda_hooks_set_and_unset_the_variable(tmp_path):
    (tmp_path / 'conda-meta').mkdir()
    activate, deactivate = weights.write_conda_hooks(tmp_path)
    env = {'PATH': os.environ['PATH'], 'CONDA_PREFIX': str(tmp_path)}
    script = f'. "{activate}"; echo "[$TOTALSEG_HOME_DIR]"; . "{deactivate}"; echo "[$TOTALSEG_HOME_DIR]"'
    out = subprocess.run(['bash', '-c', script], env=env, capture_output=True, text=True).stdout.split()
    assert out == [f'[{tmp_path}/share/totalsegmentator]', '[]']


def test_conda_hooks_leave_a_users_value_alone(tmp_path):
    (tmp_path / 'conda-meta').mkdir()
    activate, deactivate = weights.write_conda_hooks(tmp_path)
    env = {'PATH': os.environ['PATH'], 'CONDA_PREFIX': str(tmp_path), 'TOTALSEG_HOME_DIR': '/mine'}
    script = f'. "{activate}"; . "{deactivate}"; echo "[$TOTALSEG_HOME_DIR]"'
    assert subprocess.run(['bash', '-c', script], env=env, capture_output=True, text=True).stdout.strip() == '[/mine]'


def test_setup_turns_usage_statistics_off(home):
    import json
    weights.setup(prefix=home.parent, download=False)
    config = json.loads((home / 'config.json').read_text())
    assert config['send_usage_stats'] is False
```

- [ ] **Step 2: Run it.** Expected: FAIL, module missing.
- [ ] **Step 3: Implement**

```python
import os
import sys
from pathlib import Path

REQUIRED_MODELS = {
    117: 'Dataset117_lung_airways_arteries_veins_282subj',
    291: 'Dataset291_TotalSegmentator_part1_organs_1559subj',
    297: 'Dataset297_TotalSegmentator_total_3mm_1559subj',
    298: 'Dataset298_TotalSegmentator_total_6mm_1559subj',
}
HOOK_NAME = 'alr_totalseg.sh'

ACTIVATE = '''if [ -z "${TOTALSEG_HOME_DIR:-}" ]; then
    export TOTALSEG_HOME_DIR="$CONDA_PREFIX/share/totalsegmentator"
    export ALR_SET_TOTALSEG_HOME=1
fi
'''
DEACTIVATE = '''if [ "${ALR_SET_TOTALSEG_HOME:-}" = "1" ]; then
    unset TOTALSEG_HOME_DIR
    unset ALR_SET_TOTALSEG_HOME
fi
'''


def default_home(prefix=None):
    return Path(prefix or sys.prefix) / 'share' / 'totalsegmentator'


def ensure_env(prefix=None):
    """Point TotalSegmentator at a folder inside the environment unless the user chose one."""
    os.environ.setdefault('TOTALSEG_HOME_DIR', str(default_home(prefix)))
    home = Path(os.environ['TOTALSEG_HOME_DIR'])
    try:
        home.mkdir(parents=True, exist_ok=True)
    except PermissionError as error:
        raise PermissionError(f'cannot write to {home}; set TOTALSEG_HOME_DIR to a folder you own') from error
    return home


def _model_dirs():
    from totalsegmentator.config import get_weights_dir
    root = Path(get_weights_dir())
    return {task_id: root / name for task_id, name in REQUIRED_MODELS.items()}


def missing_models():
    return sorted(task_id for task_id, path in _model_dirs().items() if not path.is_dir())


def write_conda_hooks(prefix):
    prefix = Path(prefix)
    written = []
    for folder, text in (('activate.d', ACTIVATE), ('deactivate.d', DEACTIVATE)):
        path = prefix / 'etc' / 'conda' / folder / HOOK_NAME
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        written.append(path)
    return written


def setup(prefix=None, download=True):
    home = ensure_env(prefix)
    from totalsegmentator.config import set_config_key, setup_totalseg
    setup_totalseg()
    set_config_key('send_usage_stats', False)
    if download:
        from totalsegmentator.libs import download_pretrained_weights
        for task_id in REQUIRED_MODELS:
            download_pretrained_weights(task_id)
    if (Path(prefix or sys.prefix) / 'conda-meta').is_dir():
        write_conda_hooks(prefix or sys.prefix)
    return home


def check():
    import importlib.metadata as metadata
    from alr import __version__
    from alr.device import choose_device
    lines = [f'alr {__version__}']
    for package in ('TotalSegmentator', 'torch', 'nnunetv2', 'kimimaro', 'vtk', 'pyvista', 'nibabel'):
        try:
            lines.append(f'{package} {metadata.version(package)}')
        except metadata.PackageNotFoundError:
            lines.append(f'{package} NOT INSTALLED')
    lines.append(f'device {choose_device()}')
    home = ensure_env()
    lines.append(f'weights folder {home}')
    missing = missing_models()
    for task_id in REQUIRED_MODELS:
        lines.append(f'model {task_id}: ' + ('MISSING, run: alr setup' if task_id in missing else 'present'))
    ok = not missing and not any('NOT INSTALLED' in line for line in lines)
    return lines, ok
```

- [ ] **Step 4: Run it.** Expected: 9 passed (8 if the test runs as root). If `test_setup_turns_usage_statistics_off` fails because `setup_totalseg` reads another config location, read `totalsegmentator/config.py` lines 16 to 70 and adapt `setup`; do not weaken the test.
- [ ] **Step 5: Commit.** `git add src/alr/weights.py tests/unit/test_weights.py && git commit -m "feat: weights folder, setup and check"`

---

### Task 6: segment.py (step 1)

**Files:**
- Create: `src/alr/segment.py`
- Test: `tests/unit/test_segment.py`
- Source: `reference/av_phenotype/compute_totalseg_airway.py` (90 lines)

**Interfaces:**
- Consumes: `Case` (Task 3), `choose_device` (Task 4), `weights.ensure_env` (Task 5).
- Produces: `LOBE_CLASSES: list[str]`, `totalsegmentator_exe() -> Path`, `airway_command(exe, ct, out_dir, device) -> list[str]`, `lobe_command(exe, ct, out_file, device) -> list[str]`, `link_airway(case) -> None`, `run(case) -> None`.

- [ ] **Step 1: Write the failing test.** It uses a fake `TotalSegmentator` executable that writes the files the real one writes and records its arguments, so the test runs real subprocesses without a model.

```python
import os
import stat
import sys

import pytest

from alr import segment
from alr.layout import Case

FAKE = '''#!{python}
import os, pathlib, sys
args = sys.argv[1:]
pathlib.Path(os.environ['FAKE_CALLS']).open('a').write(' '.join(args) + '\\n')
out = pathlib.Path(args[args.index('-o') + 1])
task = args[args.index('-ta') + 1]
if os.environ.get('FAKE_FAIL') == task:
    sys.exit(3)
if task == 'lung_vessels':
    out.mkdir(parents=True, exist_ok=True)
    (out / 'lung_airways.nii.gz').write_bytes(b'airway')
    (out / 'lung_arteries.nii.gz').write_bytes(b'artery')
else:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(b'lobes')
'''


@pytest.fixture
def fake_ts(tmp_path, monkeypatch):
    exe = tmp_path / 'bin' / 'TotalSegmentator'
    exe.parent.mkdir()
    exe.write_text(FAKE.format(python=sys.executable))
    exe.chmod(exe.stat().st_mode | stat.S_IEXEC)
    calls = tmp_path / 'calls.txt'
    monkeypatch.setenv('FAKE_CALLS', str(calls))
    monkeypatch.setenv('TOTALSEG_HOME_DIR', str(tmp_path / 'ts_home'))
    monkeypatch.setattr(segment, 'totalsegmentator_exe', lambda: exe)
    return calls


@pytest.fixture
def case(tmp_path):
    c = Case('T1', tmp_path / 'T1' / 'T1', device='cpu')
    c.dir.mkdir(parents=True)
    c.ct.write_bytes(b'ct')
    return c


def test_commands_match_the_original_flags(tmp_path):
    exe, ct = tmp_path / 'TotalSegmentator', tmp_path / 'ct.nii.gz'
    assert segment.airway_command(exe, ct, tmp_path / 'v', 'cpu') == [
        str(exe), '-i', str(ct), '-o', str(tmp_path / 'v'), '-ta', 'lung_vessels', '-d', 'cpu']
    cmd = segment.lobe_command(exe, ct, tmp_path / 'l.nii.gz', 'gpu')
    assert cmd[:11] == [str(exe), '-i', str(ct), '-o', str(tmp_path / 'l.nii.gz'),
                        '-ta', 'total', '-ml', '-rmb', '-d', 'gpu']
    assert cmd[11:] == ['-rs'] + segment.LOBE_CLASSES
    assert segment.LOBE_CLASSES == ['lung_upper_lobe_left', 'lung_lower_lobe_left', 'lung_upper_lobe_right',
                                    'lung_middle_lobe_right', 'lung_lower_lobe_right']


def test_run_writes_both_masks_and_a_relative_link(case, fake_ts):
    segment.run(case)
    assert case.airway_mask.read_bytes() == b'airway'
    assert case.lobe_mask_ts.read_bytes() == b'lobes'
    assert case.airway_link.is_symlink()
    assert not os.path.isabs(os.readlink(case.airway_link))
    assert case.airway_link.read_bytes() == b'airway'
    assert len(fake_ts.read_text().splitlines()) == 2


def test_run_does_nothing_when_both_masks_exist(case, fake_ts):
    for p in (case.airway_mask, case.lobe_mask_ts):
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b'x')
    segment.run(case)
    assert not fake_ts.exists()


def test_run_makes_only_the_missing_lobe_mask(case, fake_ts):
    case.airway_mask.parent.mkdir(parents=True)
    case.airway_mask.write_bytes(b'x')
    segment.run(case)
    assert [line.split()[line.split().index('-ta') + 1] for line in fake_ts.read_text().splitlines()] == ['total']


def test_airway_failure_stops_the_step(case, fake_ts, monkeypatch):
    monkeypatch.setenv('FAKE_FAIL', 'lung_vessels')
    with pytest.raises(RuntimeError, match='lung_vessels'):
        segment.run(case)


def test_lobe_failure_is_reported_and_the_step_continues(case, fake_ts, monkeypatch, capsys):
    monkeypatch.setenv('FAKE_FAIL', 'total')
    segment.run(case)
    assert case.airway_mask.exists() and not case.lobe_mask_ts.exists()
    assert 'continuing' in capsys.readouterr().out


def test_missing_ct_is_an_error(tmp_path, fake_ts):
    with pytest.raises(FileNotFoundError, match='ct.nii.gz'):
        segment.run(Case('T2', tmp_path / 'nowhere'))
```

- [ ] **Step 2: Run it.** Expected: FAIL, module missing.
- [ ] **Step 3: Implement** `src/alr/segment.py`. Read `reference/av_phenotype/compute_totalseg_airway.py` lines 40 to 88 first. The logic below is that logic with the three portability changes of spec section 8.

```python
import os
import shutil
import subprocess
import sys
from pathlib import Path

from alr import weights
from alr.device import choose_device

LOBE_CLASSES = ['lung_upper_lobe_left', 'lung_lower_lobe_left', 'lung_upper_lobe_right',
                'lung_middle_lobe_right', 'lung_lower_lobe_right']


def totalsegmentator_exe():
    """The TotalSegmentator executable of the active environment."""
    beside_python = Path(sys.executable).parent / 'TotalSegmentator'
    if beside_python.exists():
        return beside_python
    found = shutil.which('TotalSegmentator')
    if found:
        return Path(found)
    raise FileNotFoundError('TotalSegmentator is not installed in this environment; see the README')


def airway_command(exe, ct, out_dir, device):
    return [str(exe), '-i', str(ct), '-o', str(out_dir), '-ta', 'lung_vessels', '-d', device]


def lobe_command(exe, ct, out_file, device):
    return [str(exe), '-i', str(ct), '-o', str(out_file), '-ta', 'total', '-ml', '-rmb',
            '-d', device, '-rs', *LOBE_CLASSES]


def link_airway(case):
    """Top level lung_airways.nii.gz, a relative link to the TotalSegmentator output."""
    if case.airway_mask.exists() and not os.path.lexists(case.airway_link):
        try:
            os.symlink(os.path.relpath(case.airway_mask, case.dir), case.airway_link)
        except OSError:
            shutil.copy2(case.airway_mask, case.airway_link)


def run(case):
    print(f'TotalSegmentator (airway and lobes)  {case.name}')
    if not case.ct.exists():
        raise FileNotFoundError(f'{case.ct} not found; put the CT there as ct.nii.gz')
    weights.ensure_env()
    device = choose_device(case.device)
    exe = totalsegmentator_exe()
    have_airway, have_lobes = case.airway_mask.exists(), case.lobe_mask_ts.exists()
    if have_airway and have_lobes:
        print('  both masks already present, skipping TotalSegmentator')
    else:
        if not have_airway:
            case.vessels_dir.mkdir(parents=True, exist_ok=True)
            if subprocess.run(airway_command(exe, case.ct, case.vessels_dir, device)).returncode != 0:
                raise RuntimeError('TotalSegmentator lung_vessels failed')
        if not have_lobes:
            case.merged_dir.mkdir(parents=True, exist_ok=True)
            if subprocess.run(lobe_command(exe, case.ct, case.lobe_mask_ts, device)).returncode != 0:
                print('  TotalSegmentator total failed; continuing without a lobe mask '
                      '(Smith dysanapsis lung volume will be skipped)')
    link_airway(case)
```

- [ ] **Step 4: Run it.** Expected: 7 passed.
- [ ] **Step 5: Commit.** `git add src/alr/segment.py tests/unit/test_segment.py && git commit -m "feat: step 1, TotalSegmentator airway and lobe masks"`

---

### Task 7: graph.py

**Files:**
- Create: `src/alr/graph.py`
- Test: `tests/unit/test_graph.py`

**Interfaces:**
- Produces: `edges_from_lines(lines) -> np.ndarray` of shape (E, 2), int; `adjacency(edges) -> defaultdict[int, list[int]]`; `connected_components(adj, n_vertices) -> list[list[int]]`, largest first, equal sizes in discovery order.

- [ ] **Step 1: Write the failing test**

```python
import numpy as np

from alr import graph

LINES = np.array([2, 0, 1, 2, 1, 2, 2, 3, 4], dtype=np.int64)   # edges 0-1, 1-2, 3-4; vertex 5 alone


def test_edges_from_vtk_lines():
    assert graph.edges_from_lines(LINES).tolist() == [[0, 1], [1, 2], [3, 4]]


def test_adjacency_is_symmetric():
    adj = graph.adjacency(graph.edges_from_lines(LINES))
    assert adj[1] == [0, 2]
    assert adj[0] == [1] and adj[2] == [1]
    assert adj[4] == [3]


def test_components_are_sorted_largest_first_and_include_isolated_vertices():
    adj = graph.adjacency(graph.edges_from_lines(LINES))
    comps = graph.connected_components(adj, 6)
    assert [sorted(c) for c in comps] == [[0, 1, 2], [3, 4], [5]]


def test_equal_sized_components_keep_discovery_order():
    lines = np.array([2, 0, 1, 2, 2, 3], dtype=np.int64)
    comps = graph.connected_components(graph.adjacency(graph.edges_from_lines(lines)), 4)
    assert [sorted(c) for c in comps] == [[0, 1], [2, 3]]
```

- [ ] **Step 2: Run it.** Expected: FAIL, module missing.
- [ ] **Step 3: Implement**

```python
from collections import defaultdict, deque

import numpy as np


def edges_from_lines(lines):
    """Edge list from a VTK lines array, where every cell is [2, v0, v1]."""
    return np.asarray(lines).reshape(-1, 3)[:, 1:3].astype(int)


def adjacency(edges):
    adj = defaultdict(list)
    for u, v in edges:
        adj[int(u)].append(int(v))
        adj[int(v)].append(int(u))
    return adj


def connected_components(adj, n_vertices):
    visited = np.zeros(n_vertices, dtype=bool)
    comps = []
    for start in range(n_vertices):
        if visited[start]:
            continue
        comp = []
        queue = deque([start])
        visited[start] = True
        while queue:
            u = queue.popleft()
            comp.append(u)
            for v in adj[u]:
                if not visited[v]:
                    visited[v] = True
                    queue.append(v)
        comps.append(comp)
    comps.sort(key=lambda c: -len(c))
    return comps
```

- [ ] **Step 4: Run it.** Expected: 4 passed.
- [ ] **Step 5: Commit.** `git add src/alr/graph.py tests/unit/test_graph.py && git commit -m "feat: shared graph helpers for the centerline"`

---

### Task 8: conftest, regression helpers and resample.py (step 2)

**Files:**
- Create: `tests/conftest.py`, `tests/regression/helpers.py`, `tests/regression/conftest.py`, `src/alr/resample.py`
- Test: `tests/regression/test_reg_resample.py`
- Source: `reference/av_phenotype/compute_airway_mask_isotropic.py` (64 lines)

**Interfaces:**
- Produces: fixtures `alr_data`, `stored` (yields `(name, stored_dir)` for CF121, CF008, NL001, CF005), `case_with_skeleton`; helpers `scratch_case(name, stored_dir, tmp_path, extra=())` and `vtk_problems(new, old)`; `alr.resample.run(case)`.

- [ ] **Step 1: Write the shared test support.**

`tests/conftest.py`:

```python
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))

CASES = ['CF121', 'CF008', 'NL001', 'CF005']


@pytest.fixture(scope='session')
def alr_data():
    value = os.environ.get('ALR_DATA')
    if not value or not Path(value).is_dir():
        pytest.skip('set ALR_DATA to the extracted regression data folder (the one that holds CF121/CF121/)')
    return Path(value)


@pytest.fixture(params=CASES)
def stored(request, alr_data):
    """(case name, folder with the stored original outputs)"""
    return request.param, alr_data / request.param / request.param
```

`tests/regression/helpers.py`:

```python
import shutil
from pathlib import Path

import pyvista as pv
import reproduce_reference as ref

from alr.layout import Case

INPUTS = ['totalseg_vessels/lung_airways.nii.gz', 'lobes_ml.nii.gz',
          'totalseg_merged/lung_lobes_multilabel.nii.gz']


def scratch_case(name, stored_dir, tmp_path, extra=()):
    """A fresh case folder holding only the stored TotalSegmentator inputs (and any extras)."""
    dst = tmp_path / name / name
    dst.mkdir(parents=True, exist_ok=True)
    for rel in (*INPUTS, *extra):
        src = Path(stored_dir) / rel
        if src.exists():
            (dst / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst / rel)
    return Case(name, dst)


def vtk_problems(new, old):
    """Compare points, lines and every array that the new VTK has with the stored VTK."""
    a, b = pv.read(new), pv.read(old)
    problems = [ref.compare_array('points', a.points, b.points),
                ref.compare_array('lines', a.lines, b.lines)]
    for key in a.point_data.keys():
        if key not in b.point_data.keys():
            problems.append(f'{key}: not in the stored VTK')
        else:
            problems.append(ref.compare_array(key, a.point_data[key], b.point_data[key]))
    return [p for p in problems if p]
```

`tests/regression/conftest.py` (the skeleton cache keeps the slow TEASAR run to once per case; it needs Task 9, so the fixture imports lazily):

```python
import shutil

import pytest
from helpers import scratch_case

from alr.layout import Case

_CACHE = {}


@pytest.fixture
def case_with_skeleton(stored, tmp_path_factory, tmp_path):
    from alr import skeleton
    name, stored_dir = stored
    if name not in _CACHE:
        base = tmp_path_factory.mktemp(f'skeleton_{name}')
        case = scratch_case(name, stored_dir, base, extra=('lung_airways_iso.nii.gz',))
        skeleton.run(case)
        _CACHE[name] = case.dir
    dst = tmp_path / name / name
    shutil.copytree(_CACHE[name], dst)
    return Case(name, dst)
```

- [ ] **Step 2: Write the failing regression test** `tests/regression/test_reg_resample.py`

```python
import reproduce_reference as ref
from helpers import scratch_case

from alr import resample


def test_iso_mask_matches_the_stored_one(stored, tmp_path):
    name, stored_dir = stored
    case = scratch_case(name, stored_dir, tmp_path)
    resample.run(case)
    assert ref.compare_nifti(case.airway_iso, stored_dir / 'lung_airways_iso.nii.gz') == []
```

Run: `ALR_DATA=$ALR_DATA python -m pytest tests/regression/test_reg_resample.py -v`. Expected: FAIL, `ImportError: cannot import name resample`.

- [ ] **Step 3: Port the module.** Start from the source and apply the recipe below; "Port recipe" is used by every later port task.

**Port recipe (all port tasks):**
1. `cp reference/av_phenotype/<source>.py src/alr/<module>.py`.
2. Replace the header comment block with a short docstring. Keep the facts about what the step reads and writes.
3. Delete the module level `CASE = ...`, `CASE_DIR = ...`, `ROOT = ...`, path constants built from them, and any `os.environ` use. Every path now comes from `case` (see `alr.layout.Case`).
4. Turn the `main()` or `process()` or `run()` function so that it takes `case` (or `case` plus the values it needs) and replace each use of a deleted global by the matching `case` property.
5. Replace the `if __name__ == '__main__':` block by a public `def run(case): ...` that does what that block did, with the original existence checks and messages kept.
6. Delete `sys.exit` calls: raise an exception for errors, or `return` where the original printed a message and returned.
7. Do not change any numerical code, any constant, any message that reports results, or any loop.
8. Check: `grep -nE "os\.environ|CASE_ID|CASE_DIR|\bROOT\b|ROOT_DIR|AIR_VTK|/Users/" src/alr/<module>.py` must print nothing, and `python -c "import alr.<module>"` must print nothing and create no file.

For this module: `SRC_CANDIDATES` becomes `[case.airway_link, case.airway_mask]`, `OUT` becomes `case.airway_iso`, `TARGET_VOXEL_MM = 0.625` and `needs_rerun` stay, and the choice of source (lines 38 to 42 of the source: the first candidate that exists and is not a symlink, otherwise the first that exists) stays as it is.

- [ ] **Step 4: Run the test.** Expected: 4 passed (one per case).
- [ ] **Step 5: Commit.** `git add tests/conftest.py tests/regression src/alr/resample.py && git commit -m "feat: step 2, isotropic airway mask, with regression support"`

---

### Task 9: skeleton.py (step 3)

**Files:**
- Create: `src/alr/skeleton.py`
- Test: `tests/regression/test_reg_skeleton.py`
- Source: `reference/av_phenotype/compute_airway_skeleton_teasar.py` (164 lines, function `run()` at line 52)

**Interfaces:**
- Consumes: `Case`. Produces: `run(case)` writing `case.teasar_nii` and `case.teasar_vtk`.

- [ ] **Step 1: Write the failing test**

```python
import reproduce_reference as ref
from helpers import scratch_case, vtk_problems

from alr import skeleton


def test_skeleton_matches_the_stored_one(stored, tmp_path):
    name, stored_dir = stored
    case = scratch_case(name, stored_dir, tmp_path, extra=('lung_airways_iso.nii.gz',))
    skeleton.run(case)
    results = stored_dir / 'airway_results'
    assert ref.compare_nifti(case.teasar_nii, results / 'lung_airways_teasar.nii.gz') == []
    assert vtk_problems(case.teasar_vtk, results / 'lung_airways_teasar.vtk') == []
```

- [ ] **Step 2: Run it.** Expected: FAIL, import error.
- [ ] **Step 3: Port by the recipe.** Path constants become `case.airway_iso`, `case.teasar_nii`, `case.teasar_vtk`, `case.results`; `os.makedirs(AIRWAY_DIR, exist_ok=True)` moves into `run`. `TEASAR_PARAMS`, `DUST_THRESHOLD`, `ANISOTROPY` and `CONN26` stay module constants and do not change. The function `run()` at line 52 keeps its body.
- [ ] **Step 4: Run it.** Expected: 4 passed. If a case fails only on float arrays by a tiny amount, read `docs/baseline.md` before changing any tolerance.
- [ ] **Step 5: Commit.** `git add src/alr/skeleton.py tests/regression/test_reg_skeleton.py && git commit -m "feat: step 3, TEASAR airway centerline"`

---

### Task 10: generations.py (step 4)

**Files:**
- Create: `src/alr/generations.py`, `tests/unit/synthetic.py`
- Test: `tests/unit/test_generations.py`, `tests/regression/test_reg_generations.py`
- Source: `reference/av_phenotype/compute_airway_generations.py` (223 lines, `process(vtk_path)` at line 57)

**Interfaces:**
- Consumes: `graph.edges_from_lines`, `graph.adjacency`, `graph.connected_components`. Produces: `run(case)`; `process(vtk_path, case_name)`; it adds `Generation`, `Generation_Type`, `Is_Root`, `Distance_From_Root_mm`.

- [ ] **Step 1: Write the failing tests.** The expected values below were produced by running the original script on this tree on 2026-09-30.

`tests/unit/synthetic.py`:

```python
import numpy as np
import pyvista as pv

from alr.layout import Case

# Trunk 0-1-2-3 with the carina at 3, left branch 3-4-5-6 splitting into 10 and 11,
# right branch 3-7-8-9, and a detached fragment 12-13.
POINTS = np.array([[0, 0, 30], [0, 0, 20], [0, 0, 10], [0, 0, 0],
                   [-10, 0, -10], [-20, 0, -20], [-30, 0, -30],
                   [10, 0, -10], [20, 0, -20], [30, 0, -30],
                   [-40, 0, -40], [-40, 10, -40],
                   [100, 100, 100], [100, 100, 110]], dtype=float)
EDGES = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (3, 7), (7, 8), (8, 9), (6, 10), (6, 11), (12, 13)]
RADIUS = [10, 10, 10, 10, 5, 5, 5, 5, 5, 5, 2, 2, 1, 1]


def tree_case(tmp_path):
    case = Case('SYN', tmp_path)
    case.results.mkdir(parents=True)
    lines = np.hstack([[2, a, b] for a, b in EDGES]).astype(np.int64)
    tree = pv.PolyData(POINTS, lines=lines)
    tree.point_data['Radius'] = np.array(RADIUS, dtype=np.float32)
    tree.save(case.teasar_vtk, binary=False)
    return case
```

`tests/unit/test_generations.py`:

```python
import numpy as np
import pyvista as pv
from synthetic import tree_case

from alr import generations


def test_generations_on_a_hand_built_tree(tmp_path):
    case = tree_case(tmp_path)
    generations.run(case)
    data = pv.read(case.teasar_vtk).point_data
    assert data['Generation'].tolist() == [1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 2, 2, -1, -1]
    assert data['Is_Root'].tolist() == [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    assert data['Generation_Type'][3] == 'Trachea'
    assert data['Generation_Type'][10] == 'Lobar_Bronchi'
    assert data['Generation_Type'][12] == 'Unknown'
    assert data['Distance_From_Root_mm'][12] == -1.0
    np.testing.assert_allclose(data['Distance_From_Root_mm'][4], 14.1421, atol=1e-3)
```

`tests/regression/test_reg_generations.py`:

```python
from helpers import vtk_problems

import pyvista as pv

from alr import generations


def test_generations_match_the_stored_ones(case_with_skeleton, stored):
    name, stored_dir = stored
    generations.run(case_with_skeleton)
    new = case_with_skeleton.teasar_vtk
    assert {'Generation', 'Generation_Type', 'Is_Root', 'Distance_From_Root_mm'} <= set(pv.read(new).point_data.keys())
    assert vtk_problems(new, stored_dir / 'airway_results' / 'lung_airways_teasar.vtk') == []
```

For `from synthetic import tree_case` to work, add an empty `tests/unit/conftest.py`.

- [ ] **Step 2: Run them.** Expected: FAIL, import error.
- [ ] **Step 3: Port by the recipe, and use `graph`.** After the recipe, replace the block that builds `lines`, `edges`, `adj` and the connected components (source lines 66 to 91) by:

```python
    edges = graph.edges_from_lines(cl.lines)
    log(f'  edges: {len(edges)}')
    adj = graph.adjacency(edges)
    components = [np.array(c) for c in graph.connected_components(adj, n_v)]
    sizes = [len(c) for c in components]
```

Keep the log lines that follow. Move `t0 = time.time()` and `def log(msg)` (source lines 44 to 46) into the top of `process` so that timing starts at each call. Change the wrong header comment "never -1" (source line 15) to: "Generation is -1 on vertices that are not connected to the largest component." Change the signature to `process(vtk_path, case_name)` and use `case_name` where `CASE` was used. `run(case)` must do what the source `__main__` block did, with `case.teasar_vtk`.

- [ ] **Step 4: Run them.** Expected: unit 1 passed; regression 4 passed.
- [ ] **Step 5: Commit.** `git add src/alr/generations.py tests && git commit -m "feat: step 4, Weibel generations"`

---

### Task 11: strahler.py (step 5)

**Files:**
- Create: `src/alr/strahler.py`
- Test: `tests/unit/test_strahler.py`, `tests/regression/test_reg_strahler.py`
- Source: `reference/av_phenotype/compute_airway_strahler.py` (127 lines, `process(vtk_path)` at line 30)

**Interfaces:**
- Produces: `run(case)`, `process(vtk_path, case_name)`; adds `Strahler_Order`. It needs `Is_Root`, so `generations.run` runs first.

- [ ] **Step 1: Write the failing tests**

`tests/unit/test_strahler.py`:

```python
import pyvista as pv
from synthetic import tree_case

from alr import generations, strahler


def test_strahler_on_a_hand_built_tree(tmp_path):
    case = tree_case(tmp_path)
    generations.run(case)
    strahler.run(case)
    order = pv.read(case.teasar_vtk).point_data['Strahler_Order'].tolist()
    assert order == [1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1, 1, -1, -1]
```

`tests/regression/test_reg_strahler.py`:

```python
import pyvista as pv
from helpers import vtk_problems

from alr import generations, strahler


def test_strahler_matches_the_stored_one(case_with_skeleton, stored):
    name, stored_dir = stored
    generations.run(case_with_skeleton)
    strahler.run(case_with_skeleton)
    new = case_with_skeleton.teasar_vtk
    assert 'Strahler_Order' in pv.read(new).point_data.keys()
    assert vtk_problems(new, stored_dir / 'airway_results' / 'lung_airways_teasar.vtk') == []
```

- [ ] **Step 2: Run them.** Expected: FAIL, import error.
- [ ] **Step 3: Port by the recipe and use `graph`.** Replace source lines 41 to 62 (edges, `adj`, components, `main = set(comps[0])`) by:

```python
    edges = graph.edges_from_lines(cl.lines)
    adj = graph.adjacency(edges)
    comps = graph.connected_components(adj, n_v)
    main = set(comps[0])
```

Move `t0` and `log` into `process` as in Task 10. The early `return` when `Is_Root` is missing stays.

- [ ] **Step 4: Run them.** Expected: all pass.
- [ ] **Step 5: Commit.** `git add src/alr/strahler.py tests && git commit -m "feat: step 5, Strahler order"`

---

### Task 12: lobe_labels.py (step 6)

**Files:**
- Create: `src/alr/lobe_labels.py`
- Test: `tests/regression/test_reg_lobe_labels.py`
- Source: `reference/av_phenotype/compute_airway_lobe_labels.py` (83 lines, `main()` at line 36)

- [ ] **Step 1: Write the failing test**

```python
import pyvista as pv
from helpers import vtk_problems

from alr import generations, lobe_labels, strahler


def test_lobe_labels_match_the_stored_ones(case_with_skeleton, stored):
    name, stored_dir = stored
    generations.run(case_with_skeleton)
    strahler.run(case_with_skeleton)
    lobe_labels.run(case_with_skeleton)
    new = case_with_skeleton.teasar_vtk
    assert {'Lobe', 'Lobe_Name'} <= set(pv.read(new).point_data.keys())
    assert vtk_problems(new, stored_dir / 'airway_results' / 'lung_airways_teasar.vtk') == []
```

- [ ] **Step 2: Run it.** Expected: FAIL, import error.
- [ ] **Step 3: Port by the recipe.** `LOBE_CANDIDATES` becomes `case.lobe_candidates`. `LOBE_NAMES` stays a module constant.
- [ ] **Step 4: Run it.** Expected: 4 passed (NL001 and CF005 use `lobes_ml.nii.gz`, CF121 and CF008 the merged file).
- [ ] **Step 5: Commit.** `git add src/alr/lobe_labels.py tests/regression/test_reg_lobe_labels.py && git commit -m "feat: step 6, lobe labels from the lobe mask"`

---

### Task 13: lobe_topology.py (step 7)

**Files:**
- Create: `src/alr/lobe_topology.py`
- Test: `tests/regression/test_reg_lobe_topology.py`
- Source: `reference/av_phenotype/compute_airway_lobe_topology.py` (268 lines: `bfs_subtree` line 55, `walk_chain_to_bifurcation` line 68, `main` line 88)

- [ ] **Step 1: Write the failing test**

```python
import pyvista as pv
from helpers import vtk_problems

from alr import generations, lobe_labels, lobe_topology, strahler


def test_lobe_topology_matches_the_stored_one(case_with_skeleton, stored):
    name, stored_dir = stored
    case = case_with_skeleton
    for step in (generations, strahler, lobe_labels, lobe_topology):
        step.run(case)
    new = case.teasar_vtk
    assert {'Lobe_Topology', 'Lobe_Topology_Name'} <= set(pv.read(new).point_data.keys())
    assert vtk_problems(new, stored_dir / 'airway_results' / 'lung_airways_teasar.vtk') == []
```

- [ ] **Step 2: Run it.** Expected: FAIL, import error.
- [ ] **Step 3: Port by the recipe.** Use `graph` for the adjacency at source lines 97 to 99: `edges = graph.edges_from_lines(cl.lines)` and `adj = graph.adjacency(edges)`, replacing the `defaultdict` loop. The two helper functions, `LBL_EXTRA` and `NAMES` stay unchanged. The early returns with printed messages stay.
- [ ] **Step 4: Run it.** Expected: 4 passed. At this point every array of the stored VTK is written, so add to the test file a second test:

```python
def test_array_names_equal_the_stored_ones(case_with_skeleton, stored):
    name, stored_dir = stored
    for step in (generations, strahler, lobe_labels, lobe_topology):
        step.run(case_with_skeleton)
    new = set(pv.read(case_with_skeleton.teasar_vtk).point_data.keys())
    old = set(pv.read(stored_dir / 'airway_results' / 'lung_airways_teasar.vtk').point_data.keys())
    assert new == old
```

- [ ] **Step 5: Commit.** `git add src/alr/lobe_topology.py tests/regression/test_reg_lobe_topology.py && git commit -m "feat: step 7, topology based lobe labels"`

---

### Task 14: dysanapsis.py (step 8)

**Files:**
- Create: `src/alr/dysanapsis.py`
- Test: `tests/regression/test_reg_dysanapsis.py`
- Source: `reference/av_phenotype/compute_smith_dysanapsis.py` (231 lines, `find_lobe_path` line 70, `main` line 77)

- [ ] **Step 1: Write the failing test**

```python
import reproduce_reference as ref

from alr import dysanapsis, generations, lobe_labels, lobe_topology, strahler


def test_smith_files_match_the_stored_ones(case_with_skeleton, stored):
    name, stored_dir = stored
    case = case_with_skeleton
    for step in (generations, strahler, lobe_labels, lobe_topology, dysanapsis):
        step.run(case)
    results = stored_dir / 'airway_results'
    assert ref.compare_csv(case.smith_csv, results / f'{name}_smith_dysanapsis.csv') == []
    assert ref.compare_json(case.smith_json, results / f'{name}_smith_dysanapsis.json') == []
    assert case.smith_png.stat().st_size > 0
```

- [ ] **Step 2: Run it.** Expected: FAIL, import error.
- [ ] **Step 3: Port by the recipe.** `AIRWAY_DIR` becomes `case.results`; `OUT_CSV`, `OUT_JSON`, `OUT_PNG` become `case.smith_csv`, `case.smith_json`, `case.smith_png`; `find_lobe_path` uses `case.lobe_candidates`. The constants `SMITH_COEFF`, `CENTRAL_GENS` and the `SMITH_REF_*` values stay exactly as they are.
- [ ] **Step 4: Run it.** Expected: 4 passed.
- [ ] **Step 5: Commit.** `git add src/alr/dysanapsis.py tests/regression/test_reg_dysanapsis.py && git commit -m "feat: step 8, Smith dysanapsis index"`

---

### Task 15: export_csv.py (step 9)

**Files:**
- Create: `src/alr/export_csv.py`
- Test: `tests/regression/test_reg_export_csv.py`
- Source: `reference/av_phenotype/export_airway_csv.py` (145 lines, `main` line 40)

- [ ] **Step 1: Write the failing test**

```python
import reproduce_reference as ref

from alr import dysanapsis, export_csv, generations, lobe_labels, lobe_topology, strahler


def test_centerline_csv_matches_the_stored_one(case_with_skeleton, stored):
    name, stored_dir = stored
    case = case_with_skeleton
    for step in (generations, strahler, lobe_labels, lobe_topology, dysanapsis, export_csv):
        step.run(case)
    assert ref.compare_csv(case.centerline_csv, stored_dir / 'airway_results' / f'{name}_airway_centerline.csv') == []
```

- [ ] **Step 2: Run it.** Expected: FAIL, import error.
- [ ] **Step 3: Port by the recipe.** `DYSANAPSIS_JSON` becomes `case.smith_json`, `OUT_CSV` becomes `case.centerline_csv`.
- [ ] **Step 4: Run it.** Expected: 4 passed.
- [ ] **Step 5: Commit.** `git add src/alr/export_csv.py tests/regression/test_reg_export_csv.py && git commit -m "feat: step 9, centerline CSV"`

---

### Task 16: export_html.py (step 10)

**Files:**
- Create: `src/alr/export_html.py`
- Test: `tests/regression/test_reg_export_html.py`
- Source: `reference/av_phenotype/export_airway_3d.py` (568 lines: `build_scene` line 52, `export_html` line 120, `export_glb` line 414, `export_pdf` line 488, `main` line 551)

- [ ] **Step 1: Write the failing test**

```python
from alr import dysanapsis, export_html, generations, lobe_labels, lobe_topology, strahler


def test_both_viewers_are_written(case_with_skeleton):
    case = case_with_skeleton
    for step in (generations, strahler, lobe_labels, lobe_topology, dysanapsis, export_html):
        step.run(case)
    for path in (case.html, case.html_strahler):
        text = path.read_text()
        assert len(text) > 100_000
        assert 'plotly' in text.lower()
```

- [ ] **Step 2: Run it.** Expected: FAIL, import error.
- [ ] **Step 3: Port by the recipe, with these extra edits.** Delete `export_glb`, `export_pdf`, `OUT_GLB`, `OUT_PDF`, every `import trimesh`, `import pymeshlab` and the `import vtk` that only they use, and the comment that says GLB and PDF are disabled (they are not ported). `build_scene(case)` and `export_html(case, ...)` receive the case: `AIR_VTK` becomes `case.teasar_vtk`, `OUT_HTML` becomes `case.html`, `OUT_HTML_STRAHLER` becomes `case.html_strahler`, `CASE` becomes `case.name`. In `build_scene`, replace the `adj = defaultdict(list)` loop over `lines` (source lines 80 to 82) by `adj = graph.adjacency(lines)`. Keep `ROOT_SPHERE_MM`, `PLOTLY_CONFIG`, `STRAHLER_COLORS_AW` and the function signature of `export_html` as they are. `run(case)` keeps the original check: if the VTK is missing, print a message and return.
- [ ] **Step 4: Run it.** Expected: 4 passed.
- [ ] **Step 5: Verify the viewer by eye once.** Open one HTML in a browser and compare it with the stored one for the same case in `ALR_DATA`. Write "checked" in `PROGRESS.md`. Do not commit HTML files.
- [ ] **Step 6: Commit.** `git add src/alr/export_html.py tests/regression/test_reg_export_html.py && git commit -m "feat: step 10, interactive 3D viewers"`

---

### Task 17: surface.py (step 11)

**Files:**
- Create: `src/alr/surface.py`
- Test: `tests/unit/test_surface.py`, `tests/regression/test_reg_surface.py`
- Source: `reference/av_phenotype/compute_airway_surface.py` and `reference/av_phenotype/tests/test_compute_airway_surface.py`

**Interfaces:**
- Produces: `mask_to_surface(src_path, dst_path) -> int | None`, `run(case)`.

- [ ] **Step 1: Write the failing tests.** Copy `reference/av_phenotype/tests/test_compute_airway_surface.py` to `tests/unit/test_surface.py`, delete the `sys.path.insert` line, and change the import to `from alr.surface import mask_to_surface`. Add `tests/regression/test_reg_surface.py`:

```python
import numpy as np
import pyvista as pv
import pytest
from helpers import scratch_case

from alr import surface


def test_surface_matches_the_stored_one(stored, tmp_path):
    name, stored_dir = stored
    old = stored_dir / 'airway_results' / f'{name}_airway_surface.stl'
    if not old.exists():
        pytest.skip(f'no stored surface for {name}')
    case = scratch_case(name, stored_dir, tmp_path, extra=('lung_airways_iso.nii.gz',))
    surface.run(case)
    a, b = pv.read(case.surface_stl), pv.read(old)
    assert a.n_points == b.n_points and a.n_cells == b.n_cells
    assert np.allclose(a.points, b.points, atol=1e-4)
```

- [ ] **Step 2: Run them.** Expected: FAIL, import error.
- [ ] **Step 3: Implement.** Copy `mask_to_surface`, `_is_axis_aligned`, `TAUBIN_ITER` and `TAUBIN_PASS_BAND` from the reference file unchanged. Drop its `_parse_case` and `main`. Add:

```python
def run(case):
    n_tri = mask_to_surface(case.airway_iso, case.surface_stl)
    if n_tri is None:
        print(f'{case.name}: skip, {case.surface_stl.name} is up to date')
    else:
        print(f'{case.name}: {case.surface_stl.name}  {n_tri} triangles')
```

- [ ] **Step 4: Run them.** Expected: 5 unit passed; regression 3 passed and 1 skipped (NL001 has no stored surface).
- [ ] **Step 5: Commit.** `git add src/alr/surface.py tests && git commit -m "feat: step 11, airway surface mesh"`

---

### Task 18: pipeline.py, cohort.py and the command line

**Files:**
- Create: `src/alr/pipeline.py`, `src/alr/cohort.py`; modify `src/alr/cli.py`
- Test: `tests/unit/test_pipeline.py`, `tests/unit/test_cohort.py`, `tests/unit/test_cli.py`, `tests/unit/test_imports.py`

**Interfaces:**
- Consumes: every step module with `run(case)`.
- Produces: `pipeline.Step(name, run, done)`, `pipeline.STEPS`, `pipeline.run(case, steps=STEPS, out=print) -> float`; `cohort.discover_cases(root, prefix) -> list[str]`, `cohort.run_cohort(root, prefix, force=False, limit=None, device='auto', runner=pipeline.run, out=print) -> tuple[Path, list[dict]]`; `cli.prepare_ad_hoc(ct, case_dir=None, name=None, force=False, device='auto') -> Case`; `cli.main(argv=None) -> int`.

- [ ] **Step 1: Write the failing pipeline test**

```python
import pytest

from alr import pipeline
from alr.layout import Case


def make_steps(log, fail_at=None):
    def step(name, done_path):
        def run(case):
            if name == fail_at:
                raise ValueError('boom')
            log.append(name)
            done_path(case).parent.mkdir(parents=True, exist_ok=True)
            done_path(case).write_text(name)
        return pipeline.Step(name, run, lambda c: done_path(c).exists())
    return (step('one', lambda c: c.dir / 'a.txt'),
            step('two', lambda c: c.dir / 'b.txt'),
            step('three', lambda c: c.dir / 'c.txt'))


@pytest.fixture
def case(tmp_path):
    c = Case('T', tmp_path)
    c.ct.write_bytes(b'ct')
    return c


def test_steps_run_in_order(case):
    log = []
    pipeline.run(case, make_steps(log), out=lambda *a: None)
    assert log == ['one', 'two', 'three']


def test_second_run_skips_finished_steps(case):
    log = []
    pipeline.run(case, make_steps(log), out=lambda *a: None)
    pipeline.run(case, make_steps(log), out=lambda *a: None)
    assert log == ['one', 'two', 'three']


def test_force_runs_everything_again(case):
    log = []
    pipeline.run(case, make_steps(log), out=lambda *a: None)
    pipeline.run(Case('T', case.dir, force=True), make_steps(log), out=lambda *a: None)
    assert log == ['one', 'two', 'three'] * 2


def test_rerun_after_a_crash_resumes_at_the_failed_step(case):
    log = []
    with pytest.raises(RuntimeError, match='two'):
        pipeline.run(case, make_steps(log, fail_at='two'), out=lambda *a: None)
    assert log == ['one']
    pipeline.run(case, make_steps(log), out=lambda *a: None)
    assert log == ['one', 'two', 'three']


def test_missing_ct_is_reported(tmp_path):
    with pytest.raises(FileNotFoundError, match='ct.nii.gz'):
        pipeline.run(Case('T', tmp_path), make_steps([]), out=lambda *a: None)


def test_real_step_order_and_skip_rules(tmp_path):
    names = [s.name for s in pipeline.STEPS]
    assert names == ['segment', 'resample', 'skeleton', 'generations', 'strahler', 'lobe_labels',
                     'lobe_topology', 'dysanapsis', 'export_csv', 'export_html', 'surface']
    case = Case('T', tmp_path)
    done = {s.name: s.done(case) for s in pipeline.STEPS}
    assert not any(done.values())
    case.airway_mask.parent.mkdir(parents=True)
    case.airway_mask.write_bytes(b'x')
    by_name = {s.name: s for s in pipeline.STEPS}
    assert not by_name['segment'].done(case)          # the lobe mask is still missing
    case.lobe_candidates[0].write_bytes(b'x')
    assert by_name['segment'].done(case)
    assert not by_name['generations'].done(case)       # steps 4 to 7 always run
```

- [ ] **Step 2: Implement `pipeline.py`**

```python
import time
from dataclasses import dataclass
from typing import Callable

from alr import (dysanapsis, export_csv, export_html, generations, lobe_labels, lobe_topology,
                 resample, segment, skeleton, strahler, surface)
from alr.layout import Case


@dataclass(frozen=True)
class Step:
    name: str
    run: Callable[[Case], None]
    done: Callable[[Case], bool]


def _has_lobe_mask(case):
    return any(path.exists() for path in case.lobe_candidates)


STEPS = (
    Step('segment', segment.run, lambda c: c.airway_mask.exists() and _has_lobe_mask(c)),
    Step('resample', resample.run, lambda c: c.airway_iso.exists()),
    Step('skeleton', skeleton.run, lambda c: c.teasar_vtk.exists()),
    Step('generations', generations.run, lambda c: False),
    Step('strahler', strahler.run, lambda c: False),
    Step('lobe_labels', lobe_labels.run, lambda c: False),
    Step('lobe_topology', lobe_topology.run, lambda c: False),
    Step('dysanapsis', dysanapsis.run, lambda c: c.smith_csv.exists()),
    Step('export_csv', export_csv.run, lambda c: c.centerline_csv.exists()),
    Step('export_html', export_html.run, lambda c: c.html.exists()),
    Step('surface', surface.run, lambda c: c.surface_stl.exists()),
)


def run(case, steps=STEPS, out=print):
    if not case.ct.exists():
        raise FileNotFoundError(f'{case.ct} not found; put the CT there as ct.nii.gz')
    started = time.time()
    total = len(steps)
    for index, step in enumerate(steps, start=1):
        if not case.force and step.done(case):
            out(f'STEP {index}/{total} {step.name}: skipped, output exists')
            continue
        out(f'STEP {index}/{total} {step.name}')
        try:
            step.run(case)
        except Exception as error:
            raise RuntimeError(f'step {step.name} failed: {error}') from error
    return time.time() - started
```

- [ ] **Step 3: Run the pipeline tests.** `python -m pytest tests/unit/test_pipeline.py -v`. Expected: 6 passed. The one deliberate change from the source runner: the segment step is also not skipped when no lobe mask exists, so a missing lobe mask heals on the next run. Mention it in `PROGRESS.md` and in the README.

- [ ] **Step 4: Write the failing cohort test** (it passes after Step 6). The cohort loop is a port of `run_airway_cohort.py`.

`tests/unit/test_cohort.py`:

```python
import csv

import pytest

from alr import cohort


def make_root(tmp_path):
    for name, with_ct in [('CF001', True), ('CF002', True), ('CF010_t2', True), ('CF003', False), ('NL001', True)]:
        d = tmp_path / name / name
        d.mkdir(parents=True)
        if with_ct:
            (d / 'ct.nii.gz').write_bytes(b'ct')
    (tmp_path / 'CF_notes').mkdir()


def test_discovery_matches_the_prefix_and_number(tmp_path):
    make_root(tmp_path)
    assert cohort.discover_cases(tmp_path, 'CF') == ['CF001', 'CF002', 'CF003', 'CF010_t2']


def test_cohort_records_ok_failed_and_skipped(tmp_path):
    make_root(tmp_path)

    def runner(case):
        if case.name == 'CF002':
            raise RuntimeError('broken mask')

    path, rows = cohort.run_cohort(tmp_path, 'CF', runner=runner, out=lambda *a: None)
    status = {r['case_id']: r['status'] for r in rows}
    assert status == {'CF001': 'ok', 'CF002': 'failed', 'CF003': 'skip_no_ct', 'CF010_t2': 'ok'}
    with open(path, newline='') as f:
        written = list(csv.DictReader(f))
    assert list(written[0].keys()) == ['case_id', 'status', 'runtime_s', 'error']
    assert 'broken mask' in [r for r in written if r['case_id'] == 'CF002'][0]['error']


def test_limit_takes_the_first_cases(tmp_path):
    make_root(tmp_path)
    _, rows = cohort.run_cohort(tmp_path, 'CF', limit=2, runner=lambda c: None, out=lambda *a: None)
    assert [r['case_id'] for r in rows] == ['CF001', 'CF002']
```

- [ ] **Step 5: Write the failing CLI tests** `tests/unit/test_cli.py`

```python
import os

import pytest

from alr import cli


def test_ad_hoc_case_uses_the_ct_folder_and_stem(tmp_path):
    ct = tmp_path / 'scan_7.nii.gz'
    ct.write_bytes(b'ct')
    case = cli.prepare_ad_hoc(ct)
    assert case.name == 'scan_7'
    assert case.dir.resolve() == tmp_path.resolve()
    assert case.ct.resolve() == ct.resolve() and case.ct.is_symlink()


def test_ad_hoc_case_with_output_folder_and_name(tmp_path):
    ct = tmp_path / 'in' / 'scan.nii.gz'
    ct.parent.mkdir()
    ct.write_bytes(b'ct')
    case = cli.prepare_ad_hoc(ct, case_dir=tmp_path / 'out', name='P1', force=True, device='cpu')
    assert (case.name, case.dir.resolve(), case.force, case.device) == ('P1', (tmp_path / 'out').resolve(), True, 'cpu')
    assert case.ct.resolve() == ct.resolve()


def test_path_with_spaces_and_non_ascii_characters(tmp_path):
    ct = tmp_path / 'my scans' / 'scän 1.nii.gz'
    ct.parent.mkdir()
    ct.write_bytes(b'ct')
    case = cli.prepare_ad_hoc(ct)
    assert case.name == 'scän 1'
    assert case.ct.resolve() == ct.resolve()


def test_ct_already_named_ct_is_not_linked_to_itself(tmp_path):
    ct = tmp_path / 'ct.nii.gz'
    ct.write_bytes(b'ct')
    case = cli.prepare_ad_hoc(ct)
    assert case.ct.resolve() == ct.resolve() and not case.ct.is_symlink()


def test_uncompressed_nifti_is_rejected_with_the_fix(tmp_path):
    ct = tmp_path / 'scan.nii'
    ct.write_bytes(b'ct')
    with pytest.raises(ValueError, match='gzip'):
        cli.prepare_ad_hoc(ct)


def test_missing_ct_is_rejected(tmp_path):
    with pytest.raises(FileNotFoundError):
        cli.prepare_ad_hoc(tmp_path / 'nope.nii.gz')


def test_run_needs_exactly_one_of_ct_or_case(capsys):
    assert cli.main(['run']) != 0
    assert cli.main(['run', '--ct', 'a.nii.gz', '--case', 'X', '--root', '.']) != 0


def test_unknown_device_is_rejected(tmp_path, capsys):
    ct = tmp_path / 'a.nii.gz'
    ct.write_bytes(b'ct')
    assert cli.main(['run', '--ct', str(ct), '--device', 'tpu']) != 0
```

- [ ] **Step 6: Implement `cohort.py` and `cli.py`**

`src/alr/cohort.py`:

```python
import csv
import datetime
import re
import time
from pathlib import Path

from alr import pipeline
from alr.layout import Case

FIELDS = ['case_id', 'status', 'runtime_s', 'error']


def discover_cases(root, prefix):
    pattern = re.compile(rf'^{re.escape(prefix)}[0-9]+(_[A-Za-z0-9]+)?$')
    return sorted(p.name for p in Path(root).iterdir() if p.is_dir() and pattern.match(p.name))


def run_cohort(root, prefix, force=False, limit=None, device='auto', runner=pipeline.run, out=print):
    root = Path(root)
    cases = discover_cases(root, prefix)
    if limit:
        cases = cases[:limit]
    out(f'Cohort {prefix}: {len(cases)} cases')
    stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    summary = root / f'{prefix}_airway_cohort_{stamp}.csv'
    rows = []
    for index, name in enumerate(cases, start=1):
        case = Case.from_root(root, name, force=force, device=device)
        if not case.ct.exists():
            out(f'[{index}/{len(cases)}] {name}: skip, no ct.nii.gz')
            rows.append(dict(case_id=name, status='skip_no_ct', runtime_s=0.0, error=''))
        else:
            out(f'[{index}/{len(cases)}] {name}')
            started = time.time()
            try:
                runner(case)
                status, error = 'ok', ''
            except Exception as exc:
                status, error = 'failed', repr(exc)[:200]
            rows.append(dict(case_id=name, status=status, runtime_s=round(time.time() - started, 1), error=error))
        with open(summary, 'w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)
    return summary, rows
```

`src/alr/cli.py` (replaces the Task 1 version):

```python
import argparse
import os
import sys
from pathlib import Path

from alr import __version__
from alr.device import choose_device
from alr.layout import Case


def prepare_ad_hoc(ct, case_dir=None, name=None, force=False, device='auto'):
    """Case for a CT that lives anywhere. The CT is linked as ct.nii.gz in the case folder."""
    ct = Path(ct).expanduser()
    if not ct.exists():
        raise FileNotFoundError(f'CT not found: {ct}')
    ct = ct.resolve()
    if not ct.name.endswith('.nii.gz'):
        raise ValueError(f'expected a .nii.gz file, got {ct.name}; compress it with: gzip -k {ct.name}')
    case_dir = Path(case_dir).expanduser().resolve() if case_dir else ct.parent
    case_dir.mkdir(parents=True, exist_ok=True)
    case = Case(name or ct.name[:-len('.nii.gz')], case_dir, force=force, device=device)
    if case.ct != ct:
        if os.path.lexists(case.ct):
            case.ct.unlink()
        os.symlink(ct, case.ct)
    return case


def build_parser():
    parser = argparse.ArgumentParser(
        prog='alr', description='ALR (airway lung ratio): airway mask, surface and labeled TEASAR centerline from a chest CT')
    parser.add_argument('--version', action='version', version=f'alr {__version__}')
    commands = parser.add_subparsers(dest='command', required=True)

    run = commands.add_parser('run', help='process one case')
    run.add_argument('--ct', help='CT as a .nii.gz file')
    run.add_argument('--case-dir', help='output folder for --ct (default: the folder of the CT)')
    run.add_argument('--name', help='case name for --ct (default: the file name)')
    run.add_argument('--case', help='case ID in the ROOT/ID/ID/ layout')
    run.add_argument('--root', help='folder that holds ID/ID/ for --case')
    run.add_argument('--force', action='store_true', help='run every step again')
    run.add_argument('--device', default='auto', help='auto, gpu, gpu:N, mps or cpu')

    cohort = commands.add_parser('cohort', help='process every case of a prefix, one at a time')
    cohort.add_argument('prefix')
    cohort.add_argument('--root', required=True)
    cohort.add_argument('--force', action='store_true')
    cohort.add_argument('--limit', type=int)
    cohort.add_argument('--device', default='auto')

    commands.add_parser('setup', help='download the TotalSegmentator models and set the weights folder')
    commands.add_parser('check', help='print versions, device and which models are present')
    return parser


def _dispatch(parser, args):
    if args.command == 'run':
        if bool(args.ct) == bool(args.case):
            parser.error('use exactly one of --ct or --case')
        if args.device != 'auto':
            choose_device(args.device)
        if args.ct:
            case = prepare_ad_hoc(args.ct, args.case_dir, args.name, args.force, args.device)
        else:
            if not args.root:
                parser.error('--case needs --root')
            case = Case.from_root(args.root, args.case, force=args.force, device=args.device)
        from alr import pipeline
        print(f'done in {pipeline.run(case):.1f} s')
        return 0
    if args.command == 'cohort':
        if args.device != 'auto':
            choose_device(args.device)
        from alr import cohort
        summary, rows = cohort.run_cohort(args.root, args.prefix, args.force, args.limit, args.device)
        failed = sum(1 for r in rows if r['status'] == 'failed')
        print(f'{len(rows)} cases, {failed} failed, summary: {summary}')
        return 0
    if args.command == 'setup':
        from alr import weights
        print(f'weights folder: {weights.setup()}')
        return 0
    from alr import weights
    lines, ok = weights.check()
    print('\n'.join(lines))
    return 0 if ok else 1


def main(argv=None):
    parser = build_parser()
    try:
        return _dispatch(parser, parser.parse_args(argv))
    except SystemExit as stop:
        return stop.code if isinstance(stop.code, int) else 1
    except (FileNotFoundError, ValueError, PermissionError, RuntimeError) as error:
        print(f'alr: {error}', file=sys.stderr)
        return 1
```

- [ ] **Step 7: Write and pass the import test** `tests/unit/test_imports.py`: for every module name in `alr`, run `python -c "import alr.<name>"` in a subprocess with `cwd=tmp_path` and assert return code 0, empty stdout and no file created in `tmp_path`. Modules: `layout device weights segment resample skeleton graph generations strahler lobe_labels lobe_topology dysanapsis export_csv export_html surface pipeline cohort cli`.

- [ ] **Step 8: Run the whole test folder.** `python -m pytest -v`. Expected: all unit tests pass; regression tests pass with `ALR_DATA` set and are skipped without it.
- [ ] **Step 9: Commit.** `git add src tests && git commit -m "feat: pipeline order, cohort loop and the alr command line"`

---

### Task 19: README and documentation checks

**Files:**
- Modify: `README.md`; create `tests/unit/test_docs.py`

- [ ] **Step 1: Write the failing test**

```python
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_readme_has_the_quick_start_and_required_statements():
    text = (ROOT / 'README.md').read_text()
    for needed in ['git clone https://github.com/kiwixiao/ALR.git', 'conda env create -f environment.yml',
                   'conda activate ALR', 'alr setup', 'alr run --ct', 'airway lung ratio',
                   'kimimaro', 'GPL', 'usage statistics', 'MIT', 'TotalSegmentator']:
        assert needed in text, needed


def test_documents_avoid_em_dashes_and_en_dashes():
    for name in ['README.md', 'PROGRESS.md', 'CLAUDE.md', 'docs/HANDOFF.md']:
        text = (ROOT / name).read_text()
        assert '\u2014' not in text and '\u2013' not in text, name
```

- [ ] **Step 2: Run it.** Expected: FAIL.
- [ ] **Step 3: Write the README.** Sections: what ALR is (airway lung ratio, one paragraph); requirements (Linux, conda, internet on the first run for about 0.8 GB of model weights, a CT as NIfTI `.nii.gz`, a GPU is optional and the CPU works but is slow); the four commands (`git clone`, `conda env create -f environment.yml`, `conda activate ALR`, `alr setup`, then `alr run --ct scan.nii.gz --case-dir out/`); what `alr setup` and `alr check` do; the list of output files from spec section 6 with one line each; the fragment convention (`-1`); a note that usage statistics of TotalSegmentator are turned off by `alr setup` and how to turn them on (`set_config_key('send_usage_stats', True)` in `totalsegmentator.config`); the license table (ALR MIT; kimimaro GPL 3 or later; pyvista, nibabel, plotly MIT; vtk, scipy, numpy, networkx BSD; matplotlib PSF; TotalSegmentator Apache 2.0) with the sentence that a bundle which redistributes kimimaro carries its GPL terms; the TotalSegmentator citation, copied from the TotalSegmentator repository README on the day you write this; an honest limits paragraph (models trained on adult CT; segmentation quality on other scans is not validated here; masks may contain small false positive pieces outside the lungs, and `-1` fragments exist). Follow the writing style in Global Constraints.
- [ ] **Step 4: Run it.** Expected: 2 passed.
- [ ] **Step 5: Commit.** `git add README.md tests/unit/test_docs.py && git commit -m "docs: README with the quick start, outputs and licenses"`

---

### Task 20: Clean machine acceptance

**Files:**
- Modify: `docs/baseline.md` (append), `PROGRESS.md`

- [ ] **Step 1: Fresh environment from the file.** On a clean Linux machine or a clean container (`docker run -it --rm condaforge/miniforge3 bash`), run exactly the README commands: clone, `conda env create -f environment.yml`, `conda activate ALR`, `alr setup`, `alr check`. Expected: `alr check` exits 0 and lists the four models as present. Record any pin that had to change.
- [ ] **Step 2: Unit tests.** `python -m pytest tests/unit -v`. Expected: all pass.
- [ ] **Step 3: Regression.** Extract the data pack (Task 2) and run `ALR_DATA=... python -m pytest tests/regression -v`. Expected: all pass.
- [ ] **Step 4: Whole pipeline from the stored masks.** Copy one case, for example CF121, into a scratch folder with its CT and its `totalseg_vessels/` and `totalseg_merged/` folders (the author supplies the CT, about 280 MB, and it is never committed). Run `alr run --ct <CT> --case-dir <scratch>`. Expected: steps 1 is skipped, steps 2 to 11 run, and the files equal the stored ones.
- [ ] **Step 5: Segmentation on this machine.** Run `alr run` on the same CT in a folder without masks. Compute the Dice between the new `totalseg_vessels/lung_airways.nii.gz` and the stored one, and the vertex count and maximum generation of the new centerline against the stored ones. Write the numbers in `docs/baseline.md`. Do not set a pass threshold; report the numbers to the author.
- [ ] **Step 6: Report.** Update `PROGRESS.md` (NEXT STEP, STATUS, SESSION LOG), commit, push the branch, and tell the author that the branch is ready for review. Do not merge.

---

## Self-Review

- **Spec coverage:** interface (Tasks 1, 5, 18); layout and file contract (3); steps 1 to 11 (6, 8 to 17); environment, weights and usage statistics (1, 5); portability changes (3, 4, 6, 8, 18); license and dependencies (1, 19); testing layers (7 to 18); rollout phase A and B (2, 8 to 17), phase C is out of scope and needs the author; Linux acceptance (20); repository housekeeping (Global Constraints).
- **Known deliberate deviation from the source:** the segment step is not considered done when no lobe mask exists (Task 18). Everything else in the pipeline order and skip rules is the same as `run_airway_pipeline.py`.
- **Placeholders:** none. Port tasks name the source file, the lines, the exact edits and a grep check that finds every leftover global.
- **Type consistency:** `Case` properties and the `run(case)` interface are defined in Task 3 and used unchanged afterwards; `graph` functions are defined in Task 7 before Tasks 10, 11, 13 and 16 use them.
