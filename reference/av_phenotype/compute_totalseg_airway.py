# Auto-invoke TotalSegmentator (conda env totalseg213) for airway + lobe masks
# if they are not already on disk for this case.
#
# Idempotent: skips immediately when both expected outputs already exist.
#
# Outputs (per the canonical layout in totalseg_lobes_pipeline.md):
#   {CASE_DIR}/totalseg_vessels/lung_airways.nii.gz    (lung_vessels task)
#   {CASE_DIR}/totalseg_vessels/lung_arteries.nii.gz
#   {CASE_DIR}/totalseg_vessels/lung_veins.nii.gz
#   {CASE_DIR}/totalseg_vessels/lung_airways_wall.nii.gz
#   {CASE_DIR}/totalseg_merged/lung_lobes_multilabel.nii.gz   (5-lobe multilabel)
#
# Also creates a top-level symlink {CASE_DIR}/lung_airways.nii.gz pointing at the
# vessels-dir copy so downstream stages have a single canonical path.

import os, subprocess, sys

CASE = os.environ.get('CASE_ID', 'CF001')
CASE_DIR = os.environ.get(
    'CASE_DIR',
    f'/path/to/av_phenotype/final_analysis/{CASE}/{CASE}')
CT = f'{CASE_DIR}/ct.nii.gz'

VESSELS_DIR = f'{CASE_DIR}/totalseg_vessels'
MERGED_DIR  = f'{CASE_DIR}/totalseg_merged'
AIRWAY_OUT  = f'{VESSELS_DIR}/lung_airways.nii.gz'
LOBES_OUT   = f'{MERGED_DIR}/lung_lobes_multilabel.nii.gz'

CONDA_ENV = 'totalseg213'   # must already exist; see totalseg_lobes_pipeline.md
DEVICE = 'mps'              # use Apple Metal GPU; change to 'cpu' or 'gpu' if needed


def conda_run(cmd):
    full = ['conda', 'run', '-n', CONDA_ENV] + cmd
    print('  $ ' + ' '.join(full))
    r = subprocess.run(full)
    return r.returncode == 0


def main():
    print(f'\n══════════════ TotalSegmentator (airway + lobes)  ({CASE}) ══════════════')
    if not os.path.exists(CT):
        print(f'  CT not found at {CT} — abort.'); sys.exit(1)

    have_airway = os.path.exists(AIRWAY_OUT)
    have_lobes  = os.path.exists(LOBES_OUT)
    if have_airway and have_lobes:
        print(f'  both outputs already present:')
        print(f'    {AIRWAY_OUT}'); print(f'    {LOBES_OUT}')
        print('  skipping TS invocation.')
    else:
        # 1) lung_vessels task → airway + arteries + veins
        if not have_airway:
            os.makedirs(VESSELS_DIR, exist_ok=True)
            print(f'  running TS lung_vessels (output: {VESSELS_DIR}) ...')
            if not conda_run(['TotalSegmentator', '-i', CT, '-o', VESSELS_DIR,
                              '-ta', 'lung_vessels', '-d', DEVICE]):
                print('  TS lung_vessels failed.'); sys.exit(1)
        else:
            print(f'  ⤷ skip lung_vessels (airway mask already at {AIRWAY_OUT})')

        # 2) total task → 5-lobe multilabel
        if not have_lobes:
            os.makedirs(MERGED_DIR, exist_ok=True)
            print(f'  running TS total (output: {LOBES_OUT}) ...')
            if not conda_run(['TotalSegmentator', '-i', CT, '-o', LOBES_OUT,
                              '-ta', 'total', '-ml', '-rmb', '-d', DEVICE,
                              '-rs', 'lung_upper_lobe_left',
                                     'lung_lower_lobe_left',
                                     'lung_upper_lobe_right',
                                     'lung_middle_lobe_right',
                                     'lung_lower_lobe_right']):
                print('  TS total failed; continuing (lobe mask absent → '
                      'Smith dysanapsis V_lung will be skipped).')
        else:
            print(f'  ⤷ skip TS total (lobe mask already at {LOBES_OUT})')

    # 3) Top-level symlinks so downstream stages have one canonical path
    canon_airway = f'{CASE_DIR}/lung_airways.nii.gz'
    if os.path.exists(AIRWAY_OUT) and not os.path.exists(canon_airway):
        try:
            os.symlink(AIRWAY_OUT, canon_airway)
            print(f'  symlinked {canon_airway} → {AIRWAY_OUT}')
        except OSError as e:
            print(f'  could not create symlink {canon_airway}: {e}')


if __name__ == '__main__':
    main()
    print('\n[Done]')
