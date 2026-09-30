#!/usr/bin/env bash
# Pack the minimum files needed to reproduce the airway chain for the four
# regression cases. Run this on the machine that holds the av_phenotype cohort,
# from the av_phenotype folder. The CTs are not included (about 74 MB in total).
#
#   bash pack_regression_data.sh [final_analysis_folder] [output_tar]
#
# Extract on the other machine with:  tar -xf alr_regression_data.tar -C DATA
# and pass  --data DATA/final_analysis  to tools/reproduce_reference.py.
set -euo pipefail

ROOT="${1:-final_analysis}"
OUT="${2:-alr_regression_data.tar}"
CASES="CF121 CF008 NL001 CF005"

files=()
for c in $CASES; do
  d="$ROOT/$c/$c"
  for f in totalseg_vessels/lung_airways.nii.gz lung_airways_iso.nii.gz \
           totalseg_merged/lung_lobes_multilabel.nii.gz lobes_ml.nii.gz; do
    [ -e "$d/$f" ] && files+=("$d/$f")
  done
  while IFS= read -r f; do files+=("$f"); done < <(find "$d/airway_results" -maxdepth 1 -type f | sort)
done

tar -cf "$OUT" "${files[@]}"
echo "wrote $OUT with ${#files[@]} files"
