#!/usr/bin/env bash
# Rebuild every notebook from nb_src/ and execute it in place (outputs saved into the .ipynb).
# Usage: tools/run_all.sh [pattern]      e.g. tools/run_all.sh 2[0-9]
#        NN_DEVICE=cpu tools/run_all.sh  to force CPU
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-.venv/bin/python}
$PY tools/build.py
for nb in notebooks/${1:-}*.ipynb; do
  start=$(date +%s)
  if $PY -m nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=1800 "$nb" >/dev/null 2>&1; then
    echo "ok    $(( $(date +%s) - start ))s  $nb"
  else
    echo "FAIL  $nb"; exit 1
  fi
done
