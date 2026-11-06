#!/usr/bin/env bash
set -euo pipefail

usage() { echo "Usage: $0 [--quick]"; exit 1; }

QUICK=FALSE
while [[ $# -gt 0 ]]; do
  case $1 in
    --quick) QUICK=TRUE; shift;;
    *) usage;;
  esac
done

# Restore renv packages if needed
Rscript -e "if (!requireNamespace('targets', quietly = TRUE)) renv::restore()"

GDPM_QUICK=$QUICK Rscript -e "targets::tar_make()"
