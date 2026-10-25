#!/bin/bash
#SBATCH --job-name=carriers-report
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err
#
# Tables and the pre-registered RQ1 evidence standard, from the saved JSON only:
# CPU, no GPU, no rerun of generation.
#
#   sbatch slurm/report.sbatch
set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
source slurm/env.sh
OUT="${TABLES:-$CARRIERS_RESULTS/tables}"
banner "report -> $OUT"

python -m analysis.report --results "$CARRIERS_RESULTS" --out "$OUT"

echo
echo "== report.md"
cat "$OUT/report.md"
echo "end=$(date)"
