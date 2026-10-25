#!/bin/bash
#SBATCH --job-name=carriers-judge
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err
#
# The intent judge, after generation. Colosseum's three prompts and its parsing,
# unchanged; only the transport is a local 14B server, so the scores are only
# comparable within this project.
#
#   sbatch slurm/judge.sbatch                     # all settings on disk
#   SETTINGS=watched,no_channel sbatch slurm/judge.sbatch
#
# Resumable: a run that already has judge.json is skipped unless OVERWRITE=1.
set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
source slurm/env.sh
banner "judging $CARRIERS_RESULTS with $JUDGE_MODEL"

PORT="$(pick_port)"
serve_vllm "$JUDGE_MODEL" "$PORT" "${GPU_UTIL:-0.90}" judge

ARGS=(--results "$CARRIERS_RESULTS" --base-url "http://127.0.0.1:${PORT}/v1" --model "$JUDGE_MODEL")
[[ -n "${SETTINGS:-}" ]] && ARGS+=(--settings "$SETTINGS")
[[ -n "${OVERWRITE:-}" ]] && ARGS+=(--overwrite)

rc=0
python -m analysis.judge "${ARGS[@]}" || rc=$?

stop_vllm
echo "== judge done (exit $rc)  $(date)"
exit $rc
