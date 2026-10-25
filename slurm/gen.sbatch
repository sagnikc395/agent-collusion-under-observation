#!/bin/bash
#SBATCH --job-name=carriers-gen
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/slurm/%x-%A_%a.out
#SBATCH --error=logs/slurm/%x-%A_%a.err
#
# Generation: one array task per setting, all of that setting's seeds inside it.
# One GPU, one vLLM server per task, so tasks are independent and a preemption
# costs at most the episodes in flight (finished ones are skipped on resubmit).
#
# Submitted by slurm/chain.sh. On its own, the array range must match the
# setting list -- chain.sh works that out for you:
#
#   SETTINGS=plan SEEDS=1-20 sbatch --array=0-10%4 slurm/gen.sbatch
#   SETTINGS=core SEEDS=1-10 sbatch --array=0-4 slurm/gen.sbatch
#   SETTINGS=watched sbatch --array=0 slurm/gen.sbatch        # one setting
#
# Re-running one failed task: sbatch --array=<i> slurm/gen.sbatch
set -euo pipefail

# Slurm runs the job script from a spool copy, so $0 is useless for finding the
# repo; it does start the job in the submitting directory.
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source slurm/env.sh

CONFIG="${CONFIG:-configs/unity.yaml}"
SETTINGS="${SETTINGS:-plan}"
SEEDS="${SEEDS:-}"
CONCURRENCY="${CONCURRENCY:-4}"

# Expand groups (`plan`, `core`, `rq4`, ...) the same way the runner does, so the
# array index means the same setting here and in chain.sh. A read loop, not
# mapfile, so this also works under the bash 3.2 of a macOS shell.
LIST=()
N_SETTINGS=0
while IFS= read -r name; do
  [[ -z "$name" ]] && continue
  LIST+=("$name")
  N_SETTINGS=$(( N_SETTINGS + 1 ))
done < <(python slurm/_settings.py "$SETTINGS")

if (( N_SETTINGS == 0 )); then
  echo "!! SETTINGS='$SETTINGS' expanded to nothing" >&2
  exit 1
fi

I="${SLURM_ARRAY_TASK_ID:-0}"
if (( I >= N_SETTINGS )); then
  echo "task $I: nothing to do ($N_SETTINGS settings in '$SETTINGS')"
  exit 0
fi
SETTING="${LIST[$I]}"

banner "generation: $SETTING (seeds=${SEEDS:-from config})"

PORT="$(pick_port)"
# The agents need tool calling. The parser is model-family-specific: hermes for
# Qwen2.5, llama3_json for Llama 3.x (see slurm/env.sh).
VLLM_EXTRA=(--enable-auto-tool-choice --tool-call-parser "$TOOL_CALL_PARSER")
if [[ -n "${VLLM_CHAT_TEMPLATE:-}" ]]; then
  VLLM_EXTRA+=(--chat-template "$VLLM_CHAT_TEMPLATE")
fi
serve_vllm "$AGENT_MODEL" "$PORT" "${GPU_UTIL:-0.85}" "agent-$SETTING" "${VLLM_EXTRA[@]}"

ARGS=(--config "$CONFIG" --settings "$SETTING" --out-dir "$CARRIERS_RESULTS"
      --max-concurrent-runs "$CONCURRENCY" --vllm-host 127.0.0.1 --vllm-port "$PORT")
[[ -n "$SEEDS" ]] && ARGS+=(--seeds "$SEEDS")
[[ -n "${OVERWRITE:-}" ]] && ARGS+=(--overwrite)

rc=0
python -m carriers.runner "${ARGS[@]}" || rc=$?

stop_vllm
echo "== $SETTING done (exit $rc)  $(date)"
# Informational only: `|| true` because a failing count must not turn a good
# task into a failed one (an empty results tree makes find exit non-zero).
N=$(find "$CARRIERS_RESULTS/runs" -path "*/$SETTING/*/metrics.json" 2>/dev/null | wc -l | tr -d ' ') || true
echo "== episodes on disk for $SETTING: ${N:-0}"
exit $rc
