#!/usr/bin/env bash
# Shared job preamble for Unity. Sourced, never executed.
#
#   source slurm/env.sh          # cd to the repo, activate the venv, set caches
#   PORT=$(pick_port)            # a free port on this node
#   serve_vllm "$MODEL" "$PORT" 0.85 agent   # start vLLM, block until healthy
#
# Everything here is cluster-shaped but not Unity-only: the paths fall back to
# the repo when /work and /scratch are absent, so the same scripts run on a
# laptop or any other Slurm site.

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$PROJECT_ROOT"

# Optional per-user settings: SLURM_ACCOUNT, HF_HOME, HF_TOKEN, CARRIERS_RESULTS.
# A value the submitting script exported wins over .env, so chain.sh's flags do
# not get quietly overwritten by a stale default.
_keep_results="${CARRIERS_RESULTS:-}"
_keep_config="${CONFIG:-}"
if [[ -f "$PROJECT_ROOT/.env" ]]; then
  set -a; source "$PROJECT_ROOT/.env"; set +a
fi
[[ -n "$_keep_results" ]] && export CARRIERS_RESULTS="$_keep_results"
[[ -n "$_keep_config" ]] && export CONFIG="$_keep_config"
unset _keep_results _keep_config

# Colosseum's venv is the one with vLLM and Terrarium in it (RUNNING.md).
VENV="${CARRIERS_VENV:-$PROJECT_ROOT/colosseum/.venv}"
if [[ -f "$VENV/bin/activate" ]]; then
  source "$VENV/bin/activate"
else
  echo "!! no venv at $VENV -- run: sbatch slurm/setup.sbatch" >&2
  return 1 2>/dev/null || exit 1
fi

export COLOSSEUM_ROOT="${COLOSSEUM_ROOT:-$PROJECT_ROOT/colosseum}"
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_DISABLE_TELEMETRY=1

# Weights are ~12GB for the two models: small, but still not the home quota.
if [[ -z "${HF_HOME:-}" ]]; then
  for candidate in "/work/pi_${SLURM_ACCOUNT:-none}/$USER" "/scratch/workspace/$USER" "/work/$USER"; do
    if [[ -d "$candidate" ]]; then export HF_HOME="$candidate/hf_cache"; break; fi
  done
  export HF_HOME="${HF_HOME:-$PROJECT_ROOT/.hf_cache}"
  if [[ "$HF_HOME" == "$PROJECT_ROOT/.hf_cache" ]]; then
    echo "!! HF_HOME unset -> caching ~12GB of weights in the repo at $HF_HOME."
    echo "!! Point HF_HOME at /work or /scratch in .env instead."
  fi
fi
mkdir -p "$HF_HOME"

N="${SLURM_CPUS_PER_TASK:-4}"
export OMP_NUM_THREADS="$N" MKL_NUM_THREADS="$N" OPENBLAS_NUM_THREADS="$N"

module load cuda/12.6 2>/dev/null || true

export CARRIERS_RESULTS="${CARRIERS_RESULTS:-$PROJECT_ROOT/results/sprint}"

# Open-weight models throughout, served locally: no API key is used anywhere in
# this project, for the agents or for the judge. Both default to Qwen2.5, which
# is ungated on the Hub; a Llama swap needs an accepted license and HF_TOKEN.
export AGENT_MODEL="${AGENT_MODEL:-Qwen/Qwen2.5-3B-Instruct}"
export JUDGE_MODEL="${JUDGE_MODEL:-Qwen/Qwen2.5-7B-Instruct-AWQ}"
# vLLM's tool-call parser is per model family; the agents are useless without it.
case "${TOOL_CALL_PARSER:-}" in
  "") case "$AGENT_MODEL" in
        *[Qq]wen*)  TOOL_CALL_PARSER=hermes ;;
        *[Ll]lama*) TOOL_CALL_PARSER=llama3_json ;;
        *[Mm]istral*) TOOL_CALL_PARSER=mistral ;;
        *) TOOL_CALL_PARSER=hermes ;;
      esac ;;
esac
export TOOL_CALL_PARSER
# Llama 3.x also needs vLLM's tool chat template; Qwen2.5 has one built in.
export VLLM_CHAT_TEMPLATE="${VLLM_CHAT_TEMPLATE:-}"

# ---------------------------------------------------------------- vLLM helpers

# A free port on this node. Array tasks can share a node, so the port cannot be
# a constant and cannot live in the config YAML; the runner takes it on the CLI.
pick_port() {
  python - <<'PY'
import socket
s = socket.socket()
s.bind(("127.0.0.1", 0))
print(s.getsockname()[1])
s.close()
PY
}

VLLM_PID=""

# serve_vllm <model> <port> <gpu-util> <name> [extra vllm args...]
serve_vllm() {
  local model="$1" port="$2" util="$3" name="$4"; shift 4
  local log="logs/vllm/${name}-${SLURM_JOB_ID:-local}.log"
  mkdir -p logs/vllm
  echo "[$name] serving $model on :$port (log: $log)"
  python -m vllm.entrypoints.openai.api_server \
    --model "$model" \
    --served-model-name "$model" \
    --host 127.0.0.1 --port "$port" \
    --max-model-len "${VLLM_MAX_MODEL_LEN:-8192}" \
    --gpu-memory-utilization "$util" \
    --download-dir "$HF_HOME" \
    --trust-remote-code \
    "$@" > "$log" 2>&1 &
  VLLM_PID=$!
  trap 'stop_vllm' EXIT
  local deadline=$(( SECONDS + ${VLLM_STARTUP_TIMEOUT:-1800} ))
  while (( SECONDS < deadline )); do
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
      echo "!! [$name] vLLM died during startup; last 40 lines:" >&2
      tail -n 40 "$log" >&2
      return 1
    fi
    if curl -sf "http://127.0.0.1:${port}/v1/models" >/dev/null 2>&1; then
      echo "[$name] ready after ${SECONDS}s"
      return 0
    fi
    sleep 10
  done
  echo "!! [$name] not healthy within ${VLLM_STARTUP_TIMEOUT:-1800}s; see $log" >&2
  tail -n 40 "$log" >&2
  return 1
}

stop_vllm() {
  if [[ -n "$VLLM_PID" ]] && kill -0 "$VLLM_PID" 2>/dev/null; then
    echo "[vllm] stopping $VLLM_PID"
    kill "$VLLM_PID" 2>/dev/null || true
    wait "$VLLM_PID" 2>/dev/null || true
  fi
}

banner() {
  echo "=============================================================="
  echo " $* "
  echo " node=$(hostname) job=${SLURM_JOB_ID:-local}${SLURM_ARRAY_TASK_ID:+ task=$SLURM_ARRAY_TASK_ID}"
  echo " gpu=${CUDA_VISIBLE_DEVICES:-none} commit=$(git rev-parse --short HEAD 2>/dev/null || echo '?')"
  echo " HF_HOME=$HF_HOME  results=$CARRIERS_RESULTS"
  echo " start=$(date)"
  echo "=============================================================="
}
