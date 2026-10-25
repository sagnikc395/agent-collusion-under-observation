#!/usr/bin/env bash
# Start the agent model (port 8020) and, with --judge, the judge model (8021).
# Terrarium can start the agent server itself, but starting it once here keeps
# concurrent episodes from racing to launch it.
set -euo pipefail

AGENT_MODEL="${AGENT_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
JUDGE_MODEL="${JUDGE_MODEL:-Qwen/Qwen2.5-14B-Instruct}"
LOG_DIR="${LOG_DIR:-logs/vllm}"
mkdir -p "$LOG_DIR"

start() {
  local model="$1" port="$2" util="$3" name="$4"
  if curl -sf "http://127.0.0.1:${port}/v1/models" >/dev/null 2>&1; then
    echo "[$name] already serving on :$port"
    return
  fi
  echo "[$name] starting $model on :$port"
  nohup python -m vllm.entrypoints.openai.api_server \
    --model "$model" \
    --port "$port" \
    --host 127.0.0.1 \
    --max-model-len 16000 \
    --gpu-memory-utilization "$util" \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    --trust-remote-code \
    > "$LOG_DIR/${name}.log" 2>&1 &
  echo "[$name] logs: $LOG_DIR/${name}.log"
}

wait_ready() {
  local port="$1" name="$2"
  for _ in $(seq 1 120); do
    if curl -sf "http://127.0.0.1:${port}/v1/models" >/dev/null 2>&1; then
      echo "[$name] ready"
      return 0
    fi
    sleep 10
  done
  echo "[$name] did not come up; see $LOG_DIR/${name}.log" >&2
  return 1
}

if [[ "${1:-agent}" == "--judge" ]]; then
  # Run after generation: the judge needs the memory the agent server holds.
  start "$JUDGE_MODEL" 8021 0.90 judge
  wait_ready 8021 judge
else
  start "$AGENT_MODEL" 8020 0.85 agent
  wait_ready 8020 agent
fi
