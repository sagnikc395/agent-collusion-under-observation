#!/bin/bash
#SBATCH --job-name=carriers-setup
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err
#
# One-time setup, on a CPU node (GPU hours are not for downloading):
#   mkdir -p logs/slurm && sbatch slurm/setup.sbatch
#
# Clones Colosseum, builds its venv with the vLLM extra, prefetches both model
# checkpoints into HF_HOME, and runs the offline test suite. Idempotent: a
# re-submit skips whatever is already there, so it is also the repair script.
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$PROJECT_ROOT"
mkdir -p logs/slurm
if [[ -f .env ]]; then set -a; source .env; set +a; fi

echo "node=$(hostname) start=$(date)"

# ---- uv ---------------------------------------------------------------------
if ! command -v uv >/dev/null 2>&1; then
  module load uv 2>/dev/null || true
fi
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
echo "uv: $(uv --version)"

# ---- Colosseum (external dependency, used unmodified) -----------------------
export COLOSSEUM_ROOT="${COLOSSEUM_ROOT:-$PROJECT_ROOT/colosseum}"
if [[ ! -d "$COLOSSEUM_ROOT/.git" ]]; then
  echo "== cloning Colosseum into $COLOSSEUM_ROOT"
  git clone --recurse-submodules https://github.com/umass-ai-safety/colosseum.git "$COLOSSEUM_ROOT"
else
  echo "== Colosseum present; updating submodules"
  git -C "$COLOSSEUM_ROOT" submodule update --init --recursive
fi

echo "== building the venv (vLLM extra)"
cd "$COLOSSEUM_ROOT"
[[ -d .venv ]] || uv venv --python 3.11 .venv
VIRTUAL_ENV="$COLOSSEUM_ROOT/.venv" uv sync --no-install-project --extra vllm --link-mode=copy
# Our analysis and tests, on top of Terrarium's own dependencies.
VIRTUAL_ENV="$COLOSSEUM_ROOT/.venv" uv pip install --link-mode=copy \
  pytest numpy scipy pandas pyyaml requests tqdm names
cd "$PROJECT_ROOT"

# ---- model weights ----------------------------------------------------------
source slurm/env.sh
banner "prefetching weights into $HF_HOME"
python - <<'PY'
import os
from huggingface_hub import snapshot_download

for model in (os.environ["AGENT_MODEL"], os.environ["JUDGE_MODEL"]):
    print(f"== {model}", flush=True)
    path = snapshot_download(model, max_workers=8)
    print(f"   -> {path}", flush=True)
PY

# ---- wiring check (no GPU, no API) ------------------------------------------
echo "== offline tests"
python -m pytest tests -q

echo "setup OK; next:  bash slurm/chain.sh"
echo "end=$(date)"
