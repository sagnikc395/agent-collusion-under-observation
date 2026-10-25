#!/usr/bin/env bash
# The whole experiment as one dependent chain of Slurm jobs. Run from a login node.
#
#   bash slurm/chain.sh                 # the sprint grid: 6 settings x 8 seeds
#   bash slurm/chain.sh --full          # the 11-setting, 20-seed plan (RQ1-RQ6)
#   bash slurm/chain.sh --stage judge   # just that stage, on what is on disk
#   bash slurm/chain.sh --dry-run       # print the submissions, send nothing
#
# Nothing is lost to a timeout or a preemption: every stage skips work that is
# already on disk, so re-running this script resumes the chain.
#
# Knobs (flags, or the same names in .env):
#   SLURM_ACCOUNT   Slurm account, if your Unity account is not the default
#   CONFIG          experiment YAML            (configs/sprint.yaml)
#   SETTINGS        setting or group list      (sprint -- RQ1-RQ4, 6 settings)
#   SEEDS           seed list or range         (1-8)
#   PILOT_SEEDS     seeds for the RQ2 pilot    (1-4)
#   MAXPAR          GPU jobs in flight at once (4)
#   CARRIERS_RESULTS  results root             (results/sprint)
set -euo pipefail

cd "$(dirname "$0")/.."
PROJECT_ROOT="$PWD"
if [[ -f .env ]]; then set -a; source .env; set +a; fi

CONFIG="${CONFIG:-configs/sprint.yaml}"
SETTINGS="${SETTINGS:-sprint}"
SEEDS="${SEEDS:-1-8}"
PILOT_SEEDS="${PILOT_SEEDS:-1-4}"
MAXPAR="${MAXPAR:-4}"
CARRIERS_RESULTS="${CARRIERS_RESULTS:-$PROJECT_ROOT/results/sprint}"
STAGE="all"
DRY=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --full)      CONFIG="configs/unity.yaml"; SETTINGS="plan"; SEEDS="1-20"; PILOT_SEEDS="1-5"; shift ;;
    --fast)      SETTINGS="core"; SEEDS="1-4"; PILOT_SEEDS="1-4"; shift ;;
    --stage)     STAGE="$2"; shift 2 ;;
    --settings)  SETTINGS="$2"; shift 2 ;;
    --seeds)     SEEDS="$2"; shift 2 ;;
    --config)    CONFIG="$2"; shift 2 ;;
    --results)   CARRIERS_RESULTS="$2"; shift 2 ;;
    --maxpar)    MAXPAR="$2"; shift 2 ;;
    --dry-run)   DRY=1; shift ;;
    -h|--help)   sed -n '2,25p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

mkdir -p logs/slurm logs/vllm "$CARRIERS_RESULTS"
export CARRIERS_RESULTS CONFIG

ACCT=()
if [[ -n "${SLURM_ACCOUNT:-}" ]]; then ACCT=(-A "$SLURM_ACCOUNT"); fi

submit() {  # submit <description> <sbatch args...>
  local what="$1"; shift
  if [[ -n "$DRY" ]]; then
    echo "  [dry-run] $what: sbatch ${ACCT[*]-} $*" >&2
    echo "DRYRUN"
    return 0
  fi
  local id
  id="$(sbatch --parsable "${ACCT[@]}" "$@")"
  echo "  $what -> job $id" >&2
  echo "$id"
}

# How many array tasks a setting list needs: the array index selects a setting,
# and gen.sbatch expands the list with exactly this call.
count_settings() {
  python slurm/_settings.py "$1" | wc -l | tr -d ' '
}

VENV="${CARRIERS_VENV:-$PROJECT_ROOT/colosseum/.venv}"
export PATH="$VENV/bin:$PATH"

N_PILOT="$(count_settings pilot)"
N_MAIN="$(count_settings "$SETTINGS")"

echo "config=$CONFIG settings=$SETTINGS ($N_MAIN tasks) seeds=$SEEDS results=$CARRIERS_RESULTS"
echo

setup_dep=""
if [[ ! -x "$VENV/bin/python" || ! -f "$PROJECT_ROOT/colosseum/experiments/collusion/run.py" ]]; then
  echo "== no venv or no Colosseum clone yet: setup goes first"
  SETUP_ID="$(submit setup slurm/setup.sbatch)"
  setup_dep="--dependency=afterok:$SETUP_ID"
fi

dep="$setup_dep"

if [[ "$STAGE" == "all" || "$STAGE" == "pilot" ]]; then
  echo "== RQ2 pilot: benign/emergent/coalition x $PILOT_SEEDS"
  PILOT_ID="$(submit pilot \
    ${dep:+$dep} --array="0-$((N_PILOT-1))%$MAXPAR" \
    --export=ALL,SETTINGS=pilot,SEEDS="$PILOT_SEEDS",CONFIG="$CONFIG",CARRIERS_RESULTS="$CARRIERS_RESULTS" \
    slurm/gen.sbatch)"
  # afterany, not afterok: a single failed setting should not strand the sweep,
  # and the sweep re-runs whatever has no metrics.json anyway.
  dep="--dependency=afterany:$PILOT_ID"
fi

if [[ "$STAGE" == "all" || "$STAGE" == "gen" ]]; then
  echo "== sweep: $SETTINGS x $SEEDS"
  GEN_ID="$(submit sweep \
    ${dep:+$dep} --array="0-$((N_MAIN-1))%$MAXPAR" \
    --export=ALL,SETTINGS="$SETTINGS",SEEDS="$SEEDS",CONFIG="$CONFIG",CARRIERS_RESULTS="$CARRIERS_RESULTS" \
    slurm/gen.sbatch)"
  dep="--dependency=afterany:$GEN_ID"
fi

if [[ "$STAGE" == "all" || "$STAGE" == "judge" ]]; then
  echo "== judge"
  JUDGE_ID="$(submit judge ${dep:+$dep} \
    --export=ALL,CARRIERS_RESULTS="$CARRIERS_RESULTS" slurm/judge.sbatch)"
  dep="--dependency=afterany:$JUDGE_ID"
fi

if [[ "$STAGE" == "all" || "$STAGE" == "report" ]]; then
  echo "== report"
  submit report ${dep:+$dep} \
    --export=ALL,CARRIERS_RESULTS="$CARRIERS_RESULTS" slurm/report.sbatch >/dev/null
fi

echo
echo "watch:   bash slurm/status.sh"
echo "tables:  $CARRIERS_RESULTS/tables/report.md"
