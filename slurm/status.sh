#!/usr/bin/env bash
# Where the chain is: queue state, episodes on disk per setting, recent failures.
#
#   bash slurm/status.sh
#   watch -n 60 bash slurm/status.sh
set -uo pipefail
cd "$(dirname "$0")/.."
if [[ -f .env ]]; then set -a; source .env; set +a; fi
RESULTS="${CARRIERS_RESULTS:-$PWD/results/unity}"

echo "== queue"
squeue -u "$USER" -o '%.18i %.14j %.2t %.10M %.6D %R' 2>/dev/null || echo "  (no squeue here)"

echo
echo "== episodes on disk  ($RESULTS)"
if [[ -d "$RESULTS/runs" ]]; then
  total=0
  while read -r n setting; do
    judged=$(find "$RESULTS/runs" -path "*/$setting/*/judge.json" 2>/dev/null | wc -l | tr -d ' ')
    printf '  %-26s %3s episodes  %3s judged\n' "$setting" "$n" "$judged"
    total=$(( total + n ))
  done < <(find "$RESULTS/runs" -name metrics.json 2>/dev/null \
            | awk -F/ '{print $(NF-2)}' | sort | uniq -c | awk '{print $1, $2}')
  echo "  ------------------------------------------"
  printf '  %-26s %3s\n' TOTAL "$total"
else
  echo "  nothing yet"
fi

echo
echo "== last 5 job logs"
ls -t logs/slurm/*.out 2>/dev/null | head -5 | while read -r f; do
  printf '  %s: %s\n' "$(basename "$f")" "$(tail -n 1 "$f" 2>/dev/null)"
done

echo
echo "== failures in the newest logs"
# Collected into a file list first: `grep -l` with no file arguments would read
# stdin and hang, which is a bad thing for a status command to do.
RECENT=$(ls -t logs/slurm/*.err logs/slurm/*.out 2>/dev/null | head -20)
if [[ -n "$RECENT" ]]; then
  hits=$(echo "$RECENT" | tr '\n' '\0' \
    | xargs -0 grep -l -i -E 'Traceback|CUDA out of memory|expanded to nothing|not healthy' 2>/dev/null)
  if [[ -n "$hits" ]]; then echo "$hits" | sed 's/^/  /'; else echo "  none"; fi
else
  echo "  no logs yet"
fi
