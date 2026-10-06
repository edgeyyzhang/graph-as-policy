#!/usr/bin/env bash
# Evaluate a policy graph on its task and write the policy card.
#
#   examples/rehearse_loop/policy_report.sh GRAPH_DIR SUITE/TASK OUT_DIR
#       [--trajs N|all] [--seed S] [--cases 1-16] [--skip-eval] [-- extra policy_card.py args]
#
# Samples N layouts at random (default 4, seed 0) from the case pool (default 1-16),
# runs `gap rehearse` on them with frames and video into OUT_DIR/eval, then writes
# policy_card.py's report into OUT_DIR/report (report.html, report.json) with a
# two-panel video of every evaluated layout. `--trajs all` evaluates the whole pool
# and embeds videos only for the failures. Pick the GPU with CUDA_VISIBLE_DEVICES.
# With --skip-eval an existing OUT_DIR/eval is reused.
#
# Example:
#   CUDA_VISIBLE_DEVICES=2 examples/rehearse_loop/policy_report.sh \
#       ../gap_loops/l10_task_00/results/graph libero_10/0 outputs/policy_report/l10_task_00 --trajs 4
set -euo pipefail

if [ "$#" -lt 3 ]; then sed -n 2,16p "$0"; exit 2; fi
graph=$(realpath "$1"); sim=$2; out=$(realpath -m "$3"); shift 3
pool=1-16; trajs=4; seed=0; skip=0; extra=()
while [ "$#" -gt 0 ]; do
  case "$1" in
    --trajs) trajs=$2; shift 2 ;;
    --seed) seed=$2; shift 2 ;;
    --cases) pool=$2; shift 2 ;;
    --skip-eval) skip=1; shift ;;
    --) shift; extra=("$@"); break ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
gap=$(cd "$here/../.." && pwd)
python=${GAP_PYTHON:-$gap/.venv/bin/python}

[ -f "$graph/workflow.json" ] || { echo "no workflow.json in $graph" >&2; exit 2; }
mkdir -p "$out"
if [ "$skip" = 0 ]; then
  if [ -d "$out/eval/cases" ]; then
    echo "evaluation exists at $out/eval; pass --skip-eval to reuse it or remove it" >&2; exit 1
  fi
  mkdir -p "$out/eval"
  if [ "$trajs" = all ]; then
    cases=$pool
    rm -f "$out/eval/sample.json"
  else
    # Draw the layouts once, record the draw, and evaluate only those.
    cases=$("$python" - "$pool" "$trajs" "$seed" "$out/eval/sample.json" <<'EOF'
import json, random, sys
pool, n, seed, path = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
ids = []
for part in pool.split(","):
    lo, _, hi = part.partition("-")
    ids.extend(range(int(lo), int(hi or lo) + 1))
if not 1 <= n <= len(ids):
    sys.exit(f"--trajs must be between 1 and {len(ids)} for the pool {pool}")
picked = sorted(random.Random(seed).sample(ids, n))
json.dump({"pool": pool, "trajs": n, "seed": seed, "cases": picked}, open(path, "w"))
print(",".join(map(str, picked)))
EOF
)
  fi
  echo "evaluating $graph on $sim cases $cases -> $out/eval  $(date +%H:%M:%S)"
  "$here/run.sh" "$graph" "$sim" "$cases" "$out/eval" --frames --video > "$out/eval.log" 2>&1 \
    || { echo "rehearsal failed; see $out/eval.log" >&2; exit 1; }
  grep 'rehearsal complete' "$out/eval.log" | tail -1
fi
[ -d "$out/eval/cases" ] || { echo "no evaluation at $out/eval" >&2; exit 1; }

echo "writing the policy card  $(date +%H:%M:%S)"
opts=()
[ -f "$graph/optimization.json" ] && opts+=(--optimization "$graph/optimization.json")
[ -f "$out/eval/sample.json" ] && opts+=(--all-videos)
export PYTHONPATH=$gap:$gap/gap-core/src MPLBACKEND=Agg
"$python" "$here/policy_card.py" "$graph" "$out/eval" --out "$out/report" "${opts[@]}" "${extra[@]}"
