#!/usr/bin/env bash
# Evaluate a policy graph on its task and write the policy card.
#
#   examples/rehearse_loop/policy_report.sh GRAPH_DIR SUITE/TASK OUT_DIR [--cases 1-16] [--skip-eval]
#                                            [-- extra policy_card.py args]
#
# Runs `gap rehearse` on the cases with frames and video into OUT_DIR/eval, then
# policy_card.py into OUT_DIR/report (report.html, report.json). Pick the GPU with
# CUDA_VISIBLE_DEVICES. With --skip-eval an existing OUT_DIR/eval is reused.
#
# Example:
#   CUDA_VISIBLE_DEVICES=2 examples/rehearse_loop/policy_report.sh \
#       ../gap_loops/l10_task_00/results/graph libero_10/0 outputs/policy_report/l10_task_00
set -euo pipefail

if [ "$#" -lt 3 ]; then sed -n 2,13p "$0"; exit 2; fi
graph=$(realpath "$1"); sim=$2; out=$(realpath -m "$3"); shift 3
cases=1-16; skip=0; extra=()
while [ "$#" -gt 0 ]; do
  case "$1" in
    --cases) cases=$2; shift 2 ;;
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
  echo "evaluating $graph on $sim cases $cases -> $out/eval  $(date +%H:%M:%S)"
  "$here/run.sh" "$graph" "$sim" "$cases" "$out/eval" --frames --video > "$out/eval.log" 2>&1 \
    || { echo "rehearsal failed; see $out/eval.log" >&2; exit 1; }
  grep 'rehearsal complete' "$out/eval.log" | tail -1
fi
[ -d "$out/eval/cases" ] || { echo "no evaluation at $out/eval" >&2; exit 1; }

echo "writing the policy card  $(date +%H:%M:%S)"
opt=()
[ -f "$graph/optimization.json" ] && opt=(--optimization "$graph/optimization.json")
export PYTHONPATH=$gap:$gap/gap-core/src MPLBACKEND=Agg
"$python" "$here/policy_card.py" "$graph" "$out/eval" --out "$out/report" "${opt[@]}" "${extra[@]}"
