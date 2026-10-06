#!/usr/bin/env bash
# Print a workflow graph as text, or draw it.
#
#   examples/rehearse_loop/show_graph.sh GRAPH_DIR            text view on the terminal
#   examples/rehearse_loop/show_graph.sh GRAPH_DIR OUT.pdf    PDF (and PNG) drawing
set -euo pipefail
[ "$#" -ge 1 ] || { sed -n 2,6p "$0"; exit 2; }
graph=$(realpath "$1"); out=${2:-}
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
gap=$(cd "$here/../.." && pwd)
python=${GAP_PYTHON:-$gap/.venv/bin/python}
export PYTHONPATH=$gap:$gap/gap-core/src
if [ -n "$out" ]; then
  "$python" -c 'import sys; from gap.viz import render; print(render(sys.argv[1], sys.argv[2]))' "$graph/workflow.json" "$out"
else
  "$python" -c 'import sys; from gap.viz import to_text; print(to_text(sys.argv[1], color=sys.stdout.isatty()))' "$graph/workflow.json"
fi
