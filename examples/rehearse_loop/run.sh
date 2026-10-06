#!/usr/bin/env bash
# Run `gap rehearse` with the environment this checkout needs.
#
#   examples/rehearse_loop/run.sh GRAPH_DIR SUITE/TASK CASES OUT_DIR [extra gap-rehearse args]
#
# Example:
#   examples/rehearse_loop/run.sh examples/libero_quickstart/graph \
#       libero_object_all_variance/0 1-8 outputs/rehearse_loop/soup/round_00 --frames
#
# The OpenRouter key is read from ~/.config/gap/credentials.env and is never printed.
# The IK backend defaults to PyRoKi because cuRobo is not installed in the run venv;
# pass `--ik curobo` to override.
set -euo pipefail

if [ "$#" -lt 4 ]; then
  sed -n 2,12p "$0"; exit 2
fi
graph=$(realpath "$1"); sim=$2; cases=$3; out=$(realpath -m "$4"); shift 4

here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# The perception cache is keyed on camera frames and arguments, not on script content, so a
# cache shared between rounds would hide an edit to a perception script. Keep one per round.
export GAP_PERCEPTION_CACHE_DIR=${GAP_PERCEPTION_CACHE_DIR:-$out/perception_cache}

ik=(--ik pyroki)
for arg in "$@"; do [ "$arg" = "--ik" ] && ik=(); done

mkdir -p "$out"
exec "$here/gap.sh" rehearse "$graph" --sim "$sim" --cases "$cases" --out "$out" "${ik[@]}" "$@"
