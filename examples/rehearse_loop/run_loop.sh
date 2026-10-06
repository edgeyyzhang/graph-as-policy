#!/usr/bin/env bash
# Run one rehearsal loop end to end: start the broker, run the authoring agent,
# stop the broker, print the ledger.
#
#   examples/rehearse_loop/run_loop.sh WORKSPACE TRUSTED [--model MODEL] [--max-usd N]
#
# WORKSPACE and TRUSTED are the two directories created by `rehearse-loop init`.
# Pick the GPU with CUDA_VISIBLE_DEVICES; the broker inherits it. Start this once
# per loop: a second broker on the same TRUSTED directory would serve every
# request as well, running each rehearsal twice.
set -uo pipefail

if [ "$#" -lt 2 ]; then sed -n 2,10p "$0"; exit 2; fi
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ws=$(realpath "$1"); trusted=$(realpath "$2"); shift 2
model=claude-opus-5-5; max_usd=15
while [ "$#" -gt 0 ]; do
  case "$1" in
    --model) model=$2; shift 2 ;;
    --max-usd) max_usd=$2; shift 2 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done
[ -f "$ws/AGENT_PROMPT.md" ] || { echo "not a loop workspace: $ws" >&2; exit 2; }
[ -f "$trusted/config.json" ] || { echo "not a trusted directory: $trusted" >&2; exit 2; }
[ -f "$trusted/agent.done" ] && { echo "agent already finished: $trusted/agent.done" >&2; exit 0; }
if pgrep -f "rehearse-loop serve $trusted" > /dev/null 2>&1; then
  echo "a broker is already serving $trusted" >&2; exit 1
fi

echo "loop start $(date +%H:%M:%S)"
"$here/gap.sh" rehearse-loop serve "$trusted" --idle-timeout 3600 > "$trusted/broker.log" 2>&1 &
broker=$!
"$here/launch_agent.sh" "$ws" --model "$model" --max-usd "$max_usd" 2> "$trusted/agent.stderr"
rc=$?
kill "$broker" 2>/dev/null; wait "$broker" 2>/dev/null
echo "agent exit $rc $(date +%H:%M:%S)"
"$here/gap.sh" rehearse-loop status "$trusted"
date +%H:%M:%S > "$trusted/agent.done"
exit "$rc"
