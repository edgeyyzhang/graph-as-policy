#!/usr/bin/env bash
# Start Claude Code headless as the authoring agent of a rehearsal loop.
#
#   examples/rehearse_loop/launch_agent.sh WORKSPACE [--model MODEL] [--prompt TEXT] [--max-usd N]
#
# The broker for the same loop must be running (gap.sh rehearse-loop serve TRUSTED).
#
# What the agent can do:
#   - read and search files inside WORKSPACE only (Claude Code restricted mode);
#   - edit files under results/ and .cache/ (edits elsewhere are denied);
#   - run commands only through ./eval/run, which confines them to the workspace
#     with no network.
# User settings, CLAUDE.md files, memory, plugins and MCP servers are not loaded.
# The transcript goes to WORKSPACE/../<name>.agent.jsonl, outside the workspace.
set -euo pipefail

if [ "$#" -lt 1 ]; then sed -n 2,15p "$0"; exit 2; fi
workspace=$(realpath "$1"); shift
model=""; prompt=""; max_usd=""
while [ "$#" -gt 0 ]; do
  case "$1" in
    --model) model=$2; shift 2 ;;
    --prompt) prompt=$2; shift 2 ;;
    --max-usd) max_usd=$2; shift 2 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done
[ -f "$workspace/AGENT_PROMPT.md" ] || { echo "not a loop workspace: $workspace" >&2; exit 2; }
[ -n "$prompt" ] || prompt="Read AGENT_PROMPT.md in the current directory and carry out the task it describes."

# Absolute paths: a rule with a relative path is matched against the shell's current
# directory, which the agent can change with `cd`.
w="/$workspace"   # a leading // marks an absolute path in a permission rule
settings=$(cat <<JSON
{
  "permissions": {
    "allow": ["Read", "Glob", "Grep",
              "Edit($w/results/**)", "Write($w/results/**)", "Edit($w/.cache/**)", "Write($w/.cache/**)",
              "Bash(./eval/run)", "Bash(./eval/run *)", "Bash($workspace/eval/run)", "Bash($workspace/eval/run *)"],
    "deny": ["Edit($w/eval/**)", "Write($w/eval/**)", "Edit($w/runtime/**)", "Write($w/runtime/**)",
             "Edit($w/.requests/**)", "Write($w/.requests/**)", "Edit($w/.venv/**)", "Write($w/.venv/**)",
             "Edit($w/AGENT_PROMPT.md)", "Write($w/AGENT_PROMPT.md)",
             "WebFetch", "WebSearch", "Agent", "Task", "NotebookEdit"]
  }
}
JSON
)

log="$(dirname "$workspace")/$(basename "$workspace").agent.jsonl"
args=(--print "$prompt" --restricted --tools "Read,Glob,Grep,Edit,Write,Bash"
      --settings "$settings" --strict-mcp-config --permission-prompts none
      --disable-slash-commands --no-session-persistence
      --output-format stream-json --verbose)
[ -n "$model" ] && args+=(--model "$model")
[ -n "$max_usd" ] && args+=(--max-budget-usd "$max_usd")

cd "$workspace"
echo "agent transcript: $log" >&2
# Nothing from the launching shell's environment that the agent does not need.
exec env -i HOME="$HOME" PATH="$HOME/.local/bin:/usr/bin:/bin" LANG=C.UTF-8 TERM=dumb \
  claude "${args[@]}" > "$log" < /dev/null
