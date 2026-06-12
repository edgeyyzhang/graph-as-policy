# Install the gap skill for AI agents

The skill ([`skills/gap/SKILL.md`](skills/gap/SKILL.md)) teaches a coding
agent to drive gap: search skill registries, check tool/model
capabilities, run graphs in sim, generate and hand-author graphs, create
unit-tested skill bundles and registries, debug traces — with real-robot
runs gated on explicit human confirmation.

## Claude Code (recommended)

```bash
claude plugin marketplace add graph-robots/graph-as-policy
claude plugin install gap@gap                  # the engine skill
claude plugin install open-robot-skills@gap    # optional: the 17 robot bundle contracts
```

Verify in a new session: ask *"What robot skills are available?"* — the
`gap` skill should be listed (or run `/gap:gap`). `claude plugin list`
shows installed plugins. After editing skill content locally:
`claude plugin marketplace update gap && claude plugin update gap@gap`,
or develop live with `claude --plugin-dir ./agent` + `/reload-plugins`.

Working inside this repo checkout? `.claude/settings.json` already
references the marketplace — trust the workspace when prompted and the
skill loads for everyone on the project. Team installs:
`claude plugin install gap@gap -s project`.

## Other agents (Cursor, Codex CLI, Gemini CLI, …)

The skill is plain markdown with no executable payload:

```bash
git clone https://github.com/graph-robots/graph-as-policy.git ~/.agents/.gap
ln -s ~/.agents/.gap/agent/skills/gap <your-agent's-skills-dir>/gap
# or copy the directory / paste SKILL.md into your rules file (AGENTS.md, .cursor/rules/...)
```

Update with `git pull`. The `references/` directory holds the deep dives
the skill links to — keep it next to SKILL.md.

## No install at all

The CLI is self-describing. Point any agent at:

```bash
gap --help          # command tree
gap check           # what can run here (use --format json programmatically)
```

or fetch the skill text directly:
<https://raw.githubusercontent.com/graph-robots/graph-as-policy/main/agent/skills/gap/SKILL.md>
