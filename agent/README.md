# agent/ — coding-agent integration

This directory packages gap for AI coding agents: a Claude Code plugin
(`.claude-plugin/plugin.json` + `skills/gap/`) teaching agents to search
skill registries, check capabilities, and run/author/test robot task
graphs. Install instructions: [INSTALL.md](INSTALL.md). Regenerate the
CLI reference after CLI changes: `uv run python agent/scripts/gen_cli_reference.py`.

Not to be confused with the `gap.agent` Python package (`gap/agent/`),
which is the engine's LLM graph-generation pipeline.
