---
name: fixture-skill
description: Produce a greeting for a named entity via a canonical script and
  a bundled prompt template. Use when a loader test needs a minimal skill
  bundle with scripts and prompts.
compatibility: requires gap>=0.1
metadata: {category: testing, tags: [fixture, greeting]}
gap:
  allowed_tools: [robot.get_observation, fixture-tool.echo]
  exit_conditions:
    greeted: Greeting produced and bound in subgraph outputs.
    failed: Could not produce a greeting.
  produces_outputs: {greeting: str}
  required_inputs: {who: str}
  canonical_scripts:
    - hello: scripts/hello.py
  prompts:
    greet: prompts/greet.md
  hard_rules:
    - Always greet politely.
  streaming: false
---

# fixture-skill

A minimal skill bundle: one canonical script, one prompt template. Exists
only for the loader test suite.

## When to use

- Never, outside of tests.
