# Skill registries

A **registry** is a local directory of bundles: manipulation strategies
under `skills/<name>/` and model-backed tool bundles under
`tools/<name>/`, each an Agent-Skills-format directory with a `SKILL.md`.
[open-robot-skills](https://github.com/graph-robots/open-robot-skills) is
the canonical public registry — GaP treats it as exactly that, an
example. Labs and projects bring their own and layer them.

## Resolution order (first hit wins the layer)

1. `--skills PATH` flags / `skills=` arguments — repeatable,
   precedence-ordered, **full override** (nothing else merges in).
2. `$GAP_SKILLS_PATH` — `os.pathsep`-separated list (a single path keeps
   its historical meaning) — also a full override; every entry must be a
   valid registry.
3. Project config: the nearest `pyproject.toml` (walking up from the cwd)
   with a `[tool.gap]` table:

   ```toml
   [tool.gap]
   registries = ["./skills", "../open-robot-skills"]  # relative to this file
   ```

4. User config `~/.config/gap/registries.toml`, managed by
   `gap registry add/remove` (most recently added first):

   ```toml
   [[registry]]
   name = "lab-skills"
   path = "/home/u/lab-skills"
   ```

5. Auto-discovery: an `open-robot-skills` checkout next to the GaP
   checkout or the cwd (the documented side-by-side layout).

Layers 3–5 merge (project, then user, then auto), deduplicated by path.
`gap registry list` shows the live set with provenance — and surfaces
configured-but-broken entries.

## Shadowing

The merged catalog is the union of all active registries. A bundle name
claimed by a higher-precedence registry **shadows** same-named bundles
below it — first wins, with a loud warning. That is the supported way to
override one public skill with your fork: put your registry earlier and
give your bundle the same name. Within one registry, a duplicate name is
a hard error. Changing precedence requires a fresh process (bundle
modules import once per process).

## Commands

```bash
gap registry init <path> [--name N] [--add]   # scaffold a new registry
gap registry add <name> <path> [--user|--project]
gap registry remove <name> [--user|--project]
gap registry list [--format json]
```

`init` scaffolds `pyproject.toml` (one-pip-extra-per-bundle convention,
pytest config with gpu/llm markers), `tools/`, `skills/`,
`tests/conftest.py` (session-scoped `skills_registry`/`tool_registry`
fixtures), and a README. Registries are **local paths** in this release —
clone remote ones yourself, then `gap registry add`.

## Registry anatomy

```
my-lab-skills/
├── pyproject.toml      # [project].name = registry/dist name; one extra per bundle
├── tools/<bundle>/     # SKILL.md + tools.py (@tool functions)
├── skills/<bundle>/    # SKILL.md + scripts/ + prompts/ + references/
└── tests/              # conftest.py + test_<bundle>.py per bundle
```

A directory counts as a registry once it has a `tools/` or `skills/`
root; bundles appear in catalogs once they carry a valid `SKILL.md`.
`gap check` derives install hints from the registry's own pyproject
(`uv sync --extra <bundle>` when it has a `uv.lock`, else
`pip install '<dist>[<bundle>]'`).
