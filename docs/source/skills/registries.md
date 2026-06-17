# Skill Registries

A **registry** is any local directory that holds skill bundles under a
`tools/` and/or `skills/` root — each immediate subdirectory containing a
`SKILL.md` is one bundle. Registries are discovered **by path**, never by
pip entry points, so a registry is just a checkout you can clone, fork,
and edit.

The canonical public registry is **open-robot-skills**, which ships
model-backed tool bundles ([sam3](gh-skills:tools/sam3),
[grounding-dino](gh-skills:tools/grounding-dino),
[curobo](gh-skills:tools/curobo), …) under
[tools/](gh-skills:tools) and manipulation skill bundles
([perceiving-objects](gh-skills:skills/perceiving-objects),
[grasping-direct-ik](gh-skills:skills/grasping-direct-ik),
[transporting-objects](gh-skills:skills/transporting-objects), …) under
[skills/](gh-skills:skills). gap treats it as exactly
that — an example, not a special case. Labs and projects bring their own
registries and layer them on top.

```text
my-registry/
├── pyproject.toml        # one distribution; one pip extra per bundle
├── tools/
│   └── <bundle>/
│       ├── SKILL.md
│       └── tools.py
├── skills/
│   └── <bundle>/
│       ├── SKILL.md
│       └── scripts/
└── tests/
```

Multiple registries are active at once, in precedence order. The merged
catalog is the union of their bundles; when two registries claim the same
bundle name, the higher-precedence one wins (see
[Merging and shadowing](#merging-and-shadowing)). This is how you
override one public skill with a lab fork without forking the whole
registry.

This page covers how gap finds registries, how they merge, the
`gap registry` lifecycle commands, and the `gap check` capability report.
For what goes *inside* a bundle, see
[Authoring bundles](authoring-bundles.md); for the catalogs themselves,
see the [skill catalog](skill-catalog.md) and
[tool catalog](tool-catalog.md).

## Resolution precedence

Every command that touches skills (`gap run`, `gap generate`,
`gap check`, `gap skills …`, `gap tools …`) resolves the active registry
set through the same five layers, implemented in
[gap/skills/registries.py](gh-engine:gap/skills/registries.py):

1. **`--skills PATH` flags / `skills=` arguments** (repeatable) —
   **full override**. Exactly the listed registries run; nothing else is
   merged in. Paths are taken as-is with no shape validation — problems
   surface loudly at load time, which lets hermetic test callers pass
   fixtures.
2. **`$GAP_SKILLS_PATH`** — an `os.pathsep`-separated list (`:` on
   Linux; a single path keeps its historical single-registry meaning).
   Also a **full override**. Every entry must be a valid registry
   checkout — a directory with a `tools/` or `skills/` root containing
   at least one `SKILL.md` bundle — or resolution raises `ValueError`.
   An explicitly set environment variable must be valid; silently
   falling through would mask typos.
3. **Project config** — the nearest `pyproject.toml` declaring a
   `[tool.gap].registries` list, walking up from the current directory.
   The walk continues past pyprojects *without* a `[tool.gap]` table, so
   a nested package inside a configured workspace still sees the
   workspace config. Paths are resolved relative to that pyproject's
   directory. Entries that are not registry directories are skipped with
   a warning.

   ```toml
   [tool.gap]
   registries = ["./skills", "../open-robot-skills"]
   ```

4. **User config** — `~/.config/gap/registries.toml` (respects
   `$XDG_CONFIG_HOME`), managed by `gap registry add/remove`. Entry
   order is precedence order. A missing or unparseable file is treated
   as empty; malformed entries are skipped with a warning so one typo
   does not take every gap command down.

   ```toml
   [[registry]]
   name = "lab-skills"
   path = "/home/you/lab-skills"

   [[registry]]
   name = "open-robot-skills"
   path = "/home/you/open-robot-skills"
   ```

5. **Sibling auto-discovery** — walking up from the gap package
   directory and the current directory, the first directory named
   exactly `open-robot-skills` that has a populated bundle root. This is
   the documented side-by-side layout from
   [Installation](../getting-started/installation.md). Running a command
   *inside* any registry checkout also counts.

Layers 1 and 2 do **not** merge with anything below them. Layers 3–5
merge — project entries first, then user entries, then the
auto-discovered sibling — deduplicated by resolved path. Registry names
come from the configured name (user config) or the registry's own
`[project].name`, falling back to the directory name; a name collision
across layers gets a `-2`, `-3`, … suffix with a warning.

:::{warning}
`--skills` and `$GAP_SKILLS_PATH` are full overrides, not additions.
When either is in force, your configured project and user registries —
and the auto-discovered sibling — are suppressed entirely.
`gap registry list` prints a note when this is happening.
:::

:::{note}
Auto-discovery is deliberately strict: only a directory named exactly
`open-robot-skills` whose `tools/` or `skills/` root actually contains a
bundle — or the populated registry checkout you are running inside — is
picked up implicitly. Arbitrary directories — including a
freshly `gap registry init`-ed, still-empty registry — must be wired in
via flags or config. The lenient test (just "has a `tools/` or `skills/`
dir") is used for `gap registry add` and for configured project/user
entries, so an empty new registry is addable and configurable but never
silently discovered.
:::

When nothing resolves and a registry is required (for example,
`gap generate`), the command fails with a message listing everything
that was tried, in order, plus the ways to fix it.

## Merging and shadowing

The resolved registry set loads into one merged catalog, in precedence
order:

- **Across registries, first wins.** A bundle name already claimed by a
  higher-precedence registry **shadows** same-named bundles below it.
  The loser is skipped with a loud warning *before* any of its modules
  import — so a lab fork of `perceiving-objects` cleanly replaces the
  public one.
- **Within one registry, duplicates are a hard error.** Two bundles with
  the same name in a single registry raise `ValueError` — there is no
  sensible winner.

Shadowed bundles still show up in listings: `gap skills list` marks them
`(shadowed)` and `gap check` reports them as `SHADOWED by registry
'<name>'`, so an unintentional override is visible rather than silent.

:::{warning}
Bundle modules import under a process-global synthetic namespace
(`gap_skills.<tools|skills>.<bundle>`). Once a bundle's modules have
imported, they stay for the life of the process — shadowing is safe
because the losing bundle never imports, but *changing registry
precedence requires a fresh process*. In long-lived processes
(notebooks, test sessions), re-resolving with a different order will not
reload bundles that already imported.
:::

## Managing registries: `gap registry`

### `gap registry list`

Shows the active set in precedence order, plus configured-but-broken
entries that resolution skipped — this is the place typos show up.

```bash
gap registry list
gap registry list --format json
```

The pretty table has columns `PRIORITY  NAME  PATH  SOURCE  DIST  TOOLS
SKILLS  STATUS`. Active registries get `status: ok` with live
tool/skill bundle counts; configured entries that did not resolve appear
with no priority and a status of `broken (not a registry checkout)` or
`shadowed path` (same path already active under another name). When
`--skills` or `$GAP_SKILLS_PATH` is in force, a note explains that
configured registries are suppressed. Always exits 0.

### `gap registry init`

Scaffolds a brand-new, empty registry ready for `gap skills new`:

```bash
gap registry init ~/lab-skills            # name defaults to the directory name
gap registry init ~/lab-skills --name lab-skills --add
```

The scaffold writes:

- `pyproject.toml` — depends on `graph-as-policy`; an empty
  `[project.optional-dependencies]` table with the one-pip-extra-per-bundle
  convention documented (this is where `gap check` install hints come
  from); pytest config that deselects `gpu`/`llm` markers by default;
  a hatchling build that bypasses file selection (the package carries
  dependency metadata — bundles load by path).
- `tools/` and `skills/` bundle roots (with `.gitkeep`).
- `tests/conftest.py` — session-scoped `skills_registry` and
  `tool_registry` fixtures, used by the tests `gap skills new`
  generates (see [Testing bundles](testing-bundles.md)).
- `README.md` and `.gitignore`.

`--add` chains straight into `gap registry add`, putting the new
registry at the top of your user config. Names must match
`^[a-z0-9]+(?:-[a-z0-9]+)*$` (bad name → exit 2), and `init` refuses to
scaffold over an existing `pyproject.toml`, `tools/`, or `skills/`
(exit 1).

### `gap registry add`

```bash
gap registry add lab-skills ~/lab-skills              # user config (default)
gap registry add lab-skills ../lab-skills --project   # pyproject [tool.gap]
```

- The path must be a local directory with a `tools/` or `skills/` root
  (exit 2 otherwise, with a pointer to `gap registry init`).
- **Remote URLs are rejected** (exit 2). Registries are local
  directories in this release; the error prints a clone-first recipe
  (`git clone <url> ~/skills/<name>`, install its deps, then `add` the
  clone). Git-managed registries are future work.
- `add` **prepends**: the new entry lands at the *top* of the config, so
  its bundles shadow same-named bundles in every later entry and in the
  auto-discovered sibling. A duplicate name in the user config is exit 1
  (`remove` it first).
- `--project` edits the nearest `pyproject.toml`, writing the path
  relative to it when possible. If there is no `[tool.gap]` table yet,
  one is appended. If the table exists, the command can only edit a
  **single-line** `registries = [...]` array (it prepends the new
  entry); a multi-line array exits 1 and asks you to edit manually.

### `gap registry remove`

```bash
gap registry remove lab-skills              # user config (default)
gap registry remove lab-skills --project
```

Unknown name → exit 1 (the message lists what *is* configured). With
`--project`, no pyproject declaring `[tool.gap].registries` → exit 2,
and the same single-line-array limitation applies.

## Capability report: `gap check`

`gap check` answers "what can run *here*?" — the `sky check` of gap. It
reports the environment, every active registry, per-bundle operational
status, and which skills are therefore runnable, with a fix hint per
failure.

```bash
gap check
gap check --registry open-robot-skills    # restrict to one active registry
gap check --format json                   # stable machine-readable schema
gap check --strict                        # CI gating
```

Every probe is fast and offline:

- **GPU** — one `nvidia-smi` subprocess (5 s timeout, deliberately
  torch-free).
- **LLM credentials** — environment lookups: `ANTHROPIC_API_KEY`,
  `OPENAI_API_KEY` (custom OpenAI-compatible endpoints may not need
  it), and Vertex via `$GOOGLE_APPLICATION_CREDENTIALS` or the gcloud
  ADC file.
- **Bundle deps** — an isolated per-bundle import probe. Bundles
  lazy-load their models, so importing never loads weights.
- **Declared requirements** — the bundle's `gap.requires` block
  (`gpu`, `env`, `env_any`, `weights`) checked statically from
  `SKILL.md`.
- **Weights** — the bundle's optional, filesystem-only
  `weights_cached()` hook in `tools.py`/`skill.py`. A bundle without
  the hook (or a hook returning `None`) reports `unknown`, not missing.

Each bundle rolls up to `READY`, `NOT READY`, or `SHADOWED`, with
per-probe detail lines and fix hints — dependency failures hint
`uv run gap skills install <bundle>` when the registry has a `uv.lock`, else
`pip install '<dist>[<bundle>]'` using the registry's own
`[project].name`. Tool and policy bundles that own their own
`pyproject.toml` (an isolated per-bundle venv) are installed via
`uv run gap skills install <name>` (or `--all` for the whole registry);
`gap skills check` annotates each row with `(venv-ready)` or
`(venv missing — run gap skills install <name>)` so the missing step is
obvious before you reach `gap check`. Skills then roll up to ready or `BLOCKED`: a skill is
blocked by any not-ready tool bundle that owns one of its
`gap.allowed_tools` (ownership by name prefix; `robot.*`/`sim.*` tools
are satisfied by the live connector at run time, and tools whose prefix
matches no bundle are flagged as unknown). The report ends with a
summary line:

```text
9 bundle(s): 7 ready, 2 not ready · 6 skill(s): 4 ready, 2 blocked
```

`--format json` emits a stable schema (`schema_version: 1`) with
top-level keys `schema_version`, `environment`, `registries`, `bundles`,
and `skills` — suitable for dashboards and CI assertions.

Exit codes:

- **0** whenever a report is produced — diagnostic by design; a laptop
  without a GPU is not an error, even if every bundle is not ready.
- **1** with `--strict`, when any non-shadowed bundle is `not-ready`
  **or zero registries are active**.
- **2** on usage or resolution errors (unknown `--registry` name,
  invalid `$GAP_SKILLS_PATH`). A bad `--skills` path is *not* an error
  here — explicit paths are taken without shape validation, so the
  report simply shows that registry with zero bundles.

:::{note}
`gap check` validates whether bundles can *run*. To validate whether
bundles are *well-formed* (frontmatter, referenced files, resolvable
tool names), use `gap skills check` — see
[Testing bundles](testing-bundles.md).
:::

## Worked example: shadow a public skill with a lab fork

Suppose your lab wants its own version of `perceiving-objects` — same
name, different prompt and thresholds — while keeping everything else
from open-robot-skills.

```bash
# 1. Scaffold a lab registry and put it at the top of the user config.
gap registry init ~/lab-skills --add

# 2. Fork the public bundle into it.
cp -r ~/open-robot-skills/skills/perceiving-objects ~/lab-skills/skills/
# ... edit ~/lab-skills/skills/perceiving-objects/SKILL.md and scripts/ ...

# 3. Confirm the layering.
gap registry list
gap check
```

`gap registry list` shows `lab-skills` at priority 1 (source `user`) and
`open-robot-skills` below it (source `auto` if it is the side-by-side
sibling). `gap check` reports your fork as the active
`perceiving-objects` and the public copy as `SHADOWED by registry
'lab-skills'`; loading logs a shadowing warning so the override is never
silent. Every gap command — `gap generate`, `gap run`, the catalogs —
now sees the lab fork wherever the bundle name `perceiving-objects` is
referenced, and the rest of the public registry is untouched.

To pin this layering for one repo instead of your whole user account,
use the project scope:

```bash
cd ~/my-robot-project
gap registry add lab-skills ~/lab-skills --project
```

and commit the resulting `[tool.gap].registries` entry. Teammates get
the same registry set from the pyproject; CI can enforce readiness with
`gap check --strict`.

## See also

- [Authoring bundles](authoring-bundles.md) — the `SKILL.md` format and
  bundle layout.
- [Testing bundles](testing-bundles.md) — `gap skills check`,
  `gap skills test`, and the scaffolded test fixtures.
- [Skill catalog](skill-catalog.md) and [tool catalog](tool-catalog.md)
  — what the public registry ships.
- [CLI reference](../reference/cli.md) — full flag listings for
  `gap registry` and `gap check`.
- [Environment variables](../reference/environment-variables.md) —
  `GAP_SKILLS_PATH` and friends.
