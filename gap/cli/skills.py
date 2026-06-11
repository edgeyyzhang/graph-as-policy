"""``gap skills`` subcommands — list / check / table / new for open-robot-skills bundles."""

from __future__ import annotations

import argparse


def register(subparsers: argparse._SubParsersAction) -> None:
    sp = subparsers.add_parser(
        "skills",
        help="Inspect, verify, and scaffold open-robot-skills bundles",
    )
    sub = sp.add_subparsers(dest="skills_command")

    lp = sub.add_parser("list", help="List discovered bundles")
    _add_skills_path(lp)
    lp.set_defaults(func=_handle_list)

    cp = sub.add_parser(
        "check",
        help="Validate every bundle: SKILL.md format + import probe "
             "(PASS/WARN/FAIL per bundle; non-zero exit on FAIL)",
    )
    _add_skills_path(cp)
    cp.add_argument(
        "--download", action="store_true",
        help="After the checks, run each bundle's optional prefetch() "
             "to download model weights",
    )
    cp.set_defaults(func=_handle_check)

    tp = sub.add_parser(
        "table",
        help="Dump a catalog table of all discovered bundles",
    )
    _add_skills_path(tp)
    tp.add_argument(
        "--format", default="pretty", choices=["pretty", "markdown", "json"],
        help="Output format (default: pretty terminal table; markdown is "
             "paste-ready for READMEs; json is machine-readable)",
    )
    tp.add_argument(
        "--kind", default=None, choices=["tool", "skill"],
        help="Only bundles of this kind (markdown output then drops the "
             "Kind column — one paste-ready table per README section)",
    )
    tp.set_defaults(func=_handle_table)

    np_ = sub.add_parser("new", help="Scaffold a new bundle from a template")
    np_.add_argument("name", help="Bundle name (== directory name)")
    np_.add_argument(
        "--kind", required=True, choices=["tool", "skill"],
        help="Bundle kind: tools/<name> or skills/<name>",
    )
    _add_skills_path(np_)
    np_.set_defaults(func=_handle_new)

    # NOTE: no sp.set_defaults(func=...) here — argparse applies parent
    # defaults to the namespace before the sub-subparser runs, which would
    # mask the per-subcommand handlers. `gap skills` with no subcommand
    # falls through to the top-level help in main().


def _add_skills_path(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--skills", default=None,
        help="open-robot-skills checkout root (default: auto-discovered — "
             "$GAP_SKILLS_PATH, the current directory, or an open-robot-skills "
             "checkout next to the graph-as-policy checkout)",
    )


def _skills_root(args: argparse.Namespace):
    """Resolve the checkout (explicit flag > $GAP_SKILLS_PATH > discovery)."""
    from gap.skills import find_skills_path

    return find_skills_path(args.skills, required=True)


# ---------------------------------------------------------------------------
# list
# ---------------------------------------------------------------------------


def _handle_list(args: argparse.Namespace) -> int:
    from gap.skills import load_skills

    try:
        root = _skills_root(args)
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}")
        return 2
    registry = load_skills(root)
    infos = registry.list_skills()
    if not infos:
        print(f"no bundles found under {root}/tools or {root}/skills")
        return 1

    rows = []
    for info in sorted(infos, key=lambda i: (i.kind, i.name)):
        desc = info.meta.description.strip().splitlines()[0] if info.meta.description else ""
        tools = ", ".join(sorted(info.meta.tools)) if info.meta.tools else "-"
        rows.append((info.kind, info.name, desc, tools))

    headers = ("KIND", "NAME", "DESCRIPTION", "TOOLS")
    widths = [
        max(len(headers[c]), *(len(r[c]) for r in rows)) for c in range(3)
    ] + [len(headers[3])]
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    print(fmt.format(*headers))
    for row in rows:
        print(fmt.format(*row))
    return 0


# ---------------------------------------------------------------------------
# check (+ --download)
# ---------------------------------------------------------------------------


def _handle_check(args: argparse.Namespace) -> int:
    """Per-bundle format validation + import probe.

    Two layers, merged into one PASS/WARN/FAIL line per bundle:

    1. **Format** (static, no imports): SKILL.md frontmatter shape per
       kind, referenced resources on disk, ``allowed_tools`` resolvable,
       declared type names, pip-extra convention — via
       :mod:`gap.skills.validate` (the same rules the open-robot-skills test
       suite enforces).
    2. **Import probe**: each bundle is registered individually so one
       broken bundle doesn't mask the rest; ImportError hints are mapped
       to the checkout's pip extras (extra name == bundle name).

    Exit status is non-zero iff any bundle FAILs.
    """
    from gap.skills import SkillsRegistry
    from gap.skills.validate import BundleIssue, load_checkout_extras, validate_checkout

    try:
        root = _skills_root(args)
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}")
        return 2
    extras = load_checkout_extras(root) or {}

    reports = validate_checkout(root)
    if not reports:
        print(f"no bundles found under {root}/tools or {root}/skills")
        return 1

    # Layer 2: import probe per bundle, folded into the same report.
    infos: dict[str, object] = {}
    for report in reports:
        reg = SkillsRegistry()
        try:
            reg.register_bundle(report.name, report.bundle_dir, kind=report.kind)
        except ImportError as exc:
            missing = getattr(exc, "name", None) or str(exc)
            hint = ""
            if report.name in extras:
                hint = f" — pip install 'open-robot-skills[{report.name}]'"
            report.issues.append(BundleIssue(
                "error", f"import probe failed: missing {missing}{hint}",
            ))
        except Exception as exc:
            report.issues.append(BundleIssue("error", f"import probe failed: {exc}"))
        else:
            infos[report.name] = reg.get(report.name)

    failures = 0
    for report in reports:
        status = report.status
        if status == "FAIL":
            failures += 1
        print(f"[{report.kind}] {report.name}: {status}")
        for issue in report.issues:
            print(f"    {issue}")

    n_pass = sum(1 for r in reports if r.status == "PASS")
    n_warn = sum(1 for r in reports if r.status == "WARN")
    print(f"\n{len(reports)} bundle(s): {n_pass} PASS, {n_warn} WARN, {failures} FAIL")

    if args.download:
        print()
        for report in reports:
            info = infos.get(report.name)
            if info is None:
                print(f"[{report.kind}] {report.name}: skipped prefetch (import failed)")
                continue
            prefetch = None
            for module in (info.tools_module, info.module):
                fn = getattr(module, "prefetch", None) if module is not None else None
                if callable(fn):
                    prefetch = fn
                    break
            if prefetch is None:
                print(f"[{report.kind}] {report.name}: declares no weights (no prefetch())")
                continue
            try:
                prefetch()
                print(f"[{report.kind}] {report.name}: prefetch OK")
            except Exception as exc:
                failures += 1
                print(f"[{report.kind}] {report.name}: prefetch FAILED: {exc}")

    return 1 if failures else 0


# ---------------------------------------------------------------------------
# table
# ---------------------------------------------------------------------------


def _first_sentence(description: str) -> str:
    """First sentence of a bundle description, whitespace-normalized."""
    import re

    text = " ".join(description.split())
    m = re.search(r"(?<=[.!?])\s", text)
    return text[: m.start()] if m else text


def _table_rows(root) -> list[dict]:
    """Catalog rows from static SKILL.md parsing (no bundle imports)."""
    from gap.skills.validate import load_checkout_extras, validate_checkout

    extras = load_checkout_extras(root) or {}
    rows = []
    for report in validate_checkout(root):
        meta = report.meta
        rows.append({
            "name": report.name,
            "kind": report.kind,
            "description": _first_sentence(meta.description) if meta else "(SKILL.md rejected)",
            "tools": sorted(meta.tools) if meta else [],
            "extra": f"open-robot-skills[{report.name}]" if report.name in extras else "",
        })
    rows.sort(key=lambda r: (r["kind"], r["name"]))
    return rows


def _handle_table(args: argparse.Namespace) -> int:
    try:
        root = _skills_root(args)
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}")
        return 2
    rows = _table_rows(root)
    kind = getattr(args, "kind", None)
    if kind:
        rows = [r for r in rows if r["kind"] == kind]
    if not rows:
        print(f"no bundles found under {root}/tools or {root}/skills")
        return 1

    if args.format == "json":
        import json

        print(json.dumps(rows, indent=2))
        return 0

    cells = [
        (
            r["name"],
            r["kind"],
            r["description"],
            ", ".join(f"`{t}`" for t in r["tools"]) if r["tools"] else "—",
            f"`{r['extra']}`" if r["extra"] else "—",
        )
        for r in rows
    ]
    headers = ("Bundle", "Kind", "Description", "Tools", "Extra")

    if args.format == "markdown":
        # Bundle names link to their directory (paste-ready for the
        # open-robot-skills README at the checkout root).
        cells = [
            (f"[{name}]({'tools' if k == 'tool' else 'skills'}/{name}/)",
             k, desc, tools, extra)
            for name, k, desc, tools, extra in cells
        ]
        if kind:  # one table per kind: the Kind column is redundant
            headers = tuple(h for h in headers if h != "Kind")
            cells = [(n, d, t, e) for n, _, d, t, e in cells]
        print("| " + " | ".join(headers) + " |")
        print("|" + "|".join("---" for _ in headers) + "|")
        for row in cells:
            print("| " + " | ".join(row) + " |")
        return 0

    # pretty terminal table (backtick markup dropped)
    cells = [tuple(c.replace("`", "") for c in row) for row in cells]
    widths = [
        max(len(headers[c]), *(len(row[c]) for row in cells))
        for c in range(len(headers))
    ]
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    print(fmt.format(*headers))
    print(fmt.format(*("-" * w for w in widths)))
    for row in cells:
        print(fmt.format(*row))
    return 0


# ---------------------------------------------------------------------------
# new
# ---------------------------------------------------------------------------

_TOOL_SKILL_MD = """\
---
name: {name}
description: TODO one-line description of what this tool bundle computes.
  Use when TODO the situation that calls for it.
compatibility: requires gap>=0.1
metadata: {{category: TODO, tags: []}}
gap:
  tools:
    - {name}.run: TODO summary of the tool function.
---

# {name}

TODO: describe the model this bundle wraps and its tool functions.

## When to use

- TODO
"""

_TOOL_TOOLS_PY = '''\
"""{name} tool bundle."""

from gap.tools import tool


@tool(name="{name}.run", summary="TODO summary of the tool function.")
def run(text: str) -> str:
    """TODO: implement."""
    raise NotImplementedError
'''

_SKILL_SKILL_MD = """\
---
name: {name}
description: TODO one-line description of this manipulation strategy.
  Use when TODO the situation that calls for it.
compatibility: requires gap>=0.1
metadata: {{category: TODO, tags: []}}
gap:
  allowed_tools: []
  exit_conditions:
    done: TODO meaning of success.
    failed: TODO meaning of failure.
  produces_outputs: {{}}
  required_inputs: {{}}
  canonical_scripts:
    - example: scripts/example.py
---

# {name}

TODO: describe the strategy.

## When to use

- TODO
"""

_SKILL_SCRIPT_PY = '''\
"""Canonical script for the {name} skill bundle."""

from typing import TypedDict


class Output(TypedDict):
    result: str


def run(ctx, *, example_input: str = "") -> Output:
    """TODO: implement."""
    return {{"result": example_input}}
'''


def _handle_new(args: argparse.Namespace) -> int:
    from pathlib import Path

    from gap.skills import find_skills_path

    try:
        root = find_skills_path(args.skills)
    except ValueError as exc:
        print(f"error: {exc}")
        return 2
    if root is None:
        # Bootstrapping a brand-new checkout: nothing discoverable yet, so
        # scaffold into the current directory.
        root = Path.cwd()
        print(f"no open-robot-skills checkout discovered; scaffolding into {root}")
    folder = "tools" if args.kind == "tool" else "skills"
    bundle_dir = root / folder / args.name
    if bundle_dir.exists():
        print(f"refusing to overwrite existing bundle: {bundle_dir}")
        return 1
    bundle_dir.mkdir(parents=True)
    if args.kind == "tool":
        (bundle_dir / "SKILL.md").write_text(
            _TOOL_SKILL_MD.format(name=args.name)
        )
        (bundle_dir / "tools.py").write_text(_TOOL_TOOLS_PY.format(name=args.name))
    else:
        (bundle_dir / "SKILL.md").write_text(_SKILL_SKILL_MD.format(name=args.name))
        scripts = bundle_dir / "scripts"
        scripts.mkdir()
        (scripts / "example.py").write_text(_SKILL_SCRIPT_PY.format(name=args.name))
        (bundle_dir / "prompts").mkdir()
        (bundle_dir / "references").mkdir()
    print(f"scaffolded {args.kind} bundle at {bundle_dir}")
    return 0
