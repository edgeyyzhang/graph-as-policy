"""``gap run`` subcommand — execute (or validate) a workflow graph."""

from __future__ import annotations

import argparse


def register(subparsers: argparse._SubParsersAction) -> None:
    sp = subparsers.add_parser(
        "run",
        help="Execute a gap workflow graph",
    )
    sp.add_argument(
        "graph",
        help="Path to a workflow directory or workflow.json",
    )
    sp.add_argument(
        "--sim", default=None, metavar="SUITE/TASK",
        help="Run against a sim connector, e.g. libero_object/0 "
             "(default: tools-only, no robot)",
    )
    sp.add_argument(
        "--real", default=None, choices=["franka", "ur_zed"],
        help="Run against a real-hardware connector (franka: robots_realtime "
             "msgpack bridge; ur_zed: perception-only UR + ZED). "
             "READ THE SAFETY NOTES in examples/real_franka_pick_place/README.md "
             "before driving hardware.",
    )
    sp.add_argument(
        "--rr-config", default=None, metavar="YAML",
        help="franka only: rr-session config (relative to "
             "third_party/robots_realtime); default "
             "configs/franka/franka_robotiq_client.yaml",
    )
    sp.add_argument(
        "--no-rr-autostart", action="store_true",
        help="franka only: do not spawn rr-session; run it yourself in a "
             "second terminal",
    )
    sp.add_argument(
        "--skills", default=None,
        help="Path to an open-robot-skills checkout (bundle discovery root). "
             "Default: auto-discovered — $GAP_SKILLS_PATH or an open-robot-skills "
             "checkout next to the graph-as-policy checkout",
    )
    sp.add_argument(
        "--validate-only", action="store_true",
        help="Validate the workflow graph without executing it",
    )
    sp.add_argument(
        "--no-trace", action="store_true",
        help="Disable trace output (default: traces into ./outputs/run_<timestamp>)",
    )
    sp.add_argument(
        "--trace-dir", default=None,
        help="Trace output directory (overrides the default outputs/run_<timestamp>)",
    )
    sp.add_argument(
        "--checkpoints", default="warn", choices=["off", "warn", "raise"],
        help="Checkpoint enforcement mode (default: warn)",
    )
    sp.add_argument(
        "--inputs", nargs="*", default=[], metavar="K=V",
        help="Initial workflow inputs as k=v pairs (values parsed as JSON "
             "when possible, else strings)",
    )
    sp.add_argument(
        "-v", "--verbose", action="store_true",
        help="Enable debug logging",
    )
    sp.set_defaults(func=_handle)


def _parse_inputs(pairs: list[str]) -> dict:
    import json

    out: dict = {}
    for pair in pairs:
        if "=" not in pair:
            raise SystemExit(f"--inputs entries must be k=v, got {pair!r}")
        key, _, raw = pair.partition("=")
        try:
            out[key] = json.loads(raw)
        except (ValueError, TypeError):
            out[key] = raw
    return out


def _validate_only(graph: str, skills: str | None) -> int:
    from pathlib import Path

    from gap.runtime.validate import validate_workflow
    from gap.runtime.workflow import load_workflow

    path = Path(graph)
    if path.is_dir():
        path = path / "workflow.json"
    try:
        wf = load_workflow(path)
    except Exception as exc:
        print(f"FAIL: {exc}")
        return 1

    skill_registry = None
    if skills is None:
        from gap.skills import find_skills_path

        skills = find_skills_path()
    if skills:
        from gap.skills import load_skills

        skill_registry = load_skills(skills)

    issues = validate_workflow(wf, skill_registry=skill_registry)
    for issue in issues:
        print(str(issue))
    errors = [i for i in issues if i.severity == "error"]
    if errors:
        print(f"FAIL: {len(errors)} error(s), {len(issues) - len(errors)} warning(s)")
        return 1
    print(f"OK: 0 errors, {len(issues)} warning(s)")
    return 0


def _handle(args: argparse.Namespace) -> int:
    import logging
    import time
    from pathlib import Path

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if args.validate_only:
        return _validate_only(args.graph, args.skills)

    # Tracing is ON by default: ./outputs/run_<timestamp> unless overridden.
    if args.no_trace:
        trace_dir = None
    elif args.trace_dir:
        trace_dir = args.trace_dir
    else:
        trace_dir = str(
            Path("outputs") / f"run_{time.strftime('%Y%m%d_%H%M%S')}"
        )

    if args.sim and args.real:
        raise SystemExit("--sim and --real are mutually exclusive")

    connector = None
    if args.sim:
        import gap.connector

        connector = gap.connector.sim("libero", task=args.sim)
    elif args.real:
        import gap.connector

        connector = gap.connector.real(
            args.real,
            rr_config=args.rr_config,
            rr_autostart=not args.no_rr_autostart,
        )

    from gap.runtime.execute import execute

    try:
        result = execute(
            args.graph,
            connector,
            skills=args.skills,
            inputs=_parse_inputs(args.inputs) or None,
            trace_dir=trace_dir,
            checkpoints=args.checkpoints,
        )
    finally:
        if connector is not None:
            connector.close()

    status = "SUCCESS" if result.success else "FAILURE"
    print(f"{status} (exit={result.exit_status}, {result.duration_s:.1f}s)")
    if result.trace_path is not None:
        print(f"trace: {result.trace_path}")
    for cp in result.checkpoint_results:
        print(f"checkpoint: {cp}")
    if result.error is not None:
        print(f"error: {result.error}")
    return 0 if result.success else 1
