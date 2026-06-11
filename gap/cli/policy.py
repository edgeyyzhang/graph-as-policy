"""``gap policy`` subcommand — serve / list learned-policy presets."""

from __future__ import annotations

import argparse


def register(subparsers: argparse._SubParsersAction) -> None:
    sp = subparsers.add_parser(
        "policy",
        help="Manage learned-policy servers (serve a preset, list presets)",
    )
    psub = sp.add_subparsers(dest="policy_command")

    serve = psub.add_parser(
        "serve",
        help="Spawn a policy server from a named preset and block until "
             "Ctrl-C",
    )
    serve.add_argument(
        "preset",
        help="Preset name (see `gap policy list`), e.g. pi05-libero",
    )
    serve.add_argument(
        "--port", type=int, default=None,
        help="Serve on this port (default: an OS-allocated free port)",
    )
    serve.add_argument(
        "--startup-timeout", type=float, default=900.0, metavar="SECS",
        help="How long to wait for the server port to open (default 900; "
             "first run downloads checkpoints)",
    )
    serve.set_defaults(func=_handle_serve)

    lst = psub.add_parser("list", help="List the known policy presets")
    lst.set_defaults(func=_handle_list)

    sp.set_defaults(func=_handle_help, _parser=sp)


def _handle_help(args: argparse.Namespace) -> int:
    args._parser.print_help()
    return 1


def _handle_list(args: argparse.Namespace) -> int:
    from gap.runtime.policy_presets import PRESETS

    print(f"{len(PRESETS)} preset(s):\n")
    for name, preset in sorted(PRESETS.items()):
        print(f"  {name}")
        print(f"    checkpoint: {preset.get('checkpoint_uri', '-')}")
        cmd = " ".join(str(preset.get("start_cmd", "")).split())
        print(f"    start_cmd:  {cmd}")
        notes = str(preset.get("notes", "")).strip()
        if notes:
            print(f"    notes:      {notes}")
        print()
    print("serve one with: gap policy serve <preset> [--port N]")
    return 0


def _handle_serve(args: argparse.Namespace) -> int:
    import logging
    import time
    from contextlib import contextmanager

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    from gap.runtime.policy_manager import PolicyManager
    from gap.runtime.policy_presets import PRESETS, resolve_policies

    if args.preset not in PRESETS:
        print(
            f"error: unknown preset {args.preset!r} "
            f"(available: {', '.join(sorted(PRESETS))})"
        )
        return 2

    policy_id = args.preset
    entries = resolve_policies({policy_id: {"preset": args.preset}})

    @contextmanager
    def _pinned_port(port: int | None):
        """Pin the manager's port allocation to a user-chosen port.

        PolicyManager always allocates a free port for the ``{port}``
        placeholder; for a long-lived `gap policy serve` the user wants
        a stable, well-known port, so we swap the allocator for the
        duration of the spawn.
        """
        if port is None:
            yield
            return
        import gap.runtime.policy_manager as pm_mod

        original = pm_mod._allocate_free_port
        pm_mod._allocate_free_port = lambda: port
        try:
            yield
        finally:
            pm_mod._allocate_free_port = original

    manager = PolicyManager(
        entries=entries, startup_timeout_s=float(args.startup_timeout),
    )
    try:
        with _pinned_port(args.port):
            manager.boot_all([policy_id])
    except Exception as exc:
        print(f"FAIL: {exc}")
        return 1

    url = manager.url_for(policy_id)
    print(f"policy {policy_id!r} serving on {url} — Ctrl-C to stop")
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\nshutting down ...")
    finally:
        manager.shutdown_all()
    return 0
