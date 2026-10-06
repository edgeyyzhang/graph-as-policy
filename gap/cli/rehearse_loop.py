"""``gap rehearse-loop`` — the loop between an authoring agent and the rehearsal.

    gap rehearse-loop init WORKSPACE TRUSTED --graph GRAPH --sim libero_object_all_variance/0 \\
        --instruction "pick up the alphabet soup and place it in the basket" \\
        --visible 1-8 --holdout 9-16 --budget 8
    gap rehearse-loop serve TRUSTED
    gap rehearse-loop status TRUSTED

``init`` creates the agent's workspace and the trusted directory. ``serve``
answers the agent's validate and rehearse requests. ``status`` prints the
ledger, including the held-out results the agent never sees.
"""

from __future__ import annotations

import argparse


def register(subparsers: argparse._SubParsersAction) -> None:
    sp = subparsers.add_parser("rehearse-loop", help="Run the loop between an authoring agent and the rehearsal")
    sub = sp.add_subparsers(dest="loop_command", required=True)

    init = sub.add_parser("init", help="Create the agent's workspace and the trusted directory")
    init.add_argument("workspace", help="Directory handed to the agent (must not exist)")
    init.add_argument("trusted", help="Directory for the configuration, full results and ledger (must not exist)")
    init.add_argument("--graph", required=True, help="Starting graph: a workflow directory")
    init.add_argument("--sim", required=True, metavar="SUITE/TASK")
    init.add_argument("--env", default="libero")
    init.add_argument("--instruction", required=True, help="The task in plain language, shown to the agent")
    init.add_argument("--visible", required=True, help="Cases the agent sees: '1-8' or '1,3,5'")
    init.add_argument("--holdout", default="", help="Cases the agent never sees")
    init.add_argument("--budget", type=int, default=8, help="Number of rehearsals the agent may use (default: 8)")
    init.add_argument("--ik", default="pyroki", choices=["curobo", "pyroki"])
    init.add_argument("--skills", action="append", default=None, metavar="PATH",
                      help="Skill library root(s) copied into the workspace")
    init.add_argument("--no-frames", action="store_true")
    init.add_argument("--no-python", action="store_true", help="Do not install a Python into the workspace")
    init.set_defaults(func=_init)

    serve = sub.add_parser("serve", help="Answer the agent's requests")
    serve.add_argument("trusted")
    serve.add_argument("--once", action="store_true", help="Handle the pending requests and return")
    serve.add_argument("--idle-timeout", type=float, default=None, metavar="S",
                       help="Stop after this many seconds without a request")
    serve.add_argument("-v", "--verbose", action="store_true")
    serve.set_defaults(func=_serve)

    status = sub.add_parser("status", help="Print the ledger, including held-out results")
    status.add_argument("trusted")
    status.set_defaults(func=_status)


def _init(args: argparse.Namespace) -> int:
    from gap.rehearse.loop import init_loop
    from gap.rehearse.run import parse_cases

    config = init_loop(
        args.workspace, args.trusted,
        graph=args.graph, sim=args.sim, env=args.env, instruction=args.instruction,
        visible_cases=parse_cases(args.visible),
        holdout_cases=parse_cases(args.holdout) if args.holdout else [],
        rehearsal_budget=args.budget, ik=args.ik, skills=args.skills,
        frames=not args.no_frames, python=not args.no_python,
    )
    print(f"workspace: {config['workspace']}")
    print(f"trusted:   {config['trusted']}")
    print(f"visible cases {config['visible_cases']}, held-out cases {config['holdout_cases']}, "
          f"budget {config['rehearsal_budget']} rehearsals")
    return 0


def _serve(args: argparse.Namespace) -> int:
    import logging

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    from gap.rehearse.loop import Broker

    try:
        Broker(args.trusted).serve(once=args.once, idle_timeout_s=args.idle_timeout)
    except KeyboardInterrupt:
        pass
    return 0


def _status(args: argparse.Namespace) -> int:
    import json
    from pathlib import Path

    from gap.rehearse.loop.workspace import read_ledger

    trusted = Path(args.trusted)
    config = json.loads((trusted / "config.json").read_text())
    ledger = read_ledger(trusted)
    print(f"task: {config['instruction']}")
    print(f"sim {config['sim']}; visible {config['visible_cases']}; held out {config['holdout_cases']}")
    print(f"rehearsals used: {len(ledger)} of {config['rehearsal_budget']}")
    for row in ledger:
        v, h = row["visible"], row.get("holdout")
        held = f"{h['successes']} of {h['cases']}" if h else "not run"
        print(f"  round {row['round']:02d}: visible {v['successes']} of {v['cases']}, held out {held}")
    return 0
