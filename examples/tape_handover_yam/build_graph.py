#!/usr/bin/env python
"""Author the reference bimanual tape-handover workflow with ``gap.builder``.

One ``tsh-bimanual-handover`` subgraph in the skill's recommended flow
(see its SKILL.md) — the same shape ``gap generate`` produces:

    observe → perceive_duct → perceive_tape → pickup → bimanual_exchange → place → placed

Everything numeric (meeting point, grasp offsets, clearances) lives in the
skill's ``scripts/constants.py``; the workflow only supplies the object keys and
the arm ids, plus the measured-data refs that make the pipeline
ground-truth-free.

Build + validate (no sim, no GPU)::

    python examples/tape_handover_yam/build_graph.py --out examples/tape_handover_yam
    gap run examples/tape_handover_yam --validate-only

Run it (sim): see ``run.py``.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from gap.builder import Ref, Subgraph, Workflow

GIVER, RECEIVER = 0, 1
YELLOW, DUCT = "yellow_tape_1", "duct_tape_1"

# Canonical scripts copied into <out>/scripts/ (what ``gap generate`` emits).
SCRIPTS = ["constants.py", "_motion.py", "_perceive.py", "perceive_duct.py",
           "perceive_tape.py", "pickup.py", "bimanual_exchange.py", "place.py",
           "return_leg.py"]


def build_workflow(roundtrip: bool = False) -> Workflow:
    sg = Subgraph(name="handover_sg", skill="tsh-bimanual-handover")
    sg.add_node("observe", type="tool", tool="robot.get_observation")
    sg.add_node("perceive_duct", type="script", script="scripts/perceive_duct.py",
                inputs={"target_key": DUCT, "cameras": Ref("observe.cameras")})
    sg.add_node("perceive_tape", type="script", script="scripts/perceive_tape.py",
                inputs={"object_key": YELLOW, "cameras": Ref("observe.cameras")})
    sg.add_node("pickup", type="script", script="scripts/pickup.py",
                inputs={"tape_xyz": Ref("perceive_tape.tape_xyz"),
                        "hole_radius": Ref("perceive_tape.hole_radius"),
                        "rim_radius": Ref("perceive_tape.rim_radius"),
                        "arm_id": GIVER})
    sg.add_node("bimanual_exchange", type="script",
                script="scripts/bimanual_exchange.py",
                inputs={"giver_arm": GIVER, "receiver_arm": RECEIVER,
                        "tape_in_giver": Ref("pickup.tape_in_giver")})
    sg.add_node("place", type="script", script="scripts/place.py",
                inputs={"receiver_offset": Ref("bimanual_exchange.receiver_offset"),
                        "duct_xyz": Ref("perceive_duct.duct_xyz"),
                        "tape_half_z": Ref("perceive_tape.tape_half_z"),
                        "tape_cloud": Ref("perceive_tape.tape_cloud"),
                        "arm_id": RECEIVER})
    # Round trip: replay the forward leg's RECORDED gripper poses in reverse
    # (roles swapped) so the tape ends back at its original spot — no second
    # perception pass, no mirrored geometry (see scripts/return_leg.py).
    exit_state, last = ("placed", "place") if not roundtrip else ("returned", None)
    if roundtrip:
        sg.add_node("return_leg", type="script", script="scripts/return_leg.py",
                    inputs={"grasp_tcp": Ref("pickup.grasp_tcp"),
                            "giver_tcp": Ref("bimanual_exchange.giver_tcp"),
                            "receiver_tcp": Ref("bimanual_exchange.receiver_tcp"),
                            "receiver_offset": Ref("bimanual_exchange.receiver_offset"),
                            "place_tcp": Ref("place.place_tcp"),
                            "tape_cloud": Ref("perceive_tape.tape_cloud"),
                            "object_key": YELLOW,
                            "giver_arm": GIVER, "receiver_arm": RECEIVER})
        last = "return_leg"
    sg.add_exit(exit_state)
    sg.set_on_error("failed")
    edges = [("START", "observe"), ("observe", "perceive_duct"),
             ("perceive_duct", "perceive_tape"),
             ("perceive_tape", "pickup"), ("pickup", "bimanual_exchange"),
             ("bimanual_exchange", "place")]
    if roundtrip:
        edges.append(("place", "return_leg"))
    edges += [(last, exit_state), (exit_state, "END")]
    for s, d in edges:
        sg.add_edge(s, d)

    wf = Workflow(name="tape_spool_handover_yam",
                  description="Non-policy bimanual tape handover on LIBERO-YAM "
                              "(scripted primitives + canonical curobo bundle).")
    wf.add_subgraph(sg)
    wf.add_node("handover", type="subgraph", ref="handover_sg")
    wf.add_node("done", type="end", status="success")
    wf.add_node("abort", type="end", status="failure",
                recovery=[{"tool": "robot.open_gripper", "inputs": {}}])
    wf.add_edge("START", "handover")
    wf.add_conditional_edges("handover", {exit_state: "done", "failed": "abort"},
                             router_field="exit")
    return wf


def main(argv=None) -> int:
    repo = Path(__file__).resolve().parents[2]  # .../graph-as-policy
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=str(repo / "examples" / "tape_handover_yam"))
    parser.add_argument("--tsh-skills", default=str(repo / "tsh-skills"))
    parser.add_argument("--roundtrip", action="store_true",
                        help="append the return leg: replay the forward poses in "
                             "reverse so the tape ends back where it was picked")
    args = parser.parse_args(argv)

    out_dir = Path(args.out)
    scripts_src = Path(args.tsh_skills) / "skills" / "tsh-bimanual-handover" / "scripts"
    (out_dir / "scripts").mkdir(parents=True, exist_ok=True)
    for name in SCRIPTS:
        src = scripts_src / name
        if not src.is_file():
            raise FileNotFoundError(f"script not found: {src}")
        dest = out_dir / "scripts" / name
        if not (dest.exists() and src.samefile(dest)):
            shutil.copy2(src, dest)

    build_workflow(roundtrip=args.roundtrip).save(out_dir / "workflow.json")
    print(f"built {out_dir}/  (workflow.json + {len(SCRIPTS)} scripts)")
    print(f"\nvalidate with:\n  gap run {out_dir} --validate-only")
    return 0


if __name__ == "__main__":
    sys.exit(main())

