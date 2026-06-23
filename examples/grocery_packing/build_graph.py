#!/usr/bin/env python
"""Author a grocery-*packing* workflow — pick up EVERY object and basket them.

Where ``examples/build_a_graph`` picks a single described object, this graph
**loops**: perceive the next object on the table, grasp it with the planner
recipe, transport it into the basket, then route **back** to perception and
repeat — until perception reports nothing left, at which point the loop exits
cleanly. The basket is perceived once up front; only the per-object subgraphs
iterate.

The loop is a genuine **backward edge** in the policy graph:

    transport --placed--> perceive_next        # ← re-enters an earlier node

The executor treats a conditional edge that resolves to an already-completed
node as a loop: it resets the loop body and re-runs it (see docs/runtime.md
§7.2). ``node_visit_cap`` bounds runaway loops.

Build + validate (no sim)::

    python examples/grocery_packing/build_graph.py --out my_packing

Execute on the LIBERO sim (needs ``uv run gap skills install --all`` with
downloaded weights, a VLM credential, and ``MUJOCO_GL=egl``)::

    MUJOCO_GL=egl python examples/grocery_packing/build_graph.py \\
        --out my_packing --execute

or run the built artifact like any other graph::

    MUJOCO_GL=egl gap run my_packing --sim libero_object_packing/0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from gap.builder import Ref, Subgraph, Workflow

# Perception prompts + the sim body the (probe) grasp/transport checkpoints
# verify against. ``TARGET_PROMPT`` names the *class* of things to pack
# ("grocery item") rather than a single described object, so each loop
# surfaces whichever item is still on the table. The class name is load-
# bearing: ``perceive_dino_vlm`` runs a VLM pairwise tournament that, for
# every comparison, keeps the crop that better matches the prompt — and a
# real grocery item beats the basket on "which is the grocery item?" every
# time. So the tournament returns the basket ONLY once every item has been
# packed and teleported away, which ``route_next_object`` reads (geometrically)
# as "table clear → stop". A bare "object" prompt breaks both halves: the
# basket is just as much an "object" as a can, so the tournament picks it (or
# any item) arbitrarily — erratic grasps — and never yields a clean stop.
TARGET_PROMPT = "grocery item"
CONTAINER_PROMPT = "basket"
TARGET_BODY = "alphabet soup"

#: Canonical bundle scripts the graph's ``type: script`` nodes reference,
#: copied verbatim from the open-robot-skills checkout into ``<out>/scripts/`` —
#: exactly the per-workflow copies ``gap generate`` emits. Same planner-grasp +
#: mask-isolated transport recipe as the grocery_fulfillment acceptance graph.
SCRIPT_SOURCES: dict[str, str] = {
    "scripts/perceive_dino_vlm.py":
        "skills/perceiving-objects/scripts/perceive_dino_vlm.py",
    "scripts/compute_drop_pose.py":
        "skills/transporting-objects/scripts/compute_drop_pose.py",
    "scripts/waypoint_move.py":
        "skills/transporting-objects/scripts/waypoint_move.py",
    "scripts/descend_release.py":
        "skills/transporting-objects/scripts/descend_release.py",
}

#: A tiny router script (materialized by ``write_graph``) that turns the
#: perception result into the loop's continue/stop decision. Keeping "none" a
#: normal router outcome (not ``on_error``) lets the loop terminate without
#: tripping the abort path. The router also supplies the loop's *termination*
#: signal: the "grocery item" prompt only resolves to the basket once every
#: real item has been packed and teleported away, so a perceived target that
#: sits inside the basket's own XY footprint means the table is clear.
_ROUTE_NEXT_OBJECT_SCRIPT = '''\
"""Loop control for grocery_packing's perceive_next subgraph.

Routes to "found" when a graspable grocery item is still on the table (grasp
it) or "none" when only the basket remains (the loop's clean exit).

Why a geometric check and not just ``found``: with the "grocery item" prompt
the VLM tournament prefers a real item over the basket on every comparison,
so it returns the basket ONLY when every item has already been packed and
teleported to the benchmark graveyard. We confirm that here — if the
perceived target's centroid falls inside the basket's XY footprint, there is
nothing left but the basket, so the table is clear. Authored by build_graph.py.
"""
from __future__ import annotations

import numpy as np

from gap import NodeContext


def run(ctx: NodeContext, found: bool, cloud: dict, container_obb: dict) -> dict:
    if not found:
        return {"route": "none"}
    pts = np.asarray(cloud["points"]) if cloud and "points" in cloud else None
    if pts is None or pts.size == 0:
        return {"route": "none"}
    cx, cy = float(pts[:, 0].mean()), float(pts[:, 1].mean())
    center, extent = container_obb["center"], container_obb["extent"]
    # OBB ``extent`` is a half-extent; anything within the basket footprint
    # (+ a small margin) is the basket itself, so the loop is done.
    basket_r = float(max(extent["x"], extent["y"])) + 0.03
    if float(np.hypot(cx - center["x"], cy - center["y"])) < basket_r:
        return {"route": "none"}
    return {"route": "found"}
'''


# ---------------------------------------------------------------------------
# Subgraphs
# ---------------------------------------------------------------------------


def container_subgraph() -> Subgraph:
    """Perceive the basket ONCE: observe → DINO+VLM+SAM3 → OBB fit → found.

    Runs a single time at the start of the workflow; its ``container_obb``
    output is read by every transport iteration (the cross-subgraph store
    keeps the most-recent producer, and only this subgraph produces it).
    """
    sg = Subgraph(name="container_sg", skill="perceiving-objects")
    sg.add_node("observe", type="tool", tool="robot.get_observation")
    sg.add_node(
        "perceive", type="script", script="scripts/perceive_dino_vlm.py",
        inputs={"cameras": Ref("observe.cameras"), "object_name": CONTAINER_PROMPT},
    )
    sg.add_node(
        "filter_obb", type="tool", tool="geometry.filter_and_compute_obb",
        inputs={"points": Ref("perceive.cloud")},
    )
    sg.add_exit("found")
    sg.set_on_error("not_found")
    for src, dst in [("START", "observe"), ("observe", "perceive"),
                     ("perceive", "filter_obb"), ("filter_obb", "found"),
                     ("found", "END")]:
        sg.add_edge(src, dst)
    sg.set_outputs(container_obb=Ref("filter_obb.obb"),
                   container_mask=Ref("perceive.mask"))
    return sg


def perceive_next_subgraph() -> Subgraph:
    """The loop head: perceive the next grocery item, or exit if the table is clear.

    observe → perceive → decide(router on the perceived target vs. the basket):
      * found → fit OBB → ``found`` exit (grasp this item)
      * none  → ``none`` exit (only the basket remains → the loop terminates)

    The basket OBB (perceived once by ``container_sg``) is threaded in as a
    declared input so the router can tell "still an item on the table" from
    "only the basket is left" — see ``_ROUTE_NEXT_OBJECT_SCRIPT``. OBB fitting
    is gated behind "found" so an empty point cloud never reaches
    ``geometry.filter_and_compute_obb``; "none" is a normal exit marker, not
    ``on_error``, so termination does not look like a failure.
    """
    sg = Subgraph(name="perceive_next_sg", skill="perceiving-objects")
    sg.add_input("container_obb", type_name="OrientedBoundingBox")
    sg.add_node("observe", type="tool", tool="robot.get_observation")
    sg.add_node(
        "perceive", type="script", script="scripts/perceive_dino_vlm.py",
        inputs={"cameras": Ref("observe.cameras"), "object_name": TARGET_PROMPT},
    )
    sg.add_node(
        "decide", type="router", script="scripts/route_next_object.py",
        inputs={"found": Ref("perceive.found"),
                "cloud": Ref("perceive.cloud"),
                "container_obb": Ref("in.container_obb")},
    )
    sg.add_node(
        "filter_obb", type="tool", tool="geometry.filter_and_compute_obb",
        inputs={"points": Ref("perceive.cloud")},
    )
    sg.add_exit("found")          # item on the table → grasp it
    sg.add_exit("none")           # only the basket left → clean loop exit
    sg.set_on_error("not_found")  # genuine perception failure → abort
    sg.add_edge("START", "observe")
    sg.add_edge("observe", "perceive")
    sg.add_edge("perceive", "decide")
    sg.add_conditional_edges(
        "decide", {"found": "filter_obb", "none": "none"})
    sg.add_edge("filter_obb", "found")
    sg.add_edge("found", "END")
    sg.add_edge("none", "END")
    sg.set_outputs(target_obb=Ref("filter_obb.obb"),
                   target_mask=Ref("perceive.mask"))
    return sg


def grasp_subgraph() -> Subgraph:
    """Direct top-down grasp (matches the simplified grocery_fulfillment recipe):
    open → top-down candidates → ``robot.go_to_pose`` with an approach-from-above
    (``z_approach``) → observe → close.

    No CuRobo collision planner: ``robot.go_to_pose`` runs the connector's
    in-process IK with a lift-then-descend approach. This is far more robust
    than the planner path (which could stall/fail per-candidate on awkward
    objects) and carries no large ``world_config`` RPC payload."""
    sg = Subgraph(name="grasp_sg", skill="grasping-with-planner")
    sg.add_input("target_obb", type_name="OrientedBoundingBox")
    sg.add_node("open", type="tool", tool="robot.open_gripper",
                inputs={"settle_steps": 40})
    sg.add_node("compute_grasp", type="tool",
                tool="geometry.top_down_grasp_candidates",
                inputs={"obb": Ref("in.target_obb")})
    sg.add_node("goto_grasp", type="tool", tool="robot.go_to_pose",
                inputs={"pose": Ref("compute_grasp.candidates.poses.0"),
                        "z_approach": 0.1})
    sg.add_node("observe", type="tool", tool="robot.get_observation", inputs={})
    sg.add_node("close", type="tool", tool="robot.close_gripper",
                inputs={"settle_steps": 60})
    sg.add_exit("grasped")
    sg.set_on_error("failed")
    for src, dst in [("START", "open"), ("open", "compute_grasp"),
                     ("compute_grasp", "goto_grasp"), ("goto_grasp", "observe"),
                     ("observe", "close"), ("close", "grasped"), ("grasped", "END")]:
        sg.add_edge(src, dst)
    sg.set_outputs(ee_pose_at_grasp=Ref("observe.arms.0.ee_pose"),
                   grasp_pose=Ref("compute_grasp.candidates.poses.0"))
    return sg


def transport_subgraph() -> Subgraph:
    """``transporting-objects`` (ported from grocery_fulfillment): drop pose
    from the basket OBB + held object + grasp pose → lift + lateral waypoint
    move → descend, release, retract."""
    sg = Subgraph(name="transport_sg", skill="transporting-objects")
    sg.add_input("container_obb", type_name="OrientedBoundingBox")
    sg.add_input("target_obb", type_name="OrientedBoundingBox")
    sg.add_input("ee_pose_at_grasp", type_name="Se3Pose")
    sg.add_node(
        "compute_drop", type="script", script="scripts/compute_drop_pose.py",
        inputs={"container_obb": Ref("in.container_obb"),
                "held_obb": Ref("in.target_obb"),
                "ee_pose_at_grasp": Ref("in.ee_pose_at_grasp")},
    )
    sg.add_node(
        "move_above", type="script", script="scripts/waypoint_move.py",
        inputs={"drop_x": Ref("compute_drop.drop_position.x"),
                "drop_y": Ref("compute_drop.drop_position.y")},
    )
    sg.add_node(
        "release", type="script", script="scripts/descend_release.py",
        inputs={"drop_position": Ref("compute_drop.drop_position")},
    )
    sg.add_exit("placed")
    sg.set_on_error("blocked")
    for src, dst in [("START", "compute_drop"), ("compute_drop", "move_above"),
                     ("move_above", "release"), ("release", "placed"),
                     ("placed", "END")]:
        sg.add_edge(src, dst)
    sg.set_outputs(drop_pose=Ref("compute_drop.drop_pose"))
    return sg


# ---------------------------------------------------------------------------
# Top-level workflow (the loop)
# ---------------------------------------------------------------------------


def build_workflow(
    target: str = TARGET_PROMPT,
    container: str = CONTAINER_PROMPT,
) -> Workflow:
    """Assemble the looping pack-everything workflow (pure, no file I/O)."""
    wf = Workflow(
        name="grocery_packing",
        description=(
            f"Pick up every {target} on the table and place each in the "
            f"{container}, looping until perception finds nothing left "
            f"(planner grasp; authored with gap.builder)."
        ),
    )
    wf.add_subgraph(container_subgraph())
    wf.add_subgraph(perceive_next_subgraph())
    wf.add_subgraph(grasp_subgraph())
    wf.add_subgraph(transport_subgraph())

    wf.add_node("container", type="subgraph", ref="container_sg")
    wf.add_node("perceive_next", type="subgraph", ref="perceive_next_sg")
    wf.add_node("grasp", type="subgraph", ref="grasp_sg")
    wf.add_node("transport", type="subgraph", ref="transport_sg")
    wf.add_node("done", type="end", status="success")
    wf.add_node(
        "abort", type="end", status="failure",
        recovery=[{"tool": "robot.open_gripper", "inputs": {}},
                  {"tool": "robot.go_home", "inputs": {}}],
    )

    wf.add_edge("START", "container")
    wf.add_conditional_edges(
        "container", {"found": "perceive_next", "not_found": "abort"},
        router_field="exit")
    wf.add_conditional_edges(
        "perceive_next",
        {"found": "grasp", "none": "done", "not_found": "abort"},
        router_field="exit")
    wf.add_conditional_edges(
        "grasp", {"grasped": "transport", "failed": "abort"},
        router_field="exit")
    wf.add_conditional_edges(
        "transport", {"placed": "perceive_next", "blocked": "abort"},
        router_field="exit")          # ← THE BACKWARD EDGE: loop to perceive_next
    return wf


# ---------------------------------------------------------------------------
# Artifact assembly (the same on-disk layout `gap generate` writes)
# ---------------------------------------------------------------------------

_CONTAINER_CHECKPOINT = '''\
"""Postcondition checkpoint for `container_sg` (authored by build_graph.py)."""

from gap.runtime.verify import Checkpoint


def _container_matches_truth(world, out) -> bool:
    """The perceived basket OBB center lines up with the sim's ground truth."""
    c = out["container_obb"]["center"]
    p = world.body("basket").position
    return abs(c["x"] - p[0]) < 0.12 and abs(c["y"] - p[1]) < 0.12


CHECKPOINTS = [
    Checkpoint(
        name="container_obb_matches_truth",
        subgraph="container_sg",
        # Probe, not a hard gate: a large open basket's OBB *center* (geometric
        # centroid of the visible rim) sits a fair bit from the sim body
        # *origin*, so an exact ground-truth match is unreliable. We still
        # surface the localization delta in feedback, but don't gate on it.
        predicate=_container_matches_truth,
        rationale="perceived basket localizes near the real basket (probe)",
        validate=False,
    ),
]
'''

# Per-iteration grasp/transport checkpoints are PROBES (validate=False): they
# surface in feedback but never gate. The packing loop handles many objects but
# the sim exposes ground truth only by named body, so a hard per-iteration gate
# would need every object's body name. They probe the canonical {target_body!r}.
_GRASP_CHECKPOINT = '''\
"""Probe checkpoint for `grasp_sg` (authored by build_graph.py)."""

from gap.runtime.verify import Checkpoint


def _target_held(world) -> bool:
    return world.body({target_body!r}).is_grasped()


CHECKPOINTS = [
    Checkpoint(
        name="target_held",
        subgraph="grasp_sg",
        predicate=_target_held,
        rationale=(
            "probe: when the canonical object is the one being grasped, the "
            "gripper closed ON it (multi-object loops need per-body names for "
            "a hard gate, so this stays a probe)"
        ),
        validate=False,
    ),
]
'''

_TRANSPORT_CHECKPOINT = '''\
"""Probe checkpoint for `transport_sg` (authored by build_graph.py)."""

from gap.runtime.verify import Checkpoint


def _target_in_container(world) -> bool:
    return world.body({target_body!r}).is_in(world.body("basket"))


CHECKPOINTS = [
    Checkpoint(
        name="target_in_container",
        subgraph="transport_sg",
        predicate=_target_in_container,
        rationale=(
            "probe: the canonical object settled inside the basket after "
            "release (probe, not a hard gate — see grasp_sg)"
        ),
        validate=False,
    ),
]
'''


def write_graph(
    wf: Workflow,
    out_dir: str | Path,
    skills_root: str | Path,
    *,
    target_body: str = TARGET_BODY,
) -> Path:
    """Materialize the workflow directory: scripts + workflow.json + checkpoints.

    Scripts are written first so ``wf.save`` (which parses and structurally
    validates, importing each script for schema introspection) sees the
    complete artifact.
    """
    import shutil

    out_dir = Path(out_dir)
    skills_root = Path(skills_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "scripts").mkdir(exist_ok=True)

    for dest_rel, src_rel in SCRIPT_SOURCES.items():
        src = skills_root / src_rel
        if not src.is_file():
            raise FileNotFoundError(
                f"canonical script not found: {src} — is {skills_root} an "
                f"open-robot-skills checkout?"
            )
        shutil.copy2(src, out_dir / dest_rel)

    # The loop-control router is new (not a bundle script): materialize it.
    (out_dir / "scripts" / "route_next_object.py").write_text(
        _ROUTE_NEXT_OBJECT_SCRIPT, encoding="utf-8")

    wf.save(out_dir / "workflow.json")        # parse + W/S validation

    checkpoints_dir = out_dir / "checkpoints"
    checkpoints_dir.mkdir(exist_ok=True)
    (checkpoints_dir / "container_sg.py").write_text(
        _CONTAINER_CHECKPOINT, encoding="utf-8")
    (checkpoints_dir / "grasp_sg.py").write_text(
        _GRASP_CHECKPOINT.format(target_body=target_body), encoding="utf-8")
    (checkpoints_dir / "transport_sg.py").write_text(
        _TRANSPORT_CHECKPOINT.format(target_body=target_body), encoding="utf-8")
    return out_dir


def validate_graph(out_dir: Path, skills_root: Path) -> int:
    """Re-validate the written artifact with the skill registry in the loop."""
    from gap.runtime.validate import validate_workflow
    from gap.runtime.workflow import load_workflow
    from gap.skills import load_skills

    wf_def = load_workflow(out_dir / "workflow.json")
    issues = validate_workflow(wf_def, skill_registry=load_skills(skills_root))
    errors = [i for i in issues if i.severity == "error"]
    for issue in issues:
        print(f"  {issue}")
    print(f"validate: {len(errors)} error(s), {len(issues) - len(errors)} warning(s)")
    return len(errors)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default="my_packing", metavar="DIR",
                        help="output workflow directory (default: my_packing)")
    parser.add_argument("--skills", default=None,
                        help="open-robot-skills checkout (default: auto-discovered "
                             "via $GAP_SKILLS_PATH or the sibling checkout)")
    parser.add_argument("--target", default=TARGET_PROMPT,
                        help="perception prompt for the objects to pick")
    parser.add_argument("--container", default=CONTAINER_PROMPT,
                        help="perception prompt for the destination basket")
    parser.add_argument("--target-body", default=TARGET_BODY,
                        help="sim body name the probe checkpoints reference")
    parser.add_argument("--execute", action="store_true",
                        help="after building, run the graph on the LIBERO sim")
    parser.add_argument("--task", default="libero_object_packing/0",
                        metavar="SUITE/ID",
                        help="sim task for --execute. Defaults to the VAB "
                             "pack-all suite, whose env teleports each delivered "
                             "object to a graveyard pose and scores "
                             "completion_rate = delivered/total.")
    parser.add_argument("--video", default=None, metavar="PATH",
                        help="with --execute, record an mp4 of the rollout to PATH")
    args = parser.parse_args(argv)

    from gap.skills import find_skills_path

    skills_root = find_skills_path(args.skills, required=True)

    wf = build_workflow(target=args.target, container=args.container)
    out_dir = write_graph(wf, args.out, skills_root, target_body=args.target_body)
    print(f"built {out_dir}/  (workflow.json + {len(SCRIPT_SOURCES)} scripts + "
          f"route_next_object.py + 3 checkpoint sidecars)")

    n_errors = validate_graph(out_dir, skills_root)
    if n_errors:
        return 1

    if not args.execute:
        print(f"\nrun it with:\n  MUJOCO_GL=egl gap run {out_dir} --sim {args.task}")
        return 0

    import gap

    conn = gap.connector.sim("libero", task=args.task, record_video=bool(args.video))
    if args.video and hasattr(conn, "start_video"):
        # record_video=True enables offscreen render but only arms capture on
        # connector.reset(); the workflow never resets (it opens with
        # get_observation), so start capture explicitly before the rollout.
        conn.start_video()
    packed = None
    completion = None
    try:
        result = gap.execute(out_dir, conn, skills=skills_root)
        try:  # VAB packing env: report all-delivered + reward (best-effort)
            packed = conn.check_success()
        except Exception:  # noqa: BLE001 — non-packing envs lack this
            packed = None
        try:  # per-object delivered/total — VAB packing env tracks it in step info
            completion = (getattr(conn.env, "_current_info", None) or {}).get(
                "completion_rate")
        except Exception:  # noqa: BLE001 — non-packing envs lack this
            completion = None
    finally:
        if args.video:
            try:
                info = conn.save_video(args.video)
                print(f"video: {args.video} ({info.get('num_frames', '?')} frames)")
            except Exception as exc:  # noqa: BLE001 — best-effort video
                print(f"video save failed: {exc}")
        conn.close()
    print(f"{'SUCCESS' if result.success else 'FAILURE'} "
          f"(exit={result.exit_status}, {result.duration_s:.1f}s)")
    if packed is not None:
        done, reward = packed
        line = f"packing: all_delivered={done} reward={reward:.3f}"
        if completion is not None:
            line += f" completion_rate={completion:.3f}"
        print(line)
    for cp in result.checkpoint_results:
        print(f"checkpoint: {cp}")
    if result.trace_path is not None:
        print(f"trace: {result.trace_path}  (browse with `gap viz`)")
    return 0 if result.success else 1


if __name__ == "__main__":
    sys.exit(main())
