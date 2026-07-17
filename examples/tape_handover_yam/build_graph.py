#!/usr/bin/env python
"""Author the routed bimanual tape-handover workflow with ``gap.builder``.

The appendix-style decomposition — generic single-purpose subgraphs, a
reachability-routed handover, verification gates with recovery loop edges,
and a next-item loop — instead of one monolithic handover chain:

    gripper_geometry → station_geometry → perceive_dest → perceive_target
      → ring_geometry → route ──direct/needs_handover──▶ pickup → verify_grasp
      → [dispatch] ──direct──▶ place
                   └─handover─▶ present → handover → place
      → verify_place → [next_item] → done

Recovery edges (the agent-testing surface):
  verify_grasp.retry  → perceive_target   (re-perceive + re-grasp, bounded)
  verify_place.retry  → perceive_target   (re-perceive wherever it landed)
  next_item.next      → perceive_dest     (multi-item loop; TSH runs once)

A handover is a ROUTED strategy, not a fixed stage: ``route`` probes each
arm's grasp + place reachability with the canonical curobo bundle (plans
only) and picks ``direct`` (one arm does both) or ``needs_handover``.

Everything numeric lives in the skills' ``scripts/constants.py``; the
workflow only supplies object queries, wiring, and the measured-data refs
that keep the pipeline ground-truth-free. Postcondition checkpoints live in
``checkpoints/<sg>.py`` sidecars (hand-authored, this directory) evaluated
against privileged sim state (``gap run --checkpoints warn|raise``).

Build + validate (no sim, no GPU)::

    python examples/tape_handover_yam/build_graph.py
    gap run examples/tape_handover_yam --validate-only

Run it (sim): see ``run.py``.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from gap.builder import Ref, Subgraph, Workflow

TARGET_QUERY = "yellow tape"   # DINO noun phrase for the pickup object
DEST_QUERY = "gray tape"       # colour-anchored query for the duct (place dest)

# Canonical scripts each subgraph materializes into <out>/scripts/<sg>/,
# copied from its skill's scripts dir: {sg_node: (skill_dir, [files])}.
SG_SCRIPTS: dict[str, tuple[str, list[str]]] = {
    "gripper_geometry": ("tsh-gripper-geometry",
                         ["gripper_geometry.py", "_gripper_geometry.py", "constants.py"]),
    "station_geometry": ("tsh-station-geometry",
                         ["station_geometry.py", "_station_geometry.py"]),
    "perceive_dest":    ("tsh-perceive",
                         ["perceive_object.py", "_perceive.py", "constants.py"]),
    "perceive_target":  ("tsh-perceive",
                         ["perceive_object.py", "_perceive.py", "constants.py"]),
    "ring_geometry":    ("tsh-ring-geometry",
                         ["ring_geometry.py", "_ring.py", "constants.py"]),
    "route":            ("tsh-route",
                         ["route.py", "_ring.py", "_motion.py", "constants.py"]),
    "pickup":           ("tsh-pickup",
                         ["pickup.py", "_ring.py", "_motion.py", "constants.py"]),
    "verify_grasp":     ("tsh-verify-grasp", ["verify_grasp.py"]),
    "present":          ("tsh-transport-held",
                         ["transport_held.py", "_held.py", "_motion.py"]),
    "handover":         ("tsh-handover",
                         ["bimanual_exchange.py", "_motion.py", "constants.py"]),
    "place":            ("tsh-place",
                         ["place.py", "_held.py", "_motion.py", "constants.py"]),
    "verify_place":     ("tsh-verify-place",
                         ["verify_place.py", "_perceive.py", "constants.py"]),
}

# Canonical pick-and-place stage tags (consumed by the refine loop's
# mechanical-swap engine; the runtime ignores them).
SG_STAGES = {"pickup": "grasp", "present": "transport",
             "handover": "transport", "place": "place"}

DISPATCH_ROUTE_SRC = '''\
"""Top-level router: after a verified grasp, dispatch on the probed route."""

from gap import NodeContext


def run(ctx: NodeContext, *, route: str) -> dict:
    if route not in ("direct", "handover"):
        raise RuntimeError(f"dispatch: unknown route {route!r}")
    return {"route": "place" if route == "direct" else "present"}
'''

NEXT_ITEM_SRC = '''\
"""Top-level router: loop to the next item, or finish.

TSH is a single-item task (``total_items=1`` → done after the first placed
item). A sorting port raises ``total_items`` (or replaces this counter with
a perceive-not_found loop exit) and the same edge re-enters at
``perceive_dest`` — re-perceiving the destination each item also makes the
growing stack height come for free. The counter is per-process (one run =
one episode).
"""

import itertools

from gap import NodeContext

_COUNT = itertools.count(1)


def run(ctx: NodeContext, *, total_items: int = 1) -> dict:
    placed = next(_COUNT)
    return {"route": "done" if placed >= int(total_items) else "next"}
'''


def _perceive_subgraph(name: str, query: str, prefix: str, *,
                       center_field: str) -> Subgraph:
    """One generic perceive-object instance (appendix perception pattern):
    the query is a literal, the outputs are name-prefixed."""
    sg = Subgraph(name=name, skill="tsh-perceive")
    sg.add_node("observe", type="tool", tool="robot.get_observation")
    sg.add_node("perceive", type="script",
                script=f"scripts/{name}/perceive_object.py",
                inputs={"cameras": Ref("observe.cameras"),
                        "object_query": query})
    sg.add_exit("perceived")
    for s, d in [("START", "observe"), ("observe", "perceive"),
                 ("perceive", "perceived"), ("perceived", "END")]:
        sg.add_edge(s, d)
    sg.set_outputs(**{
        f"{prefix}_found": Ref("perceive.found"),
        f"{prefix}_cloud": Ref("perceive.cloud"),
        f"{prefix}_xyz": Ref(f"perceive.{center_field}"),
        f"{prefix}_top_xyz": Ref("perceive.top_xyz"),
        f"{prefix}_half_z": Ref("perceive.half_z"),
    })
    sg.set_on_error("not_found")
    return sg


def _verdict_subgraph(name: str, skill: str, script: str, check_inputs: dict,
                      inputs: dict[str, str], verdicts: dict[str, str],
                      outputs: dict[str, Ref]) -> Subgraph:
    """A verification gate: observe → check, routed on the check's
    ``verdict`` field to one noop exit per verdict (recovery is ROUTED,
    never raised)."""
    sg = Subgraph(name=name, skill=skill)
    for in_name, type_name in inputs.items():
        sg.add_input(in_name, type_name=type_name)
    sg.add_node("observe", type="tool", tool="robot.get_observation")
    sg.add_node("check", type="script", script=script,
                inputs={**check_inputs, "cameras": Ref("observe.cameras")})
    for exit_name in verdicts.values():
        sg.add_exit(exit_name)
    sg.add_edge("START", "observe")
    sg.add_edge("observe", "check")
    sg.add_conditional_edges("check", dict(verdicts), router_field="verdict")
    for exit_name in verdicts.values():
        sg.add_edge(exit_name, "END")
    sg.set_outputs(**outputs)
    sg.set_on_error("failed")
    return sg


def build_workflow() -> Workflow:
    wf = Workflow(
        name="tape_spool_handover_yam",
        description="Routed bimanual tape handover on LIBERO-YAM: probe "
                    "reachability, hand over only when the destination is "
                    "out of the picking arm's reach, verify each stage, "
                    "recover by re-perceiving.")

    # -- self-model (run-once calibration) ---------------------------------
    sg = Subgraph(name="gripper_geometry", skill="tsh-gripper-geometry")
    sg.add_node("derive", type="script",
                script="scripts/gripper_geometry/gripper_geometry.py")
    sg.add_exit("derived")
    for s, d in [("START", "derive"), ("derive", "derived"), ("derived", "END")]:
        sg.add_edge(s, d)
    sg.set_outputs(fingertip_axial=Ref("derive.fingertip_axial"),
                   finger_half_gap=Ref("derive.finger_half_gap"))
    sg.set_on_error("failed")
    wf.add_subgraph(sg)

    sg = Subgraph(name="station_geometry", skill="tsh-station-geometry")
    sg.add_node("derive", type="script",
                script="scripts/station_geometry/station_geometry.py")
    sg.add_exit("derived")
    for s, d in [("START", "derive"), ("derive", "derived"), ("derived", "END")]:
        sg.add_edge(s, d)
    sg.set_outputs(meet_xyz=Ref("derive.meet_xyz"),
                   giver_quat=Ref("derive.giver_quat"),
                   recv_quat=Ref("derive.recv_quat"))
    sg.set_on_error("failed")
    wf.add_subgraph(sg)

    # -- perception (generic, one instance per object; dest FIRST — its
    #    view is clean before anything is held over it) ---------------------
    wf.add_subgraph(_perceive_subgraph(
        "perceive_dest", DEST_QUERY, "dest", center_field="top_xyz"))
    # dest_xyz = TOP-face centre (the place rest target).
    wf.add_subgraph(_perceive_subgraph(
        "perceive_target", TARGET_QUERY, "target", center_field="center_xyz"))
    # target_xyz = body-centroid height (the grasp point).

    # -- ring geometry (tape-specific post-processor) -----------------------
    sg = Subgraph(name="ring_geometry", skill="tsh-ring-geometry")
    for n, t in [("target_cloud", "PointCloud"), ("target_xyz", "Vec3"),
                 ("target_half_z", "float")]:
        sg.add_input(n, type_name=t)
    sg.add_node("ring", type="script",
                script="scripts/ring_geometry/ring_geometry.py",
                inputs={"cloud": Ref("in.target_cloud"),
                        "center_xyz": Ref("in.target_xyz"),
                        "half_z": Ref("in.target_half_z")})
    sg.add_exit("derived")
    for s, d in [("START", "ring"), ("ring", "derived"), ("derived", "END")]:
        sg.add_edge(s, d)
    sg.set_outputs(hole_radius=Ref("ring.hole_radius"),
                   rim_radius=Ref("ring.rim_radius"))
    sg.set_on_error("degenerate")
    wf.add_subgraph(sg)

    # -- route: is a handover needed at all? -------------------------------
    sg = Subgraph(name="route", skill="tsh-route")
    for n, t in [("target_xyz", "Vec3"), ("dest_xyz", "Vec3"),
                 ("target_half_z", "float"), ("hole_radius", "float"),
                 ("rim_radius", "float"), ("fingertip_axial", "float"),
                 ("finger_half_gap", "float")]:
        sg.add_input(n, type_name=t)
    sg.add_node("route", type="script", script="scripts/route/route.py",
                inputs={"target_xyz": Ref("in.target_xyz"),
                        "dest_xyz": Ref("in.dest_xyz"),
                        "half_z": Ref("in.target_half_z"),
                        "hole_radius": Ref("in.hole_radius"),
                        "rim_radius": Ref("in.rim_radius"),
                        "fingertip_axial": Ref("in.fingertip_axial"),
                        "finger_half_gap": Ref("in.finger_half_gap")})
    sg.add_exit("direct")
    sg.add_exit("needs_handover")
    sg.add_edge("START", "route")
    sg.add_conditional_edges(
        "route", {"direct": "direct", "handover": "needs_handover"},
        router_field="route")
    sg.add_edge("direct", "END")
    sg.add_edge("needs_handover", "END")
    sg.set_outputs(route=Ref("route.route"),
                   pick_arm=Ref("route.pick_arm"),
                   place_arm=Ref("route.place_arm"),
                   giver_arm=Ref("route.giver_arm"),
                   receiver_arm=Ref("route.receiver_arm"))
    sg.set_on_error("unreachable")
    wf.add_subgraph(sg)

    # -- grasp --------------------------------------------------------------
    sg = Subgraph(name="pickup", skill="tsh-pickup")
    for n, t in [("target_xyz", "Vec3"), ("hole_radius", "float"),
                 ("rim_radius", "float"), ("fingertip_axial", "float"),
                 ("finger_half_gap", "float"), ("pick_arm", "int")]:
        sg.add_input(n, type_name=t)
    sg.add_node("pickup", type="script", script="scripts/pickup/pickup.py",
                inputs={"tape_xyz": Ref("in.target_xyz"),
                        "hole_radius": Ref("in.hole_radius"),
                        "rim_radius": Ref("in.rim_radius"),
                        "fingertip_axial": Ref("in.fingertip_axial"),
                        "finger_half_gap": Ref("in.finger_half_gap"),
                        "arm_id": Ref("in.pick_arm")})
    sg.add_exit("grasped")
    for s, d in [("START", "pickup"), ("pickup", "grasped"), ("grasped", "END")]:
        sg.add_edge(s, d)
    sg.set_outputs(grasped=Ref("pickup.grasped"),
                   held_offset=Ref("pickup.held_offset"),
                   tape_in_giver=Ref("pickup.tape_in_giver"),
                   grasp_tcp=Ref("pickup.grasp_tcp"),
                   rim_radius=Ref("pickup.rim_radius"),
                   pick_arm=Ref("pickup.pick_arm"))
    sg.set_on_error("failed")
    wf.add_subgraph(sg)

    # -- verify the grasp (routed recovery) ---------------------------------
    wf.add_subgraph(_verdict_subgraph(
        "verify_grasp", "tsh-verify-grasp",
        "scripts/verify_grasp/verify_grasp.py",
        {"arm_id": Ref("in.pick_arm"), "object_query": TARGET_QUERY},
        inputs={"pick_arm": "int"},
        verdicts={"holding": "holding", "retry": "retry", "give_up": "give_up"},
        outputs={"verdict": Ref("check.verdict"),
                 "holding": Ref("check.holding"),
                 "fraction": Ref("check.fraction")}))

    # -- handover branch: present (shared held-transport), then exchange ----
    sg = Subgraph(name="present", skill="tsh-transport-held")
    for n, t in [("giver_arm", "int"), ("held_offset", "Vec3"),
                 ("meet_xyz", "Vec3"), ("giver_quat", "Quaternion"),
                 ("target_cloud", "PointCloud")]:
        sg.add_input(n, type_name=t)
    sg.add_node("transport", type="script",
                script="scripts/present/transport_held.py",
                inputs={"arm_id": Ref("in.giver_arm"),
                        "held_offset": Ref("in.held_offset"),
                        "target_xyz": Ref("in.meet_xyz"),
                        "target_quat": Ref("in.giver_quat"),
                        "held_cloud": Ref("in.target_cloud")})
    sg.add_exit("transported")
    for s, d in [("START", "transport"), ("transport", "transported"),
                 ("transported", "END")]:
        sg.add_edge(s, d)
    sg.set_outputs(transported=Ref("transport.transported"),
                   held_tcp=Ref("transport.held_tcp"))
    sg.set_on_error("failed")
    wf.add_subgraph(sg)

    sg = Subgraph(name="handover", skill="tsh-handover")
    for n, t in [("giver_arm", "int"), ("receiver_arm", "int"),
                 ("tape_in_giver", "Vec3"), ("rim_radius", "float"),
                 ("meet_xyz", "Vec3"), ("giver_quat", "Quaternion"),
                 ("recv_quat", "Quaternion")]:
        sg.add_input(n, type_name=t)
    sg.add_node("exchange", type="script",
                script="scripts/handover/bimanual_exchange.py",
                inputs={"giver_arm": Ref("in.giver_arm"),
                        "receiver_arm": Ref("in.receiver_arm"),
                        "tape_in_giver": Ref("in.tape_in_giver"),
                        "rim_radius": Ref("in.rim_radius"),
                        "meet_xyz": Ref("in.meet_xyz"),
                        "giver_quat": Ref("in.giver_quat"),
                        "recv_quat": Ref("in.recv_quat"),
                        "skip_present": True})
    sg.add_exit("handed_over")
    for s, d in [("START", "exchange"), ("exchange", "handed_over"),
                 ("handed_over", "END")]:
        sg.add_edge(s, d)
    sg.set_outputs(handed_over=Ref("exchange.handed_over"),
                   receiver_offset=Ref("exchange.receiver_offset"),
                   held_offset=Ref("exchange.held_offset"),
                   giver_tcp=Ref("exchange.giver_tcp"),
                   receiver_tcp=Ref("exchange.receiver_tcp"),
                   giver_arm=Ref("exchange.giver_arm"),
                   receiver_arm=Ref("exchange.receiver_arm"))
    sg.set_on_error("failed")
    wf.add_subgraph(sg)

    # -- place (either branch; held_offset is the LATEST holder's grip) -----
    sg = Subgraph(name="place", skill="tsh-place")
    for n, t in [("held_offset", "Vec3"), ("dest_xyz", "Vec3"),
                 ("target_half_z", "float"), ("target_cloud", "PointCloud"),
                 ("place_arm", "int")]:
        sg.add_input(n, type_name=t)
    sg.add_node("place", type="script", script="scripts/place/place.py",
                inputs={"held_offset": Ref("in.held_offset"),
                        "dest_xyz": Ref("in.dest_xyz"),
                        "arm_id": Ref("in.place_arm"),
                        "tape_half_z": Ref("in.target_half_z"),
                        "tape_cloud": Ref("in.target_cloud")})
    sg.add_exit("placed")
    for s, d in [("START", "place"), ("place", "placed"), ("placed", "END")]:
        sg.add_edge(s, d)
    sg.set_outputs(placed=Ref("place.placed"),
                   place_tcp=Ref("place.place_tcp"),
                   place_arm=Ref("place.place_arm"))
    sg.set_on_error("failed")
    wf.add_subgraph(sg)

    # -- verify the place (routed recovery) ---------------------------------
    wf.add_subgraph(_verdict_subgraph(
        "verify_place", "tsh-verify-place",
        "scripts/verify_place/verify_place.py",
        {"object_query": TARGET_QUERY, "dest_xyz": Ref("in.dest_xyz"),
         "half_z": Ref("in.target_half_z")},
        inputs={"dest_xyz": "Vec3", "target_half_z": "float"},
        verdicts={"placed": "placed", "retry": "retry", "give_up": "give_up"},
        outputs={"verdict": Ref("check.verdict"),
                 "placed": Ref("check.placed"),
                 "xy_error_m": Ref("check.xy_error_m"),
                 "z_error_m": Ref("check.z_error_m")}))

    # -- top level -----------------------------------------------------------
    for name in SG_SCRIPTS:
        wf.add_node(name, type="subgraph", ref=name)
    wf.add_node("dispatch", type="router", script="scripts/dispatch_route.py",
                inputs={"route": Ref("route.route")})
    wf.add_node("next_item", type="router", script="scripts/next_item.py",
                inputs={"total_items": 1})
    wf.add_node("done", type="end", status="success")
    wf.add_node("abort", type="end", status="failure",
                recovery=[{"tool": "robot.open_gripper", "inputs": {"arm_id": 0}},
                          {"tool": "robot.open_gripper", "inputs": {"arm_id": 1}}])

    wf.add_edge("START", "gripper_geometry")
    top_edges = {
        "gripper_geometry": {"derived": "station_geometry", "failed": "abort"},
        "station_geometry": {"derived": "perceive_dest", "failed": "abort"},
        "perceive_dest": {"perceived": "perceive_target", "not_found": "abort"},
        "perceive_target": {"perceived": "ring_geometry", "not_found": "abort"},
        "ring_geometry": {"derived": "route", "degenerate": "abort"},
        # Both routes grasp first; the dispatch after the verified grasp
        # decides whether an exchange happens.
        "route": {"direct": "pickup", "needs_handover": "pickup",
                  "unreachable": "abort"},
        "pickup": {"grasped": "verify_grasp", "failed": "abort"},
        # Recovery loop edges: a failed verify re-enters at perception.
        "verify_grasp": {"holding": "dispatch", "retry": "perceive_target",
                         "give_up": "abort", "failed": "abort"},
        "present": {"transported": "handover", "failed": "abort"},
        "handover": {"handed_over": "place", "failed": "abort"},
        "place": {"placed": "verify_place", "failed": "abort"},
        "verify_place": {"placed": "next_item", "retry": "perceive_target",
                         "give_up": "abort", "failed": "abort"},
    }
    for src, mapping in top_edges.items():
        wf.add_conditional_edges(src, mapping, router_field="exit")
    # Router nodes: the script's returned route IS the mapping key.
    wf.add_conditional_edges("dispatch", {"place": "place", "present": "present"})
    wf.add_conditional_edges("next_item", {"done": "done", "next": "perceive_dest"})
    return wf


def main(argv=None) -> int:
    repo = Path(__file__).resolve().parents[2]  # .../graph-as-policy
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=str(repo / "examples" / "tape_handover_yam"))
    parser.add_argument("--tsh-skills", default=str(repo / "tsh-skills"))
    args = parser.parse_args(argv)

    out_dir = Path(args.out)
    skills_root = Path(args.tsh_skills) / "skills"

    # The legacy layout symlinked <out>/scripts to the deprecated
    # tsh-bimanual-handover scripts dir; the routed workflow namespaces
    # scripts per subgraph, so replace the symlink with a real directory.
    scripts_dir = out_dir / "scripts"
    if scripts_dir.is_symlink():
        scripts_dir.unlink()
    scripts_dir.mkdir(parents=True, exist_ok=True)

    for sg_name, (skill, files) in SG_SCRIPTS.items():
        src_dir = skills_root / skill / "scripts"
        dest_dir = scripts_dir / sg_name
        dest_dir.mkdir(parents=True, exist_ok=True)
        for name in files:
            src = src_dir / name
            if not src.is_file():
                raise FileNotFoundError(f"canonical script not found: {src}")
            shutil.copy2(src, dest_dir / name)

    (scripts_dir / "dispatch_route.py").write_text(DISPATCH_ROUTE_SRC)
    (scripts_dir / "next_item.py").write_text(NEXT_ITEM_SRC)

    wf = build_workflow()
    out_path = out_dir / "workflow.json"
    wf.save(out_path)
    # Stage tags for the refine loop's swap engine — valid workflow.json keys
    # the builder does not (yet) expose; injected post-validation.
    raw = json.loads(out_path.read_text())
    for sg_name, stage in SG_STAGES.items():
        raw["subgraphs"][sg_name]["stage"] = stage
    out_path.write_text(json.dumps(raw, indent=2))

    n_scripts = sum(len(files) for _, files in SG_SCRIPTS.values()) + 2
    print(f"built {out_dir}/  (workflow.json + {n_scripts} scripts in "
          f"{len(SG_SCRIPTS)} subgraph dirs; checkpoints/ ride along)")
    print(f"\nvalidate with:\n  gap run {out_dir} --validate-only")
    return 0


if __name__ == "__main__":
    sys.exit(main())
