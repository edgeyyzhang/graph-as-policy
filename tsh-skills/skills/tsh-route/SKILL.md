---
name: tsh-route
description: >
  Decide the transport route for a pick-and-place: which arm grasps, and
  whether a bimanual handover is needed at all. Probes reachability with the
  canonical curobo bundle (plans only, execute=False, no sim time) using the
  EXACT ring-grasp poses tsh-pickup will execute and the place hover's yaw
  sweep with the predicted held-tape offset, per arm. Exits ``direct`` (one
  arm grasps AND places), ``needs_handover`` (one arm grasps, the other
  places), or raises when nothing reaches. Use between perception and the
  grasp on any bimanual station — a handover is a routed strategy, not a
  fixed stage.
compatibility: requires gap>=0.1
metadata:
  category: planning
  tags: [tsh, routing, reachability, bimanual, curobo, yam]
gap:
  allowed_tools:
    - libero-yam.arm_base_pose
    - curobo.plan_to_pose
  exit_conditions:
    direct: One arm can grasp the target and reach the destination — no exchange.
    needs_handover: Pick and place need different arms — run the exchange.
    unreachable: No arm can grasp, or nothing can place (raise routes to on_error).
  required_inputs:
    target_xyz: Vec3          # tape grasp point, from the target perceive subgraph
    dest_xyz: Vec3            # destination top-face centre, from the dest perceive subgraph
    target_half_z: float      # perceived tape half-thickness (place rest height)
    hole_radius: float        # from tsh-ring-geometry
    rim_radius: float         # from tsh-ring-geometry
    fingertip_axial: float    # from tsh-gripper-geometry
    finger_half_gap: float    # from tsh-gripper-geometry
  produces_outputs:
    route: str                # "direct" | "handover" (the internal routing field)
    pick_arm: int             # consumed by the pickup subgraph
    place_arm: int            # consumed by the place subgraph
    giver_arm: int            # aliases consumed by the handover subgraph
    receiver_arm: int
  hard_rules:
    - >
      Probe with the SHARED ``_ring.ring_grasp_poses`` — never re-derive
      grasp poses here; the probe's answer must be the pose pickup executes.
    - >
      ``execute=False`` everywhere — this subgraph plans, it never moves.
  canonical_scripts:
    - route: scripts/route.py
  streaming: false
---

# tsh-route

The "reachability probe first" practice, promoted from a debugging habit
into a graph node. A bimanual handover exists only because the destination
is outside the picking arm's workspace — so whether to run one is a
per-scene routing decision, probed in milliseconds of GPU planning instead
of discovered minutes into a sim rollout.

For each arm it plans (without executing) the pickup's hover + seat poses
(`_ring.ring_grasp_poses` — the same function pickup executes) and the place
hover's yaw sweep with the predicted held-tape offset, then routes:

| exit | meaning | downstream |
|---|---|---|
| `direct` | pick_arm == place_arm | pickup → verify → place |
| `needs_handover` | different arms | pickup → verify → handover → place |
| raise (`unreachable`) | dead zone | on_error → abort / re-station |

Porting note: for a sorting task this same node runs per item — tapes on the
placing side route `direct`, far-side tapes route `needs_handover`; the
graph topology does not change.

## When to use

- Between perception (+ ring geometry) and the grasp, on any bimanual
  station where the destination may or may not be in the picking arm's
  workspace.

## When NOT to use

- Single-arm platforms (there is no routing decision).
- When the task PRESCRIBES a handover regardless of reach (a demo of the
  exchange itself) — wire pickup → handover directly and skip this skill.

## Recommended subgraph state flow

Route on the script's `route` field to two noop exits:

```text
route ──(route=="direct")──▶ direct → END
      └─(route=="handover")▶ needs_handover → END
```

```python
sg.add_node("route", type="script", script="scripts/<sg>/route.py", inputs={...})
sg.add_exit("direct")
sg.add_exit("needs_handover")
sg.add_conditional_edges("route", {"direct": "direct", "handover": "needs_handover"},
                         router_field="route")
sg.add_edge("direct", END)
sg.add_edge("needs_handover", END)
sg.set_on_error("unreachable")
```

## Checkpoints

- validate=True: a route was produced with a coherent arm assignment
  (`pick_arm != place_arm` iff the exit was `needs_handover`).

## See also

- `tsh-ring-geometry/scripts/_ring.py` — the shared grasp-pose derivation.
- `tsh-pickup`, `tsh-handover`, `tsh-place` — the consumers of the arm ids.
