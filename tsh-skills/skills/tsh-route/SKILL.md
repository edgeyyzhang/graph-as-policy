---
name: tsh-route
description: >
  Decide the transport route: which arm grasps, and whether a bimanual
  handover is needed. Probes reachability with the canonical curobo bundle
  (plans only, execute=False) using the EXACT ring-grasp poses tsh-pickup
  will execute and the place hover's yaw sweep with the predicted held-tape
  offset, per arm. Exits ``direct`` (one arm grasps AND places),
  ``needs_handover`` (different arms), or raises. THIS SKILL ONLY DECIDES —
  it does not enforce, and it must run BEFORE tsh-pickup (pickup needs its
  pick_arm). To ACT on the decision, always add the companion skill
  ``tsh-dispatch-route`` immediately after tsh-pickup: it re-reads ``route``
  and branches ``direct`` to the place chain vs ``needs_handover`` to the
  exchange chain. Pair them always — without tsh-dispatch-route the route is
  inert and the graph is hardwired to one strategy regardless of the scene.
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
      REQUIRES a dispatch node at the TOP LEVEL (after pickup) that routes this
      exit straight to the place chain, skipping station_geometry/transport_held/
      handover entirely — this skill only DECIDES the route, it does not enforce
      it. If nothing branches on `route`, the handover runs unconditionally
      regardless of this answer (see hard_rules).
    needs_handover: Pick and place need different arms — run the exchange.
    unreachable: No arm can grasp, or nothing can place (raise routes to on_error).
  required_inputs:
    target_xyz: Vec3          # tape grasp point, from the target perceive subgraph
    container_xyz: Vec3            # destination top-face centre, from the dest perceive subgraph
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
    - >
      This skill's ``route`` output is INERT unless something consumes it. The
      TOP-LEVEL graph MUST add a dispatch node/conditional edge immediately
      after ``tsh-pickup`` that reads ``route`` and sends ``direct`` straight
      to the place chain (``place_pose`` → ``transport_held`` → ``place``),
      bypassing ``tsh-station-geometry`` / the present ``tsh-transport-held``
      instance / ``tsh-handover`` entirely — see "Wiring the route decision"
      below. Do NOT let ``pickup``'s success exit flow unconditionally into
      the handover chain: a real run confirmed this silently forces a
      handover even when ``route == "direct"``, handing the tape to an arm
      whose place-reachability was only ESTIMATED (a synthetic receiver
      offset, not the measured one) — in that run the estimate was wrong and
      the place step failed outright, on a scene route itself had correctly
      judged solvable by a single arm.
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
| `direct` | pick_arm == place_arm | pickup → **dispatch** → place_pose (station_geometry/handover SKIPPED) |
| `needs_handover` | different arms | pickup → **dispatch** → station_geometry → present → handover → place_pose |
| raise (`unreachable`) | dead zone | on_error → abort / re-station |

The **dispatch** step is not optional decoration — see "Wiring the route
decision at the top level" below for why, and the required node.

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

## Wiring the route decision at the top level

`route`'s answer only matters if the graph branches on it — and because this
skill runs BEFORE `tsh-pickup` (pickup needs `pick_arm`) while the branch
belongs AFTER it, the decision has to be carried across the grasp. That is
exactly what the companion skill **`tsh-dispatch-route`** does. Always compose
the pair:

```text
route → pickup → dispatch_route ─┬─(direct)─────────▶ place_pose ─▶ transport_held ─▶ place
                                 └─(needs_handover)─▶ station_geometry ─▶ present ─▶ handover ─▶ place_pose ─▶ …
```

```python
wf.add_conditional_edges("pickup", {"grasped": "dispatch_route", "failed": "abort"})
wf.add_conditional_edges("dispatch_route", {
    "direct":         "place_pose",
    "needs_handover": "station_geometry",
    "failed":         "abort",
})
```

Do NOT author a bespoke router node for this — `tsh-dispatch-route` is a
registered skill; select it from the catalog like any other. Both branches
converge on ONE shared `place_pose → transport_held → place` chain (never two
parallel copies): `held_offset`/`place_arm` resolve to whichever holder ran
last — `tsh-pickup`'s on the direct path, `tsh-handover`'s aliases on the
routed path — under the standard latest-producer cross-subgraph binding.

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
- `examples/tape_handover_yam/scripts/dispatch_route.py` — a working reference
  implementation of the top-level dispatch node (see "Wiring the route
  decision" above); the hand-built example graph wires it this way today.
- `tsh-station-geometry`, `tsh-place-pose` — see their own SKILL.md's ordering
  requirements; both assume they run downstream of the dispatch decision.
