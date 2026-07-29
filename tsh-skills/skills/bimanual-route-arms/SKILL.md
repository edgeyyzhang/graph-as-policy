---
name: bimanual-route-arms
description: >
  Decide the transport route: which arm grasps, and whether a bimanual handover
  is needed. Probes reachability with the canonical curobo bundle (plans only,
  execute=False) using the grasp poses pickup will execute — consumed from a
  single calculate-grasp-ring node — and the place hover's yaw sweep
  with the predicted held-tape offset. Exits ``direct`` (one arm grasps AND
  places) or ``needs_handover`` (different arms), or raises when nothing is
  reachable. It also owns the handover station geometry (meet point +
  giver/receiver presentation orientations, derived from the two arm-base
  poses), which the exchange chain consumes. Runs before pickup (pickup
  needs its pick_arm); dispatch-route acts on the ``route`` decision after
  the grasp, where the two chains actually diverge.
compatibility: requires gap>=0.1
metadata:
  category: planning
  tags: [tsh, routing, reachability, bimanual, curobo, yam, self-model]
gap:
  allowed_tools:
    - libero-yam.arm_base_pose
    - curobo.plan_to_pose
  exit_conditions:
    direct: One arm can grasp the target and reach the destination — no exchange.
    needs_handover: Pick and place need different arms — run the exchange.
    unreachable: No arm can grasp, or nothing can place (raise routes to on_error).
  required_inputs:
    target_xyz: Vec3
    container_xyz: Vec3
    target_half_z: float
    rim_radius: float
    arm0_hover_xyz: Vec3
    arm0_seat_xyz: Vec3
    arm0_lift_xyz: Vec3
    arm0_grasp_quat: Quaternion
    arm0_held_offset: Vec3
    arm1_hover_xyz: Vec3
    arm1_seat_xyz: Vec3
    arm1_lift_xyz: Vec3
    arm1_grasp_quat: Quaternion
    arm1_held_offset: Vec3
  produces_outputs:
    route: str
    pick_arm: int
    place_arm: int
    giver_arm: int
    receiver_arm: int
    pick_hover_xyz: Vec3
    pick_seat_xyz: Vec3
    pick_lift_xyz: Vec3
    pick_grasp_quat: Quaternion
    meet_xyz: Vec3
    giver_quat: Quaternion
    recv_quat: Quaternion
    hold_target_xyz: Vec3
    hold_target_quat: Quaternion
  hard_rules:
    - >
      Consume the grasp legs from the single calculate-grasp-ring node
      (its arm0_* and arm1_* outputs) — never re-derive grasp poses here. Route
      replays those exact poses, so its feasibility answer matches the grasp
      pickup runs, and relays the CHOSEN arm's legs as the pick_* outputs so
      pickup executes them (no separate grasp instance on the pickup side).
    - >
      execute=False everywhere — this subgraph plans, it never moves.
    - >
      station_geometry runs UNCONDITIONALLY — never gate it behind the
      needs_handover exit, and never instantiate it per branch. It is ~0ms of
      arithmetic on two arm-base poses, and running it on every path is what
      keeps meet_xyz / giver_quat / recv_quat / hold_target_* BOUND on every
      path. Gating it validates clean (the names have declared producers) and
      then dies at runtime with "input 'hold_target_xyz' has no upstream
      producer" — bound-ness is a runtime property W8 cannot check.
    - >
      The subgraph branches on station_geometry's relayed `route` field
      (router_field="route"), NOT on route.py's exit — that is what lets
      station_geometry sit on every path while the subgraph still exposes two
      exits. Keep the node chain linear: START -> route -> station_geometry.
    - >
      The station-geometry node also emits hold_target_xyz/hold_target_quat
      (aliases of meet_xyz/giver_quat) — the canonical destination pair
      transport-held-with-object requires, which calculate-place-pose emits too, so the
      PRESENT leg auto-wires by exact name exactly like the place legs. Never
      alias these to a bare target_xyz/target_quat: that generic name collides
      with the perception role prefix (perceive-tape-* also produces target_xyz,
      for the picked object), and under the latest-producer rule the giver would
      silently carry the tape back to its pickup spot instead of the meet point.
    - >
      The route output only takes effect if a downstream node branches on it
      after the grasp; on its own this skill decides but does not enforce.
  canonical_scripts:
    - route: scripts/route.py
    - station_geometry: scripts/station_geometry.py
  streaming: false
---

# bimanual-route-arms

The "reachability probe first" practice promoted into a graph node. A bimanual
handover exists only because the destination is outside the picking arm's
workspace — so whether to run one is a per-scene routing decision, probed in
milliseconds of GPU planning instead of discovered minutes into a sim rollout.
The station geometry (meet point, presentation orientations) lives in this
skill rather than a separate one because it is derived from the same
arm-base geometry the routing decision already reads.

For each arm it plans (without executing) the pickup's hover + seat poses (from
`calculate-grasp-ring`) and the place hover's yaw sweep
with the predicted held-tape offset, then routes `direct` (one arm does both) or
`needs_handover` (pick and place need different arms). It also derives the meet
point + giver/receiver presentation quats from the two arm-base poses
(self-configuring for a restationed pair, no tuned rendezvous constants).

## When to use

- Between perception and the grasp, on any bimanual station where the
  destination may or may not be in the picking arm's workspace.

## When NOT to use

- Single-arm platforms (there is no routing decision).
- When the task PRESCRIBES a handover regardless of reach — skip this skill
  (wire a `tsh-station-geometry`-equivalent step directly, or bind the meet
  pose as a literal).

## Recommended subgraph state flow

Strictly linear, then branch at the end. Both scripts run on every path.

```text
START → route → station_geometry ──(route=="direct")────────▶ direct → END
                                 └─(route=="needs_handover")▶ needs_handover → END
```

1. **`route`** — `type: script`, file `scripts/<sg>/route.py`. Inputs: the scene
   values (`target_xyz`, `container_xyz`, `target_half_z`, `rim_radius`) and the
   per-arm grasp legs (`arm0_*` / `arm1_*`). Returns `{route, pick_arm,
   place_arm, giver_arm, receiver_arm, pick_hover_xyz, pick_seat_xyz,
   pick_lift_xyz, pick_grasp_quat}` — the last four are the CHOSEN arm's grasp
   legs, relayed for `pickup` to execute directly.
2. **`station_geometry`** — `type: script`, file
   `scripts/<sg>/station_geometry.py`. Inputs: `route=Ref("route.route")`,
   `giver_arm=Ref("route.giver_arm")`, `receiver_arm=Ref("route.receiver_arm")`.
   Returns `{route, meet_xyz, giver_quat, recv_quat, hold_target_xyz,
   hold_target_quat}`.

Then `add_conditional_edges("station_geometry", {"direct": "direct",
"needs_handover": "needs_handover"}, router_field="route")`, two noop exits, and
`set_on_error("unreachable")`. Bind every output above in `set_outputs`.

## Required end states

| End state | Meaning |
|---|---|
| `direct` | One arm grasps and places. |
| `needs_handover` | Pick and place need different arms — run the exchange. |
| `unreachable` | No arm can grasp or place (raise → abort). |

## See also

- `calculate-grasp-ring` — one node supplying both arms' grasp legs.
- `pickup`, `bimanual-handover`, `place` — consumers of the arm ids.
- `bimanual-handover` — consumer of `giver_quat` / `recv_quat`.
- `transport-held-with-object` — its PRESENT instance consumes `meet_xyz` / `giver_quat` directly.
- `scripts/_station_geometry.py` — the station-geometry derivation.
