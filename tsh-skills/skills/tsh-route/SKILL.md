---
name: tsh-route
description: >
  Decide the transport route: which arm grasps, and whether a bimanual handover
  is needed. Probes reachability with the canonical curobo bundle (plans only,
  execute=False) using the grasp poses tsh-pickup will execute — consumed from a
  tsh-calculate-grasp-ring instance per arm — and the place hover's yaw sweep
  with the predicted held-tape offset. Exits ``direct`` (one arm grasps AND
  places) or ``needs_handover`` (different arms), or raises when nothing is
  reachable. Runs before tsh-pickup (pickup needs its pick_arm); a downstream
  dispatch acts on the ``route`` decision after the grasp.
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
  hard_rules:
    - >
      Consume the grasp legs from one tsh-calculate-grasp-ring instance per arm
      (the arm0_* and arm1_* inputs) — never re-derive grasp poses here. Route
      replays those exact poses, so its feasibility answer matches the grasp
      pickup runs, and relays the CHOSEN arm's legs as the pick_* outputs so
      tsh-pickup executes them (no separate grasp instance on the pickup side).
    - >
      execute=False everywhere — this subgraph plans, it never moves.
    - >
      The route output only takes effect if a downstream node branches on it
      after the grasp; on its own this skill decides but does not enforce.
  canonical_scripts:
    - route: scripts/route.py
  streaming: false
---

# tsh-route

The "reachability probe first" practice promoted into a graph node. A bimanual
handover exists only because the destination is outside the picking arm's
workspace — so whether to run one is a per-scene routing decision, probed in
milliseconds of GPU planning instead of discovered minutes into a sim rollout.

For each arm it plans (without executing) the pickup's hover + seat poses (from
that arm's `tsh-calculate-grasp-ring` instance) and the place hover's yaw sweep
with the predicted held-tape offset, then routes `direct` (one arm does both) or
`needs_handover` (pick and place need different arms).

## When to use

- Between perception and the grasp, on any bimanual station where the
  destination may or may not be in the picking arm's workspace.

## When NOT to use

- Single-arm platforms (there is no routing decision).
- When the task PRESCRIBES a handover regardless of reach — skip this skill.

## Recommended subgraph state flow

```text
route ──(route=="direct")────────▶ direct → END
      └─(route=="needs_handover")▶ needs_handover → END
```

1. **`route`** — `type: script`, file `scripts/<sg>/route.py`. Inputs: the scene
   values (`target_xyz`, `container_xyz`, `target_half_z`, `rim_radius`) and the
   per-arm grasp legs (`arm0_*` / `arm1_*`). Returns `{route, pick_arm,
   place_arm, giver_arm, receiver_arm, pick_hover_xyz, pick_seat_xyz,
   pick_lift_xyz, pick_grasp_quat}` — the last four are the CHOSEN arm's grasp
   legs, relayed for `tsh-pickup` to execute directly.

Route on the script's `route` field to two noop exits (`direct`,
`needs_handover`); `on_error: "unreachable"`.

## Required end states

| End state | Meaning |
|---|---|
| `direct` | One arm grasps and places. |
| `needs_handover` | Pick and place need different arms — run the exchange. |
| `unreachable` | No arm can grasp or place (raise → abort). |

## See also

- `tsh-calculate-grasp-ring` — supplies the per-arm grasp legs route probes.
- `tsh-pickup`, `tsh-handover`, `tsh-place` — consumers of the arm ids.
