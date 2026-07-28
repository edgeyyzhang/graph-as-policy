---
name: tsh-route-arms-bimanual
description: >
  Decide the transport route: which arm grasps, and whether a bimanual handover
  is needed. Probes reachability with the canonical curobo bundle (plans only,
  execute=False) using the grasp poses tsh-pickup will execute — consumed from a
  tsh-calculate-grasp-ring instance per arm — and the place hover's yaw sweep
  with the predicted held-tape offset. Exits ``direct`` (one arm grasps AND
  places) or ``needs_handover`` (different arms), or raises when nothing is
  reachable. On ``needs_handover`` it additionally derives the handover station
  geometry (meet point + giver/receiver presentation orientations) from the two
  arm-base poses — station geometry is only meaningful when an exchange is
  actually going to happen, so it is wired as a conditional second node rather
  than a separate always-run skill. Runs before tsh-pickup (pickup needs its
  pick_arm), so wire each exit to its OWN tsh-pickup instance — direct into the
  place chain, needs_handover into the present/exchange chain. Only one runs.
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
    needs_handover: Pick and place need different arms — station geometry derived, run the exchange.
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
      Consume the grasp legs from one tsh-calculate-grasp-ring instance per arm
      (the arm0_* and arm1_* inputs) — never re-derive grasp poses here. Route
      replays those exact poses, so its feasibility answer matches the grasp
      pickup runs, and relays the CHOSEN arm's legs as the pick_* outputs so
      tsh-pickup executes them (no separate grasp instance on the pickup side).
    - >
      execute=False everywhere — this subgraph plans, it never moves.
    - >
      meet_xyz / giver_quat / recv_quat are ONLY bound on the needs_handover
      exit — the derive_station_geometry node does not run on the direct
      route. Never wire them as a required input on a node that also runs on
      the direct exit; only tsh-transport-held's present instance and
      tsh-handover consume them, and both are only reachable via
      needs_handover.
    - >
      The station-geometry node also emits hold_target_xyz/hold_target_quat
      (aliases of meet_xyz/giver_quat) — the canonical destination pair
      tsh-transport-held requires, which tsh-place-pose emits too, so the
      PRESENT leg auto-wires by exact name exactly like the place legs. Never
      alias these to a bare target_xyz/target_quat: that generic name collides
      with the perception role prefix (tsh-perceive-* also produces target_xyz,
      for the picked object), and under the latest-producer rule the giver would
      silently carry the tape back to its pickup spot instead of the meet point.
    - >
      Wire the two exits to SEPARATE tsh-pickup instances — direct to the one
      feeding the place chain, needs_handover to the one feeding the
      present/exchange chain. Do NOT collapse both exits into a single shared
      pickup: pickup exits only grasped/failed, so the route decision would be
      erased at the join and the graph could no longer branch on it. Only one
      pickup ever executes, exactly as only one place chain does.
  canonical_scripts:
    - route: scripts/route.py
    - station_geometry: scripts/station_geometry.py
  streaming: false
---

# tsh-route-arms-bimanual

The "reachability probe first" practice promoted into a graph node. A bimanual
handover exists only because the destination is outside the picking arm's
workspace — so whether to run one is a per-scene routing decision, probed in
milliseconds of GPU planning instead of discovered minutes into a sim rollout.
Because the station geometry (meet point, presentation orientations) is only
meaningful when that decision comes back `needs_handover`, deriving it is
wired as a second node reached only down that branch, not a separate skill
that runs unconditionally regardless of whether an exchange happens.

For each arm it plans (without executing) the pickup's hover + seat poses (from
that arm's `tsh-calculate-grasp-ring` instance) and the place hover's yaw sweep
with the predicted held-tape offset, then routes `direct` (one arm does both) or
`needs_handover` (pick and place need different arms). On `needs_handover` it
also derives the meet point + giver/receiver presentation quats from the two
arm-base poses (self-configuring for a restationed pair, no tuned rendezvous
constants).

## When to use

- Between perception and the grasp, on any bimanual station where the
  destination may or may not be in the picking arm's workspace.

## When NOT to use

- Single-arm platforms (there is no routing decision).
- When the task PRESCRIBES a handover regardless of reach — skip this skill
  (wire a `tsh-station-geometry`-equivalent step directly, or bind the meet
  pose as a literal).

## Recommended subgraph state flow

```text
route ──(route=="direct")─────────────────────▶ direct → END
      └─(route=="needs_handover")─▶ derive_station_geometry → needs_handover → END
```

1. **`route`** — `type: script`, file `scripts/<sg>/route.py`. Inputs: the scene
   values (`target_xyz`, `container_xyz`, `target_half_z`, `rim_radius`) and the
   per-arm grasp legs (`arm0_*` / `arm1_*`). Returns `{route, pick_arm,
   place_arm, giver_arm, receiver_arm, pick_hover_xyz, pick_seat_xyz,
   pick_lift_xyz, pick_grasp_quat}` — the last four are the CHOSEN arm's grasp
   legs, relayed for `tsh-pickup` to execute directly.
2. **`derive_station_geometry`** — `type: script`,
   file `scripts/<sg>/station_geometry.py`. Only reached on the
   `needs_handover` branch. Inputs: `giver_arm=Ref("route.giver_arm")`,
   `receiver_arm=Ref("route.receiver_arm")` (both already produced by `route`
   on every path). Returns `{meet_xyz, giver_quat, recv_quat}`.

Route on the script's `route` field to two exits: `direct` goes straight to
the `direct` noop; `handover` goes through `derive_station_geometry` first,
then the `needs_handover` noop. `on_error: "unreachable"`.

## Required end states

| End state | Meaning |
|---|---|
| `direct` | One arm grasps and places. |
| `needs_handover` | Pick and place need different arms; station geometry derived — run the exchange. |
| `unreachable` | No arm can grasp or place (raise → abort). |

## See also

- `tsh-calculate-grasp-ring` — supplies the per-arm grasp legs route probes.
- `tsh-pickup`, `tsh-handover`, `tsh-place` — consumers of the arm ids.
- `tsh-handover` — consumer of `giver_quat` / `recv_quat`.
- `tsh-transport-held` — its PRESENT instance consumes `meet_xyz` / `giver_quat` directly.
- `scripts/_station_geometry.py` — the station-geometry derivation.
