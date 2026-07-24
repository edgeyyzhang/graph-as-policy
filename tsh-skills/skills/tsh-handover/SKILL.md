---
name: tsh-handover
description: >
  Bimanual insertion handover: the RECEIVER leg only — the giver is assumed to
  already be presenting the tape "o" face-on at the meet point (a preceding
  tsh-transport-held instance carries it there, composing tape_in_giver into
  its tool_offset; fed by tsh-station-geometry's target_xyz/target_quat
  aliases). This node sweeps a rim clock angle, threads a finger through the
  same hole and closes, then the giver releases and retracts clear. BEFORE the
  giver releases it measures the tape centre in the receiver TCP frame —
  receiver_offset — so place can track the held tape by FK with no ground
  truth. Presentation orientations come from tsh-station-geometry. Use for the
  giver→receiver exchange of a perceived tape ring on LIBERO-YAM, always
  preceded by station_geometry → transport_held (present) in the same chain.
  giver_arm/receiver_arm come from tsh-route; when the task pins them, bind
  literal ints inside the subgraph and omit those inputs.
compatibility: requires gap>=0.1
metadata:
  category: motion
  tags: [tsh, handover, bimanual, insertion, curobo, yam]
gap:
  allowed_tools:
    - libero-yam.arm_base_pose
    - libero-yam.execute_trajectory
    - curobo.plan_to_pose
    - curobo.plan_directed_linear
    - robot.open_gripper
    - robot.close_gripper
    - robot.get_ee_pose
  exit_conditions:
    handed_over: Receiver holds the tape; receiver_offset bound in the outputs.
    failed: No reachable receiver clock angle, or an exchange leg had no plan (raise routes to abort).
  required_inputs:
    tape_in_giver: Vec3            # from tsh-pickup (the giver's tool offset; a.k.a. held_offset)
    rim_radius: float              # from tsh-pickup (relayed perceived rim radius); derives the
                                    # receiver's thread offset (stable run-to-run, unlike hole_radius)
    giver_quat: Quaternion        # from tsh-station-geometry; REQUIRED, no tuned fallback, wxyz.
                                    # Used for the giver's RETRACT move at the end — the present
                                    # leg itself already happened in the preceding transport_held.
    recv_quat: Quaternion         # from tsh-station-geometry; REQUIRED, no tuned fallback, wxyz canonical
    giver_arm: int                 # giving arm from tsh-route (0=left, 1=right). Task pins the
                                    # arms? Pass LITERALS in the node inputs and OMIT these two
                                    # subgraph inputs.
    receiver_arm: int              # receiving arm from tsh-route. Default 1.
    # NOT an input: meet_xyz. This skill no longer does its own present leg —
    # that's ALWAYS a preceding tsh-transport-held instance's job now (fed by
    # tsh-station-geometry's target_xyz/target_quat aliases). Compose that
    # chain ahead of this node; never wire meet_xyz here directly.
  produces_outputs:
    handed_over: bool
    receiver_offset: Vec3          # tape centre in the receiver TCP frame (m), measured before release
    held_offset: Vec3              # alias of receiver_offset — REBINDS the cross-subgraph held_offset
                                    # name to the receiver's grip, so a downstream place tracks the
                                    # latest holder either route (latest-producer binding)
    giver_tcp: Se3Pose             # world giver TCP at the present
    receiver_tcp: Se3Pose          # world receiver TCP at the grab
    giver_arm: int                 # echo of the giving arm (checkpoint anchor)
    receiver_arm: int              # echo of the receiving arm (the holder after this subgraph)
    place_arm: int                 # alias of receiver_arm — REBINDS the cross-subgraph place_arm
                                    # name (tsh-route's, on the direct route) to the receiver, so
                                    # place_pose/transport_held/place auto-wire arm_id to whoever
                                    # holds the tape now BY EXACT NAME — never reference this
                                    # subgraph directly by name from elsewhere (e.g.
                                    # Ref("handover.receiver_arm") from another subgraph's node) —
                                    # that is not valid; only Ref("in.<name>") cross-subgraph
                                    # wiring is supported. Declare place_arm as an input on the
                                    # consuming subgraph and let it auto-wire.
  hard_rules:
    - >
      `recv_quat` is CANONICAL (as if the receiver sits on -Y); `_rim_grasp`
      mirrors it for a +Y receiver. Pass the canonical quat, not a pre-mirrored one.
    - >
      Measure `receiver_offset` BEFORE the giver releases (grip still rigid),
      anchored to the receiver's designed rim seat — not to the over-inserted
      pose — so place tracks the settled tape, not the transient dual-pinch.
  canonical_scripts:
    - bimanual_exchange: scripts/bimanual_exchange.py
  streaming: false
---

# tsh-handover

This is the RECEIVER half of the exchange only. The giver's present move — the
tape (`tool_offset = tape_in_giver`) planned to `meet_xyz` at `giver_quat` — is
now always a preceding `tsh-transport-held` instance's job, fed by
`tsh-station-geometry`'s `target_xyz`/`target_quat` aliases (same trick
`tsh-place-pose` uses for the place-chain instance). See
`tsh-station-geometry`'s SKILL.md for the ordering requirement (that instance
must run after `tsh-pickup`, the last consumer of the perception `target_xyz`).

This node picks up from there: the receiver opens, sweeps a rim clock angle
for a reachable thread (the ring's symmetry is the reach margin), slides in
behind the hole and inserts purely along +X, then closes. Before the giver
lets go, the tape centre is expressed in the receiver TCP frame
(`receiver_offset`) so place can plan the held tape by FK. Finally the giver
retracts perpendicular to the tape axis (using `giver_quat` — still required
here even though the present leg itself already happened upstream) and the
receiver backs off on its own side — de-conflicted by geometry.

`giver_quat` / `recv_quat` are REQUIRED, supplied base-geometry-derived by
`tsh-station-geometry` (no tuned fallback).

## When to use

- On the `needs_handover` route (from `tsh-route`), after `tsh-pickup` →
  `tsh-station-geometry` → a `tsh-transport-held` present instance, to
  transfer the tape to the receiver arm — feeding `held_offset` into
  `tsh-place` (via `tsh-place-pose`, which must run AFTER this node).
- The direct route (one arm reaches both tape and destination) SKIPS this
  subgraph entirely: `dispatch` routes the verified grasp straight to `place`.

## When NOT to use

- The `direct` route — no exchange is needed; do not force a handover.
- Single-arm regrasps, or learned/policy exchanges.
- Non-ring objects (the capture threads a hole).
- Standalone, without a preceding `tsh-station-geometry` → `tsh-transport-held`
  present chain — this node has no way to get the tape to the meet point on
  its own anymore.

## Recommended subgraph state flow

1 state, always preceded in the SAME top-level chain by
`station_geometry_sg → present_sg (tsh-transport-held)`:

```text
station_geometry ─▶ present (tsh-transport-held) ─▶ bimanual_exchange
```

1. **`bimanual_exchange`** — `type: script`, `scripts/<sg>/bimanual_exchange.py`.
   Inputs: `tape_in_giver=Ref("in.tape_in_giver")`, `rim_radius=Ref("in.rim_radius")`,
   `giver_quat=Ref("in.giver_quat")`, `recv_quat=Ref("in.recv_quat")`,
   `giver_arm=Ref("in.giver_arm")`, `receiver_arm=Ref("in.receiver_arm")`
   (or literal ints with the arm subgraph inputs omitted, when the task pins
   them). Do NOT wire `meet_xyz` — it's no longer a script parameter.
   Returns `{handed_over, receiver_offset, held_offset, giver_tcp, receiver_tcp,
   giver_arm, receiver_arm, place_arm}`.

Bind the outputs (`held_offset` AND `place_arm` — the receiver's grip and arm —
feed `tsh-place-pose`/`tsh-place`/the place-chain `tsh-transport-held` by the
latest-producer cross-subgraph rule; declare `place_arm` as an input on those
subgraphs and let it auto-wire — do not reference `bimanual_exchange` or this
subgraph by name from anywhere else):

```python
sg.set_outputs(
    receiver_offset=Ref("bimanual_exchange.receiver_offset"),
    held_offset=Ref("bimanual_exchange.held_offset"),
    giver_tcp=Ref("bimanual_exchange.giver_tcp"),
    receiver_tcp=Ref("bimanual_exchange.receiver_tcp"),
    giver_arm=Ref("bimanual_exchange.giver_arm"),
    receiver_arm=Ref("bimanual_exchange.receiver_arm"),
    place_arm=Ref("bimanual_exchange.place_arm"),
)
```

## Required end states

| End state | Meaning |
|---|---|
| `handed_over` | Receiver holds the tape; route to the place subgraph. |
| `failed` | No reachable clock angle / plan failed; route to `abort`. |

## See also

- `tsh-route` — decides `direct` vs `needs_handover` and the giver/receiver arms.
- `tsh-transport-held` — ALWAYS composed immediately ahead of this node now,
  as the required PRESENT leg (fed by `tsh-station-geometry`'s `target_xyz`/
  `target_quat` aliases); also reused by `tsh-place-pose`/`tsh-place` for the
  collision-aware place-hover approach.
- `tsh-pickup` — supplies `tape_in_giver` (a.k.a. `held_offset`) + `rim_radius`.
- `tsh-station-geometry` — supplies `giver_quat` / `recv_quat`, plus the
  `target_xyz`/`target_quat` aliases the present-leg `tsh-transport-held`
  instance consumes.
- `tsh-place` — consumes `held_offset` (the receiver's grip after the exchange).
