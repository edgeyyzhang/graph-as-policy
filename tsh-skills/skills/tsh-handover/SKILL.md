---
name: tsh-handover
description: >
  Bimanual insertion handover: the RECEIVER leg only. The giver is assumed to
  already be presenting the tape face-on at the meet point (a preceding
  tsh-transport-held instance carries it there). This node sweeps a rim clock
  angle, threads a finger through the hole and closes, measures the tape centre
  in the receiver TCP frame (receiver_offset) before the giver releases, then the
  giver releases and retracts clear. Presentation orientations come from
  tsh-route-arms-bimanual's station-geometry node. giver_arm/receiver_arm come
  from tsh-route-arms-bimanual; when the task pins them, bind literal ints and
  omit the inputs.
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
    tape_in_giver: Vec3
    rim_radius: float
    giver_quat: Quaternion
    recv_quat: Quaternion
    giver_arm: int
    receiver_arm: int
  produces_outputs:
    handed_over: bool
    receiver_offset: Vec3
    held_offset: Vec3
    giver_tcp: Se3Pose
    receiver_tcp: Se3Pose
    giver_arm: int
    receiver_arm: int
    place_arm: int
    holding_arm: int
  hard_rules:
    - >
      held_offset and holding_arm are ONE datum and are always emitted together:
      held_offset is the tape centre in the HOLDER's TCP frame, so it is
      meaningless without the arm whose frame it is in. This node is where the
      holder changes, so it re-anchors BOTH from the giver to the receiver.
      A downstream node that takes the offset but selects its own arm plans the
      held tape onto the wrong gripper.
    - >
      recv_quat is CANONICAL (as if the receiver sits on -Y); the script mirrors
      it for a +Y receiver. Pass the canonical quat, not a pre-mirrored one.
    - >
      Measure receiver_offset BEFORE the giver releases (grip still rigid),
      anchored to the receiver's designed rim seat — so place tracks the settled
      tape, not the transient dual-pinch.
    - >
      This skill has no present leg — it needs a preceding tsh-station-geometry →
      tsh-transport-held (present) chain to bring the tape to the meet point.
      held_offset and place_arm alias the receiver's grip/arm so the place chain
      auto-wires to whoever holds the tape after the exchange.
  canonical_scripts:
    - bimanual_exchange: scripts/bimanual_exchange.py
  streaming: false
---

# tsh-handover

The RECEIVER half of the exchange. The giver's present move is a preceding
`tsh-transport-held` instance's job (fed by `tsh-station-geometry`). This node
picks up from there: the receiver opens, sweeps a rim clock angle for a reachable
thread (the ring's symmetry is the reach margin), inserts along +X, and closes.
Before the giver lets go, the tape centre is expressed in the receiver TCP frame
(`receiver_offset`) so place can plan the held tape by FK. Finally the giver
retracts and the receiver backs off on its own side.

## When to use

- On the `needs_handover` route, after `tsh-pickup` → `tsh-station-geometry` → a
  `tsh-transport-held` present instance, to transfer the tape to the receiver.

## When NOT to use

- The `direct` route — no exchange is needed.
- Single-arm regrasps, learned/policy exchanges, or non-ring objects.
- Standalone, without a preceding present chain — it can't reach the meet point.

## Recommended subgraph state flow

```text
bimanual_exchange
```

1. **`bimanual_exchange`** — `type: script`, `scripts/<sg>/bimanual_exchange.py`.
   Inputs: `tape_in_giver=Ref("in.tape_in_giver")`, `rim_radius=Ref("in.rim_radius")`,
   `giver_quat=Ref("in.giver_quat")`, `recv_quat=Ref("in.recv_quat")`,
   `giver_arm=Ref("in.giver_arm")`, `receiver_arm=Ref("in.receiver_arm")` (or
   literals). Returns `{handed_over, receiver_offset, held_offset, giver_tcp,
   receiver_tcp, giver_arm, receiver_arm, place_arm}`.

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
| `handed_over` | Receiver holds the tape; route to the place chain. |
| `failed` | No reachable clock angle / plan failed; route to `abort`. |

## See also

- `tsh-route-arms-bimanual` — decides `needs_handover` and the giver/receiver
  arms, and derives the station geometry (meet point + presentation quats)
  consumed here.
- `tsh-transport-held` — the required present chain (carries the tape to the meet point).
- `tsh-pickup` — supplies `tape_in_giver` + `rim_radius`.
- `tsh-place` — consumes `held_offset` (the receiver's grip) after the exchange.
