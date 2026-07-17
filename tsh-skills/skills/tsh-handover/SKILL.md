---
name: tsh-handover
description: >
  Bimanual insertion handover: the giver presents the tape "o" face-on at the
  meet point by planning the HELD TAPE as the end effector (tape_in_giver
  composed into the tcp_offset), the receiver sweeps a rim clock angle, threads
  its finger through the same hole and closes, then the giver releases and
  retracts clear. BEFORE the giver releases it measures the tape centre in the
  receiver TCP frame — receiver_offset — so place can track the held tape by FK
  with no ground truth. Meet point and presentation orientations come from
  tsh-station-geometry (or the tuned constants). Use for the giver→receiver
  exchange of a perceived tape ring on LIBERO-YAM.
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
    meet_xyz: Vec3                 # from tsh-station-geometry; REQUIRED, no tuned fallback
    giver_quat: Quaternion        # from tsh-station-geometry; REQUIRED, no tuned fallback, wxyz
    recv_quat: Quaternion         # from tsh-station-geometry; REQUIRED, no tuned fallback, wxyz canonical
    giver_arm: int                 # route-decided giving arm (0=left, 1=right). Default 0.
    receiver_arm: int              # route-decided receiving arm. Default 1.
    skip_present: bool             # True when a preceding tsh-transport-held (present) node already
                                    # carried the tape to meet_xyz; the exchange then starts at the
                                    # receiver approach. Default False (exchange presents it itself).
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

The giver presents the tape as the end effector (`tool_offset = tape_in_giver`),
so cuRobo plans the tape *centre* to `meet_xyz` directly at `giver_quat` —
repeatable regardless of how the ring settled on the finger. The receiver opens,
sweeps a rim clock angle for a reachable thread (the ring's symmetry is the reach
margin), slides in behind the hole and inserts purely along +X, then closes.
Before the giver lets go, the tape centre is expressed in the receiver TCP frame
(`receiver_offset`) so place can plan the held tape by FK. Finally the giver
retracts perpendicular to the tape axis and the receiver backs off on its own
side — de-conflicted by geometry.

`meet_xyz` / `giver_quat` / `recv_quat` are REQUIRED, supplied
base-geometry-derived by `tsh-station-geometry` (no tuned fallback); a
`GAP_HANDOVER_XYZ` env override is taken verbatim over the wired `meet_xyz`.

## When to use

- On the `needs_handover` route (from `tsh-route`), after `tsh-pickup` and a
  `tsh-transport-held` (`present`) node has carried the tape to `meet_xyz`, to
  transfer it to the receiver arm — feeding `held_offset` into `tsh-place`.
- The direct route (one arm reaches both tape and destination) SKIPS this
  subgraph entirely: `dispatch` routes the verified grasp straight to `place`.

## When NOT to use

- The `direct` route — no exchange is needed; do not force a handover.
- Single-arm regrasps, or learned/policy exchanges.
- Non-ring objects (the capture threads a hole).

## Recommended subgraph state flow

1 state:

```text
bimanual_exchange
```

1. **`bimanual_exchange`** — `type: script`, `scripts/<sg>/bimanual_exchange.py`.
   Inputs: `tape_in_giver=Ref("in.tape_in_giver")`, `rim_radius=Ref("in.rim_radius")`,
   `meet_xyz=Ref("in.meet_xyz")`, `giver_quat=Ref("in.giver_quat")`,
   `recv_quat=Ref("in.recv_quat")`, `giver_arm=Ref("in.giver_arm")`,
   `receiver_arm=Ref("in.receiver_arm")`, and `skip_present=True` when a
   `present` node already carried the tape to `meet_xyz`.
   Returns `{handed_over, receiver_offset, held_offset, giver_tcp, receiver_tcp,
   giver_arm, receiver_arm}`.

Bind the outputs (`held_offset` — the receiver's grip — feeds `tsh-place` by the
latest-producer cross-subgraph rule):

```python
sg.set_outputs(
    receiver_offset=Ref("bimanual_exchange.receiver_offset"),
    held_offset=Ref("bimanual_exchange.held_offset"),
    giver_tcp=Ref("bimanual_exchange.giver_tcp"),
    receiver_tcp=Ref("bimanual_exchange.receiver_tcp"),
    giver_arm=Ref("bimanual_exchange.giver_arm"),
    receiver_arm=Ref("bimanual_exchange.receiver_arm"),
)
```

## Required end states

| End state | Meaning |
|---|---|
| `handed_over` | Receiver holds the tape; route to the place subgraph. |
| `failed` | No reachable clock angle / plan failed; route to `abort`. |

## See also

- `tsh-route` — decides `direct` vs `needs_handover` and the giver/receiver arms.
- `tsh-transport-held` (`present`) — carries the tape to `meet_xyz` first; set
  `skip_present=True` so the exchange starts at the receiver approach.
- `tsh-pickup` — supplies `tape_in_giver` (a.k.a. `held_offset`) + `rim_radius`.
- `tsh-station-geometry` — supplies `meet_xyz` / `giver_quat` / `recv_quat`.
- `tsh-place` — consumes `held_offset` (the receiver's grip after the exchange).
