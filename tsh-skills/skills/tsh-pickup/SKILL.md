---
name: tsh-pickup
description: >
  Ring-grasp the tape from above with the giver arm — near finger into the hole,
  far finger onto the outer rim wall. The three grasp legs (hover, seat, lift)
  come ready-to-plan from tsh-calculate-grasp-ring; pickup executes them, closes
  the jaws gently, and lifts, every leg planned by the canonical curobo bundle
  and streamed onto the sim. Before closing it anchors the tape centre in the
  giver TCP frame and emits it as the rigid tape_in_giver offset the exchange
  tracks by FK. pick_arm comes from tsh-route; when the task pins the arm
  instead, bind arm_id literally and omit the input.
compatibility: requires gap>=0.1
metadata:
  category: grasping
  tags: [tsh, grasping, ring, bimanual, curobo, yam]
gap:
  allowed_tools:
    - libero-yam.arm_base_pose
    - libero-yam.execute_trajectory
    - curobo.plan_to_pose
    - robot.open_gripper
    - robot.close_gripper
    - robot.get_ee_pose
  exit_conditions:
    grasped: Tape ring held; giver_held_offset + grasp_tcp bound in the outputs.
    failed: A grasp leg had no cuRobo plan (raise routes to abort).
  required_inputs:
    target_xyz: Vec3
    pick_hover_xyz: Vec3
    pick_seat_xyz: Vec3
    pick_lift_xyz: Vec3
    pick_grasp_quat: Quaternion
    pick_arm: int
  produces_outputs:
    grasped: bool
    giver_held_offset: Vec3
    tape_in_giver: Vec3
    grasp_tcp: Se3Pose
    pick_arm: int
  hard_rules:
    - >
      ALWAYS begin the grasp with `robot.open_gripper` before the descent —
      a gripper left closed from a previous step silently fails the ring grasp.
    - >
      The grasp legs are named `pick_*` (not the bare `hover_xyz`/`seat_xyz`/...)
      because `tsh-place-pose` also produces a plain `hover_xyz` — the `pick_`
      prefix avoids colliding with it under the latest-producer cross-subgraph
      rule. `tsh-route` relays the chosen arm's legs under these exact names;
      without `tsh-route`, wire a `tsh-calculate-grasp-ring` instance and rename
      its outputs to match.
    - >
      Measure `tape_in_giver` at the seat, BEFORE `robot.close_gripper` — the
      rigid offset anchors on the perceived grasp point, not a shifted grip.
    - >
      This node deliberately does NOT produce a bare `held_offset` — that name
      is reserved for whichever holder is current at a given graph point.
      `tsh-dispatch-route` relays `giver_held_offset` forward as `held_offset`
      on the direct exit; `tsh-handover` provides it after an exchange. Do not
      wire the shared place chain to this node directly.
  canonical_scripts:
    - pickup: scripts/pickup.py
  streaming: false
---

# tsh-pickup

A ring grasp: the near finger drops into the hole and the far finger seats on the
outer rim, then the jaws close gently (ramped, to avoid kicking the light ring)
and lift. The three grasp legs come ready-to-plan from `tsh-calculate-grasp-ring`
— pickup only executes them. The grip is rigid, so the tape centre expressed in
the giver TCP frame — `tape_in_giver`, measured at the seat before the jaws close
— lets the exchange track and plan the held tape by forward kinematics with no
ground truth.

## When to use

- After `tsh-calculate-grasp-ring` and `tsh-route`, to acquire the tape before
  `tsh-handover`.

## When NOT to use

- Non-ring objects (this is a hole/rim straddle, not a generic top-down grasp —
  use `grasping-with-planner`).
- Learned/policy grasps.

## Recommended subgraph state flow

```text
pickup
```

1. **`pickup`** — `type: script`, file `scripts/<sg>/pickup.py`. Inputs:
   `tape_xyz=Ref("in.target_xyz")`, `hover_xyz=Ref("in.pick_hover_xyz")`,
   `seat_xyz=Ref("in.pick_seat_xyz")`, `lift_xyz=Ref("in.pick_lift_xyz")`,
   `grasp_quat=Ref("in.pick_grasp_quat")`, plus `arm_id=Ref("in.pick_arm")` (or a
   literal `arm_id=0` with `pick_arm` omitted when the task pins the arm).
   Returns `{grasped, giver_held_offset, tape_in_giver, grasp_tcp, pick_arm}`.

```python
sg.set_outputs(
    giver_held_offset=Ref("pickup.giver_held_offset"),
    tape_in_giver=Ref("pickup.tape_in_giver"),
    grasp_tcp=Ref("pickup.grasp_tcp"),
    pick_arm=Ref("pickup.pick_arm"),
)
```

## Required end states

| End state | Meaning |
|---|---|
| `grasped` | Tape held; route to the handover / place chain. |
| `failed` | Grasp plan failed; route to `abort`. |

## See also

- `tsh-route` — relays the chosen arm's grasp legs as `pick_*`.
- `tsh-calculate-grasp-ring` — the underlying producer of the grasp legs (via `tsh-route`, or directly when there is no route).
- `tsh-perceive-cv` — supplies `target_xyz` (rebound to `tape_xyz`).
- `tsh-dispatch-route` — relays `giver_held_offset` forward as the shared `held_offset`.
- `tsh-handover` — consumes `tape_in_giver` directly (the giver's grip, unambiguous regardless of route).
