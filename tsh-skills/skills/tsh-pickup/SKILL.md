---
name: tsh-pickup
description: >
  Ring-grasp the tape from above with the giver arm — near finger into the hole,
  far finger onto the outer rim wall — using the perceived ring geometry and the
  model-derived gripper offsets. Hover, descend, gently close, and lift, every
  leg planned by the canonical curobo bundle and executed on the sim. Before
  closing it anchors the tape centre in the giver TCP frame and emits it as the
  rigid ``tape_in_giver`` offset the exchange tracks by FK. Use to acquire a
  perceived tape ring for a scripted bimanual handover on LIBERO-YAM.
  ``pick_arm`` comes from tsh-route; when the task pins the arm instead, bind
  the script's arm literally inside the subgraph and omit the input.
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
    grasped: Tape ring held; held_offset + grasp_tcp + rim_radius bound in the outputs.
    failed: A grasp leg had no cuRobo plan (raise routes to abort).
  required_inputs:
    # Names follow the upstream producers' output names exactly (tsh-perceive's
    # target_ prefix, tsh-route's pick_arm) so the coordinator can wire by name;
    # the subgraph rebinds them to the script's kwargs (see the state flow below).
    target_xyz: Vec3               # world grasp point — tsh-perceive's <name>_xyz (target_ prefix);
                                    # rebind to the script's tape_xyz kwarg
    hole_radius: float             # perceived inner radius (m); REQUIRED, no tuned fallback
    rim_radius: float              # perceived outer radius (m); REQUIRED, no tuned fallback
    fingertip_axial: float         # from tsh-gripper-geometry; REQUIRED, no tuned fallback
    finger_half_gap: float         # from tsh-gripper-geometry; REQUIRED, no tuned fallback
    pick_arm: int                  # picking arm from tsh-route (0=left, 1=right); rebind to the
                                    # script's arm_id kwarg. Task pins the arm? Pass arm_id a
                                    # LITERAL in the node inputs and OMIT this subgraph input.
  produces_outputs:
    grasped: bool
    held_offset: Vec3             # tape centre in the picking arm's TCP frame (m), rigid under
                                    # the grip; the shared transport/place skills track it by FK.
                                    # This is the canonical name; the handover rebinds it to the
                                    # receiver's measured offset (latest-producer cross-sg binding).
    tape_in_giver: Vec3           # alias of held_offset (legacy graphs / exchange input)
    grasp_tcp: Se3Pose             # world TCP pose at close (return-leg replay)
    rim_radius: float              # passthrough of the perceived rim radius used for this grasp;
                                    # relayed to tsh-handover (stable across runs, unlike hole_radius)
    pick_arm: int                 # echo of the arm that holds the tape (checkpoint anchor)
  hard_rules:
    - >
      ALWAYS begin the grasp with `robot.open_gripper` before the descent —
      a gripper left closed from a previous step silently fails the ring grasp.
    - >
      The finger placement uses `fingertip_axial` + `finger_half_gap` (from
      tsh-gripper-geometry) and the PERCEIVED `hole_radius`/`rim_radius` for the
      wall midpoint — all four REQUIRED, no tuned fallback. Do not hard-code
      ring offsets.
  canonical_scripts:
    - pickup: scripts/pickup.py
  streaming: false
---

# tsh-pickup

A ring grasp: the TCP is centred on the wall midpoint `(hole_r + rim_r)/2` (from
perception) so both fingers reach their wall with the same travel, the near
finger drops into the hole and the far finger seats on the outer rim, then the
jaws close gently (ramped, to avoid kicking the light ring) and lift. The grip is
rigid, so the tape centre expressed in the giver TCP frame — `tape_in_giver` —
lets the exchange track and plan the held tape by forward kinematics with no
ground truth.

Finger geometry comes from `tsh-gripper-geometry` (`gripper_offsets`): the
fingertip axial offset and the finger half-gap. Combined with the perceived
radii, the grasp self-configures; both are required — there is no tuned
fallback for either.

## When to use

- After `tsh-perceive` (tape) and `tsh-gripper-geometry`, to acquire the tape
  before `tsh-handover`.

## When NOT to use

- Non-ring objects (this is a hole/rim straddle, not a generic top-down grasp —
  use `grasping-with-planner`).
- Learned/policy grasps.

## Recommended subgraph state flow

1 state:

```text
pickup
```

1. **`pickup`** — `type: script`, `scripts/<sg>/pickup.py`. Inputs:
   `tape_xyz=Ref("in.target_xyz")`, `hole_radius=Ref("in.hole_radius")`,
   `rim_radius=Ref("in.rim_radius")`,
   `fingertip_axial=Ref("in.fingertip_axial")`,
   `finger_half_gap=Ref("in.finger_half_gap")`, plus
   `arm_id=Ref("in.pick_arm")` (route-decided picking arm — or a literal
   `arm_id=0` with the `pick_arm` subgraph input omitted, when the task pins
   the arm).
   Returns `{grasped, held_offset, tape_in_giver, grasp_tcp, rim_radius, pick_arm}`.

Bind the outputs (`held_offset` + `rim_radius` feed the shared transport / the
handover; `grasp_tcp` feeds a return leg):

```python
sg.set_outputs(
    held_offset=Ref("pickup.held_offset"),
    grasp_tcp=Ref("pickup.grasp_tcp"),
    rim_radius=Ref("pickup.rim_radius"),
    pick_arm=Ref("pickup.pick_arm"),
)
```

## Required end states

| End state | Meaning |
|---|---|
| `grasped` | Tape held; route to the handover subgraph. |
| `failed` | Grasp plan failed; route to `abort`. |

## See also

- `tsh-perceive` — supplies `tape_xyz` + ring radii.
- `tsh-gripper-geometry` — supplies `fingertip_axial` + `finger_half_gap`.
- `tsh-transport-held` — consumes `held_offset` to carry the tape to the meet.
- `tsh-handover` — consumes `held_offset` (a.k.a. `tape_in_giver`) + `rim_radius`.
