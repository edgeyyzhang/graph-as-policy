---
name: tsh-bimanual-handover
description: Scripted (non-policy) bimanual tape-spool handover on the LIBERO-YAM
  YAM arms. Perceives the tape and the place target from RGB-D (no ground truth,
  no object dimensions), ring-grasps the tape with the giver arm, presents it
  face-on at a validated meeting point by planning the HELD TAPE as the end
  effector, threads the receiver's finger into the same hole for the exchange,
  and lays the tape flat on the duct roll. All motion is planned by the
  canonical curobo tool bundle and executed by the LIBERO-YAM connector. Use for
  a two-tape handover on the LIBERO-YAM env when no TSH policy checkpoint is
  available.
compatibility: requires gap>=0.1
metadata: {category: motion, tags: [tsh, handover, bimanual, scripted, yam, curobo]}
gap:
  allowed_tools:
    # RGB-D perception (agentview camera): DINO detect -> SAM box segment ->
    # depth back-projection (inlined in scripts/_perceive.py).
    - robot.get_observation
    - grounding-dino.detect
    - sam3.segment_box
    # Gripper + FK primitives from the LIBERO-YAM connector.
    - robot.open_gripper
    - robot.close_gripper
    - robot.get_ee_pose
    # Motion is planned by the canonical open-robot-skills curobo bundle and
    # executed on the sim by the connector: libero-yam.arm_base_pose supplies
    # the frame info (base pose, TCP offset, current joints) and
    # libero-yam.execute_trajectory streams the planned joints PD-controlled.
    - libero-yam.arm_base_pose
    - libero-yam.execute_trajectory
    - curobo.plan_to_pose
    - curobo.plan_directed_linear
    - curobo.plan_with_grasped_object
  # This is a MONOLITHIC single subgraph (perceive -> pickup -> exchange ->
  # place, all internal) with NO upstream producer subgraph, so it declares NO
  # required_inputs. The scene parameters (object_key, target_key, giver_arm,
  # receiver_arm) are LITERALS the coordinator inlines directly on the internal
  # script nodes — see "### Parameter reference". They are NOT subgraph inputs:
  # GaP's cross-subgraph rule (W8) only satisfies a subgraph input from an
  # upstream subgraph output of the same name, and nothing produces these, so
  # declaring them as required_inputs is unsatisfiable and fails validation.
  exit_conditions:
    placed: Tape transferred giver -> receiver and laid flat on the place target.
    # ROUND-TRIP ONLY — un-advertised by default. The coordinator wires up every
    # declared exit_condition, so leaving `returned` here makes it append the
    # `return_leg` state even for a one-way "pick up / hand over / place" task.
    # To author a round-trip graph, un-comment `returned` here AND `return_leg`
    # under canonical_scripts below (see the "### Return / round-trip" section).
    # returned: tape handed back and re-placed at its pickup spot.
    failed: Perception, planning, or motion error (a raise propagates to on_error).
  hard_rules:
    - >
      Do NOT invent numeric tuning values. Every offset, height, angle and the
      default handover point live in the skill's scripts/constants.py and are
      validated for this scene. The ONLY values the coordinator authors are the
      object keys and the arm ids, written as LITERAL inputs directly on the
      internal script nodes (see the parameter reference) — NOT as subgraph
      required_inputs. In particular do NOT pass meet_xyz unless the task
      explicitly names a handover location.
    - >
      perceive_duct MUST be the FIRST state — before the grasp, while the view
      is clean. At place time the held tape + gripper hover over the duct and
      leak into the DINO box / SAM mask, skewing the perceived top face low.
    - >
      Perception consumes ONE shared observe node (robot.get_observation):
      perceive_duct and perceive_tape take `cameras = Ref("observe.cameras")`
      plus their object key — nothing else. The script selects the external
      agentview view itself; do NOT name a camera (the wrist cameras are not
      wired as a fallback).
    - >
      Keep the exact data-flow refs from the state table: pickup consumes
      perceive_tape's tape_xyz/hole_radius/rim_radius; bimanual_exchange
      consumes pickup's tape_in_giver; place consumes bimanual_exchange's
      receiver_offset, perceive_duct's duct_xyz and perceive_tape's tape_cloud.
      These measured values are what make the pipeline ground-truth-free —
      dropping one silently degrades the handover.
    - >
      giver_arm picks the tape and must be the arm on the tape's side of the
      table (arm 0 = left / +Y, arm 1 = right / -Y); receiver_arm places.
      arm_id on pickup = giver_arm; arm_id on place = receiver_arm.
  canonical_scripts:
    - perceive_duct: scripts/perceive_duct.py
    - perceive_tape: scripts/perceive_tape.py
    - pickup: scripts/pickup.py
    - bimanual_exchange: scripts/bimanual_exchange.py
    - place: scripts/place.py
    # ROUND-TRIP ONLY — un-comment together with `returned` in exit_conditions
    # above to author a round-trip graph (see "### Return / round-trip").
    # - return_leg: scripts/return_leg.py
  streaming: false
---

# tsh-bimanual-handover

The non-policy path for the two-tape handover: hand-scripted motion primitives
inside the GaP pipeline (graph + connector tools). Perception is RGB-D
(DINO+SAM+depth back-projection); every motion is planned by the canonical
curobo bundle (`curobo.plan_to_pose` / `plan_directed_linear` /
`plan_with_grasped_object`) and streamed onto the sim via
`libero-yam.execute_trajectory`. No ground-truth object poses anywhere.

The held tape is treated as the **end effector**: the pickup measures the tape
centre in the giver's TCP frame (`tape_in_giver`, rigid under the grip), the
exchange composes that offset into the planner's `tcp_offset` so cuRobo plans
"tape centre to the meeting point" directly, and the exchange in turn measures
`receiver_offset` (tape centre in the receiver's TCP frame) at the grab instant
so the place can plan the tape — not the gripper — onto the duct.

## When to use

- A two-tape handover on the LIBERO-YAM env with no TSH policy checkpoint.

## When NOT to use

- Any other robot/env: the scripts assume the YAM connector tools
  (`libero-yam.arm_base_pose`, `libero-yam.execute_trajectory`) and the
  `yam.yml` curobo robot config.
- A TSH VLA checkpoint is available and preferred — use `tsh-pi05`.

## Recommended subgraph state flow



(`placed` is the success-marker `noop` from `sg.add_exit("placed")`, with an
edge to `END`; `on_error: "failed"` catches any raise.)

State details (the perception + manipulation states are `type: script` from this
bundle's canonical_scripts; `observe` is a `type: tool` node):

1. **`observe`** — `type: tool`, `tool: "robot.get_observation"`. Takes one
   RGB-D observation of the clean scene and emits `cameras`, the shared camera
   list both perceive states consume. No inputs. MUST run first.
2. **`perceive_duct`** — inputs: `target_key = <target_key>`,
   `cameras = Ref("observe.cameras")`. Localizes the place target's top-face
   centre from RGB-D while the view is clean; emits `duct_xyz`. MUST run before
   the grasp (see hard rules).
3. **`perceive_tape`** — inputs: `object_key = <object_key>`,
   `cameras = Ref("observe.cameras")`. Localizes the tape from RGB-D; emits the
   grasp point `tape_xyz`, the world-frame `tape_cloud`, and the measured
   `hole_radius` / `rim_radius`.
4. **`pickup`** — inputs:
   `tape_xyz = Ref("perceive_tape.tape_xyz")`,
   `hole_radius = Ref("perceive_tape.hole_radius")`,
   `rim_radius = Ref("perceive_tape.rim_radius")`,
   `arm_id = <giver_arm>`. Ring-grasps from above (one finger in the hole, one
   on the outer wall), lifts, and emits `tape_in_giver` — the measured held-tape
   offset that makes the tape the giver's end effector downstream.
5. **`bimanual_exchange`** — inputs:
   `giver_arm = <giver_arm>`, `receiver_arm = <receiver_arm>`,
   `tape_in_giver = Ref("pickup.tape_in_giver")`. Presents the tape face-on at
   the validated meeting point (tape planned as the EE), sweeps the receiver
   grasp angle around the ring for a reachable insertion, threads, swaps grips
   (receiver closes before the giver opens), measures and emits
   `receiver_offset`, then retracts the giver first. Do NOT pass `meet_xyz`
   unless the task explicitly specifies a handover location (the env var
   `GAP_HANDOVER_XYZ="x,y,z"` also overrides it at runtime for sweeps).
6. **`place`** — inputs:
   `receiver_offset = Ref("bimanual_exchange.receiver_offset")`,
   `duct_xyz = Ref("perceive_duct.duct_xyz")`,
   `tape_cloud = Ref("perceive_tape.tape_cloud")`,
   `arm_id = <receiver_arm>`. Sweeps the free yaw about the tape's vertical
   axis for a reachable place pose, approaches with the tape attached as a
   collision body, sets it straight down onto the duct, releases, retracts.

```json
"edges": [["START", "observe"], ["observe", "perceive_duct"],
          ["perceive_duct", "perceive_tape"],
          ["perceive_tape", "pickup"], ["pickup", "bimanual_exchange"],
          ["bimanual_exchange", "place"], ["place", "placed"],
          ["placed", "END"]],
"conditional_edges": {},
"exit": { "router_field": null, "success_values": ["placed"] },
"on_error": "failed"
```

### Return / round-trip

**`return_leg` is un-advertised by default** — the `returned` exit_condition and
the `return_leg` canonical_script are commented out in the frontmatter, so a
one-way "pick up / hand over / place" task generates cleanly (ENDS at
`place → placed`, `conditional_edges: {}`, `place` has the single unconditional
successor `placed`; its `placed` output is a postcondition value, NOT a routing
key). To author a round-trip graph, un-comment BOTH frontmatter entries first,
then append the `return_leg` state as below.

To return the tape to where it was picked (a round trip), append ONE state —
`return_leg` (`scripts/return_leg.py`) — after `place`, with exit `returned`:

```text
… place → return_leg → returned
```

The return leg re-perceives the resting ring ONCE (right after the place, to
centre the re-pick — the fingertips clear the duct walls by only a few mm) and
derives no other geometry. The forward states
RECORD their key world-frame TCP poses (`pickup.grasp_tcp`,
`bimanual_exchange.giver_tcp` / `receiver_tcp`, `place.place_tcp`), and the
return leg replays them in reverse with the arm roles swapped: the forward
receiver descends onto its own recorded place pose and re-closes (re-acquiring
the tape with the measured `receiver_offset` grip), carries it back to its
recorded exchange grab pose, the forward giver reverses its own retract and
re-closes at its recorded present pose, the forward receiver reverses its
insert, and the forward giver reverses its pickup — landing the tape back on
its original spot. Every pose was demonstrated by the forward leg, so
reachability and arm-arm clearance hold by construction.

- **`return_leg`** — inputs: `grasp_tcp = Ref("pickup.grasp_tcp")`,
  `giver_tcp = Ref("bimanual_exchange.giver_tcp")`,
  `receiver_tcp = Ref("bimanual_exchange.receiver_tcp")`,
  `receiver_offset = Ref("bimanual_exchange.receiver_offset")`,
  `place_tcp = Ref("place.place_tcp")`,
  `tape_cloud = Ref("perceive_tape.tape_cloud")`,
  `object_key = <object_key>` (same value as the forward `perceive_tape`, for
  the single post-place re-perceive),
  `giver_arm` / `receiver_arm` = the FORWARD roles (the script swaps them).

The exit is `returned` instead of `placed`; there is NO perception at or
before the hand-back itself and NO second `perceive_duct`.

### Parameter reference

`object_key`, `target_key`, `giver_arm`, `receiver_arm` are scene-constant
values the coordinator determines from the current task/scene and writes as
**literal `inputs` directly on the internal script nodes** — NOT as subgraph
`required_inputs`. This subgraph is monolithic (it has no upstream producer),
and GaP's cross-subgraph rule (W8) only satisfies a subgraph input from an
upstream subgraph output of the same name; since nothing produces these
scene constants, declaring them as subgraph inputs is unsatisfiable and fails
validation. Inline them instead, exactly like `open-robot-skills` reserves
`required_inputs` for upstream-produced typed data (OBB / Mask / Se3Pose) and
inlines scene constants on the nodes that consume them.

Concretely, on each node's `inputs` map write the literal value, e.g.
`perceive_tape.object_key: "yellow_tape_1"`, `perceive_duct.target_key:
"duct_tape_1"`, `pickup.arm_id: 0`, `bimanual_exchange.giver_arm: 0` /
`receiver_arm: 1`, `place.arm_id: 1`. `object_key` / `target_key` are the MJCF
body names of the tape and place target; `giver_arm` / `receiver_arm` follow
the hard rule above (arm on the tape's side picks, arm on the target's side
places). The `<...>` placeholders in the state flow above stand for these
literal values — substitute the actual body names / arm ids for the scene,
do NOT emit them as `Ref(...)` or as subgraph inputs.

Everything else (meeting point, grasp offsets, insertion depths, place
clearances) is a validated constant in `scripts/constants.py` — never set these
from the workflow.

## Required end states

| End state | Meaning |
|---|---|
| `placed` | Tape handed over and resting on the place target. Route to the workflow's success end. |
| `failed` | Any perception/planning/motion raise, via `on_error`. Route to abort (recovery: `robot.open_gripper`). Lives only in `on_error` — never declare a `failed` node. |
