---
name: tsh-transport-held
description: >
  Move a HELD object's centre to a world target pose — the shared
  "tape-as-EE" transport. The measured rigid held_offset (object centre in
  the holder's TCP frame, from the pickup's pre-close FK anchor or the
  exchange's grab-instant measurement) is composed into the planner's
  tcp_offset, so the target is an OBJECT-centre pose, never a bare TCP pose.
  With the perceived object cloud supplied, the move is planned with the
  object attached as a cuRobo collision body (plain tool-offset plan as
  fallback). The handover's giver PRESENT leg and the place approach are
  both this operation; use it standalone for any "hold the object at X"
  step (above a sorting stack, at a camera, at a rendezvous). Its arm_id
  script param must NEVER be a subgraph input literally named ``arm_id`` —
  declare ``giver_arm`` on the present instance or ``place_arm`` on the
  place instance instead (both exist upstream; a bare ``arm_id`` input has
  no producer and silently fails to wire).
compatibility: requires gap>=0.1
metadata:
  category: motion
  tags: [tsh, transport, held-object, curobo, yam]
gap:
  allowed_tools:
    - libero-yam.arm_base_pose
    - libero-yam.execute_trajectory
    - curobo.plan_to_pose
    - curobo.plan_with_grasped_object
    - robot.get_ee_pose
  exit_conditions:
    transported: Held-object centre at the target; held_tcp bound in the outputs.
    failed: No plan for the move (raise routes to on_error).
  required_inputs:
    held_offset: Vec3            # object centre in the holder's TCP frame (measured)
    target_xyz: Vec3             # world OBJECT-centre target
    target_quat: Quaternion      # holder TCP orientation at the target, wxyz
    # NOTE: the script also takes an arm_id kwarg (the holding arm), but it is
    # deliberately NOT listed here as a bare "arm_id" required input — there
    # is no upstream producer of a value literally named arm_id, and a
    # subgraph input by that name will never auto-wire (confirmed: this has
    # silently produced a dangling W8 validation error in real generations).
    # Declare the ACTUAL upstream name instead, per instance role:
    #   PRESENT-leg instance (after tsh-station-geometry, before
    #   tsh-handover): declare subgraph input `giver_arm: int` and wire
    #   arm_id=Ref("in.giver_arm") — giver_arm comes from tsh-route (or a
    #   literal when the task pins arms).
    #   PLACE-leg instance (after tsh-place-pose): declare subgraph input
    #   `place_arm: int` and wire arm_id=Ref("in.place_arm") — place_arm
    #   auto-wires to tsh-route's place_arm on the direct route OR
    #   tsh-handover's place_arm alias after a handover (latest-producer
    #   cross-subgraph rule; see tsh-handover's SKILL.md). Never hardcode a
    #   literal arm id — it will not generalize across routes.
  produces_outputs:
    transported: bool
    held_tcp: Se3Pose            # world TCP pose after the move
  hard_rules:
    - >
      The target is an OBJECT-centre pose; ALWAYS compose ``held_offset``
      into the plan — never plan the bare TCP to the target and hope the
      object lands there.
    - >
      FOR NOW, do not wire ``held_cloud`` — leave it unset so the move always
      takes the plain tool-offset plan (no attached-collision-body path).
      The attached-object planning exists and works (``approach_with_attached``
      in ``_held.py``), but it is untested in this task's graphs; revisit and
      wire it in once the plain path is validated end-to-end.
    - >
      ``arm_id`` MUST come from a declared subgraph input (``giver_arm`` on
      the present instance, ``place_arm`` on the place instance) bound via
      ``Ref("in.<name>")`` — the standard auto-wire-by-exact-name mechanism.
      NEVER reference another subgraph directly by name from a node's inputs
      (e.g. ``Ref("handover.receiver_arm")`` written inside THIS subgraph) —
      that is not valid syntax; a ``Ref``'s head is only ever a node within
      the CURRENT subgraph or the reserved ``in`` pseudostate. A confirmed
      real mistake: an agent wrote exactly that direct cross-subgraph Ref,
      which silently degraded into a dangling, unwired ``arm_id`` input with
      no upstream producer — declare the input properly instead.
  canonical_scripts:
    - transport_held: scripts/transport_held.py
  streaming: false
---

# tsh-transport-held

The factored-out common core of "move the thing I'm holding somewhere":
`_held.py` carries `plan_held_move` (tool-offset composition) and
`approach_with_attached` (cloud attached as a cuRobo collision body,
recentred at the live FK-tracked object centre). The handover's giver
present leg, the place hover approach, and this standalone node are all the
same operation on different targets — one implementation, so they cannot
drift apart.

## When to use

- The collision-aware hover approach in the place chain
  (`place_pose_sg → transport_held_sg → place_sg`) — `target_xyz`/
  `target_quat` auto-wire from `tsh-place-pose`'s alias outputs of the same
  name (it is the LATEST producer of those names at that point in the DAG,
  so it correctly wins over any earlier `target_xyz` from a perceive
  instance); `arm_id` wires from a declared `place_arm` input. This is the
  validated composition — use it.
- The giver PRESENT leg of a handover — now the ONLY way to get the tape to
  the meet point; `tsh-handover` has no present-leg logic of its own
  anymore. Compose `station_geometry_sg → present_sg (this skill) →
  handover_sg`: `target_xyz`/`target_quat` auto-wire from
  `tsh-station-geometry`'s aliases of `meet_xyz`/`giver_quat`; `arm_id` wires
  from a declared `giver_arm` input. This is mandatory, validated, and run
  successfully in sim — not a "someday" composition.
- Any standalone held-object move: hold the tape above stack N, present an
  object to a camera, stage it at a rendezvous.

## When NOT to use

- Nothing is held (use plain motion / `robot.plan_move`).
- The final set-down legs of a place — those are constrained straight-line
  descents (`curobo_linear_move`), not free transports; use `tsh-place`.

## Recommended subgraph state flow

1 state:

```text
transport_held
```

1. **`transport_held`** — `type: script`, `scripts/<sg>/transport_held.py`.
   Inputs: `held_offset=Ref("in.held_offset")`, `target_xyz=Ref("in.target_xyz")`,
   `target_quat=Ref("in.target_quat")`, and `arm_id`:
   - present instance: declare subgraph input `giver_arm: int`, wire
     `arm_id=Ref("in.giver_arm")`.
   - place instance: declare subgraph input `place_arm: int`, wire
     `arm_id=Ref("in.place_arm")`.
   Do NOT wire `held_cloud` for now (see hard_rules) — leave it unset. Do NOT
   write a literal int here, and do NOT reference another subgraph by name
   (see hard_rules) — always a declared input, rebound by exact name.
   Returns `{transported, held_tcp}`.

## Checkpoints

- validate=True: the held object's privileged centre is within tolerance of
  the commanded target (2-arg predicate on outputs vs world).
- probe: the object is still grasped after the move (`is_grasped()`).

## See also

- `scripts/_held.py` — the shared held-object motion core.
- `tsh-handover` — always preceded by this skill's present instance now;
  supplies `place_arm` (aliased from `receiver_arm`) for the place-leg
  instance to auto-wire against.
- `tsh-station-geometry` — supplies the present instance's `target_xyz`/
  `target_quat`/`giver_arm`.
- `tsh-place-pose` / `tsh-place` — the validated composition: this node runs
  between them for the collision-aware hover approach, consuming `place_arm`.
