---
name: tsh-station-geometry
description: >
  Derive the bimanual handover station geometry — the meet point and the
  giver/receiver presentation orientations — from the two arm-base poses. The
  meet point is the arm-base midpoint offset forward (+X) and up (+Z) by ~half
  the base separation; the giver/receiver quaternions are the base-geometry
  presentation frame (approach ⊥ baseline, finger-spread along baseline, up =
  world-up). Self-configures for a restationed pair rather than hard-coding a
  demo meet pose. Emits ONLY ``meet_xyz``/``giver_quat``/``recv_quat`` — no
  generic ``target_xyz`` alias, since that name collides with the perception
  role prefix. The tsh-transport-held PRESENT instance instead declares its own
  ``meet_xyz``/``giver_quat`` inputs directly. Use up-front before a scripted
  bimanual exchange on LIBERO-YAM.
compatibility: requires gap>=0.1
metadata:
  category: calibration
  tags: [tsh, self-model, bimanual, handover, station, yam]
gap:
  allowed_tools:
    - libero-yam.arm_base_pose
  exit_conditions:
    derived: Meet point + presentation quats computed and bound in the outputs.
    failed: Arm base poses unavailable (raise routes to abort).
  produces_outputs:
    meet_xyz: Vec3
    giver_quat: Quaternion
    recv_quat: Quaternion
  hard_rules:
    - >
      Do NOT add a target_xyz/target_quat alias of meet_xyz/giver_quat — that
      generic name collides with the perception role prefix (tsh-perceive-*
      also produces target_xyz, for the picked object). Whichever node runs
      LATER wins under the cross-subgraph auto-wire rule, so this skill's
      position relative to perception would silently determine which value a
      consumer gets. Have the tsh-transport-held PRESENT instance declare
      meet_xyz/giver_quat as its own inputs instead.
  canonical_scripts:
    - station_geometry: scripts/station_geometry.py
  streaming: false
---

# tsh-station-geometry

A **self-model** skill for the *bimanual workspace*: it computes where the two
arms should meet and how they should be oriented, from the arm-base geometry
alone — no tuned rendezvous constants. The handover is a symmetric bimanual
gesture, so the meet point is the arm midpoint pushed forward and up, and the
grippers point along the shared forward-reach axis with fingers spread along the
baseline.

A `tsh-transport-held` present instance placed after this node carries the tape
to the meet point (`tsh-handover` has no present-leg logic of its own) — it
declares its own `meet_xyz`/`giver_quat` inputs rather than consuming a generic
aliased name, so there is no position-dependent ambiguity with perception's
own `target_xyz`.

## When to use

- Up-front before any `tsh-handover` — `giver_quat` / `recv_quat` feed the
  exchange, and `meet_xyz` feeds the present leg.

## When NOT to use

- Single-arm tasks (there is no rendezvous to derive).

## Recommended subgraph state flow

```text
derive_station_geometry
```

1. **`derive_station_geometry`** — `type: script`,
   file `scripts/<sg>/station_geometry.py`, `inputs: {}` (optionally
   `giver_arm` / `receiver_arm` literals). Returns
   `{meet_xyz, giver_quat, recv_quat}`.

```python
sg.set_outputs(
    meet_xyz=Ref("derive_station_geometry.meet_xyz"),
    giver_quat=Ref("derive_station_geometry.giver_quat"),
    recv_quat=Ref("derive_station_geometry.recv_quat"),
)
```

## Required end states

| End state | Meaning |
|---|---|
| `derived` | Meet + quats bound; route to the present/exchange subgraph. |
| `failed` | Arm bases unavailable; route to `abort`. |

## See also

- `scripts/_station_geometry.py` — the derivation.
- `tsh-handover` — consumer of `giver_quat` / `recv_quat`.
- `tsh-transport-held` — its PRESENT instance consumes `meet_xyz` / `giver_quat` directly.
