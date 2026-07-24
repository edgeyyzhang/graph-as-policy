---
name: tsh-station-geometry
description: >
  Derive the bimanual handover station geometry — the meet point and the
  giver/receiver presentation orientations — from the two arm-base poses. The
  meet point is the arm-base midpoint offset forward (+X) and up (+Z) by ~half
  the base separation; the giver/receiver quaternions are the base-geometry
  presentation frame (approach ⊥ baseline, finger-spread along baseline, up =
  world-up; receiver canonical, splay 0). Reproduces the tuned MEET_XYZ /
  GIVER_QUAT / RECV_QUAT and self-configures for a restationed pair. Emits
  ``giver_quat``/``recv_quat`` for tsh-handover directly, plus ``target_xyz``/
  ``target_quat`` aliases of ``meet_xyz``/``giver_quat`` so a mandatory
  ``tsh-transport-held`` instance can auto-wire immediately after this node as
  the giver's collision-aware PRESENT leg — ``tsh-handover`` itself no longer
  has any present-leg logic of its own. Use as the up-front self-model step
  before a scripted bimanual exchange on LIBERO-YAM.
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
    meet_xyz: Vec3          # world-frame handover point
    giver_quat: Quaternion  # giver present orientation, wxyz
    recv_quat: Quaternion   # receiver thread orientation, wxyz (canonical; exchange mirrors)
    target_xyz: Vec3        # alias of meet_xyz — ONLY so a tsh-transport-held
                             # present instance can auto-wire right after this
                             # node (same trick tsh-place-pose uses for the
                             # place-chain transport_held). Do not reuse for
                             # anything else.
    target_quat: Quaternion # alias of giver_quat — see target_xyz above.
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

The Y-midpoint (no "own-side" bias) was validated reachable AND collision-free
across the grid, so the meet point is pure base geometry. The receiver uses the
splay-0 frame (the tuned RECV_QUAT's ~8.5° wrist splay was compensating for an
off-centre tuned meet, and is unnecessary once the meet is the symmetric
midpoint). A reachability-refined variant lives in
`_station_geometry.derive_station_geometry` for a station that needs it.

## When to use

- Always, as a mandatory up-front step before any `tsh-handover` — `giver_quat`/
  `recv_quat` feed the exchange directly, and `meet_xyz` (via the `target_xyz`/
  `target_quat` aliases) feeds the required `tsh-transport-held` PRESENT
  instance composed right after this node: `station_geometry → present
  (tsh-transport-held) → handover`. `tsh-handover` no longer has any way to
  get the tape to the meet point itself, so this subgraph is not optional
  once a handover is in the graph.
- Any scripted bimanual handover on LIBERO-YAM that should self-configure to the
  arm placement rather than hard-code a demo meet pose.
- **Ordering requirement**: this instance must run AFTER the last consumer of
  the perception `target_xyz` (i.e. after `tsh-pickup`) — cross-subgraph
  wiring binds by exact name to the LATEST producer at that point in the DAG,
  so if this subgraph ran before `tsh-pickup`, its `target_xyz` alias would
  incorrectly shadow the perceived tape position for the grasp.

## When NOT to use

- Single-arm tasks (there is no rendezvous to derive).

## Recommended subgraph state flow

1 state:

```text
derive_station_geometry
```

1. **`derive_station_geometry`** — `type: script`,
   file `scripts/<sg>/station_geometry.py` from this bundle,
   `inputs: {}` (optionally `giver_arm` / `receiver_arm` literals).
   Returns `{meet_xyz, giver_quat, recv_quat}`.

Bind the subgraph outputs:

```python
sg.set_outputs(
    meet_xyz=Ref("derive_station_geometry.meet_xyz"),
    giver_quat=Ref("derive_station_geometry.giver_quat"),
    recv_quat=Ref("derive_station_geometry.recv_quat"),
    # Aliases so a tsh-transport-held PRESENT instance can auto-wire right
    # after this node — see the ordering requirement under "When to use".
    target_xyz=Ref("derive_station_geometry.meet_xyz"),
    target_quat=Ref("derive_station_geometry.giver_quat"),
)
```

Downstream, `tsh-handover` wires `giver_quat`/`recv_quat` via `Ref("in.<name>")`
(it no longer takes `meet_xyz` at all). `tsh-transport-held`, composed right
after this node as the mandatory PRESENT leg, wires its `target_xyz`/
`target_quat` inputs from this node's aliases the same way `tsh-place-pose`'s
aliases feed the place-chain instance.

## Required end states

| End state | Meaning |
|---|---|
| `derived` | Meet + quats bound; route to the exchange subgraph. |
| `failed` | Arm bases unavailable; route to `abort`. |

## See also

- `scripts/_station_geometry.py` — the derivation (`derive_meet_xyz`,
  `derive_presentation_quats`, and a reachability-refined `derive_station_geometry`).
- `tsh-handover` — consumer of `giver_quat` / `recv_quat` directly.
- `tsh-transport-held` — consumer of the `target_xyz`/`target_quat` aliases,
  composed between this node and `tsh-handover` as the mandatory present leg.
