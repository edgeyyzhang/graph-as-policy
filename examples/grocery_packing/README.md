# grocery_packing — pack EVERY object into the basket with a loop

> **What:** a pick-and-place graph with a real **backward edge** — loop until
> every grocery item is in the basket · **Artifact:** a static `workflow.json`
> (v3) you run directly.

Where [build_a_graph](../build_a_graph/) picks one described object, this
example **loops**: perceive the next object on the table, grasp it with the
planner recipe, transport it into the basket, then route **back** to perception
and repeat — until the benchmark reports every item delivered.

This is a self-contained static graph: `packing_graph/workflow.json` plus the
`scripts/` it references. There is no build step — load and run it like any
other v3 graph.

```
packing_graph/
├── workflow.json            # the v3 graph: 4 subgraphs + done/abort ends
└── scripts/
    ├── perceive_dino_vlm.py # DINO + VLM tournament + SAM3 → world-frame cloud
    ├── exterior_view.py     # keep only the agentview cam (drop the wrist)
    ├── route_next_object.py # loop control: all-packed → done, else found/none
    ├── grasp_move.py        # VAB-style grasp approach (rise → XY → straight-Z)
    ├── transport_move.py    # VAB-style transport into the basket (straight-Z)
    └── place_release.py     # open gripper + linear retract
```

## The loop

```
START → container ──found──→ perceive_next ──found──→ grasp ──grasped──→ transport ┐
                  │             │  │ none                 │ failed          │ placed │
        not_found │             │  ↓                      ↓                 │ blocked│
                  ↓             │ done  ←──────────────── abort ←───────────┘        │
                abort           │ (clean success exit)                              │
                                └────────────────── perceive_next ←─────────────────┘
                                          THE BACKWARD EDGE (transport → perceive_next)
```

`transport --placed--> perceive_next` is a genuine cycle. The executor treats a
conditional edge that resolves to an already-completed node as a loop: it resets
the loop body (`perceive_next`, `grasp`, `transport`) and re-runs it. The
cross-subgraph store keeps the most-recent producer, so each iteration grasps
the freshly-perceived `target_obb` while the once-perceived `container_obb`
stays fixed. `GAP_ITERATION_CAP` bounds runaway loops. See `docs/runtime.md`
§7.2 for the back-edge semantics.

## The subgraphs

- **container_sg** (`perceiving-objects`) runs **once** and localizes the basket
  (`container_obb`): `get_observation → exterior_view → perceive("basket") →
  filter_and_compute_obb`.

- **perceive_next_sg** is the **loop head**: `get_observation → exterior_view →
  perceive("grocery item") → decide`. Two things keep it honest:
  - The perceive node passes an `object_description` — *"a packaged grocery
    product such as a can, box, carton, jar, or bottle; never the wicker basket
    or storage container"* — so the VLM pairwise tournament prefers a real item
    over the basket whenever one is on the table.
  - `decide` (`route_next_object.py`) is the loop's stop signal, and it is
    **driven by the benchmark, not geometry** (see below).

- **grasp_sg** — `open → top_down_grasp_candidates → grasp_move → observe →
  close`. `grasp_move.py` is the **VAB approach recipe**: rise to a hover height
  (straight-line cartesian), translate in XY over the object (cartesian), then
  descend straight down onto it with cuRobo's **axis-constrained** linear plan
  (`plan_directed_linear`, `allowed_axes=["Z"]`, `orientation_mode="LOCK"`) — a
  guaranteed vertical drop, not a curved IK path.

- **transport_sg** — `transport_move → release`. `transport_move.py` lifts,
  translates in XY over the basket, then does the same axis-locked straight-Z
  descend *into* the basket; `place_release.py` opens the gripper and retracts
  linearly.

## Termination — the benchmark's own completion signal

The VAB packing env teleports each delivered item to a graveyard and flips
`task_completed` once everything is packed. `route_next_object.py` reads that
directly via the `sim.check_success` tool:

1. `sim.check_success().task_completed` is True → **`none` → done (success)** —
   authoritative, no false-rejects, so the loop never spins on the basket after
   the table is clear and never stops early while items remain.
2. otherwise route on perception: an item was returned → `found` (grasp it);
   nothing → `none`.

The `sim.check_success` call is wrapped so a non-sim connector (real robot)
falls back to perception cleanly. `none` is a normal exit (not `on_error`), so
finishing never looks like a failure.

## The packing benchmark (teleport-on-delivery)

The default task is the **Variational-Automation-Benchmark** pack-all suite
(`libero_object_packing`). The moment an object settles inside the basket, the
`pack_all_into` predicate marks it *delivered* and the env **teleports it to a
graveyard pose** (out of the scene), scoring `completion_rate = delivered /
total`. Each delivered object disappears from the table, so the next
`perceive_next` finds the *next* object; when all are packed `task_completed`
fires and the loop exits. The graph stays a pure policy (perceive / grasp /
transport); the *benchmark* owns the teleport and the score.

## Execute it

Needs the `perceiving-objects` weights, CuRobo (`uv sync --extra grocery`), a
VLM credential, and the executor's back-edge support (in this checkout):

```bash
# Runs against the VAB pack-all suite; records an mp4.
MUJOCO_GL=egl uv run gap run examples/grocery_packing/packing_graph \
    --sim libero_object_packing/0 --video packing.mp4
uv run gap viz     # browse the recorded trace — perceive_next is visited once per object
```

The `open-robot-skills` checkout is auto-discovered (`$GAP_SKILLS_PATH` or the
sibling checkout); pass `--skills /path/to/open-robot-skills` to override. The
VLM provider/model come from `GAP_VLM_PROVIDER` / `GAP_VLM_MODEL` (+ the
provider's credentials).

On the default `seed=3` arrangement this delivers all six items
(`completion_rate = 1.0`) and exits cleanly the iteration after the last
delivery.

## Where to go next

- [build_a_graph](../build_a_graph/) — the single-object version of this graph.
- [grocery_fulfillment](../grocery_fulfillment/) — the LLM-generated acceptance
  benchmark whose grasp/transport recipes this example mirrors.
- `docs/runtime.md` §7.2 — the loop (backward-edge) semantics the executor
  implements.
