# grocery_packing — pack EVERY object into the basket with a loop

> **What:** A pick-and-place graph with a real **backward edge** — loop until the table is clear · **Needs:** `uv sync` to build · **Time:** ~1 min

Where [build_a_graph](../build_a_graph/) picks one described object, this
example **loops**: perceive the next object, grasp it with the planner recipe,
transport it into the basket, then route **back** to perception and repeat —
until perception reports nothing left. It is authored with the same
`gap.builder` pipeline (the identical `workflow.json` + `scripts/` +
`checkpoints/` layout, the same parser/validation, the same `gap run` /
`gap.execute`).

```bash
uv run python examples/grocery_packing/build_graph.py --out my_packing   # build + validate
```

leaves a runnable artifact:

```
my_packing/
├── workflow.json              # the v3 graph (4 subgraphs + done/abort ends)
├── scripts/                   # canonical bundle scripts, copied verbatim
│   ├── perceive_dino_vlm.py   #   from the open-robot-skills checkout
│   ├── compute_drop_pose.py
│   ├── waypoint_move.py
│   ├── descend_release.py
│   └── route_next_object.py   # NEW: loop control (item → grasp, only-basket → done)
└── checkpoints/
    ├── container_sg.py        # ground-truth basket-localization gate (validate=True)
    ├── grasp_sg.py            # per-iteration grasp probe
    └── transport_sg.py        # per-iteration placement probe
```

## The loop

```
START → container ──found──→ perceive_next ──found──→ grasp ──grasped──→ transport ┐
                  │             │  │ none                 │ failed          │ placed │
        not_found │             │  ↓                      ↓                 │ blocked│
                  ↓             │ done  ←──────────────── abort ←───────────┘        │
                abort           │ (open gripper, go home)                            │
                                └────────────────── perceive_next ←─────────────────┘
                                          THE BACKWARD EDGE (transport → perceive_next)
```

- **container_sg** (`perceiving-objects`) runs **once** and localizes the
  basket (`container_obb`).
- **perceive_next_sg** (`perceiving-objects`) is the **loop head**:
  `robot.get_observation` → DINO+VLM+SAM3 → a `router`. It perceives a
  **`"grocery item"`** (the *class*, not a single described object), which is
  load-bearing: `perceive_dino_vlm` runs a VLM pairwise tournament that keeps
  the crop better matching the prompt, and a real item beats the basket on
  "which is the grocery item?" every time — so it returns the basket **only**
  once every item has been packed and teleported away. The router (which is
  also handed the once-perceived `container_obb`) turns that into the loop's
  stop signal: a perceived target sitting inside the basket's XY footprint
  means the table is clear → exit `none`; otherwise → fit the OBB and exit
  `found` (grasp it). A bare `"object"` prompt breaks both halves — the
  tournament ranks the basket as just another "object", so perception grasps
  the basket and the loop never terminates. Gating the OBB fit behind `found`
  keeps an empty point cloud away from `geometry.filter_and_compute_obb`, and
  `none` is a normal exit (not `on_error`), so finishing never looks like a
  failure.
- **grasp_sg** and **transport_sg** mirror the simplified grocery_fulfillment
  recipes: a **planner-free** direct top-down grasp
  (`top_down_grasp_candidates → robot.go_to_pose(z_approach=0.1) → close`),
  then drop-pose + lift/lateral move + release. The direct grasp avoids the
  CuRobo per-candidate planning that can stall on awkward objects.
- **transport → perceive_next** is a genuine cycle. The executor treats a
  conditional edge that resolves to an already-completed node as a loop: it
  resets the loop body (`perceive_next`, `grasp`, `transport`) and re-runs it.
  Because the cross-subgraph store keeps the most-recent producer, each
  iteration grasps the freshly-perceived `target_obb` while the once-perceived
  `container_obb` stays fixed. `node_visit_cap` bounds runaway loops. See
  `docs/runtime.md` §7.2.

## The packing benchmark (teleport-on-delivery)

The default task is the **Variational-Automation-Benchmark** pack-all suite
(`libero_object_packing`), whose env (`VABControlEnv`) implements
**teleport-on-In monotonic delivery**: the moment an object settles inside the
basket, the `pack_all_into` predicate marks it *delivered* and the env
**teleports it to a graveyard pose** (out of the scene). It scores
`completion_rate = delivered / total`.

This is what makes the loop well-defined: each delivered object disappears from
the table, so the next `perceive_next` finds the *next* object, and when the
table is clear perception returns nothing → `none → done`. The graph stays a
pure policy (perceive / grasp / transport); the *benchmark* owns the teleport
and the score. For a scored grid, run the `grocery_packing` benchmark family
(`object` = single-In, `permutation` = full-table conjunction).

## Assumptions & limitations

- **Checkpoints.** The per-iteration `grasp_sg` / `transport_sg` and the
  `container_sg` checkpoints are **probes** (`validate=False`) — the authoritative
  packing metric is the env's `completion_rate`, not body-name ground truth.
  (Note: once an object is delivered it is teleported to the graveyard, so a
  `body(name).is_in(basket)` probe reads False afterwards — another reason these
  are probes, not gates.)

## Execute it

Same requirements as grocery_fulfillment (`uv sync --extra grocery` for CuRobo,
downloaded weights, a VLM credential) **plus** the executor's back-edge support
(in this checkout):

```bash
# Runs against the VAB pack-all suite by default; --video records an mp4.
MUJOCO_GL=egl uv run python examples/grocery_packing/build_graph.py \
    --out my_packing --execute --video my_packing.mp4
# or run the built artifact directly:
MUJOCO_GL=egl uv run gap run my_packing --sim libero_object_packing/0
uv run gap viz     # browse the recorded trace — perceive_next is visited once per object
```

The open-robot-skills checkout is auto-discovered (`$GAP_SKILLS_PATH` or the
sibling checkout); pass `--skills /path/to/open-robot-skills` to override.

## Where to go next

- [build_a_graph](../build_a_graph/) — the single-object version of this graph.
- [grocery_fulfillment](../grocery_fulfillment/) — the LLM-generated acceptance
  benchmark whose grasp/transport recipes this example reuses.
- `docs/runtime.md` §7.2 — the loop (backward-edge) semantics the executor
  implements.
