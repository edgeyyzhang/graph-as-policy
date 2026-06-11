# Collect demonstrations → train a policy → run it in a graph

The full data loop: a verified gap graph acts as a **scripted expert**, its
rollouts become a demonstration dataset, an external recipe trains a VLA
policy on them, and the trained policy comes back into gap as a
`running-policies` node — steered by the same perception that collected the
data (see [`../steered_policy/`](../steered_policy)).

## 1. Collect

```bash
uv sync --extra quickstart --extra policy   # sim + perception + openpi client

uv run python examples/collect_and_train/collect.py --episodes 50 --out demos.hdf5 \
    --graph examples/libero_quickstart/graph
```

Each episode resets to a fresh baked variation seed and records, at every
control step: per-camera RGB (`/observations/<camera>_rgb`, uint8),
proprioceptive state (`/observations/state`, float32), the commanded action
(`/actions`), rewards, and episode boundaries with success flags — written
by `gap.connector.collector.DataCollector`.

## 2. Convert + train (external)

The HDF5 layout maps directly onto LeRobot's dataset fields
(`observation.images.*`, `observation.state`, `action`, episode indices).
Convert with LeRobot's `lerobot/scripts/push_dataset_to_hub.py`-style
loaders or a ~30-line custom `LeRobotDataset.from_raw` adapter, filter to
`success=True` episodes, then train with your recipe of choice — e.g.
[LeRobot](https://github.com/huggingface/lerobot) ACT/diffusion baselines, or
[OpenPI](https://github.com/Physical-Intelligence/openpi) fine-tuning
(`pi05` configs). Model hosting is your responsibility — gap only needs a
websocket policy server.

## 3. Serve + run the trained policy

```bash
uv run gap policy serve pi05-libero --port 9100   # or your own checkpoint:
# policies: {my_policy: {start_cmd: "... --port {port}"}} in the task yaml

MUJOCO_GL=egl uv run gap run examples/steered_policy/graph_loop \
    --sim libero_object_all_variance/0
```

The steered graph hovers above the perceived object and hands control to
your policy — the hand-off pose matches the distribution this collection
script produced, because both use the same OBB perception + approach.
