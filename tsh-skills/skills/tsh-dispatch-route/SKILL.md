---
name: tsh-dispatch-route
description: >
  Branch the graph on the route tsh-route-arms-bimanual already probed, after the grasp.
  tsh-route-arms-bimanual runs before tsh-pickup (pickup needs its pick_arm), but the branch
  it implies happens after (both routes grasp first, then diverge). This node
  carries that decision across the pickup: it re-reads the ``route`` field and
  exposes it as two exits, ``direct`` and ``needs_handover``. It also relays
  tsh-pickup's ``giver_held_offset`` forward as ``held_offset``, the name the
  shared place chain (tsh-place-pose/tsh-transport-held/tsh-place) requires —
  tsh-pickup does not produce that name directly, so the place chain can only
  be wired downstream of this node (or of tsh-handover, on the other route) —
  and pairs it with ``holding_arm`` (= pick_arm), the arm whose TCP frame that
  offset is expressed in. Pure control flow otherwise — no planning, no motion.
compatibility: requires gap>=0.1
metadata:
  category: planning
  tags: [tsh, routing, dispatch, control-flow, bimanual, yam]
gap:
  allowed_tools: []
  exit_conditions:
    direct: One arm grasps AND places — go to the place chain.
    needs_handover: Different arms — go to the exchange chain.
    failed: route value missing or unrecognized (raise routes to on_error).
  required_inputs:
    route: str
    giver_held_offset: Vec3
    pick_arm: int
  produces_outputs:
    held_offset: Vec3
    holding_arm: int
  hard_rules:
    - >
      Place this node immediately after tsh-pickup whenever tsh-route-arms-bimanual is in the
      graph — it re-reads the route decision and branches on it. Without it the
      route answer is inert and the graph runs one fixed strategy.
    - >
      held_offset (relayed from giver_held_offset) is this node's contribution
      to the shared place chain on the direct route — wire the place chain
      downstream of this node's direct exit, never directly from tsh-pickup.
    - >
      held_offset and holding_arm are ONE datum and are always emitted together:
      held_offset is the object centre in the HOLDER's TCP frame, so it is
      meaningless without the arm whose frame it is in. Never relay one without
      the other — a consumer that gets the offset but picks its own arm plans
      the held object onto the wrong gripper.
  canonical_scripts:
    - dispatch_route: scripts/dispatch_route.py
  streaming: false
---

# tsh-dispatch-route

A control-flow node, not a decision-maker: `tsh-route-arms-bimanual` already did the
reachability probing. This exists because the *decision point* (route, before the
grasp) and the *branch point* (after the grasp) are separated by the pickup. It
re-reads the decision on the far side of the pickup and turns it back into two
exits, which the top level maps like any other subgraph's exits.

## When to use

- Immediately after `tsh-pickup`, whenever `tsh-route-arms-bimanual` is in the graph.

## When NOT to use

- Graphs with no `tsh-route-arms-bimanual` (fixed strategy, pinned arms).
- Single-arm platforms.

## Recommended subgraph state flow

```text
dispatch ──(route=="direct")────────▶ direct → END
         └─(route=="needs_handover")▶ needs_handover → END
```

1. **`dispatch`** — `type: script`, file `scripts/<sg>/dispatch_route.py`.
   Inputs: `route=Ref("in.route")`, `giver_held_offset=Ref("in.giver_held_offset")`,
   `pick_arm=Ref("in.pick_arm")`.
   Route on the script's `route` field to two noop exits (`direct`,
   `needs_handover`); `on_error: "failed"`. Returns
   `{route, held_offset, holding_arm}`.

## Required end states

| End state | Meaning |
|---|---|
| `direct` | Route to the place chain. |
| `needs_handover` | Route to the exchange chain. |
| `failed` | Unrecognized/missing route; route to `abort`. |

## See also

- `tsh-route-arms-bimanual` — the probe that produced `route`.
- `tsh-pickup` — supplies `giver_held_offset` / `pick_arm` (relayed here as
  `held_offset` / `holding_arm`).
- `tsh-transport-held`, `tsh-place-pose` — the two branch destinations; both
  consume the `held_offset` + `holding_arm` pair.
