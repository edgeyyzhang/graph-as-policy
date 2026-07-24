---
name: tsh-dispatch-route
description: >
  Branch the graph on the route tsh-route already probed, AFTER the grasp.
  tsh-route must run BEFORE tsh-pickup (pickup needs its pick_arm), but the
  branch it implies happens AFTER (both routes grasp first, then diverge).
  This node carries that decision across the pickup: it re-reads the ``route``
  field and exposes it as two exits. ALWAYS place it immediately after
  tsh-pickup whenever tsh-route is in the graph: exit ``direct`` goes to the
  place chain (tsh-place-pose), exit ``needs_handover`` goes to the exchange
  chain (tsh-station-geometry). Without this node the graph cannot act on the
  route at all — the handover either always runs or never runs, regardless of
  what tsh-route decided. Pure control flow; no planning, no motion.
compatibility: requires gap>=0.1
metadata:
  category: planning
  tags: [tsh, routing, dispatch, control-flow, bimanual, yam]
gap:
  allowed_tools: []          # pure control flow, no tools
  exit_conditions:
    direct: One arm grasps AND places — go straight to the place chain (tsh-place-pose).
    needs_handover: Different arms — go to the exchange chain (tsh-station-geometry).
    failed: route value missing or unrecognized (raise routes to on_error).
  required_inputs:
    route: str                # tsh-route's ``route`` output, auto-wired by exact name
  produces_outputs: {}        # pure control flow — the exit IS the output
  hard_rules:
    - >
      Place this node immediately after ``tsh-pickup``'s success exit whenever
      ``tsh-route`` is in the graph. ``tsh-pickup`` must NOT wire its
      ``grasped`` exit straight into ``tsh-station-geometry`` (forces a
      handover on every scene) NOR straight into ``tsh-place-pose`` (leaves
      the whole exchange chain unreachable) — both are confirmed real
      failures. It wires to THIS node, which then branches.
    - >
      Both exits converge again on the SAME place chain
      (``tsh-place-pose`` → ``tsh-transport-held`` → ``tsh-place``). Do NOT
      build two parallel place chains — one is enough, because
      ``held_offset``/``place_arm`` resolve to whichever holder ran last
      (``tsh-pickup``'s on the direct route, ``tsh-handover``'s aliases after
      an exchange) under the latest-producer cross-subgraph rule.
  canonical_scripts:
    - dispatch_route: scripts/dispatch_route.py
  streaming: false
---

# tsh-dispatch-route

A control-flow node, not a decision-maker: `tsh-route` already did all the
reachability probing. This exists purely because the *decision point* and the
*branch point* are separated by the grasp.

`tsh-route` has to run early — `tsh-pickup` needs its `pick_arm` to know which
arm grasps. But both routes grasp, so the graph can only diverge afterwards.
This node re-reads the decision on the far side of the pickup and turns it back
into two exits, which the top level maps like any other subgraph's exits.

## When to use

- Always, immediately after `tsh-pickup`, whenever `tsh-route` is in the graph.
  Without it, `route`'s answer is inert — nothing reads it, and the graph is
  hardwired to one route regardless of the scene.

## When NOT to use

- Graphs with no `tsh-route` (the task prescribes a fixed strategy and the
  arms are pinned — then there is no decision to carry).
- Single-arm platforms.

## Recommended subgraph state flow

Route on the script's `route` field to two noop exits — identical in shape to
`tsh-route`'s own subgraph:

```text
dispatch ──(route=="direct")──▶ direct → END
         └─(route=="handover")▶ needs_handover → END
```

```python
sg = Subgraph(name="dispatch_route", skill="tsh-dispatch-route")
sg.add_input("route", type_name="str")
sg.add_node("dispatch", type="script", script="scripts/<sg>/dispatch_route.py",
            inputs={"route": Ref("in.route")})
sg.add_exit("direct")
sg.add_exit("needs_handover")
sg.add_conditional_edges("dispatch", {"direct": "direct", "handover": "needs_handover"},
                         router_field="route")
sg.add_edge(START, "dispatch")
sg.add_edge("direct", END)
sg.add_edge("needs_handover", END)
sg.set_on_error("failed")
```

## Top-level wiring (the whole point of this skill)

```python
# pickup branches HERE, not into either chain directly:
wf.add_conditional_edges("pickup", {"grasped": "dispatch_route", "failed": "abort"})
wf.add_conditional_edges("dispatch_route", {
    "direct":         "place_pose",        # skip the exchange entirely
    "needs_handover": "station_geometry",  # station → present → handover → place_pose
    "failed":         "abort",
})
```

Both branches converge on the one shared `place_pose → transport_held → place`
chain. On the direct route `held_offset`/`place_arm` come from `tsh-pickup`; on
the handover route `tsh-handover` rebinds those same names to the receiver, so
the identical chain serves both with no duplication.

## Required end states

| End state | Meaning |
|---|---|
| `direct` | Route to the place chain (`tsh-place-pose`). |
| `needs_handover` | Route to the exchange chain (`tsh-station-geometry`). |
| `failed` | Unrecognized/missing route value; route to `abort`. |

## See also

- `tsh-route` — the probe that produced `route`; runs before `tsh-pickup`.
- `tsh-station-geometry` → `tsh-transport-held` (present) → `tsh-handover` —
  the `needs_handover` chain.
- `tsh-place-pose` → `tsh-transport-held` → `tsh-place` — the shared place
  chain both exits converge on.
