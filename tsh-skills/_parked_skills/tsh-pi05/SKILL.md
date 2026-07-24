---
name: tsh-pi05
description: Run the frozen TSH π0.5 VLA in closed loop for the contact-rich
  grasp + mutual handover of the tape spool, on the bimanual Franka rig. Each
  window it reads the observation, asks the policy server for a chunk of 16-D
  bimanual actions ([L j0..6, L grip, R j0..6, R grip]), and forwards them via
  sim.apply_policy_action. Use when a TSH pickup / handover / placement segment
  is delegated to the learned policy AFTER tsh-canonicalize has put the arm at a
  canonical in-distribution pose. NOT for OOD tape positions reached cold (steer
  it with tsh-canonicalize first), and NOT a planner — it has no collision model.
compatibility: requires gap>=0.1
metadata: {category: policy, tags: [tsh, policy, vla, pi, bimanual, gpu, long-running]}
gap:
  requires: {gpu: true, weights: true}
  serving:
    # TODO(serving): no TSH checkpoint exists yet. Fill --policy.dir once trained
    # and graduate this script loop to a class-based PolicyLoopSkill if desired.
    command: ["python", "server.py", "policy:checkpoint",
              "--policy.config=pi05_tsh",
              "--policy.dir=TODO_TSH_CHECKPOINT_URI",
              "--port", "{port}"]
    protocol: websocket
    requires_gpu: true
  allowed_tools:
    - robot.get_observation
    - sim.apply_policy_action
    - sim.tsh_phase            # read the mutual-grasp phase for the handover checkpoint
  exit_conditions:
    completed: Closed loop finished (max_steps reached or VLA signalled done).
    failed: Inference / execution error (raise propagates to on_error).
  required_inputs:
    prompt: str
  produces_outputs:
    status: str
    num_steps: int
  canonical_scripts:
    - run_vla: scripts/run_vla.py
---

# tsh-pi05

Closed-loop driver for the frozen TSH π0.5 VLA. Kept deliberately as a **script**
(not the class-based `PolicyLoopSkill`) for the skeleton: the obs encoding and
action space are bimanual / 16-D and differ from `pi05-libero`, and there is no
checkpoint to serve yet. The loop structure is real; only the inference call is a
`TODO(serving)` stub that currently holds position.

## Loop

1. `robot.get_observation` → build the TSH policy input (state + 3 cameras).
2. `infer(...)` → `[H, 16]` action chunk. **TODO(serving)** — wire the policy
   server client here; the stub returns hold actions so the graph validates and
   the loop is unit-testable without a server.
3. Forward `replan_every` rows via `sim.apply_policy_action`, then replan.

Success is a **checkpoint** (e.g. `sim.tsh_phase` ≥ PHASE_GRASP1 after handover),
never an exit value — see the `examples/tape_handover` workflow.

## When to use

- A TSH grasp / handover / placement segment, after `tsh-canonicalize` has
  reached the canonical pre-pose.
