"""Closed-loop driver for the frozen TSH π0.5 VLA (skeleton).

Reads observations, asks the policy for a chunk of 16-D bimanual actions, and
forwards them to the env via ``sim.apply_policy_action``. The inference call is a
``TODO(serving)`` stub (holds position) until a TSH checkpoint exists, so the
graph validates and the loop is unit-testable without a server.
"""

from __future__ import annotations

from typing import TypedDict

import numpy as np

from gap import NodeContext

# Bimanual action layout: [L j0..6, L grip, R j0..6, R grip].
_ACTION_DIM = 16


class Output(TypedDict):
    status: str
    num_steps: int


def _infer(observation: dict, prompt: str, horizon: int) -> np.ndarray:
    """Return an ``[horizon, 16]`` action chunk for the current observation.

    TODO(serving): replace with the policy-server client call. The real version
    builds the TSH input (state + exo/wrist images, analogous to openpi's
    ``build_tsh_pi_obs``), calls ``client.infer(obs)["actions"]``, and returns
    the 16-D rows. The stub holds the current pose (zeros) so the loop runs
    end-to-end without a server.
    """
    return np.zeros((horizon, _ACTION_DIM), dtype=np.float64)


def run(
    ctx: NodeContext,
    *,
    prompt: str = "",
    max_steps: int = 200,
    horizon: int = 10,
    replan_every: int = 5,
) -> Output:
    """Run the VLA in closed loop until ``max_steps`` actions have been applied.

    Args:
        prompt: task instruction passed to the policy.
        max_steps: hard cap on actions forwarded to the env.
        horizon: action-chunk length requested per inference.
        replan_every: rows consumed from each chunk before re-observing/replanning.
    """
    steps = 0
    while steps < max_steps:
        obs = ctx.tool("robot.get_observation")
        chunk = _infer(obs, prompt, horizon)
        for row in chunk[:replan_every]:
            if steps >= max_steps:
                break
            ctx.tool("sim.apply_policy_action", action=[float(v) for v in row])
            steps += 1
    return {"status": "completed", "num_steps": steps}
