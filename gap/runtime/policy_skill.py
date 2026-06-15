"""Per-checkpoint VLA-policy skill base.

A learned policy is *a skill the robot has*: ``pi05-libero`` and
``molmoact-libero`` are concrete skills, each owning one model checkpoint.
This module is the shared, capability-agnostic machinery they subclass —
the closed-loop body lives in :func:`gap.runtime.policy.run_policy_loop`;
each policy's capability (what it can do, its exit conditions, its
postcondition checkpoint) lives in its own bundle ``SKILL.md``.

The base is deliberately thin and lives in its own module (not
``gap.runtime.policy``) so importing it only pulls in ``gap.skills`` at
bundle-discovery time — when both packages are already loaded — which
avoids a ``gap.runtime`` ⇄ ``gap.skills`` import cycle at engine startup.
"""

from __future__ import annotations

from typing import Any, ClassVar, TypedDict

from gap_core.skills.meta import Skill

from .context import NodeContext
from .policy import (
    _DEFAULT_ARM_ID,
    _DEFAULT_MAX_WINDOWS,
    _DEFAULT_REPLAN_EVERY,
    _DEFAULT_SETTLE_STEPS,
    _DEFAULT_TERM_PERIOD,
    _DEFAULT_VLM_CAMERA,
    run_policy_loop,
)


class PolicyLoopOutput(TypedDict):
    """Return shape of one policy-loop invocation (the ``run_policy_loop``
    contract). Subclasses and the per-bundle ``@tool`` forms share it so the
    flat tool catalog renders typed outputs."""

    status: str
    num_windows: int
    num_steps: int


class PolicyLoopSkill(Skill):
    """Stateful base for a single-checkpoint VLA-policy skill.

    A concrete policy skill is a tiny subclass that sets :attr:`preset` to
    its bundle name (== the registered preset / ``policy_id``) and supplies
    a capability-rich ``meta``. The skill **owns its model**: it takes no
    ``policy_id`` argument — it resolves the cached websocket client for
    :attr:`preset` through the executor's
    :class:`gap.runtime.policy.PolicyExecutor` (one connection per preset,
    reused across visits within one workflow execution) and drives the
    shared closed-loop replan/execute/terminate body
    :func:`gap.runtime.policy.run_policy_loop`.

    The launcher boots the matching preset server automatically (see
    :func:`gap.runtime.policy_boot.required_policies`); a ``policies:``
    config entry of the same name overrides the serving recipe (e.g. an
    external ``url:``).
    """

    #: Registered preset / policy id this skill drives. Subclasses MUST set
    #: it; by convention it equals the bundle directory name.
    preset: ClassVar[str] = ""

    def __init__(self) -> None:
        # Lazily bound on the first run() call, then reused for every visit
        # to this state within one workflow execution.
        self._policy_executor: Any | None = None

    def run(
        self,
        ctx,
        observation_stream: Any,
        prompt: str,
        termination_prompt: str = "",
        max_windows: int = _DEFAULT_MAX_WINDOWS,
        replan_every: int = _DEFAULT_REPLAN_EVERY,
        term_period: int = _DEFAULT_TERM_PERIOD,
        arm_id: int = _DEFAULT_ARM_ID,
        vlm_camera: int = _DEFAULT_VLM_CAMERA,
        settle_steps: int = _DEFAULT_SETTLE_STEPS,
        gripper_cycle_termination: bool = False,
    ) -> PolicyLoopOutput:
        if not self.preset:
            raise RuntimeError(
                f"{type(self).__name__}: the `preset` class attribute is "
                f"unset — a policy skill must name the preset/policy id it "
                f"drives (by convention, its bundle name)."
            )
        client = self._client(ctx)
        result = run_policy_loop(
            ctx,
            client=client,
            policy_id=self.preset,
            prompt=prompt,
            termination_prompt=termination_prompt,
            max_windows=max_windows,
            replan_every=replan_every,
            term_period=term_period,
            arm_id=arm_id,
            vlm_camera=vlm_camera,
            settle_steps=settle_steps,
            gripper_cycle_termination=gripper_cycle_termination,
            obs_provider=lambda: observation_stream.latest(),
        )
        return {
            "status": result["status"],
            "num_windows": result["num_windows"],
            "num_steps": result["num_steps"],
        }

    # ------------------------------------------------------------------

    def _client(self, ctx: NodeContext) -> Any:
        """Resolve the cached websocket client for :attr:`preset`."""
        if self._policy_executor is None:
            self._policy_executor = getattr(ctx, "policy_executor", None)
        if self._policy_executor is not None:
            return self._policy_executor.client_for(self.preset)
        raise RuntimeError(
            f"{type(self).__name__}: no PolicyExecutor available to resolve "
            f"preset={self.preset!r}; the launcher must construct a "
            f"PolicyManager and PolicyExecutor before running a workflow that "
            f"uses a policy skill."
        )


__all__ = ["PolicyLoopSkill", "PolicyLoopOutput"]
