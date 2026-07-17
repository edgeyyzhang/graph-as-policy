"""Unit tests for the tsh-pi05 skill bundle (CPU-only via FakeContext)."""

from gap.testing import FakeContext


def test_run_vla_loop_forwards_actions(skills_registry):
    info = skills_registry.get("tsh-pi05")
    script = info.canonical_scripts["run_vla"].module

    ctx = FakeContext({
        "robot.get_observation": {"cameras": [], "arms": []},
        "sim.apply_policy_action": None,
    })
    out = script.run(ctx, prompt="hand over the tape", max_steps=8, replan_every=5)

    assert out["status"] == "completed"
    assert out["num_steps"] == 8
    # max_steps actions forwarded; replanned (re-observed) every replan_every rows.
    assert ctx.call_count("sim.apply_policy_action") == 8
    assert ctx.call_count("robot.get_observation") == 2  # ceil(8 / 5)
    # Stub holds position: each action is a 16-D zero vector.
    first = ctx.calls_to("sim.apply_policy_action")[0]
    assert len(first.kwargs["action"]) == 16
