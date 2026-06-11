"""Tests for Checkpoint / evaluate_checkpoint / load_checkpoints."""

from __future__ import annotations

import textwrap

import numpy as np
import pytest

from gap.runtime.verify import (
    Body,
    Checkpoint,
    CheckpointResult,
    StubWorld,
    World,
    evaluate_checkpoint,
    load_checkpoints,
)


def _make_body(name, *, pos=(0.0, 0.0, 0.0), half_extents=(0.05, 0.05, 0.05),
               contacts=frozenset()):
    pos_a = np.asarray(pos, dtype=np.float64)
    he_a = np.asarray(half_extents, dtype=np.float64)
    return Body(
        name=str(name),
        position=pos_a,
        quaternion_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
        aabb_lower=pos_a - he_a,
        aabb_upper=pos_a + he_a,
        linear_velocity=np.zeros(3, dtype=np.float64),
        angular_velocity=np.zeros(3, dtype=np.float64),
        contacts=frozenset(contacts),
    )


def _fixture_world() -> World:
    return World(
        env_id=0,
        bodies={
            "can": _make_body("can", pos=(0.4, 0.0, 0.5),
                              contacts={"panda_finger_1"}),
            "table_top": _make_body("table_top", half_extents=(0.5, 0.5, 0.005)),
        },
    )


# ---------------------------------------------------------------------------
# Arity dispatch
# ---------------------------------------------------------------------------


def test_arity_1_predicate():
    cp = Checkpoint(
        name="grasped", subgraph="grasp_sg",
        predicate=lambda w: w.body("can").is_grasped(),
        rationale="can must be held",
    )
    result = evaluate_checkpoint(cp, _fixture_world())
    assert isinstance(result, CheckpointResult)
    assert result.passed is True
    assert result.eval_error is None
    assert result.eval_time_s >= 0.0


def test_arity_2_predicate_receives_outputs():
    cp = Checkpoint(
        name="output_match", subgraph="perceive_sg",
        predicate=lambda w, outputs: outputs.get("target") == "can"
        and w.has_body(outputs["target"]),
    )
    result = evaluate_checkpoint(cp, _fixture_world(), outputs={"target": "can"})
    assert result.passed is True
    # Without the expected output the same predicate fails.
    assert evaluate_checkpoint(cp, _fixture_world(), outputs={}).passed is False


def test_arity_1_predicate_ignores_outputs():
    cp = Checkpoint(
        name="one_arg", subgraph="sg",
        predicate=lambda w: True,
    )
    result = evaluate_checkpoint(cp, _fixture_world(), outputs={"unused": 1})
    assert result.passed is True


# ---------------------------------------------------------------------------
# Eval-error capture
# ---------------------------------------------------------------------------


def test_raising_predicate_becomes_failure_with_error():
    cp = Checkpoint(
        name="raises", subgraph="sg",
        predicate=lambda w: w.body("missing_body").is_grasped(),
    )
    result = evaluate_checkpoint(cp, _fixture_world())
    assert result.passed is False
    assert result.eval_error is not None
    assert "BodyNotFoundError" in result.eval_error


def test_raising_predicate_does_not_propagate():
    def boom(w):
        raise RuntimeError("kaboom")

    cp = Checkpoint(name="boom", subgraph="sg", predicate=boom)
    result = evaluate_checkpoint(cp, _fixture_world())  # must not raise
    assert result.passed is False
    assert "RuntimeError: kaboom" in result.eval_error


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


def test_diagnostics_fn_output_surfaces():
    cp = Checkpoint(
        name="d", subgraph="sg",
        predicate=lambda w: True,
        diagnostics_fn=lambda w: {"can_z": w.body("can").z},
    )
    result = evaluate_checkpoint(cp, _fixture_world())
    assert result.diagnostics == {"can_z": 0.5}


def test_diagnostics_fn_two_arg():
    cp = Checkpoint(
        name="d2", subgraph="sg",
        predicate=lambda w: True,
        diagnostics_fn=lambda w, outputs: {"echo": outputs.get("k")},
    )
    result = evaluate_checkpoint(cp, _fixture_world(), outputs={"k": 7})
    assert result.diagnostics == {"echo": 7}


def test_raising_diagnostics_does_not_fail_checkpoint():
    def bad(w):
        raise RuntimeError("oops")

    cp = Checkpoint(
        name="d", subgraph="sg",
        predicate=lambda w: True,
        diagnostics_fn=bad,
    )
    result = evaluate_checkpoint(cp, _fixture_world())
    assert result.passed is True
    assert result.diagnostics is None


def test_non_dict_diagnostics_dropped():
    cp = Checkpoint(
        name="d", subgraph="sg",
        predicate=lambda w: True,
        diagnostics_fn=lambda w: ["not", "a", "dict"],
    )
    assert evaluate_checkpoint(cp, _fixture_world()).diagnostics is None


# ---------------------------------------------------------------------------
# validate flag
# ---------------------------------------------------------------------------


def test_validate_flag_defaults_true_and_carries_through():
    hard = Checkpoint(name="hard", subgraph="sg", predicate=lambda w: True)
    probe = Checkpoint(name="probe", subgraph="sg", predicate=lambda w: True,
                       validate=False)
    assert hard.validate is True
    assert probe.validate is False
    # Probes still evaluate normally; enforcement policy is the caller's job.
    assert evaluate_checkpoint(probe, _fixture_world()).passed is True


# ---------------------------------------------------------------------------
# load_checkpoints (sidecar .py exec)
# ---------------------------------------------------------------------------


_SIDECAR = textwrap.dedent(
    """
    from gap.runtime.verify import Checkpoint

    def _grasped(world):
        return world.body("can").is_grasped()

    def _on_table(world, outputs):
        return world.body(outputs.get("target", "can")).is_on(
            world.body("table_top"), tol_m=1.0,
        )

    CHECKPOINTS = [
        Checkpoint(
            name="can_grasped", subgraph="grasp_sg", predicate=_grasped,
            rationale="gripper closed on the can",
        ),
        Checkpoint(
            name="target_on_table", subgraph="place_sg", predicate=_on_table,
            validate=False,
        ),
    ]
    """
)


def test_load_checkpoints_and_evaluate(tmp_path):
    mod = tmp_path / "grasp_sg.py"
    mod.write_text(_SIDECAR)
    cps = load_checkpoints(mod)
    assert [c.name for c in cps] == ["can_grasped", "target_on_table"]
    assert cps[0].validate is True
    assert cps[1].validate is False

    world = _fixture_world()
    r0 = evaluate_checkpoint(cps[0], world)
    assert r0.passed is True and r0.eval_error is None
    # Arity-2 sidecar predicate gets the outputs dict.
    r1 = evaluate_checkpoint(cps[1], world, outputs={"target": "can"})
    assert r1.passed is True


def test_load_checkpoints_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_checkpoints(tmp_path / "nope.py")


def test_load_checkpoints_requires_checkpoints_list(tmp_path):
    mod = tmp_path / "bad.py"
    mod.write_text("CHECKPOINTS = 'not a list'\n")
    with pytest.raises(ValueError):
        load_checkpoints(mod)


def test_load_checkpoints_rejects_non_checkpoint_entries(tmp_path):
    mod = tmp_path / "bad_entries.py"
    mod.write_text("CHECKPOINTS = [42]\n")
    with pytest.raises(ValueError):
        load_checkpoints(mod)


def test_load_checkpoints_exec_is_isolated(tmp_path):
    """The sidecar module executes in its own namespace, not this test's."""
    mod = tmp_path / "isolated.py"
    mod.write_text(_SIDECAR + "\nLEAKY_GLOBAL = 1\n")
    cps = load_checkpoints(mod)
    assert len(cps) == 2
    assert "LEAKY_GLOBAL" not in globals()


def test_loaded_checkpoint_works_against_stub_world():
    """Sidecar checkpoints can be smoke-validated against a StubWorld."""
    world = StubWorld(body_names=["can", "table_top"], robot_body="robot")
    cp = Checkpoint(
        name="exists", subgraph="sg",
        predicate=lambda w: w.has_body("can"),
    )
    assert evaluate_checkpoint(cp, world).passed is True
