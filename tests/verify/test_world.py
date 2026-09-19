"""Tests for gap.runtime.verify.world (World/Body fixtures built by hand)."""

from __future__ import annotations

import numpy as np
import pytest

from gap.runtime.verify import (
    Articulation,
    Body,
    BodyNotFoundError,
    Robot,
    StubWorld,
    World,
    contacts_from_pairs,
)


def _make_body(
    name: str,
    *,
    pos=(0.0, 0.0, 0.0),
    half_extents=(0.05, 0.05, 0.05),
    contacts=frozenset(),
    cavity_lower=None,
    cavity_upper=None,
    lin_vel=(0.0, 0.0, 0.0),
    is_region=False,
):
    pos_a = np.asarray(pos, dtype=np.float64)
    he_a = np.asarray(half_extents, dtype=np.float64)
    return Body(
        name=str(name),
        position=pos_a,
        quaternion_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
        aabb_lower=pos_a - he_a,
        aabb_upper=pos_a + he_a,
        linear_velocity=np.asarray(lin_vel, dtype=np.float64),
        angular_velocity=np.zeros(3, dtype=np.float64),
        contacts=frozenset(contacts),
        cavity_lower=np.asarray(cavity_lower, dtype=np.float64) if cavity_lower is not None else None,
        cavity_upper=np.asarray(cavity_upper, dtype=np.float64) if cavity_upper is not None else None,
        is_region=is_region,
    )


# ---------------------------------------------------------------------------
# Construction from plain numpy data
# ---------------------------------------------------------------------------


def test_world_constructible_from_plain_data():
    can = _make_body("can", pos=(0.4, 0.0, 0.04))
    world = World(env_id=0, bodies={"can": can})
    assert world.body("can").name == "can"
    assert world.body_names() == ["can"]


def test_body_not_found_carries_available_names():
    world = World(
        env_id=0,
        bodies={"a": _make_body("a"), "b": _make_body("b")},
    )
    with pytest.raises(BodyNotFoundError) as exc_info:
        world.body("zzz")
    assert exc_info.value.name == "zzz"
    assert "a" in exc_info.value.available
    assert "b" in exc_info.value.available


# ---------------------------------------------------------------------------
# is_grasped via contacts containing a gripper-link name
# ---------------------------------------------------------------------------


def test_is_grasped_via_gripper_link_contact():
    can = _make_body("can", contacts={"panda_finger_1", "table_top"})
    world = World(env_id=0, bodies={"can": can})
    assert world.body("can").is_grasped()


def test_is_grasped_uses_world_prefixes():
    can = _make_body("can", contacts={"Robotiq_finger_left"})
    world = World(env_id=0, bodies={"can": can})
    assert world.body("can").is_grasped()


def test_is_grasped_negative_when_only_scene_contacts():
    can = _make_body("can", contacts={"basket", "table_top"})
    world = World(env_id=0, bodies={"can": can})
    assert not world.body("can").is_grasped()


def test_is_grasped_custom_prefixes():
    can = _make_body("can", contacts={"ur5_gripper_pad"})
    world = World(
        env_id=0, bodies={"can": can}, robot_link_prefixes=("ur5_",),
        robot_body_name="ur5",
    )
    assert world.body("can").is_grasped()


def test_is_grasped_by_explicit_prefix():
    can = _make_body("can", contacts={"panda_finger_1", "table_top"})
    assert can.is_grasped_by("panda_")
    assert not can.is_grasped_by("ur5_")


# ---------------------------------------------------------------------------
# is_in with cavity bounds
# ---------------------------------------------------------------------------


def test_is_in_uses_cavity_when_available():
    basket = _make_body(
        "basket", pos=(0.0, 0.0, 0.0),
        half_extents=(0.06, 0.06, 0.05),
        cavity_lower=(-0.05, -0.05, -0.045),
        cavity_upper=(0.05, 0.05, 0.045),
    )
    can = _make_body("can", pos=(0.0, 0.0, 0.0), half_extents=(0.02, 0.02, 0.02),
                     contacts={"basket"})
    assert can.is_in(basket, require_contact=True)


def test_is_in_requires_contact_by_default():
    basket = _make_body(
        "basket",
        half_extents=(0.06, 0.06, 0.05),
        cavity_lower=(-0.05, -0.05, -0.045),
        cavity_upper=(0.05, 0.05, 0.045),
    )
    can = _make_body("can", contacts=frozenset())
    assert not can.is_in(basket, require_contact=True)
    assert can.is_in(basket, require_contact=False)


def test_is_in_outside_cavity_fails():
    basket = _make_body(
        "basket",
        half_extents=(0.06, 0.06, 0.05),
        cavity_lower=(-0.05, -0.05, -0.045),
        cavity_upper=(0.05, 0.05, 0.045),
    )
    # 10 cm out in y: beyond the cavity edge + default 2 cm slack.
    can = _make_body("can", pos=(0.0, 0.15, 0.0), contacts={"basket"})
    assert not can.is_in(basket)


def test_is_in_falls_back_to_union_aabb():
    basket = _make_body("basket", half_extents=(0.06, 0.06, 0.05))
    can = _make_body("can", contacts={"basket"})
    assert can.is_in(basket, require_contact=True)


def test_is_in_region_skips_contact_requirement():
    region = _make_body(
        "drop_zone",
        half_extents=(0.06, 0.06, 0.05),
        cavity_lower=(-0.05, -0.05, -0.045),
        cavity_upper=(0.05, 0.05, 0.045),
        is_region=True,
    )
    can = _make_body("can", contacts=frozenset())
    assert can.is_in(region, require_contact=True)


# ---------------------------------------------------------------------------
# is_on / is_above AABB relations
# ---------------------------------------------------------------------------


def test_is_on_true_for_block_resting_on_table():
    table = _make_body("table", pos=(0.0, 0.0, 0.0), half_extents=(0.5, 0.5, 0.005))
    block = _make_body("block", pos=(0.1, 0.1, 0.05), half_extents=(0.03, 0.03, 0.04))
    # block.bottom_z = 0.01, table.top_z = 0.005, dz = 0.005, within tol=0.05.
    assert block.is_on(table)


def test_is_on_false_when_lifted_too_high():
    table = _make_body("table", half_extents=(0.5, 0.5, 0.005))
    block = _make_body("block", pos=(0.0, 0.0, 0.20), half_extents=(0.03, 0.03, 0.04))
    assert not block.is_on(table, tol_m=0.05)


def test_is_on_false_without_xy_overlap():
    table = _make_body("table", half_extents=(0.1, 0.1, 0.005))
    block = _make_body("block", pos=(0.5, 0.5, 0.045), half_extents=(0.03, 0.03, 0.04))
    assert not block.is_on(table)


def test_is_above_with_clearance():
    a = _make_body("a", half_extents=(0.05, 0.05, 0.02))
    b = _make_body("b", pos=(0.0, 0.0, 0.20), half_extents=(0.05, 0.05, 0.02))
    # b above a, clearance ~0.16.
    assert b.is_above(a, min_clearance_m=0.10)
    assert not a.is_above(b)


def test_is_above_requires_xy_overlap_by_default():
    a = _make_body("a", half_extents=(0.05, 0.05, 0.02))
    b = _make_body("b", pos=(1.0, 1.0, 0.20), half_extents=(0.05, 0.05, 0.02))
    assert not b.is_above(a)
    assert b.is_above(a, require_xy_overlap=False)


def test_distance_helpers():
    a = _make_body("a", pos=(0.0, 0.0, 0.0))
    b = _make_body("b", pos=(0.3, 0.4, 1.0))
    assert a.xy_distance_to(b) == pytest.approx(0.5)
    assert a.distance_to(b) == pytest.approx(np.sqrt(0.25 + 1.0))


def test_is_settled_uses_speed_thresh():
    fast = _make_body("a", lin_vel=(0.5, 0.0, 0.0))
    slow = _make_body("b", lin_vel=(0.01, 0.0, 0.0))
    assert not fast.is_settled()
    assert slow.is_settled()
    assert fast.is_settled(speed_thresh=1.0)


# ---------------------------------------------------------------------------
# eventually / always over a 3-snapshot history
# ---------------------------------------------------------------------------


def _three_snapshot_world():
    """t0: can on table; t1: can grasped + lifted; t2: can in basket."""
    basket = _make_body(
        "basket", pos=(0.6, 0.2, 0.06),
        half_extents=(0.07, 0.07, 0.06),
        cavity_lower=(0.54, 0.14, 0.005),
        cavity_upper=(0.66, 0.26, 0.115),
    )
    w0 = World(env_id=0, time_s=0.0, bodies={
        "can": _make_body("can", pos=(0.4, 0.0, 0.04), contacts={"table_top"}),
        "basket": basket,
    })
    w1 = World(env_id=0, time_s=1.0, bodies={
        "can": _make_body("can", pos=(0.4, 0.0, 0.25), contacts={"panda_finger_1"}),
        "basket": basket,
    })
    w2 = World(env_id=0, time_s=2.0, bodies={
        "can": _make_body("can", pos=(0.6, 0.2, 0.06), contacts={"basket"}),
        "basket": basket,
    })
    return World.from_history([w0, w1, w2])


def test_eventually_over_history():
    world = _three_snapshot_world()
    # Grasped only in the middle snapshot — eventually sees it.
    assert world.eventually(lambda w: w.body("can").is_grasped())
    # Final state: can is in the basket but never on the moon.
    assert world.eventually(lambda w: w.body("can").is_in(w.body("basket")))
    assert not world.eventually(lambda w: w.body("can").z > 1.0)


def test_always_over_history():
    world = _three_snapshot_world()
    assert world.always(lambda w: w.has_body("can"))
    # Grasped only at t1, so "always grasped" is False.
    assert not world.always(lambda w: w.body("can").is_grasped())


def test_history_is_chronological():
    world = _three_snapshot_world()
    h = world.history()
    assert len(h) == 2
    assert [w.time_s for w in h] == [0.0, 1.0]
    assert world.time_s == 2.0


def test_at_end_evaluates_self_only():
    world = _three_snapshot_world()
    # At the end the can is released into the basket.
    assert world.at_end(lambda w: not w.body("can").is_grasped())


# ---------------------------------------------------------------------------
# held_body
# ---------------------------------------------------------------------------


def test_held_body_returns_grasped_object():
    can = _make_body("can", pos=(0.4, 0.0, 0.5), contacts={"panda_finger_1"})
    block = _make_body("block", pos=(0.7, 0.0, 0.04))
    world = World(
        env_id=0,
        bodies={"can": can, "block": block, "robot": _make_body("robot")},
    )
    held = world.held_body()
    assert held is not None and held.name == "can"


def test_held_body_returns_none_when_gripper_empty():
    can = _make_body("can", pos=(0.4, 0.0, 0.04), contacts=frozenset())
    world = World(env_id=0, bodies={"can": can})
    assert world.held_body() is None


def test_held_body_tie_breaks_by_ee_distance():
    near = _make_body("near", pos=(0.4, 0.0, 0.5), contacts={"panda_finger_1"})
    far = _make_body("far", pos=(0.9, 0.0, 0.5), contacts={"panda_finger_2"})
    robot_view = Robot(
        body_name="robot",
        joint_pos=np.zeros(7),
        joint_names=tuple(f"panda_joint{i + 1}" for i in range(7)),
        ee_position=np.array([0.4, 0.0, 0.5]),
        ee_quaternion_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
        gripper_open_fraction=0.0,
    )
    world = World(
        env_id=0,
        bodies={"near": near, "far": far},
        robot_view=robot_view,
    )
    held = world.held_body()
    assert held is not None and held.name == "near"


# ---------------------------------------------------------------------------
# Contact-pair canonicalization
# ---------------------------------------------------------------------------


def test_contacts_from_pairs_is_symmetric():
    pairs = [("can", "basket"), ("can", "panda_finger_1")]
    by_body = contacts_from_pairs(pairs)
    assert by_body["can"] == frozenset({"basket", "panda_finger_1"})
    assert by_body["basket"] == frozenset({"can"})
    assert by_body["panda_finger_1"] == frozenset({"can"})
    assert by_body.get("table_top", frozenset()) == frozenset()


def test_contacts_from_pairs_none_means_empty():
    assert contacts_from_pairs(None) == {}


# ---------------------------------------------------------------------------
# StubWorld sanity (validation dry-run helper)
# ---------------------------------------------------------------------------


def test_stub_world_has_bodies_robot_and_cavities():
    world = StubWorld(
        body_names=["table_top", "soup_can", "basket", "robot"],
        container_names=["basket"],
        robot_body="robot",
    )
    assert world.has_body("soup_can")
    assert world.robot().body_name == "robot"
    basket = world.body("basket")
    assert basket.cavity_lower is not None and basket.cavity_upper is not None
    assert world.body("soup_can").cavity_lower is None


def test_world_exposes_named_articulations():
    hinge = Articulation(
        name="lid_hinge", parent="machine", child="lid", kind="revolute",
        position=0.2, lower=0.0, upper=2.1, progress=0.2 / 2.1,
        axis=np.array([0.0, 1.0, 0.0]), pivot=np.array([0.1, 0.0, 0.2]),
    )
    world = World(env_id=0, bodies={}, articulations={hinge.name: hinge})
    assert world.articulation_names() == ["lid_hinge"]
    assert world.has_articulation("lid_hinge")
    assert world.articulation("lid_hinge").position == pytest.approx(0.2)
    with pytest.raises(KeyError, match="available"):
        world.articulation("missing")
