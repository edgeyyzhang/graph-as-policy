"""Unit tests for gap.tools.guards — tag classification and call limits."""

import pytest

from gap_core.errors import GuardLimitExceeded
from gap_core.tools import guards

_ENV_VARS = (
    "GAP_MAX_PERCEPTION_CALLS",
    "GAP_MAX_PLANNING_CALLS",
    "GAP_MAX_SIM_STEPS",
)


@pytest.fixture(autouse=True)
def _clean_guards(monkeypatch):
    """Clear env limits, programmatic limits, and counters around each test."""
    for var in _ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    guards.set_limits()
    guards.reset_counters()
    yield
    guards.set_limits()
    guards.reset_counters()


def test_classify_tags():
    assert guards.classify_tags(("perception",)) is guards.CallCategory.PERCEPTION
    assert guards.classify_tags(("planning",)) is guards.CallCategory.PLANNING
    assert guards.classify_tags(("sim_step",)) is guards.CallCategory.SIM_STEP
    # First rate-limited tag wins; non-guard tags are ignored.
    assert guards.classify_tags(("geometry", "planning")) is guards.CallCategory.PLANNING
    assert guards.classify_tags(("geometry", "evaluation")) is None
    assert guards.classify_tags(()) is None


def test_unlimited_when_unconfigured():
    for _ in range(50):
        guards.check_and_increment(guards.CallCategory.PERCEPTION)


def test_env_var_fallback(monkeypatch):
    monkeypatch.setenv("GAP_MAX_PERCEPTION_CALLS", "2")
    guards.check_and_increment_if_applicable(("perception",))
    guards.check_and_increment_if_applicable(("perception",))
    with pytest.raises(GuardLimitExceeded):
        guards.check_and_increment_if_applicable(("perception",))


def test_unparseable_env_var_means_unlimited(monkeypatch):
    monkeypatch.setenv("GAP_MAX_SIM_STEPS", "not-a-number")
    for _ in range(10):
        guards.check_and_increment(guards.CallCategory.SIM_STEP)


def test_guard_escapes_except_exception(monkeypatch):
    monkeypatch.setenv("GAP_MAX_SIM_STEPS", "0")
    with pytest.raises(GuardLimitExceeded):
        try:
            guards.check_and_increment(guards.CallCategory.SIM_STEP)
        except Exception:  # noqa: BLE001 — the point: this must NOT catch it
            pytest.fail("GuardLimitExceeded must escape `except Exception`")


def test_set_limits_overrides_env(monkeypatch):
    monkeypatch.setenv("GAP_MAX_PLANNING_CALLS", "100")
    guards.set_limits(planning=1)
    guards.check_and_increment(guards.CallCategory.PLANNING)
    with pytest.raises(GuardLimitExceeded):
        guards.check_and_increment(guards.CallCategory.PLANNING)


def test_set_limits_clears_previous_overrides():
    guards.set_limits(perception=1)
    guards.set_limits()  # clear: back to env (unset) -> unlimited
    for _ in range(5):
        guards.check_and_increment(guards.CallCategory.PERCEPTION)


def test_reset_counters():
    guards.set_limits(perception=1)
    guards.check_and_increment(guards.CallCategory.PERCEPTION)
    guards.reset_counters()
    # Fresh budget after reset; the second post-reset call exceeds again.
    guards.check_and_increment(guards.CallCategory.PERCEPTION)
    with pytest.raises(GuardLimitExceeded):
        guards.check_and_increment(guards.CallCategory.PERCEPTION)


def test_categories_count_independently(monkeypatch):
    monkeypatch.setenv("GAP_MAX_PERCEPTION_CALLS", "1")
    guards.check_and_increment(guards.CallCategory.PERCEPTION)
    guards.check_and_increment(guards.CallCategory.PLANNING)  # unlimited
    guards.check_and_increment(guards.CallCategory.SIM_STEP)  # unlimited
    with pytest.raises(GuardLimitExceeded):
        guards.check_and_increment(guards.CallCategory.PERCEPTION)
