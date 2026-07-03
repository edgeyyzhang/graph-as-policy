"""Object-state success criteria for the zero-shot prompt matrix.

Each criterion evaluates the recorded state trace (``state.jsonl`` dicts
from ``state_log.StateRecorder``) — never the LLM's own claims and never
the env's built-in task reward (which measures a different task and is
recorded as advisory metadata only).

Conventions — every constant below is measured from the live scene or
mirrors an engine default, with the source noted:

- Frame: robot base == MuJoCo world in this env (probed: robot0_base at
  the origin, identity rotation). Tabletop surface z ~= 0 (resting
  object bottoms probed at -0.005..0).
- Containment: the basket registers NO interior cavity on LIBERO
  (probed), so "inside" is center-in-outer-AABB with the same +/-2 cm
  slack as ``gap.runtime.verify.world.Body.contains`` defaults.
- Settled: linear speed < 0.08 m/s (``Body.is_settled`` default).
- Grasp-and-lift ("picked up"): the engine's ``World.held_body()``
  reports this object, the gripper is < 0.9 open, and the object's
  center is >= 3 cm above its initial height. Requiring the lift
  rejects nudges (contact without pickup) and means an object that
  teleports into the basket without ever being carried does NOT count
  as grasped.
- "Right" (two_inches_right): the agentview camera's image-right axis
  is +y in the base frame (measured from cam_xmat; the camera sits at
  world (0.90, 0, 0.65) looking back toward the robot). The ROBOT's
  right is -y. Both readings of "right" are accepted; the direction
  actually taken is logged.
- Upright: the body-local direction that pointed world-up at t0 must be
  within 30 degrees of world-up at the end. (Asset-agnostic: the soup
  can's local z is horizontal at rest — probed — so comparing local z
  to vertical would be wrong.)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

BASKET = "basket"
#: Movables in the libero_object_all_variance scenes (probed). Other
#: scenes derive their movables from the trace via :func:`scene_objects`
#: using the case's ``container``/``statics`` vocabulary.
OBJECTS = (
    "alphabet_soup",
    "butter",
    "cream_cheese",
    "milk",
    "salad_dressing",
    "tomato_sauce",
)
#: Bodies never treated as task movables in any scene.
ALWAYS_STATIC = ("table",)

#: The VAB packing suites' pack_all_into success predicate TELEPORTS a
#: contained object to a far-away graveyard (measured: (50.6, 50.0)) the
#: moment it registers as packed — often between two of our snapshots.
#: Positions beyond this bound mark an object the ENV consumed, not one
#: the robot flung. Only honored when the case sets env_consumes_packed.
GRAVEYARD_ABS = 5.0

CONTAIN_TOL_XY = 0.02      # Body.contains default tol_xy_m
CONTAIN_TOL_Z = 0.02       # Body.contains default tol_z_m
SETTLED_SPEED = 0.08       # Body.is_settled default
GRIP_CLOSED_MAX = 0.9      # gripper open-fraction while actually holding
LIFT_MIN = 0.03            # center must rise >= 3 cm to count as picked up
ON_TABLE_MAX_BOTTOM_Z = 0.03

TWO_INCHES = 0.0508
ALONG_TOL = 0.03           # |dy| within 5.1 +/- 3 cm
PERP_TOL = 0.05            # |dx| <= 5 cm
UPRIGHT_MAX_TILT_DEG = 30.0

BASKET_MIN_DISP = 0.20
BASKET_LATERAL_FLIP_MIN_DY = 0.20
BASKET_DEPTH_CROSS_MIN_DX = 0.30

# Workspace sanity envelope (generous around the probed layout:
# objects spawn x in [0.39, 0.75], |y| <= 0.26, z >= -0.01).
SANE_MIN_Z = -0.15
SANE_MAX_ABS_Y = 0.70
SANE_X = (-0.10, 1.25)


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------


@dataclass
class Check:
    name: str
    passed: bool
    score: float
    detail: str
    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "passed": bool(self.passed),
            "score": round(float(self.score), 4),
            "detail": self.detail,
            "data": self.data,
        }


@dataclass
class SeedEvaluation:
    verdict: str            # PASS | FAIL | TIMEOUT | CRASH | NO_TRACE
    score: float
    summary: str
    checks: list[Check]
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict,
            "score": round(float(self.score), 4),
            "summary": self.summary,
            "checks": [c.to_dict() for c in self.checks],
            "diagnostics": self.diagnostics,
        }


# ---------------------------------------------------------------------------
# State-trace helpers (operate on state.jsonl dicts)
# ---------------------------------------------------------------------------


def _body(s: dict, name: str) -> dict | None:
    return (s.get("bodies") or {}).get(name)


def _pos(s: dict, name: str) -> np.ndarray | None:
    b = _body(s, name)
    return None if b is None else np.asarray(b["p"], dtype=float)


def in_basket(s: dict, name: str, container: str = BASKET) -> bool:
    """Center-in-container-AABB with Body.contains' +/-2 cm slack."""
    b, k = _body(s, name), _body(s, container)
    if b is None or k is None:
        return False
    p = b["p"]
    lo, hi = k["lo"], k["hi"]
    return (
        lo[0] - CONTAIN_TOL_XY <= p[0] <= hi[0] + CONTAIN_TOL_XY
        and lo[1] - CONTAIN_TOL_XY <= p[1] <= hi[1] + CONTAIN_TOL_XY
        and lo[2] - CONTAIN_TOL_Z <= p[2] <= hi[2] + CONTAIN_TOL_Z
    )


def resting_in_basket(s: dict, name: str, container: str = BASKET) -> bool:
    b = _body(s, name)
    return (
        b is not None
        and in_basket(s, name, container)
        and s.get("held") != name
        and float(b.get("v", 0.0)) < SETTLED_SPEED
    )


def on_table(s: dict, name: str, container: str = BASKET) -> bool:
    b = _body(s, name)
    return (
        b is not None
        and float(b["lo"][2]) <= ON_TABLE_MAX_BOTTOM_Z
        and not in_basket(s, name, container)
    )


def resting_on(s: dict, name: str, support: str,
               max_above: float = 0.035) -> bool:
    """``name`` resting on top of ``support``: bottom within
    [-1.5 cm, +max_above] of the support's top (mirrors
    ``Body.is_on_strict``'s asymmetric window, widened for AABB noise),
    xy centroid inside the support's xy AABB (+2 cm), released and
    settled. ``max_above`` is raised by callers that accept nesting
    (bowls stacked into each other still count as 'on the plate')."""
    b, sup = _body(s, name), _body(s, support)
    if b is None or sup is None:
        return False
    dz = float(b["lo"][2]) - float(sup["hi"][2])
    p = b["p"]
    lo, hi = sup["lo"], sup["hi"]
    inside_xy = (
        lo[0] - CONTAIN_TOL_XY <= p[0] <= hi[0] + CONTAIN_TOL_XY
        and lo[1] - CONTAIN_TOL_XY <= p[1] <= hi[1] + CONTAIN_TOL_XY
    )
    return (
        -0.015 <= dz <= max_above
        and inside_xy
        and s.get("held") != name
        and float(b.get("v", 0.0)) < SETTLED_SPEED
    )


def grasp_lift_index(states: list[dict], name: str) -> int | None:
    """First snapshot where ``name`` is held (engine held_body), the
    gripper is closed on it, and it has risen >= LIFT_MIN above its
    initial height. None = never picked up."""
    z0 = None
    for s in states:
        p = _pos(s, name)
        if p is not None:
            z0 = float(p[2])
            break
    if z0 is None:
        return None
    for i, s in enumerate(states):
        if s.get("held") != name:
            continue
        grip = s.get("grip")
        if grip is not None and float(grip) >= GRIP_CLOSED_MAX:
            continue
        p = _pos(s, name)
        if p is not None and float(p[2]) >= z0 + LIFT_MIN:
            return i
    return None


def first_resting_in_basket_index(states: list[dict], name: str,
                                  container: str = BASKET) -> int | None:
    for i, s in enumerate(states):
        if resting_in_basket(s, name, container):
            return i
    return None


def _quat_rotmat(q) -> np.ndarray:
    w, x, y, z = (float(v) for v in np.asarray(q, dtype=float)[:4])
    n = math.sqrt(w * w + x * x + y * y + z * z)
    if n == 0.0:
        return np.eye(3)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def tilt_from_initial_up_deg(q0, q1) -> float:
    """Angle between world-up and the body direction that pointed up at
    t0. 0 = same lean as at start; ~90 = knocked onto its side."""
    up0_local = _quat_rotmat(q0).T @ np.array([0.0, 0.0, 1.0])
    now = _quat_rotmat(q1) @ up0_local
    c = float(np.clip(now[2] / (np.linalg.norm(now) or 1.0), -1.0, 1.0))
    return math.degrees(math.acos(c))


def scene_objects(states: list[dict], case: Any = None) -> list[str]:
    """Task movables observed in the trace: every body minus the case's
    container, its declared statics, and the always-static set."""
    if not states:
        return []
    names: set[str] = set()
    for s in states:
        names |= set((s.get("bodies") or {}).keys())
    container = getattr(case, "container", None) or BASKET
    statics: set[str] = set(getattr(case, "statics", ()) or ()) | set(ALWAYS_STATIC)
    statics.add(container)
    return sorted(n for n in names if n not in statics)


# ---------------------------------------------------------------------------
# Universal checks
# ---------------------------------------------------------------------------


def at_graveyard(p) -> bool:
    return abs(float(p[0])) > GRAVEYARD_ABS or abs(float(p[1])) > GRAVEYARD_ABS


def check_scene_sane(states: list[dict], case: Any = None) -> Check:
    env_consumes = bool(getattr(case, "env_consumes_packed", False))
    violations = []
    for i, s in enumerate(states):
        for name in scene_objects(states, case):
            p = _pos(s, name)
            if p is None:
                continue
            if env_consumes and at_graveyard(p):
                continue  # env teleported a packed object out, not a fling
            x, y, z = (float(v) for v in p)
            if (z < SANE_MIN_Z or abs(y) > SANE_MAX_ABS_Y
                    or not (SANE_X[0] <= x <= SANE_X[1])):
                violations.append(
                    f"{name}@{i}: ({x:.2f},{y:.2f},{z:.2f})")
    ok = not violations
    return Check(
        "scene_sane", ok, 1.0 if ok else 0.0,
        "no object left the workspace" if ok
        else f"{len(violations)} violation(s): {violations[0]}",
        {"violations": violations[:10]},
    )


def _grasped_check(states: list[dict], name: str) -> Check:
    idx = grasp_lift_index(states, name)
    return Check(
        f"{name}_grasped_and_lifted",
        idx is not None,
        1.0 if idx is not None else 0.0,
        f"{name} held+closed-gripper+lifted at snapshot {idx}"
        if idx is not None else
        f"{name} was never picked up (held+lifted never observed)",
    )


def _in_basket_check(states: list[dict], name: str) -> Check:
    final = states[-1]
    ok = resting_in_basket(final, name)
    return Check(
        f"{name}_in_basket_final", ok, 1.0 if ok else 0.0,
        f"{name} {'is' if ok else 'is NOT'} settled inside the basket "
        "at the end",
    )


def _no_stray_check(states: list[dict], allowed: set[str],
                    case: Any = None) -> Check:
    # Only NEWLY-packed strays count: an object already in the basket at
    # snapshot 0 was not packed by the policy.
    container = getattr(case, "container", None) or BASKET
    final = states[-1]
    stray = sorted(
        n for n in scene_objects(states, case)
        if n not in allowed
        and resting_in_basket(final, n, container)
        and not resting_in_basket(states[0], n, container)
    )
    ok = not stray
    return Check(
        "no_unrequested_packing", ok, 1.0 if ok else 0.0,
        "nothing else was packed" if ok
        else f"unrequested object(s) packed: {stray}",
        {"stray": stray},
    )


# ---------------------------------------------------------------------------
# Per-case criteria
# ---------------------------------------------------------------------------


def _pick_place(states: list[dict], target: str,
                case: Any = None) -> SeedEvaluation:
    checks = [
        check_scene_sane(states, case),
        _grasped_check(states, target),
        _in_basket_check(states, target),
        _no_stray_check(states, {target}, case),
    ]
    task = checks[1:]
    passed = all(c.passed for c in checks)
    score = (
        0.3 * task[0].score + 0.5 * task[1].score + 0.2 * task[2].score
    )
    return SeedEvaluation(
        "PASS" if passed else "FAIL", score,
        f"{target}: picked={task[0].passed}, "
        f"in_basket={task[1].passed}, stray_free={task[2].passed}",
        checks,
    )


def eval_soup_basket(states: list[dict], case: Any = None,
                     **_: Any) -> SeedEvaluation:
    return _pick_place(states, "alphabet_soup", case)


def eval_pick_single(states: list[dict], case: Any = None,
                     **_: Any) -> SeedEvaluation:
    """Generic single pick-into-container: target = case.targets[0]."""
    return _pick_place(states, case.targets[0], case)


def eval_cream_cheese(states: list[dict], case: Any = None,
                      **_: Any) -> SeedEvaluation:
    return _pick_place(states, "cream_cheese", case)


def eval_all_objects(states: list[dict], case: Any = None,
                     **_: Any) -> SeedEvaluation:
    objs = scene_objects(states, case)
    final = states[-1]
    packed = [n for n in objs if resting_in_basket(final, n)]
    grasped = [n for n in objs if grasp_lift_index(states, n) is not None]
    n = len(objs)
    frac = len(packed) / max(1, n)
    per_obj = {
        name: {
            "picked_up": name in grasped,
            "grasp_snapshot": grasp_lift_index(states, name),
            "in_basket_final": name in packed,
            "first_in_basket_snapshot":
                first_resting_in_basket_index(states, name),
        }
        for name in objs
    }
    sane = check_scene_sane(states, case)
    all_in = Check(
        "all_objects_in_basket", len(packed) == n, frac,
        f"{len(packed)}/{n} objects settled in basket: {sorted(packed)}",
        per_obj,
    )
    grasp_cov = Check(
        "objects_picked_up", len(grasped) >= len(packed),
        len(grasped) / max(1, n),
        f"{len(grasped)}/{n} objects were picked up: {sorted(grasped)}",
    )
    passed = sane.passed and all_in.passed
    return SeedEvaluation(
        "PASS" if passed else "FAIL", frac,
        all_in.detail, [sane, all_in, grasp_cov],
    )


def eval_soup_then_milk(states: list[dict], case: Any = None,
                        **_: Any) -> SeedEvaluation:
    a, b = "alphabet_soup", "milk"
    sane = check_scene_sane(states, case)
    g_a, g_b = _grasped_check(states, a), _grasped_check(states, b)
    in_a, in_b = _in_basket_check(states, a), _in_basket_check(states, b)
    ia = first_resting_in_basket_index(states, a)
    ib = first_resting_in_basket_index(states, b)
    order_ok = ia is not None and ib is not None and ia < ib
    order = Check(
        "soup_enters_basket_before_milk", order_ok,
        1.0 if order_ok else 0.0,
        f"first settled-in-basket snapshot: soup@{ia}, milk@{ib}",
    )
    checks = [sane, g_a, in_a, g_b, in_b, order]
    passed = all(c.passed for c in checks)
    score = (
        0.15 * g_a.score + 0.25 * in_a.score
        + 0.15 * g_b.score + 0.25 * in_b.score + 0.2 * order.score
    )
    return SeedEvaluation(
        "PASS" if passed else "FAIL", score,
        f"soup packed={in_a.passed}, milk packed={in_b.passed}, "
        f"order_ok={order_ok}",
        checks,
    )


def eval_two_inches_right(states: list[dict], case: Any = None,
                          **_: Any) -> SeedEvaluation:
    target = "alphabet_soup"
    sane = check_scene_sane(states, case)
    grasped = _grasped_check(states, target)
    first, final = states[0], states[-1]
    b0, b1 = _body(first, target), _body(final, target)
    if b0 is None or b1 is None:
        return SeedEvaluation(
            "FAIL", 0.0, f"{target} missing from trace", [sane])
    dx = float(b1["p"][0]) - float(b0["p"][0])
    dy = float(b1["p"][1]) - float(b0["p"][1])
    # 'right' accepted as either y direction (viewer's +y / robot's -y);
    # gate on magnitude along y and small x drift.
    placement_ok = abs(abs(dy) - TWO_INCHES) <= ALONG_TOL and abs(dx) <= PERP_TOL
    direction = (
        "+y (image-right in agentview video)" if dy > 0
        else "-y (robot's right)"
    )
    placement = Check(
        "displaced_two_inches_along_y", placement_ok,
        1.0 if placement_ok else 0.0,
        f"dy={dy * 100:+.1f} cm (target |dy|={TWO_INCHES * 100:.1f}"
        f" +/- {ALONG_TOL * 100:.0f}), dx={dx * 100:+.1f} cm"
        f" (|dx| <= {PERP_TOL * 100:.0f}); moved toward {direction}",
        {"dx_m": round(dx, 4), "dy_m": round(dy, 4)},
    )
    tilt = tilt_from_initial_up_deg(b0["q"], b1["q"])
    upright = Check(
        "still_upright", tilt <= UPRIGHT_MAX_TILT_DEG,
        1.0 if tilt <= UPRIGHT_MAX_TILT_DEG else 0.0,
        f"initially-up axis tilted {tilt:.1f} deg from vertical at the "
        f"end ({'upright' if tilt <= UPRIGHT_MAX_TILT_DEG else 'KNOCKED OVER'})",
        {"tilt_deg": round(tilt, 1)},
    )
    rest_ok = (
        on_table(final, target)
        and final.get("held") != target
        and float(b1.get("v", 0.0)) < SETTLED_SPEED
    )
    resting = Check(
        "released_on_table", rest_ok, 1.0 if rest_ok else 0.0,
        "released and settled on the table, outside the basket" if rest_ok
        else "not resting on the table (still held, lifted, or in basket)",
    )
    checks = [sane, grasped, placement, upright, resting]
    passed = all(c.passed for c in checks)
    score = (
        0.25 * grasped.score + 0.35 * placement.score
        + 0.25 * upright.score + 0.15 * resting.score
    )
    return SeedEvaluation(
        "PASS" if passed else "FAIL", score,
        f"dy={dy * 100:+.1f} cm, tilt={tilt:.0f} deg, "
        f"picked={grasped.passed}, on_table={rest_ok}",
        checks,
    )


def eval_move_basket_ranch(states: list[dict], case: Any = None,
                           **_: Any) -> SeedEvaluation:
    target = "salad_dressing"
    sane = check_scene_sane(states, case)
    first, final = states[0], states[-1]
    k0, k1 = _body(first, BASKET), _body(final, BASKET)
    if k0 is None or k1 is None:
        return SeedEvaluation("FAIL", 0.0, "basket missing from trace", [sane])
    d = np.asarray(k1["p"][:2]) - np.asarray(k0["p"][:2])
    disp = float(np.linalg.norm(d))
    y0, y1 = float(k0["p"][1]), float(k1["p"][1])
    lateral_flip = (y0 * y1 < 0) and abs(y1 - y0) >= BASKET_LATERAL_FLIP_MIN_DY
    depth_cross = abs(float(d[0])) >= BASKET_DEPTH_CROSS_MIN_DX
    moved_ok = disp >= BASKET_MIN_DISP and (lateral_flip or depth_cross)
    moved = Check(
        "basket_moved_to_opposite_side", moved_ok,
        1.0 if moved_ok else min(1.0, disp / BASKET_MIN_DISP) * 0.5,
        f"basket displaced {disp * 100:.1f} cm, y {y0:+.2f} -> {y1:+.2f} "
        f"(lateral_flip={lateral_flip}, depth_cross={depth_cross})",
        {"displacement_m": round(disp, 4)},
    )
    grasped = _grasped_check(states, target)
    packed = _in_basket_check(states, target)
    checks = [sane, moved, grasped, packed]
    passed = all(c.passed for c in checks)
    score = 0.45 * moved.score + 0.2 * grasped.score + 0.35 * packed.score
    return SeedEvaluation(
        "PASS" if passed else "FAIL", score,
        f"basket moved {disp * 100:.0f} cm (opposite={moved_ok}), "
        f"ranch picked={grasped.passed}, ranch in basket={packed.passed}",
        checks,
    )


def eval_pack_scene(states: list[dict], case: Any = None,
                    **_: Any) -> SeedEvaluation:
    """Pack-all on the VAB packing suites, whose success predicate
    TELEPORTS each contained object to a graveyard (measured (50.6,
    50.0)) the instant it registers as packed — usually between two of
    our snapshots, so a final-state containment check reads an empty
    basket on a perfect run. An object counts as packed here when it
    was genuinely picked up (grasp-and-lift event) AND either was
    observed settled in the basket or ended at the graveyard (consumed
    by the env's own packed-marker)."""
    objs = scene_objects(states, case)
    container = case.container or BASKET
    final = states[-1]
    per = {}
    packed = []
    grasped = []
    for n in objs:
        gi = grasp_lift_index(states, n)
        if gi is not None:
            grasped.append(n)
        ever_in = first_resting_in_basket_index(states, n, container) is not None
        consumed = at_graveyard(final["bodies"][n]["p"]) \
            if n in final.get("bodies", {}) else False
        ok = gi is not None and (ever_in or consumed)
        if ok:
            packed.append(n)
        per[n] = {"picked_up": gi is not None, "grasp_snapshot": gi,
                  "seen_in_basket": ever_in, "env_consumed": consumed}
    sane = check_scene_sane(states, case)
    frac = len(packed) / max(1, len(objs))
    all_in = Check(
        "all_objects_packed_or_consumed", len(packed) == len(objs), frac,
        f"{len(packed)}/{len(objs)} objects picked and packed "
        f"(env consumes packed objects on this suite): {sorted(packed)}",
        per,
    )
    grasp_cov = Check(
        "objects_picked_up", len(grasped) >= len(packed),
        len(grasped) / max(1, len(objs)),
        f"{len(grasped)}/{len(objs)} objects had grasp-and-lift events",
    )
    passed = sane.passed and all_in.passed
    return SeedEvaluation(
        "PASS" if passed else "FAIL", frac, all_in.detail,
        [sane, all_in, grasp_cov],
    )


def eval_pack_listed(states: list[dict], case: Any = None,
                     **_: Any) -> SeedEvaluation:
    """``case.targets`` must all finish settled in ``case.container``;
    ``case.forbidden`` must all stay OUT (negation / category
    grounding). Per-target grasp events reported (non-gating for lists
    longer than two — the manipulation long tail is scored by
    containment, not by touch counts)."""
    container = case.container or BASKET
    targets = list(case.targets)
    forbidden = list(case.forbidden)
    final = states[-1]
    sane = check_scene_sane(states, case)
    packed = [n for n in targets if resting_in_basket(final, n, container)]
    grasped = [n for n in targets
               if grasp_lift_index(states, n) is not None]
    frac = len(packed) / max(1, len(targets))
    in_ok = Check(
        "listed_targets_in_container", len(packed) == len(targets), frac,
        f"{len(packed)}/{len(targets)} listed targets settled in "
        f"{container}: {sorted(packed)}",
        {n: {"picked_up": n in grasped,
             "in_container": n in packed} for n in targets},
    )
    kept_out = [n for n in forbidden
                if not resting_in_basket(final, n, container)
                or resting_in_basket(states[0], n, container)]
    excl_ok = Check(
        "excluded_objects_stay_out",
        len(kept_out) == len(forbidden),
        len(kept_out) / max(1, len(forbidden)) if forbidden else 1.0,
        "all excluded objects stayed out of the container"
        if len(kept_out) == len(forbidden) else
        f"excluded object(s) packed anyway: "
        f"{sorted(set(forbidden) - set(kept_out))}",
    )
    grasp_cov = Check(
        "targets_picked_up", len(grasped) >= len(packed),
        len(grasped) / max(1, len(targets)),
        f"{len(grasped)}/{len(targets)} targets were picked up: "
        f"{sorted(grasped)}",
    )
    passed = sane.passed and in_ok.passed and excl_ok.passed
    score = 0.75 * frac + 0.25 * excl_ok.score
    return SeedEvaluation(
        "PASS" if passed else "FAIL", score,
        f"{in_ok.detail}; excluded-out={excl_ok.passed}",
        [sane, in_ok, excl_ok, grasp_cov],
    )


def eval_stack_on(states: list[dict], case: Any = None,
                  **_: Any) -> SeedEvaluation:
    """``case.targets[0]`` finishes resting on top of ``case.base``
    (is_on_strict-style window), both settled, target actually picked
    up, base not dragged away (> 10 cm)."""
    target, base = case.targets[0], case.base
    sane = check_scene_sane(states, case)
    grasped = _grasped_check(states, target)
    final, first = states[-1], states[0]
    on = resting_on(final, target, base)
    tb, bb = _body(final, target), _body(final, base)
    dz = (float(tb["lo"][2]) - float(bb["hi"][2])
          if tb is not None and bb is not None else float("nan"))
    on_chk = Check(
        f"{target}_on_{base}", on, 1.0 if on else 0.0,
        f"{target} bottom is {dz * 100:+.1f} cm relative to {base} top; "
        f"{'resting on it' if on else 'NOT resting on it'}",
    )
    b0, b1 = _body(first, base), _body(final, base)
    base_disp = (
        float(np.linalg.norm(np.asarray(b1["p"][:2]) - np.asarray(b0["p"][:2])))
        if b0 is not None and b1 is not None else float("nan"))
    base_ok = Check(
        f"{base}_not_dragged", base_disp <= 0.10,
        1.0 if base_disp <= 0.10 else 0.0,
        f"{base} moved {base_disp * 100:.1f} cm (<= 10 allowed)",
    )
    checks = [sane, grasped, on_chk, base_ok]
    passed = all(c.passed for c in checks)
    score = 0.3 * grasped.score + 0.55 * on_chk.score + 0.15 * base_ok.score
    return SeedEvaluation(
        "PASS" if passed else "FAIL", score,
        f"{target} on {base}: {on} (picked={grasped.passed})",
        checks,
    )


def eval_lateral_of(states: list[dict], case: Any = None,
                    **_: Any) -> SeedEvaluation:
    """``case.targets[0]`` set down beside ``case.reference`` along y.
    'Left' is accepted in either y direction (viewer's left is -y in
    the agentview video, the robot's left is +y — both measured); the
    side taken is logged. Gates: target picked up, moved >= 4 cm,
    finishes settled on the table 6-30 cm from the reference in y and
    within 15 cm in x, not in the container."""
    target, ref = case.targets[0], case.reference
    container = case.container or BASKET
    sane = check_scene_sane(states, case)
    grasped = _grasped_check(states, target)
    final, first = states[-1], states[0]
    tb, rb, t0 = _body(final, target), _body(final, ref), _body(first, target)
    if tb is None or rb is None or t0 is None:
        return SeedEvaluation("FAIL", 0.0, "bodies missing from trace",
                              [sane])
    dy = float(tb["p"][1]) - float(rb["p"][1])
    dx = float(tb["p"][0]) - float(rb["p"][0])
    moved = float(np.linalg.norm(
        np.asarray(t0["p"][:2]) - np.asarray(tb["p"][:2])))
    beside = 0.06 <= abs(dy) <= 0.30 and abs(dx) <= 0.15
    side = ("-y (viewer's left in the agentview video)" if dy < 0
            else "+y (robot's left)")
    rel = Check(
        f"{target}_beside_{ref}", beside, 1.0 if beside else 0.0,
        f"offset from {ref}: dy={dy * 100:+.1f} cm (6-30 accepted), "
        f"dx={dx * 100:+.1f} cm (<= 15); side taken: {side}",
        {"dy_m": round(dy, 4), "dx_m": round(dx, 4)},
    )
    moved_chk = Check(
        f"{target}_moved", moved >= 0.04, 1.0 if moved >= 0.04 else 0.0,
        f"{target} displaced {moved * 100:.1f} cm from start",
    )
    rest = (
        on_table(final, target, container)
        and final.get("held") != target
        and float(tb.get("v", 0.0)) < SETTLED_SPEED
    )
    rest_chk = Check(
        "released_on_table", rest, 1.0 if rest else 0.0,
        "released and settled on the table, outside the container"
        if rest else "not settled on the table",
    )
    checks = [sane, grasped, rel, moved_chk, rest_chk]
    passed = all(c.passed for c in checks)
    score = (0.25 * grasped.score + 0.4 * rel.score
             + 0.15 * moved_chk.score + 0.2 * rest_chk.score)
    return SeedEvaluation(
        "PASS" if passed else "FAIL", score,
        f"dy={dy * 100:+.1f} cm vs {ref}, picked={grasped.passed}, "
        f"settled={rest}",
        checks,
    )


def eval_on_support(states: list[dict], case: Any = None,
                    **_: Any) -> SeedEvaluation:
    """Every ``case.targets`` body finishes over/on ``case.support``
    (nesting allowed: bottom up to 12 cm above the support top, xy
    centroid inside the support), each picked up at some point, and the
    support itself not dragged > 10 cm."""
    support = case.support
    targets = list(case.targets)
    sane = check_scene_sane(states, case)
    final, first = states[-1], states[0]
    per = {}
    on_count = 0
    for t in targets:
        ok = resting_on(final, t, support, max_above=0.12)
        per[t] = {
            "on_support": ok,
            "picked_up": grasp_lift_index(states, t) is not None,
        }
        on_count += ok
    frac = on_count / max(1, len(targets))
    on_chk = Check(
        f"targets_on_{support}", on_count == len(targets), frac,
        f"{on_count}/{len(targets)} targets settled on {support}",
        per,
    )
    grasped_all = all(v["picked_up"] for v in per.values())
    grasp_chk = Check(
        "targets_picked_up", grasped_all,
        sum(v["picked_up"] for v in per.values()) / max(1, len(targets)),
        f"picked up: {sorted(t for t, v in per.items() if v['picked_up'])}",
    )
    s0, s1 = _body(first, support), _body(final, support)
    disp = (float(np.linalg.norm(
        np.asarray(s1["p"][:2]) - np.asarray(s0["p"][:2])))
        if s0 is not None and s1 is not None else float("nan"))
    sup_chk = Check(
        f"{support}_not_dragged", disp <= 0.10,
        1.0 if disp <= 0.10 else 0.0,
        f"{support} moved {disp * 100:.1f} cm (<= 10 allowed)",
    )
    checks = [sane, on_chk, grasp_chk, sup_chk]
    passed = all(c.passed for c in checks)
    score = 0.6 * frac + 0.25 * grasp_chk.score + 0.15 * sup_chk.score
    return SeedEvaluation(
        "PASS" if passed else "FAIL", score, on_chk.detail, checks,
    )


CRITERIA = {
    "soup_basket": eval_soup_basket,
    "all_objects": eval_all_objects,
    "soup_then_milk": eval_soup_then_milk,
    "two_inches_right": eval_two_inches_right,
    "cream_cheese": eval_cream_cheese,
    "move_basket_ranch": eval_move_basket_ranch,
    "pick_single": eval_pick_single,
    "pack_scene": eval_pack_scene,
    "pack_listed": eval_pack_listed,
    "stack_on": eval_stack_on,
    "lateral_of": eval_lateral_of,
    "on_support": eval_on_support,
}


# ---------------------------------------------------------------------------
# Entry point (handles non-executed shapes, attaches diagnostics)
# ---------------------------------------------------------------------------


def evaluate_seed(
    case: Any,
    states: list[dict],
    events: list[dict],
    execution: dict[str, Any],
    extras: dict[str, Any] | None = None,
) -> SeedEvaluation:
    from state_log import hang_site

    case_id = getattr(case, "id", str(case))
    checker = getattr(case, "checker", "") or case_id
    timed_out = bool(execution.get("timed_out"))
    if len(states) < 2:
        verdict = "TIMEOUT" if timed_out else (
            "CRASH" if execution.get("harness_crash") else "NO_TRACE")
        ev = SeedEvaluation(
            verdict, 0.0,
            f"no usable state trace ({verdict.lower()}); "
            f"{str(execution.get('harness_crash') or execution.get('error') or '')[:200]}",
            [],
        )
    else:
        ev = CRITERIA[checker](states, case=case)
        if timed_out:
            ev.verdict = "TIMEOUT"
            ev.summary = "run killed at timeout — " + ev.summary
        elif execution.get("harness_crash"):
            ev.verdict = "CRASH"
            ev.summary = (
                f"harness crash ({str(execution['harness_crash'])[:120]}) — "
                + ev.summary
            )
    site = hang_site(events)
    if timed_out and site is not None:
        ev.diagnostics["hang"] = {
            "tool": site.get("tool"),
            "node": site.get("node"),
            "kwargs": site.get("kw"),
            "started_at_s": site.get("t"),
        }
        ev.summary += (
            f" | hung in tool {site.get('tool')!r}"
            f" (node {site.get('node')!r})"
        )
    tool_errors = [
        {"tool": e.get("tool"), "node": e.get("node"),
         "err": e.get("err", "")[:200]}
        for e in events if e.get("ev") == "error"
    ]
    if tool_errors:
        ev.diagnostics["tool_errors"] = tool_errors[:20]
    planner_errors = [e for e in tool_errors
                      if "curobo" in str(e.get("tool", ""))
                      or "plan" in str(e.get("tool", ""))]
    if planner_errors:
        ev.diagnostics["planner_errors"] = planner_errors[:10]
    if case_id == "cream_cheese":
        geo = [
            {"tool": e.get("tool"), "summary": e.get("summary")}
            for e in events
            if e.get("ev") == "done"
            and str(e.get("tool", "")).startswith("geometry.")
        ]
        if geo:
            ev.diagnostics["geometry_outputs"] = geo[:30]
        ev.diagnostics["cream_cheese_gt_aabb_ext_m"] = [0.083, 0.043, 0.020]
    if extras:
        ev.diagnostics.update(extras)
    return ev


# ---------------------------------------------------------------------------
# Generated-graph structure diagnostics (evidence for the cycles question)
# ---------------------------------------------------------------------------


def _edge_pairs(graph: dict) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for e in graph.get("edges") or []:
        if isinstance(e, (list, tuple)) and len(e) >= 2:
            pairs.append((str(e[0]), str(e[1])))
    cond = graph.get("conditional_edges") or {}
    if isinstance(cond, dict):
        for src, targets in cond.items():
            if isinstance(targets, str):
                pairs.append((str(src), targets))
            elif isinstance(targets, dict):
                # Generated graphs use {"router_field": ..., "mapping":
                # {label: target}}; older/flat forms map label -> target
                # directly.
                mapping = targets.get("mapping")
                if not isinstance(mapping, dict):
                    mapping = {k: v for k, v in targets.items()
                               if k != "router_field"}
                pairs.extend((str(src), str(t)) for t in mapping.values()
                             if isinstance(t, str))
            elif isinstance(targets, (list, tuple)):
                pairs.extend((str(src), str(t)) for t in targets
                             if isinstance(t, str))
    return pairs


def _has_cycle(pairs: list[tuple[str, str]]) -> bool:
    adj: dict[str, list[str]] = {}
    for a, b in pairs:
        adj.setdefault(a, []).append(b)
    WHITE, GRAY, BLACK = 0, 1, 2
    color: dict[str, int] = {}

    def dfs(u: str) -> bool:
        color[u] = GRAY
        for v in adj.get(u, ()):  # noqa: B023
            c = color.get(v, WHITE)
            if c == GRAY:
                return True
            if c == WHITE and dfs(v):
                return True
        color[u] = BLACK
        return False

    return any(
        color.get(u, WHITE) == WHITE and dfs(u) for u in list(adj)
    )


def workflow_structure(workflow: dict | None) -> dict[str, Any]:
    """Cycle report for a generated workflow: top level + per subgraph."""
    if not workflow:
        return {"available": False}
    top = _has_cycle(_edge_pairs(workflow))
    subs = {
        name: _has_cycle(_edge_pairs(sg or {}))
        for name, sg in (workflow.get("subgraphs") or {}).items()
    }
    return {
        "available": True,
        "top_level_has_cycle": top,
        "subgraph_has_cycle": subs,
        "any_cycle": bool(top or any(subs.values())),
    }


def max_node_iterations(trace_dir) -> int | None:
    """Max per-node iteration count from the executor's own node_data
    layout (node_data/<node>/iters/<k>). >1 proves the runtime revisited
    a node (i.e. executed a loop) in this run."""
    from pathlib import Path

    nd = Path(trace_dir) / "node_data"
    if not nd.is_dir():
        return None
    best = 0
    for node in nd.iterdir():
        iters = node / "iters"
        if iters.is_dir():
            best = max(best, sum(1 for _ in iters.iterdir()))
    return best or None
