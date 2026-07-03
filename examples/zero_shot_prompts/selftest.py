"""Criteria self-test on synthetic state traces (no sim, no LLM, ~1 s).

    uv run python examples/zero_shot_prompts/selftest.py

Each case gets a passing trace and its characteristic failure modes —
including the exact failures the tester matrix reported (only-soup
packed, mid-grasp hang, knocked-over can, basket never moved) — plus
anti-gaming traces (teleport without a grasp, unrequested packing,
touch-without-lift).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

import criteria  # noqa: E402
from criteria import evaluate_seed  # noqa: E402

# Layout mirroring the probed seed-0 scene (robot-base frame).
SPREAD = {
    "alphabet_soup": (0.61, 0.25, 0.038),
    "butter": (0.72, -0.24, 0.009),
    "cream_cheese": (0.65, -0.10, 0.009),
    "milk": (0.40, -0.08, 0.070),
    "salad_dressing": (0.39, -0.26, 0.073),
    "tomato_sauce": (0.45, 0.06, 0.038),
}
BASKET_P = (0.75, 0.03, -0.005)
BASKET_EXT = (0.176, 0.184, 0.165)
IDQ = (1.0, 0.0, 0.0, 0.0)
ROT90X = (math.sqrt(0.5), math.sqrt(0.5), 0.0, 0.0)  # knocked onto side


def body(p, q=IDQ, ext=(0.06, 0.06, 0.06), v=0.0, g=False):
    half = [e / 2 for e in ext]
    return {
        "p": list(p), "q": list(q),
        "lo": [p[i] - half[i] for i in range(3)],
        "hi": [p[i] + half[i] for i in range(3)],
        "v": v, "g": g,
    }


def snap(*, held=None, grip=1.0, basket_p=BASKET_P, overrides=None):
    bodies = {"basket": body(basket_p, ext=BASKET_EXT)}
    for name, p in SPREAD.items():
        bodies[name] = body(p)
    for name, b in (overrides or {}).items():
        bodies[name] = b
    return {"seq": 0, "t": 0.0, "tool": "x", "node": None,
            "held": held, "grip": grip, "ee": [0.45, 0.0, 0.26],
            "bodies": bodies}


def in_basket_p(basket_p=BASKET_P, dx=0.0, dy=0.0, dz=0.03):
    return (basket_p[0] + dx, basket_p[1] + dy, basket_p[2] + dz)


def lifted(name, dz=0.12, q=IDQ):
    p = SPREAD[name]
    return body((p[0], p[1], p[2] + dz), q=q, g=True)


EXEC_OK = {"ran": True, "success": True, "exit_status": "success",
           "harness_crash": None}

N_PASS = 0


def expect(label, ev, verdict, score_min=None, score_max=None):
    global N_PASS
    problems = []
    if ev.verdict != verdict:
        problems.append(f"verdict={ev.verdict}, want {verdict}")
    if score_min is not None and ev.score < score_min:
        problems.append(f"score={ev.score:.2f} < {score_min}")
    if score_max is not None and ev.score > score_max:
        problems.append(f"score={ev.score:.2f} > {score_max}")
    if problems:
        print(f"FAIL {label}: {'; '.join(problems)} — {ev.summary}")
        for c in ev.checks:
            print(f"    [{'ok' if c.passed else 'XX'}] {c.name}: {c.detail}")
        sys.exit(1)
    N_PASS += 1
    print(f"ok   {label} (verdict={ev.verdict} score={ev.score:.2f})")


def ev(case_id, states, events=(), execution=EXEC_OK):
    from cases import CASES
    return evaluate_seed(CASES[case_id], states, list(events),
                         dict(execution))


def main() -> None:
    init = snap()

    # ---- soup_basket -----------------------------------------------------
    held_soup = snap(held="alphabet_soup", grip=0.4,
                     overrides={"alphabet_soup": lifted("alphabet_soup")})
    soup_in = snap(overrides={
        "alphabet_soup": body(in_basket_p())})
    expect("soup_basket pick+place passes",
           ev("soup_basket", [init, held_soup, soup_in]),
           "PASS", score_min=0.99)
    expect("soup_basket teleport (never picked) fails",
           ev("soup_basket", [init, soup_in]), "FAIL", score_max=0.75)
    stray_in = snap(overrides={
        "alphabet_soup": body(in_basket_p(dx=-0.04)),
        "butter": body(in_basket_p(dx=0.04), ext=(0.078, 0.04, 0.018))})
    expect("soup_basket unrequested packing fails",
           ev("soup_basket", [init, held_soup, stray_in]), "FAIL")
    touch_only = snap(held="alphabet_soup", grip=0.4)  # contact, no lift
    expect("soup_basket touch-without-lift fails",
           ev("soup_basket", [init, touch_only, soup_in]), "FAIL")

    # scene sanity gate: an object flung below the floor fails everything
    flung = snap(overrides={
        "alphabet_soup": body(in_basket_p()),
        "milk": body((0.7, 0.0, -0.5), ext=(0.058, 0.06, 0.145))})
    expect("scene_sane catches flung object",
           ev("soup_basket", [init, held_soup, flung]), "FAIL")

    # ---- all_objects -----------------------------------------------------
    names = list(SPREAD)
    trace = [init]
    packed: dict[str, dict] = {}
    for i, n in enumerate(names):
        trace.append(snap(held=n, grip=0.4,
                          overrides={**packed, n: lifted(n)}))
        packed[n] = body(in_basket_p(dx=0.05 * (i % 3) - 0.05,
                                     dy=0.05 * (i // 3) - 0.02,
                                     dz=0.03 + 0.02 * (i // 3)))
        trace.append(snap(overrides=dict(packed)))
    expect("all_objects 6/6 passes", ev("all_objects", trace),
           "PASS", score_min=0.99)
    only_soup = [init, held_soup, soup_in]  # the tester-observed baseline
    expect("all_objects only-soup fails at 1/6",
           ev("all_objects", only_soup), "FAIL", score_max=0.2)

    # ---- soup_then_milk ---------------------------------------------------
    milk_held = snap(held="milk", grip=0.4, overrides={
        "alphabet_soup": body(in_basket_p(dx=-0.04)),
        "milk": lifted("milk")})
    both_in = snap(overrides={
        "alphabet_soup": body(in_basket_p(dx=-0.04)),
        "milk": body(in_basket_p(dx=0.04, dz=0.05),
                     ext=(0.058, 0.06, 0.145))})
    good = [init, held_soup,
            snap(overrides={"alphabet_soup": body(in_basket_p(dx=-0.04))}),
            milk_held, both_in]
    expect("soup_then_milk in order passes",
           ev("soup_then_milk", good), "PASS", score_min=0.99)

    milk_first = [
        init,
        snap(held="milk", grip=0.4, overrides={"milk": lifted("milk")}),
        snap(overrides={"milk": body(in_basket_p(dx=0.04, dz=0.05),
                                     ext=(0.058, 0.06, 0.145))}),
        snap(held="alphabet_soup", grip=0.4, overrides={
            "milk": body(in_basket_p(dx=0.04, dz=0.05),
                         ext=(0.058, 0.06, 0.145)),
            "alphabet_soup": lifted("alphabet_soup")}),
        both_in,
    ]
    expect("soup_then_milk wrong order fails",
           ev("soup_then_milk", milk_first), "FAIL", score_max=0.9)

    # the tester-observed hang: soup placed, then killed mid milk-grasp
    partial = [init, held_soup,
               snap(overrides={"alphabet_soup": body(in_basket_p(dx=-0.04))})]
    hang_events = [
        {"ev": "call", "seq": 1, "t": 1.0, "tool": "robot.get_observation",
         "node": "grasp.observe"},
        {"ev": "done", "seq": 1, "t": 2.0, "tool": "robot.get_observation",
         "node": "grasp.observe"},
        {"ev": "call", "seq": 2, "t": 3.0, "tool": "curobo.plan",
         "node": "grasp.plan", "kw": {"target": "milk"}},
    ]
    hung = ev("soup_then_milk", partial, hang_events,
              {"ran": True, "timed_out": True, "harness_crash": None})
    expect("soup_then_milk hang -> TIMEOUT", hung, "TIMEOUT", score_max=0.7)
    assert hung.diagnostics.get("hang", {}).get("tool") == "curobo.plan", \
        f"hang site not recovered: {hung.diagnostics.get('hang')}"
    print("ok   hang site recovered from events tail (curobo.plan)")

    # ---- two_inches_right --------------------------------------------------
    def soup_at(dy, dx=0.0, q=IDQ):
        p = SPREAD["alphabet_soup"]
        return snap(overrides={
            "alphabet_soup": body((p[0] + dx, p[1] + dy, p[2]), q=q)})

    expect("two_inches_right +y (viewer right) passes",
           ev("two_inches_right", [init, held_soup, soup_at(+0.05)]),
           "PASS", score_min=0.99)
    expect("two_inches_right -y (robot right) passes",
           ev("two_inches_right", [init, held_soup, soup_at(-0.05)]),
           "PASS", score_min=0.99)
    expect("two_inches_right knocked over fails",
           ev("two_inches_right",
              [init, held_soup, soup_at(+0.05, q=ROT90X)]),
           "FAIL", score_max=0.8)
    expect("two_inches_right overshoot fails",
           ev("two_inches_right", [init, held_soup, soup_at(+0.12)]),
           "FAIL")
    expect("two_inches_right never picked fails",
           ev("two_inches_right", [init, soup_at(+0.05)]), "FAIL")

    # ---- cream_cheese --------------------------------------------------------
    cc_held = snap(held="cream_cheese", grip=0.6, overrides={
        "cream_cheese": lifted("cream_cheese", dz=0.10)})
    cc_in = snap(overrides={
        "cream_cheese": body(in_basket_p(), ext=(0.083, 0.043, 0.020))})
    expect("cream_cheese pick+place passes",
           ev("cream_cheese", [init, cc_held, cc_in]),
           "PASS", score_min=0.99)
    expect("cream_cheese untouched fails",
           ev("cream_cheese", [init, init]), "FAIL")

    # ---- move_basket_ranch -----------------------------------------------------
    # seed-1-like start: basket on the -y side.
    b0 = (0.71, -0.23, -0.005)
    b1 = (0.71, +0.10, -0.005)
    start = snap(basket_p=b0)
    ranch_held = snap(basket_p=b1, held="salad_dressing", grip=0.4,
                      overrides={"salad_dressing":
                                 lifted("salad_dressing", dz=0.12)})
    ranch_in = snap(basket_p=b1, overrides={
        "salad_dressing": body(in_basket_p(b1, dz=0.05),
                               ext=(0.045, 0.066, 0.148))})
    expect("move_basket_ranch relocated+packed passes",
           ev("move_basket_ranch", [start, snap(basket_p=b1),
                                    ranch_held, ranch_in]),
           "PASS", score_min=0.99)
    unmoved_in = snap(basket_p=b0, overrides={
        "salad_dressing": body(in_basket_p(b0, dz=0.05),
                               ext=(0.045, 0.066, 0.148))})
    ranch_held_unmoved = snap(basket_p=b0, held="salad_dressing", grip=0.4,
                              overrides={"salad_dressing":
                                         lifted("salad_dressing", dz=0.12)})
    expect("move_basket_ranch basket never moved fails",
           ev("move_basket_ranch", [start, ranch_held_unmoved, unmoved_in]),
           "FAIL", score_max=0.8)

    # ---- stack_butter_on_cream_cheese --------------------------------------
    cc_p = SPREAD["cream_cheese"]
    butter_on_cc = snap(overrides={
        "cream_cheese": body(cc_p, ext=(0.083, 0.043, 0.020)),
        "butter": body((cc_p[0], cc_p[1], cc_p[2] + 0.019),
                       ext=(0.078, 0.040, 0.018))})
    butter_held = snap(held="butter", grip=0.4,
                       overrides={"butter": lifted("butter", dz=0.10)})
    expect("stack butter-on-cream-cheese passes",
           ev("stack_butter_on_cream_cheese",
              [init, butter_held, butter_on_cc]), "PASS", score_min=0.99)
    butter_beside = snap(overrides={
        "butter": body((cc_p[0] + 0.09, cc_p[1], 0.009),
                       ext=(0.078, 0.040, 0.018))})
    expect("stack beside (not on) fails",
           ev("stack_butter_on_cream_cheese",
              [init, butter_held, butter_beside]), "FAIL")

    # ---- all_except_milk -----------------------------------------------------
    five = [n for n in SPREAD if n != "milk"]
    trace = [init]
    packed5: dict[str, dict] = {}
    for i, n in enumerate(five):
        trace.append(snap(held=n, grip=0.4,
                          overrides={**packed5, n: lifted(n)}))
        packed5[n] = body(in_basket_p(dx=0.05 * (i % 3) - 0.05,
                                      dy=0.05 * (i // 3) - 0.02,
                                      dz=0.03 + 0.02 * (i // 3)))
        trace.append(snap(overrides=dict(packed5)))
    expect("all_except_milk 5-in milk-out passes",
           ev("all_except_milk", trace), "PASS", score_min=0.99)
    with_milk = snap(overrides={**packed5,
                                "milk": body(in_basket_p(dz=0.08),
                                             ext=(0.058, 0.06, 0.145))})
    expect("all_except_milk milk packed anyway fails",
           ev("all_except_milk", trace + [with_milk]), "FAIL",
           score_max=0.85)

    # ---- soup_left_of_milk ----------------------------------------------------
    milk_p = SPREAD["milk"]

    def soup_beside_milk(dy, dx=0.0):
        return snap(overrides={"alphabet_soup": body(
            (milk_p[0] + dx, milk_p[1] + dy, 0.038))})

    expect("soup_left_of_milk viewer-left (-y) passes",
           ev("soup_left_of_milk",
              [init, held_soup, soup_beside_milk(-0.12)]),
           "PASS", score_min=0.99)
    expect("soup_left_of_milk robot-left (+y) passes",
           ev("soup_left_of_milk",
              [init, held_soup, soup_beside_milk(+0.12)]),
           "PASS", score_min=0.99)
    expect("soup_left_of_milk wrong axis (x offset) fails",
           ev("soup_left_of_milk",
              [init, held_soup, soup_beside_milk(0.0, dx=0.25)]), "FAIL")

    # ---- pick_single on other VAB scenes (probed layouts) ----------------------
    # variance task 3 scene: different 6-item subset incl. chocolate_pudding
    T3 = {
        "basket": body((0.400, -0.081, -0.005), ext=BASKET_EXT),
        "chocolate_pudding": body((0.750, 0.030, 0.013), ext=(0.078, 0.055, 0.028)),
        "orange_juice": body((0.700, -0.200, 0.070), ext=(0.079, 0.079, 0.145)),
        "bbq_sauce": body((0.596, 0.260, 0.055), ext=(0.045, 0.065, 0.149)),
        "ketchup": body((0.727, -0.226, 0.072), ext=(0.045, 0.065, 0.149)),
        "salad_dressing": body((0.619, -0.121, 0.073), ext=(0.045, 0.066, 0.148)),
        "alphabet_soup": body((0.481, -0.240, 0.038), ext=(0.085, 0.089, 0.085)),
    }

    def t3_snap(over=None, *, held=None, grip=1.0):
        bodies = dict(T3)
        bodies.update(over or {})
        return {"seq": 0, "t": 0.0, "tool": "x", "node": None,
                "held": held, "grip": grip, "ee": [0.45, 0.0, 0.26],
                "bodies": bodies}

    t3_basket = (0.400, -0.081, -0.005)
    t3_0 = t3_snap()
    t3_held = t3_snap({"chocolate_pudding": body(
        (0.750, 0.030, 0.15), ext=(0.078, 0.055, 0.028), g=True)},
        held="chocolate_pudding", grip=0.4)
    t3_in = t3_snap({"chocolate_pudding": body(
        (t3_basket[0], t3_basket[1], t3_basket[2] + 0.03),
        ext=(0.078, 0.055, 0.028))})
    expect("pudding_basket pick+place passes",
           ev("pudding_basket", [t3_0, t3_held, t3_in]),
           "PASS", score_min=0.99)
    expect("pudding_basket untouched fails",
           ev("pudding_basket", [t3_0, t3_0]), "FAIL")

    # swap-suite task 5 scene (probed: no ketchup here; basket relocates)
    SW = {
        "basket": body((0.633, -0.066, -0.005), ext=BASKET_EXT),
        "bbq_sauce": body((0.396, -0.079, 0.055), ext=(0.045, 0.065, 0.149)),
        "butter": body((0.451, 0.063, 0.009), ext=(0.078, 0.04, 0.018)),
        "chocolate_pudding": body((0.750, 0.030, 0.013), ext=(0.078, 0.055, 0.028)),
        "milk": body((0.478, -0.237, 0.070), ext=(0.058, 0.06, 0.145)),
        "orange_juice": body((0.700, -0.200, 0.070), ext=(0.079, 0.079, 0.145)),
        "tomato_sauce": body((0.613, 0.260, 0.038), ext=(0.083, 0.087, 0.087)),
    }

    def sw_snap(over=None, *, held=None, grip=1.0):
        bodies = dict(SW)
        bodies.update(over or {})
        return {"seq": 0, "t": 0.0, "tool": "x", "node": None,
                "held": held, "grip": grip, "ee": [0.45, 0.0, 0.26],
                "bodies": bodies}

    sw_held = sw_snap({"orange_juice": body(
        (0.700, -0.200, 0.19), ext=(0.079, 0.079, 0.145), g=True)},
        held="orange_juice", grip=0.4)
    sw_in = sw_snap({"orange_juice": body(
        (0.633, -0.066, 0.075), ext=(0.079, 0.079, 0.145))})
    expect("oj_swap_scene pick+place passes",
           ev("oj_swap_scene", [sw_snap(), sw_held, sw_in]),
           "PASS", score_min=0.99)

    # ---- both_boxes (category grounding on the task-0 scene) --------------------
    boxes_held1 = snap(held="butter", grip=0.4,
                       overrides={"butter": lifted("butter", dz=0.10)})
    boxes_in1 = snap(overrides={"butter": body(
        in_basket_p(dx=-0.04), ext=(0.078, 0.04, 0.018))})
    boxes_held2 = snap(held="cream_cheese", grip=0.4, overrides={
        "butter": body(in_basket_p(dx=-0.04), ext=(0.078, 0.04, 0.018)),
        "cream_cheese": lifted("cream_cheese", dz=0.10)})
    boxes_both = snap(overrides={
        "butter": body(in_basket_p(dx=-0.04), ext=(0.078, 0.04, 0.018)),
        "cream_cheese": body(in_basket_p(dx=0.04), ext=(0.083, 0.043, 0.02))})
    expect("both_boxes exact pair passes",
           ev("both_boxes", [init, boxes_held1, boxes_in1, boxes_held2,
                             boxes_both]), "PASS", score_min=0.99)
    boxes_stray = snap(overrides={
        "butter": body(in_basket_p(dx=-0.04), ext=(0.078, 0.04, 0.018)),
        "cream_cheese": body(in_basket_p(dx=0.04), ext=(0.083, 0.043, 0.02)),
        "alphabet_soup": body(in_basket_p(dy=0.05, dz=0.06))})
    expect("both_boxes soup packed too fails",
           ev("both_boxes", [init, boxes_held1, boxes_in1, boxes_held2,
                             boxes_stray]), "FAIL", score_max=0.96)

    # ---- pack_scene (env consumes packed objects via graveyard teleport) -------
    GY = (50.6, 50.0, 0.01)
    trace = [init]
    consumed: dict[str, dict] = {}
    for i, n in enumerate(names):
        trace.append(snap(held=n, grip=0.4,
                          overrides={**consumed, n: lifted(n)}))
        consumed[n] = body(GY)
        trace.append(snap(overrides=dict(consumed)))
    ev_ps = ev("pack_scene_all", trace)
    expect("pack_scene all consumed passes", ev_ps, "PASS", score_min=0.99)

    # one object never grasped and left on the table -> fail at 5/6
    trace5 = [init]
    consumed5: dict[str, dict] = {}
    for i, n in enumerate(names[:5]):
        trace5.append(snap(held=n, grip=0.4,
                           overrides={**consumed5, n: lifted(n)}))
        consumed5[n] = body(GY)
        trace5.append(snap(overrides=dict(consumed5)))
    expect("pack_scene 5/6 fails", ev("pack_scene_all", trace5), "FAIL",
           score_max=0.9)

    # graveyard teleport WITHOUT a grasp event does not count as packed
    tele = [init, snap(overrides={n: body(GY) for n in names})]
    expect("pack_scene teleport-without-grasp fails",
           ev("pack_scene_all", tele), "FAIL", score_max=0.1)

    # ---- graph-structure diagnostic ------------------------------------------
    dag = {"nodes": {"a": {}, "b": {}},
           "edges": [["START", "a"], ["a", "b"], ["b", "END"]],
           "subgraphs": {}}
    loop = {"nodes": {"a": {}, "b": {}},
            "edges": [["START", "a"], ["a", "b"], ["b", "a"]],
            "subgraphs": {}}
    cond_loop = {"nodes": {"a": {}, "b": {}},
                 "edges": [["START", "a"], ["a", "b"]],
                 "conditional_edges": {"b": {"again": "a", "done": "END"}},
                 "subgraphs": {}}
    # the schema generated graphs actually use (router_field + mapping)
    routed_loop = {
        "nodes": {"perceive": {}, "grasp": {}, "transport": {}},
        "edges": [["START", "perceive"]],
        "conditional_edges": {
            "perceive": {"router_field": "exit",
                         "mapping": {"found": "grasp", "none": "done"}},
            "grasp": {"router_field": "exit",
                      "mapping": {"grasped": "transport",
                                  "failed": "abort"}},
            "transport": {"router_field": "exit",
                          "mapping": {"placed": "perceive",
                                      "blocked": "abort"}},
        },
        "subgraphs": {},
    }
    routed_chain = {
        "nodes": {"a": {}, "b": {}},
        "edges": [["START", "a"]],
        "conditional_edges": {
            "a": {"router_field": "exit", "mapping": {"ok": "b"}},
            "b": {"router_field": "exit", "mapping": {"ok": "done"}},
        },
        "subgraphs": {},
    }
    assert not criteria.workflow_structure(dag)["any_cycle"]
    assert criteria.workflow_structure(loop)["any_cycle"]
    assert criteria.workflow_structure(cond_loop)["any_cycle"]
    assert criteria.workflow_structure(routed_loop)["any_cycle"]
    assert not criteria.workflow_structure(routed_chain)["any_cycle"]
    print("ok   workflow cycle detection (DAG / back-edge / conditional "
          "flat + router_field/mapping loop + routed chain)")

    print(f"\nself-test: {N_PASS + 6} assertions passed")


if __name__ == "__main__":
    main()
