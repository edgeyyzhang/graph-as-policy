"""The zero-shot prompt matrix for the grocery_fulfillment scene.

Six natural-language prompts, fed VERBATIM to ``gap.agent.generate``
(no scene hints, no curated descriptors). Each case carries:

- ``baseline``: the tester-observed outcome this suite was built from,
  quoted faithfully. The suite does not assume the baseline is still
  true — it re-measures every claim from simulator ground truth.
- ``criterion``: the object-state success criterion, evaluated by
  ``criteria.py`` over the recorded per-tool-call state trace.
- ``seeds``: sim seeds to execute the (single) generated graph on.
  ``soup_basket`` runs three seeds because its baseline claim is
  "robust to variations in position" — the layout genuinely varies per
  seed (measured: the basket moves between (0.75, 0.03) and
  (0.71, -0.23) across seeds 0..2).

All cases run on ``libero_object_all_variance/0`` — the environment the
grocery_fulfillment example targets. Scene bodies (measured live):
alphabet_soup, basket, butter, cream_cheese, milk, salad_dressing,
tomato_sauce.
"""

from __future__ import annotations

from dataclasses import dataclass


SIM_TASK = "libero_object_all_variance/0"


@dataclass(frozen=True)
class Case:
    id: str
    prompt: str
    baseline: str
    criterion: str
    seeds: tuple[int, ...] = (0,)
    timeout_min: int = 45
    sim_task: str = SIM_TASK
    env_consumes_packed: bool = False
    # criterion wiring (checker defaults to the case id; the scene
    # vocabulary below is measured per scene — see README)
    checker: str = ""
    container: str = "basket"
    targets: tuple[str, ...] = ()
    forbidden: tuple[str, ...] = ()
    base: str = ""
    reference: str = ""
    support: str = ""
    statics: tuple[str, ...] = ()


_CASES: list[Case] = [
    Case(
        id="soup_basket",
        prompt="Pick up the alphabet soup can and place it in the basket",
        baseline="Success, robust to variations in position",
        criterion=(
            "per seed: alphabet_soup grasped-and-lifted at some point, "
            "and finishes settled inside the basket with no other object "
            "packed; case passes only if ALL seeds pass (the robustness "
            "claim)"
        ),
        seeds=(0, 1, 2),
        timeout_min=60,
    ),
    Case(
        id="all_objects",
        prompt="Pick up all objects and place them in the basket",
        baseline=(
            "Failure — picks up the alphabet soup can and places it in "
            "the basket; perception fails for other objects (unable to "
            "pick up any object other than alphabet soup). Reported "
            "cause: the generated graph must contain cycles (tasks that "
            "must be repeated), but GaP seems to only support directed "
            "acyclic graphs."
        ),
        criterion=(
            "all 6 objects finish settled inside the basket (score = "
            "fraction packed); per-object grasp events reported. "
            "Diagnostics (not gated): whether the generated workflow "
            "contains cycles, and how many times the executor actually "
            "revisited a node at runtime — direct evidence for or "
            "against the reported DAG-only cause."
        ),
        timeout_min=90,
    ),
    Case(
        id="soup_then_milk",
        prompt=(
            "Pick up the alphabet soup and place it in the basket, then "
            "pick up the milk and place it in the basket"
        ),
        baseline=(
            "Failure — alphabet soup is successfully placed, and "
            "perception succeeds for the milk, but there seems to be a "
            "mesh loading error when grasping the milk (simulation "
            "consistently hangs at this point)."
        ),
        criterion=(
            "two distinct grasp-and-lift events (soup, milk); both "
            "finish settled in the basket; soup enters the basket "
            "strictly before milk. A wall-clock timeout is classified "
            "as a HANG and the last in-flight tool call (the hang site) "
            "is reported from the event log."
        ),
        timeout_min=45,
    ),
    Case(
        id="two_inches_right",
        prompt=(
            "Pick up the alphabet soup and place it down two inches to "
            "the right"
        ),
        baseline=(
            "Failure — picks up the alphabet soup can, but knocks it "
            "down. (Later reported as success after codegen-catalog "
            "introspection, arms[0] $ref, and curobo joint-state fixes; "
            "tester unsure all were necessary.)"
        ),
        criterion=(
            "soup grasped-and-lifted; finishes on the table (not in the "
            "basket, released, settled) displaced 5.1 +/- 3 cm along y "
            "with <= 5 cm drift in x; still upright (the body axis that "
            "initially pointed up is within 30 degrees of vertical at "
            "the end — a knocked-over can reads ~90). 'Right' is "
            "accepted in either y direction: +y is image-right in the "
            "agentview video (measured from the camera frame), -y is "
            "the robot's right; the direction taken is logged."
        ),
        seeds=(0, 1),
        timeout_min=75,
    ),
    Case(
        id="cream_cheese",
        prompt="Pick up the cream cheese and put it in the basket",
        baseline=(
            "Failure — CuRobo planning failure, unable to calculate "
            "trajectory without collisions (possibly because the cream "
            "cheese is a lower object); geometry.top_down_grasp_candidates "
            "generated a larger than expected bounding box."
        ),
        criterion=(
            "cream_cheese grasped-and-lifted and finishes settled inside "
            "the basket, nothing else packed. Diagnostics (not gated): "
            "planner tool errors, and the numeric outputs of geometry.* "
            "calls so any perceived bounding box can be compared against "
            "the ground-truth extents (measured GT AABB: 8.3 x 4.3 x "
            "2.0 cm — it really is a low object). Three seeds: the "
            "low-object edge is the one to test across layouts."
        ),
        seeds=(0, 1, 2),
        timeout_min=75,
    ),
    Case(
        id="move_basket_ranch",
        prompt=(
            "Move the basket to the opposite side, then pick up the "
            "ranch and place it in the basket"
        ),
        baseline="(no outcome recorded in the tester matrix)",
        criterion=(
            "the basket's final position is displaced >= 20 cm and on "
            "the opposite side (lateral sign flip with |dy| >= 20 cm, "
            "or >= 30 cm across the table in x); the ranch "
            "(salad_dressing — the only dressing bottle in the scene) "
            "is grasped-and-lifted and finishes settled inside the "
            "basket at the basket's FINAL location. How the basket is "
            "moved (carried or pushed) is not gated. Runs on seed 1, "
            "where the basket starts clearly on the -y side (measured "
            "y=-0.23) so 'opposite side' is well-defined; at seed 0 it "
            "spawns near the centerline."
        ),
        seeds=(1,),
        timeout_min=60,
    ),
    # -- extended matrix (run2): new capability probes -----------------------
    Case(
        id="stack_butter_on_cream_cheese",
        prompt="Stack the butter on top of the cream cheese",
        baseline="(new case — no tester baseline)",
        criterion=(
            "butter grasped-and-lifted and finishes resting ON the cream "
            "cheese (bottom within the is_on_strict window, xy centroid "
            "inside the base), both settled, cream cheese not dragged "
            "more than 10 cm. Both are ~2 cm flat boxes — a precision "
            "place, not a basket drop."
        ),
        checker="stack_on",
        targets=("butter",),
        base="cream_cheese",
    ),
    Case(
        id="all_except_milk",
        prompt="Put all the objects except the milk into the basket",
        baseline="(new case — no tester baseline)",
        criterion=(
            "negation probe: the five non-milk objects finish settled in "
            "the basket AND the milk stays out. Score is 0.75*packed "
            "fraction + 0.25*milk-kept-out."
        ),
        checker="pack_listed",
        targets=("alphabet_soup", "butter", "cream_cheese",
                 "salad_dressing", "tomato_sauce"),
        forbidden=("milk",),
        timeout_min=90,
    ),
    Case(
        id="soup_left_of_milk",
        prompt="Place the alphabet soup to the left of the milk",
        baseline="(new case — no tester baseline)",
        criterion=(
            "spatial-relation probe: soup grasped-and-lifted, set down "
            "settled on the table 6-30 cm from the milk along y and "
            "within 15 cm in x, not in the basket. Either reading of "
            "'left' is accepted (viewer's left is -y in the agentview "
            "video, robot's left is +y — both measured); side logged."
        ),
        checker="lateral_of",
        targets=("alphabet_soup",),
        reference="milk",
    ),
    # -- extended matrix (run2): more VAB tasks / suites ----------------------
    # (scene contents below are PROBED live, not assumed: the VAB grocery
    # vocabulary spans 10 items across tasks; each task carries a
    # different 6-item subset + basket, and the swap suites relocate the
    # basket across inits.)
    Case(
        id="pudding_basket",
        prompt="Pick up the chocolate pudding and place it in the basket",
        baseline="(new case — no tester baseline)",
        criterion=(
            "single pick on a DIFFERENT variance task's scene "
            "(libero_object_all_variance/3 — probed: chocolate_pudding, "
            "orange_juice, bbq_sauce, ketchup, salad_dressing, "
            "alphabet_soup + basket): chocolate_pudding grasped-and-lifted, "
            "settled in the basket, nothing else packed. First coverage of "
            "the pudding box and of a scene whose distractor set differs "
            "from task 0's."
        ),
        sim_task="libero_object_all_variance/3",
        checker="pick_single",
        targets=("chocolate_pudding",),
    ),
    Case(
        id="oj_swap_scene",
        prompt="Pick up the orange juice and place it in the basket",
        baseline="(new case — no tester baseline)",
        criterion=(
            "single pick on the basket-SWAP variance suite "
            "(libero_object_target_basket_swap_variance/5 — probed scene: "
            "bbq_sauce, butter, chocolate_pudding, milk, orange_juice, "
            "tomato_sauce + basket, with the basket relocating across "
            "inits): orange_juice grasped-and-lifted, settled in the "
            "basket, nothing else packed. Two seeds = two basket "
            "placements."
        ),
        sim_task="libero_object_target_basket_swap_variance/5",
        checker="pick_single",
        targets=("orange_juice",),
        seeds=(0, 1),
        timeout_min=75,
    ),
    Case(
        id="pack_scene_all",
        prompt="Pick all the objects and place them in the basket",
        baseline="(new case — no tester baseline)",
        criterion=(
            "the native VAB packing suite (libero_object_packing/0), "
            "prompt VERBATIM from its task language. MEASURED env "
            "mechanics: the suite's pack_all_into predicate TELEPORTS "
            "each contained object to (50.6, 50.0) as its packed-marker, "
            "so the criterion is per-object grasp-and-lift AND "
            "(observed-in-basket OR env-consumed); the env's own reward "
            "is the external cross-check."
        ),
        sim_task="libero_object_packing/0",
        checker="pack_scene",
        env_consumes_packed=True,
        timeout_min=90,
    ),
    Case(
        id="both_boxes",
        prompt="Put both boxes in the basket",
        baseline="(new case — no tester baseline)",
        criterion=(
            "category grounding within the probed task-0 scene: the only "
            "two box-shaped items — butter (7.8x4.0x1.8 cm) and "
            "cream_cheese (8.3x4.3x2.0 cm) — finish settled in the "
            "basket; the can/bottle/carton items all stay out."
        ),
        checker="pack_listed",
        targets=("butter", "cream_cheese"),
        forbidden=("alphabet_soup", "milk", "salad_dressing",
                   "tomato_sauce"),
        timeout_min=60,
    ),
]

CASES: dict[str, Case] = {c.id: c for c in _CASES}
