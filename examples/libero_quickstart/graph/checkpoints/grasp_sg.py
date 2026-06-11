"""Postcondition checkpoints for the quickstart's grasp subgraph.

Loaded by the executor's checkpoint hook (``--checkpoints warn|raise``)
when the connector exposes ground-truth ``world_snapshot`` (sim only).
Evaluated once, against the post-exit World snapshot, every time
``grasp_sg`` completes on its success path.
"""

from gap.runtime.verify import Checkpoint


def _target_held(world) -> bool:
    """The soup can is in contact with a robot link after `close`."""
    return world.body("alphabet soup").is_grasped()


def _target_held_diagnostics(world) -> dict:
    body = world.body("alphabet soup")
    held = world.held_body()
    return {
        "resolved_body": body.name,
        "contacts": sorted(body.contacts),
        "held_body": held.name if held is not None else None,
        "gripper_open_fraction": (
            world.robot().gripper_open_fraction
            if world.robot_view is not None else None
        ),
        "body_z": body.z,
    }


CHECKPOINTS = [
    Checkpoint(
        name="target_held",
        subgraph="grasp_sg",
        predicate=_target_held,
        rationale=(
            "After close, the grasp target must be held: ground-truth "
            "contacts between the can and a robot finger link prove the "
            "gripper actually closed ON the object (not an empty grip)."
        ),
        validate=True,
        diagnostics_fn=_target_held_diagnostics,
    ),
]
