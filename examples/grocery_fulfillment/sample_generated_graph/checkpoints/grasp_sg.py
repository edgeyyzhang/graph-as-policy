"""Checkpoint sidecar for subgraph `grasp_sg`.

The builder block below mirrors the (hand-modified) direct top-down grasp
subgraph and is re-exec'd by the harness to capture the `validate=True`
predicate lambdas with their closure cells. The postconditions
(`grasp_pose_above_table`, `target_held`) are unchanged from the original
generated graph; only the node scaffolding was updated to match the direct
`go_to_pose` grasp.
"""

from __future__ import annotations

import math

import numpy as np

from gap.builder import Subgraph, Ref, START, END
from gap.runtime.verify import Checkpoint as _Checkpoint

# --- original builder block ---
from gap.builder import Subgraph, Ref, START, END
sg = Subgraph(name='grasp_sg', skill='grasping-with-planner')
sg.add_input('target_obb', type_name='OrientedBoundingBox')
sg.add_input('target_mask', type_name='Mask')
sg.add_node('open', type='tool', tool='robot.open_gripper', inputs={'settle_steps': 40})
sg.add_node('compute_grasp', type='tool', tool='geometry.top_down_grasp_candidates', inputs={'obb': Ref('in.target_obb')})
sg.add_node('goto_grasp', type='tool', tool='robot.go_to_pose', inputs={'pose': Ref('compute_grasp.candidates.poses.0'), 'z_approach': 0.1})
sg.add_node('observe', type='tool', tool='robot.get_observation', inputs={})
sg.add_node('close', type='tool', tool='robot.close_gripper', inputs={'settle_steps': 60})
sg.add_exit('grasped')
edges = [(START, 'open'), ('open', 'compute_grasp'), ('compute_grasp', 'goto_grasp'), ('goto_grasp', 'observe'), ('observe', 'close'), ('close', 'grasped'), ('grasped', END)]
for u, v in edges:
    sg.add_edge(u, v)
sg.set_outputs(ee_pose_at_grasp=Ref('observe.arms.0.ee_pose'), grasp_pose=Ref('compute_grasp.candidates.poses.0'))
sg.set_on_error('failed')

sg.add_checkpoint('grasp_pose_above_table', predicate=lambda w, o: o['grasp_pose']['position']['z'] > 0.01, rationale='planned grasp z-height is above the tabletop', validate=True)
sg.add_checkpoint('target_held', predicate=lambda w: w.body('alphabet soup').is_grasped(), rationale='the alphabet soup can is held by the robot gripper', validate=True)
# --- end original builder block ---

CHECKPOINTS: list[_Checkpoint] = [
    _Checkpoint(
        name=_c.name,
        subgraph=sg.name,
        predicate=_c.predicate,
        diagnostics_fn=_c.diagnostics,
        rationale=_c.rationale,
        validate=_c.validate,
        weight=_c.weight,
    )
    for _c in sg._checkpoints
]
