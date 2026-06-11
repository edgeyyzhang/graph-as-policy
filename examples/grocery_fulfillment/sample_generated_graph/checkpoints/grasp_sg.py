"""Auto-generated checkpoint sidecar for subgraph `grasp_sg`.

Do not edit. The original builder block is preserved verbatim
below so the harness can re-exec it and capture predicate
lambdas with their original closure cells.
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
sg.add_node('approach', type='script', script='scripts/approach_above.py', inputs={'target_position': Ref('compute_grasp.candidates.poses.0.position'), 'rotation': Ref('compute_grasp.candidates.poses.0.rotation'), 'target_obb': Ref('in.target_obb')})
sg.add_node('observe', type='tool', tool='robot.get_observation', inputs={})
sg.add_node('build_world', type='script', script='scripts/build_world.py', inputs={'observation': Ref('observe'), 'target_mask': Ref('in.target_mask'), 'target_obb': Ref('in.target_obb'), 'target_name': 'target'})
sg.add_node('plan', type='script', script='scripts/plan_grasp.py', inputs={'world_config': Ref('build_world.config'), 'observation': Ref('observe'), 'grasp_poses': Ref('compute_grasp.candidates.poses'), 'target_name': 'target'})
sg.add_node('execute', type='tool', tool='robot.execute_trajectory', inputs={'trajectory': Ref('plan.trajectory')})
sg.add_node('close', type='tool', tool='robot.close_gripper', inputs={'settle_steps': 60})
sg.add_exit('grasped')
edges = [(START, 'open'), ('open', 'compute_grasp'), ('compute_grasp', 'approach'), ('approach', 'observe'), ('observe', 'build_world'), ('build_world', 'plan'), ('plan', 'execute'), ('execute', 'close'), ('close', 'grasped'), ('grasped', END)]
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
