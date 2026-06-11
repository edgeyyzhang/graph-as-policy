"""Auto-generated checkpoint sidecar for subgraph `transport_sg`.

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
sg = Subgraph(name='transport_sg', skill='transporting-objects')
sg.add_input('target_obb', type_name='OrientedBoundingBox')
sg.add_input('target_mask', type_name='Mask')
sg.add_input('container_obb', type_name='OrientedBoundingBox')
sg.add_input('container_mask', type_name='Mask')
sg.add_input('ee_pose_at_grasp', type_name='Se3Pose')
sg.add_node('compute_drop', type='script', script='scripts/compute_drop_pose.py', inputs={'container_obb': Ref('in.container_obb'), 'held_obb': Ref('in.target_obb'), 'ee_pose_at_grasp': Ref('in.ee_pose_at_grasp')})
sg.add_node('move_above', type='script', script='scripts/waypoint_move.py', inputs={'drop_x': Ref('compute_drop.drop_position.x'), 'drop_y': Ref('compute_drop.drop_position.y')})
sg.add_node('release', type='script', script='scripts/descend_release.py', inputs={'drop_position': Ref('compute_drop.drop_position')})
sg.add_exit('placed')
sg.add_edge(START, 'compute_drop')
sg.add_edge('compute_drop', 'move_above')
sg.add_edge('move_above', 'release')
sg.add_edge('release', 'placed')
sg.add_edge('placed', END)
sg.set_on_error('blocked')
sg.set_outputs(drop_pose=Ref('compute_drop.drop_pose'))

sg.add_checkpoint('drop_inside_cavity', predicate=lambda w, o: w.body('basket').cavity_lower[0] < o['drop_pose']['position']['x'] < w.body('basket').cavity_upper[0] and w.body('basket').cavity_lower[1] < o['drop_pose']['position']['y'] < w.body('basket').cavity_upper[1], rationale='planned drop xy lands inside the basket cavity AABB', validate=True)
sg.add_checkpoint('drop_z_valid', predicate=lambda w, o: o['drop_pose']['position']['z'] > w.body('basket').cavity_lower[2] - 0.01, rationale='planned drop z is at or above the basket cavity floor', validate=True)
sg.add_checkpoint('target_in_container', predicate=lambda w: w.body('alphabet soup').is_in(w.body('basket')), rationale='alphabet soup settled inside the basket after release', validate=True)
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
