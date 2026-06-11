"""Auto-generated checkpoint sidecar for subgraph `container_sg`.

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
sg = Subgraph(name='container_sg', skill='perceiving-objects')
sg.add_node('observe', type='tool', tool='robot.get_observation')
sg.add_node('perceive', type='script', script='scripts/perceive_dino_vlm.py', inputs={'cameras': Ref('observe.cameras'), 'object_name': 'basket'})
sg.add_node('filter_obb', type='tool', tool='geometry.filter_and_compute_obb', inputs={'points': Ref('perceive.cloud')})
sg.add_exit('found')
sg.add_edge(START, 'observe')
sg.add_edge('observe', 'perceive')
sg.add_edge('perceive', 'filter_obb')
sg.add_edge('filter_obb', 'found')
sg.add_edge('found', END)
sg.set_on_error('not_found')
sg.set_outputs(container_obb=Ref('filter_obb.obb'), container_mask=Ref('perceive.mask'), container_cloud=Ref('perceive.cloud'))

sg.add_checkpoint('container_obb_matches_truth', predicate=lambda w, o: abs(o['container_obb']['center']['x'] - w.body('basket').position[0]) < 0.05 and abs(o['container_obb']['center']['y'] - w.body('basket').position[1]) < 0.05, rationale='container OBB centered over the privileged basket pose', validate=True)
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
