"""The rehearsal loop: an authoring agent edits a graph, the broker rehearses it.

Two sides, connected only by request files:

- the **workspace** is what the agent sees: the working graph, the skill
  library, the feedback and trajectories of the visible cases, and two
  commands (``validate`` and ``rehearse``);
- the **trusted side** holds the simulator, the perception servers, the
  credentials, the held-out cases and the ledger of every round.

:mod:`.workspace` creates both, :mod:`.broker` serves the requests, and
:mod:`.protocol` holds the checks a submitted graph must pass.
"""

from .broker import Broker
from .protocol import GraphRejected, check_graph
from .workspace import init_loop

__all__ = ["Broker", "init_loop", "check_graph", "GraphRejected"]
