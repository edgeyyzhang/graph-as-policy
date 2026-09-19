"""Privileged ``sim.*`` tools — scene geometry read straight from the sim.

In simulation the answer to *"where is the mug and how big is it"* is already
in the MjModel; there is nothing to detect. These tools hand that answer over
in exactly the shapes the perception tools produce, so a skill written against
``geometry.filter_and_compute_obb`` runs unchanged against
``sim.get_object_obb``. That interchangeability is the point: it is what lets
one graph be developed with privileged state and later ported to cameras
without rewriting the skills in between.

Four rules this module keeps:

- **Connector world frame.** LIBERO exposes camera geometry and motion targets
  in the robot-base frame and calls that its public world frame. The adapter
  converts raw MuJoCo-world coordinates into that same frame, so privileged
  OBBs, perceived OBBs, and ``robot.go_to_pose`` agree. Payloads retain
  ``frame="world"`` for API compatibility and add ``frame_convention``
  to name the concrete basis (``robot_base`` or ``mujoco_world``).
- **Half-extents.** ``OrientedBoundingBox.extent`` is half the side length,
  matching :mod:`gap_core.types` and the ``geometry.*`` bundle. Emitting full
  extents here would double every grasp offset, silently.
- **Never guess a name.** An unresolved ``object_name`` raises with the list
  of names that *do* resolve, rather than returning the nearest match.
- **An object is not its parts.** ``sim.get_object_obb`` bounds the whole
  subtree; ``sim.get_part_obb`` bounds one body inside it. For a drawer those
  are the cabinet and the sliding link, and only one of them is graspable.

``sim.clearance`` and ``sim.query`` are deliberately absent: the skills that
want them treat a missing tool as "this gate could not run", which is the
correct behaviour until they are backed by something better than a guess.
"""

from __future__ import annotations

import math
import re
from typing import Any

import numpy as np
from gap_core.errors import ToolError
from gap_core.tools import ToolRegistry
from gap_core.types import matrix_to_pose, pose_to_matrix

#: Word split for name matching. Sim identifiers are snake/camel-free
#: lowercase with underscores or digits ("akita_black_bowl_1"); noun phrases
#: are spaced. Splitting both on non-alphanumerics puts them in one alphabet.
_WORD_RE = re.compile(r"[a-z0-9]+")

#: Words that carry no identity and would otherwise create spurious overlap
#: between "the top drawer" and any body whose name happens to contain "top".
_STOPWORDS = frozenset({"the", "a", "an", "of", "on", "in", "to", "and"})


def _words(text: str) -> list[str]:
    return [w for w in _WORD_RE.findall(text) if w not in _STOPWORDS]


def _pick(tool: str, want: Any, names: list[str], *, context: str) -> str:
    """Choose the one name in *names* that *want* refers to.

    Callers write noun phrases ("the top drawer"); the model holds
    identifiers ("DrawerObject_drawer_link"). Four tiers bridge that, each
    looser than the last and every one of them still requiring a UNIQUE
    winner: exact, case-insensitive, substring in either direction, then
    shared words. An ambiguous match is an error, not a coin toss, and no
    match at all raises listing *names* — the caller is handed the
    vocabulary rather than left guessing at it.

    What this deliberately does not do is fall back to "well, there is only
    one candidate, so it must be that one". That inference is sometimes right
    and is exactly how a mistyped name becomes a confident wrong answer, so
    it belongs to a skill that can log and own it, not here.
    """
    if not want:
        raise ToolError(tool, f"name required; {context} are {sorted(names)}")
    want = str(want)
    if want in names:
        return want
    lowered = {n.lower(): n for n in names}
    if want.lower() in lowered:
        return lowered[want.lower()]

    low = want.lower()
    tiers = [
        [n for n in names if low in n.lower()],
        [n for n in names if n.lower() in low],
    ]
    words = set(_words(low))
    if words:
        overlap = {n: len(words & set(_words(n.lower()))) for n in names}
        best = max(overlap.values(), default=0)
        if best:
            tiers.append([n for n, c in overlap.items() if c == best])
    for hits in tiers:
        if len(hits) == 1:
            return hits[0]
        if len(hits) > 1:
            raise ToolError(
                tool, f"{want!r} matches {sorted(hits)}; name one of them exactly"
            )
    raise ToolError(tool, f"nothing named {want!r}; {context} are {sorted(names)}")


def _vec3(v: Any) -> dict[str, float]:
    return {"x": float(v[0]), "y": float(v[1]), "z": float(v[2])}


def _quat_xyzw(wxyz: np.ndarray) -> dict[str, float]:
    """MuJoCo stores wxyz; ``Quaternion`` on the wire is keyed x/y/z/w."""
    w, x, y, z = (float(c) for c in wxyz[:4])
    return {"x": x, "y": y, "z": z, "w": w}


def _rotmat(wxyz: np.ndarray) -> np.ndarray:
    from gap.connector.world_adapter import _quat_wxyz_to_rotmat

    return _quat_wxyz_to_rotmat(np.asarray(wxyz, dtype=np.float64))


def _grasp_yaw_deg(half: np.ndarray, quat_wxyz: np.ndarray) -> float:
    """Compass heading of the box's longest HORIZONTAL axis, in degrees.

    This is the object's long axis — *not* the direction the jaws should
    close. They are ninety degrees apart, and the grasp-proposal skills derive
    the closing heading square to this one. Reported in (-90, 90] because a
    box axis and its negation are the same axis.
    """
    R = _rotmat(quat_wxyz)
    # The axis most aligned with world +z is "up"; the long axis is the longer
    # of the other two. Same rule refine_top_down_grasp uses, so the numbers
    # agree between the privileged and the perceived path.
    vertical = int(np.argmax(np.abs(R.T @ np.array([0.0, 0.0, 1.0]))))
    horizontal = [i for i in range(3) if i != vertical]
    long_axis = max(horizontal, key=lambda i: float(half[i]))
    d = R[:, long_axis]
    yaw = math.degrees(math.atan2(float(d[1]), float(d[0])))
    while yaw > 90.0:
        yaw -= 180.0
    while yaw <= -90.0:
        yaw += 180.0
    return yaw


class PrivilegedSimTools:
    """The ``sim.*`` ground-truth tools, backed by the world adapter.

    Held by :class:`gap.connector.sim.SimConnector`, which owns the adapter's
    lifecycle (it is rebuilt on every reset, because a robosuite hard reset
    rebuilds the MjModel).
    """

    def __init__(self, connector: Any) -> None:
        self._connector = connector

    # ------------------------------------------------------------------

    def register(self, reg: ToolRegistry) -> None:
        rc = reg.register_callable
        rc("sim.list_objects", self.list_objects,
           summary="List every scene body the privileged tools can resolve.")
        rc("sim.get_object_pose", self.get_object_pose,
           summary="Ground-truth world pose of a scene object.")
        rc("sim.get_object_obb", self.get_object_obb,
           summary="Ground-truth oriented bounding box of a scene object, "
                   "in the same shape geometry.compute_obb returns.")
        rc("sim.get_part_obb", self.get_part_obb,
           summary="Ground-truth oriented bounding box of one sub-body of a "
                   "scene object, e.g. a drawer's sliding link.")
        rc("sim.list_entities", self.list_entities,
           summary="List portable scene entities and their capabilities.")
        rc("sim.resolve_entity", self.resolve_entity,
           summary="Resolve a semantic query to one exact scene entity.")
        rc("sim.get_entity_state", self.get_entity_state,
           summary="Portable pose, OBB, parts and features of an entity.")
        rc("sim.list_features", self.list_features,
           summary="List exact geometric features owned by an entity.")
        rc("sim.get_feature", self.get_feature,
           summary="Get a portable SceneFeature, optionally with mating semantics.")
        rc("sim.list_articulations", self.list_articulations,
           summary="List prismatic and revolute DOFs without backend joint ids.")
        rc("sim.get_articulation", self.get_articulation,
           summary="Get axis, limits, position and progress of an articulation.")
        rc("sim.get_contacts", self.get_contacts,
           summary="Get exact live contacts for one semantic entity.")
        rc("sim.get_task_flags", self.get_task_flags,
           summary="Get benchmark task flags and history-dependent latches.")
        rc("sim.relative_pose", self.relative_pose,
           summary="Express one entity pose in another entity's frame.")
        rc("sim.evaluate_condition", self.evaluate_condition,
           summary="Evaluate a portable graph milestone from privileged state.")
        rc("sim.describe_condition", self.describe_condition,
           summary="Expand one named benchmark condition without stepping the simulator.")

    # ------------------------------------------------------------------

    def _adapter(self) -> Any:
        ensure = getattr(self._connector, "_ensure_world_adapter", None)
        if callable(ensure):
            adapter = ensure()
        else:
            # Lightweight test connectors and third-party sim connectors may
            # predate SimConnector's shared lazy-initialization seam.
            adapter = self._connector._world_adapter
            if adapter is None:
                from gap.connector.world_adapter import LiberoWorldAdapter

                adapter = LiberoWorldAdapter(self._connector.env)
                self._connector._world_adapter = adapter
        required = ("object_names", "object_pose", "object_box")
        missing = [name for name in required if not callable(getattr(adapter, name, None))]
        if missing:
            raise ToolError(
                "sim",
                "scene adapter does not implement the portable privileged "
                f"provider contract; missing {missing}. Check "
                "`capabilities.world_state` before routing through sim.* or "
                "use a perception provider instead",
            )
        return adapter

    def _resolve(self, tool: str, object_name: Any) -> tuple[Any, str]:
        """Map a caller's object name onto a body the sim actually has.

        Matching tiers and their rationale live in :func:`_pick`; this only
        supplies the scene's object vocabulary and the live adapter.
        """
        adapter = self._adapter()
        names = adapter.object_names()
        return adapter, _pick(tool, object_name, names, context="scene objects")

    # ------------------------------------------------------------------

    def list_objects(self) -> dict:
        """Every resolvable body name, with its live connector-world position and kind.

        Call this first when the object's name is not already known from the
        task: it is the scene's vocabulary, and the other two tools refuse
        anything outside it.

        ``kind`` is ``"object"`` for a body the task itself declares and
        ``"fixture"`` for scene furniture (the tabletop). It is there because
        sim body names are not noun phrases — LIBERO calls the one manipulable
        body ``object`` — so a caller holding "the white mug" needs to narrow
        by role when it cannot narrow by text.

        ``movable`` separates the objects a hand could carry away (free
        joint) from the ones bolted to the scene that merely articulate — the
        mug rather than the drawer it goes into. Both are ``kind: "object"``,
        and only one of them is ever the target of a pick.

        ``parts`` lists the object's own sub-bodies, if any, for
        :meth:`get_part_obb`. An articulated object's box is the box of the
        whole cabinet; the piece a grasp targets is one of these.
        """
        adapter = self._adapter()
        task = set(adapter.task_object_names())
        movable = set(adapter.movable_object_names())
        objects = []
        for name in adapter.object_names():
            pose = adapter.object_pose(name)
            objects.append({
                "name": name,
                "position": _vec3(pose[0]) if pose else None,
                "kind": "object" if name in task else "fixture",
                "movable": name in movable,
                "parts": list(adapter.part_names(name)),
            })
        return {
            "objects": objects,
            "frame": "world",
            "frame_convention": self._frame_convention(adapter),
        }

    def get_object_pose(self, object_name: str) -> dict:
        """The body origin's connector-world pose — position plus orientation.

        Note this is the *origin*, which is not the centre of the geometry
        when the shape is off-axis. For anything that has to line up with the
        object's bulk (a grasp, a drop, a clearance), use
        :meth:`get_object_obb` and read its centre.
        """
        adapter, name = self._resolve("sim.get_object_pose", object_name)
        pose = adapter.object_pose(name)
        if pose is None:
            raise ToolError("sim.get_object_pose", f"{name!r} has no live body state")
        position, quat = pose
        return {
            "name": name,
            "pose": {"position": _vec3(position), "rotation": _quat_xyzw(quat)},
            "frame": "world",
            "frame_convention": self._frame_convention(adapter),
        }

    def get_object_obb(self, object_name: str) -> dict:
        """The object's live oriented bounding box, in connector world frame.

        Same payload shape as ``geometry.filter_and_compute_obb`` so the two
        are drop-in substitutes: ``obb.extent`` is **half** the side length,
        and ``grasp_yaw`` is the heading of the long horizontal axis.
        """
        adapter, name = self._resolve("sim.get_object_obb", object_name)
        return self._box_payload("sim.get_object_obb", adapter, name)

    def get_part_obb(self, object_name: str, part_name: str) -> dict:
        """The live oriented box of ONE sub-body of *object_name*.

        An articulated object is one entry in :meth:`list_objects` but several
        bodies in the model, and the distinction is the whole game when the
        object moves internally: ``sim.get_object_obb("drawer")`` bounds the
        entire cabinet, while the piece a hand has to pull is the sliding
        link. Ask for the object when you want where it *is*; ask for the part
        when you want what to *touch*.

        *part_name* is matched against that object's parts by the same tiers
        :meth:`get_object_obb` uses on object names, so "drawer link" finds
        ``DrawerObject_drawer_link``. The search is scoped to this object's
        subtree, so a part name never collides with another object's.
        """
        adapter, root = self._resolve("sim.get_part_obb", object_name)
        parts = adapter.part_names(root)
        if not parts:
            raise ToolError(
                "sim.get_part_obb",
                f"{root!r} is a single rigid body with no sub-parts; use "
                f"sim.get_object_obb instead",
            )
        part = _pick("sim.get_part_obb", part_name, parts,
                     context=f"parts of {root!r}")
        payload = self._box_payload("sim.get_part_obb", adapter, part)
        payload["object"] = root
        return payload

    # ------------------------------------------------------------------
    # Portable scene-provider surface. The legacy object tools above remain
    # for graph compatibility; new skills should consume these typed payloads.

    def list_entities(self) -> dict:
        """List scene entities without exposing backend object kinds."""
        adapter = self._adapter()
        task = set(adapter.task_object_names())
        movable = set(adapter.movable_object_names())
        entities = []
        for name in adapter.object_names():
            entities.append({
                "name": name,
                "kind": "object" if name in task else "fixture",
                "movable": name in movable,
                "parts": list(adapter.part_names(name)),
                "features": self._feature_names(adapter, name),
                "articulations": self._articulation_names(adapter, name),
            })
        return {
            "entities": entities,
            "frame": "world",
            "frame_convention": self._frame_convention(adapter),
            "source": "privileged",
        }

    def resolve_entity(
        self,
        query: str,
        kind: str = "",
        movable: bool | None = None,
    ) -> dict:
        """Resolve *query* after optional semantic filtering.

        Ambiguity is an error. Object-role aliases belong in task metadata or
        a provider, never in a generic skill.
        """
        adapter = self._adapter()
        task = set(adapter.task_object_names())
        movable_names = set(adapter.movable_object_names())
        names = list(adapter.object_names())
        if kind:
            if kind not in {"object", "fixture"}:
                raise ToolError("sim.resolve_entity", "kind must be object or fixture")
            names = [n for n in names if (n in task) == (kind == "object")]
        if movable is not None:
            names = [n for n in names if (n in movable_names) == movable]
        name = _pick("sim.resolve_entity", query, names, context="filtered entities")
        return self.get_entity_state(name)

    def get_entity_state(self, entity: str) -> dict:
        """Return the shared :class:`SceneEntity` representation."""
        adapter, name = self._resolve("sim.get_entity_state", entity)
        pose = adapter.object_pose(name)
        box = adapter.object_box(name)
        if pose is None or box is None:
            raise ToolError("sim.get_entity_state", f"{name!r} has incomplete live state")
        position, quat = pose
        center, half, box_quat = box
        task = set(adapter.task_object_names())
        result = {
            "name": name,
            "kind": "object" if name in task else "fixture",
            "movable": name in set(adapter.movable_object_names()),
            "pose": {"position": _vec3(position), "rotation": _quat_xyzw(quat)},
            "obb": {
                "center": _vec3(center),
                "extent": _vec3(half),
                "orientation": _quat_xyzw(box_quat),
            },
            "frame": "world",
            "source": "privileged",
            "backend_id": name,
            "parts": list(adapter.part_names(name)),
            "features": self._feature_names(adapter, name),
        }
        return {
            "entity": result,
            "frame_convention": self._frame_convention(adapter),
        }

    @staticmethod
    def _feature_names(adapter: Any, root: str) -> list[str]:
        fn = getattr(adapter, "feature_names", None)
        return list(fn(root)) if callable(fn) else list(adapter.part_names(root))

    @staticmethod
    def _articulation_names(adapter: Any, root: str | None = None) -> list[str]:
        fn = getattr(adapter, "articulation_names", None)
        return list(fn(root)) if callable(fn) else []

    def list_features(self, entity: str) -> dict:
        adapter, root = self._resolve("sim.list_features", entity)
        return {
            "entity": root,
            "features": self._feature_names(adapter, root),
            "source": "privileged",
        }

    def get_feature(
        self,
        entity: str,
        feature: str,
        kind: str = "part",
        axis: str = "long",
        axis_sign: int = 1,
        radius_inner: float = 0.0,
        insertion_depth: float = 0.0,
    ) -> dict:
        """Return exact feature geometry and optional mating semantics.

        ``kind`` is semantic task metadata, not inferred from a simulator
        name. For loop/aperture features the caller must state the usable
        inner radius; silently deriving a hole from a solid geom is unsafe.
        """
        adapter, root = self._resolve("sim.get_feature", entity)
        names = self._feature_names(adapter, root)
        selected = _pick("sim.get_feature", feature, names, context=f"features of {root!r}")
        box_fn = getattr(adapter, "feature_box", None)
        if not callable(box_fn):
            raise ToolError("sim.get_feature", "scene provider has no feature geometry capability")
        box = box_fn(root, selected)
        if box is None:
            raise ToolError("sim.get_feature", f"feature {selected!r} has no live geometry")
        center, half, quat = box
        pose = {"position": _vec3(center), "rotation": _quat_xyzw(quat)}
        obb = {
            "center": _vec3(center),
            "extent": _vec3(half),
            "orientation": _quat_xyzw(quat),
        }
        payload = {
            "name": selected,
            "parent": root,
            "kind": kind,
            "pose": pose,
            "obb": obb,
            "frame": "world",
            "source": "privileged",
            "backend_id": selected,
        }
        functional_kinds = {"loop", "shaft", "tip", "aperture", "surface", "region"}
        semantic_kind = "shaft" if kind == "handle" else kind
        if semantic_kind in functional_kinds:
            if semantic_kind in {"loop", "aperture"} and radius_inner <= 0.0:
                raise ToolError(
                    "sim.get_feature",
                    f"{semantic_kind} requires radius_inner from task/provider metadata",
                )
            index = self._feature_axis(axis, half)
            direction = _rotmat(quat)[:, index] * (1.0 if axis_sign >= 0 else -1.0)
            other = [float(half[i]) for i in range(3) if i != index]
            functional = {
                "kind": semantic_kind,
                "pose": pose,
                "axis": _vec3(direction),
                "description": f"privileged feature {selected} of {root}",
                "confidence": 1.0,
                "fit_quality": 1.0,
            }
            if semantic_kind in {"shaft", "tip"}:
                functional["radius_outer"] = min(other)
                functional["length"] = 2.0 * float(half[index])
            if semantic_kind in {"loop", "aperture"}:
                functional["radius_inner"] = float(radius_inner)
            if insertion_depth > 0.0:
                functional["insertion_depth"] = float(insertion_depth)
            payload["functional"] = functional
        return {"feature": payload}

    @staticmethod
    def _feature_axis(axis: str, half: np.ndarray) -> int:
        if axis in {"x", "y", "z"}:
            return {"x": 0, "y": 1, "z": 2}[axis]
        if axis == "long":
            return int(np.argmax(half))
        if axis == "short":
            return int(np.argmin(half))
        raise ToolError("sim.get_feature", "axis must be x, y, z, long, or short")

    def list_articulations(self, entity: str = "") -> dict:
        adapter = self._adapter()
        root = None
        if entity:
            adapter, root = self._resolve("sim.list_articulations", entity)
        names = self._articulation_names(adapter, root)
        states = []
        state_fn = getattr(adapter, "articulation_state", None)
        if callable(state_fn):
            states = [self._articulation_payload(adapter, state_fn(n)) for n in names]
        return {"entity": root, "articulations": states, "source": "privileged"}

    def get_articulation(self, entity: str, articulation: str = "") -> dict:
        adapter, root = self._resolve("sim.get_articulation", entity)
        names = self._articulation_names(adapter, root)
        if not names:
            raise ToolError("sim.get_articulation", f"{root!r} has no scalar articulation")
        if not articulation and len(names) == 1:
            name = names[0]
        else:
            name = _pick("sim.get_articulation", articulation, names,
                         context=f"articulations of {root!r}")
        state_fn = getattr(adapter, "articulation_state", None)
        state = state_fn(name) if callable(state_fn) else None
        if state is None:
            raise ToolError("sim.get_articulation", "scene provider has no articulation state capability")
        return {"articulation": self._articulation_payload(adapter, state)}

    def _articulation_payload(self, adapter: Any, state: dict) -> dict:
        out = dict(state)
        out["axis"] = _vec3(state["axis"])
        out["pivot"] = _vec3(state["pivot"])
        out["frame"] = "world"
        out["source"] = "privileged"
        out["backend_id"] = state["name"]
        return out

    def get_contacts(self, entity: str) -> dict:
        adapter, root = self._resolve("sim.get_contacts", entity)
        fn = getattr(adapter, "contact_pairs", None)
        if not callable(fn):
            raise ToolError("sim.get_contacts", "scene provider has no contact capability")
        pairs = [tuple(pair) for pair in fn()]
        touching = sorted({b if a == root else a for a, b in pairs if root in (a, b)})
        return {"entity": root, "touching": touching, "pairs": pairs, "source": "privileged"}

    def relative_pose(self, source: str, reference: str) -> dict:
        """Pose of *source* expressed in the live *reference* frame."""
        source_state = self.get_entity_state(source)["entity"]
        ref_state = self.get_entity_state(reference)["entity"]
        relative = np.linalg.inv(pose_to_matrix(ref_state["pose"])) @ pose_to_matrix(source_state["pose"])
        return {
            "source": source_state["name"],
            "reference": ref_state["name"],
            "pose": matrix_to_pose(relative),
            "frame": f"entity:{ref_state['name']}",
            "source_type": "privileged",
        }

    def get_task_flags(self) -> dict:
        """Return the benchmark adapter's live flags without interpreting them."""
        getter = getattr(self._connector, "get_task_flags", None)
        if not callable(getter):
            raise ToolError(
                "sim.get_task_flags",
                "connector has no task-flags provider; inject one from the task adapter",
            )
        flags = getter()
        if not isinstance(flags, dict):
            try:
                flags = dict(flags)
            except Exception as exc:
                raise ToolError(
                    "sim.get_task_flags", "task-flags provider returned no mapping"
                ) from exc
        return {"flags": dict(flags), "source": "task_adapter"}

    def evaluate_condition(self, condition: dict) -> dict:
        """Evaluate a portable relation tree against exact live task state.

        Supports both semantic scene relations and the complete Agent2Policy
        task-contract vocabulary: ref, flag, joint, offset, in_box, touching,
        all, any, and not. Backend names are resolved by the provider.
        """
        adapter = self._adapter()
        snapshot = getattr(adapter, "snapshot", None)
        if not callable(snapshot):
            raise ToolError(
                "sim.evaluate_condition", "scene provider has no snapshot capability"
            )
        world = snapshot()
        satisfied, diagnostics = self._eval_condition(
            adapter, world, condition, references=(),
        )
        return {
            "satisfied": bool(satisfied),
            "condition": condition,
            "diagnostics": diagnostics,
        }

    def describe_condition(self, name: str) -> dict:
        """Return a named task condition with every nested ref expanded.

        This is deliberately read-only and simulator-neutral. It exposes no
        information beyond the task contract already injected by the benchmark,
        but saves graph authors from hunting through adapter source or guessing
        which geom/site names a compact milestone refers to.
        """
        definitions = self._condition_definitions()
        if name not in definitions:
            raise ToolError(
                "sim.describe_condition",
                f"no named condition {name!r}; available: {sorted(definitions)}",
            )

        def expand(condition: Any, chain: tuple[str, ...]) -> Any:
            if not isinstance(condition, dict) or len(condition) != 1:
                return condition
            op, args = next(iter(condition.items()))
            if op == "ref":
                ref = str(args)
                if ref not in definitions:
                    raise ToolError(
                        "sim.describe_condition",
                        f"condition {chain[-1]!r} refers to unknown condition {ref!r}",
                    )
                if ref in chain:
                    cycle = " -> ".join((*chain, ref))
                    raise ToolError(
                        "sim.describe_condition", f"circular condition reference: {cycle}"
                    )
                return {"ref": ref, "expanded": expand(definitions[ref], (*chain, ref))}
            if op in {"all", "any"} and isinstance(args, list):
                return {op: [expand(child, chain) for child in args]}
            if op == "not":
                return {op: expand(args, chain)}
            return condition

        return {
            "name": name,
            "condition": definitions[name],
            "expanded": expand(definitions[name], (name,)),
            "available": sorted(definitions),
            "source": "task_adapter",
        }

    def _condition_definitions(self) -> dict[str, Any]:
        getter = getattr(self._connector, "get_condition_definitions", None)
        return dict(getter() or {}) if callable(getter) else {}

    @staticmethod
    def _condition_names(value: Any) -> tuple[str, ...]:
        if isinstance(value, str):
            return (value,)
        if isinstance(value, (list, tuple)):
            return tuple(str(item) for item in value)
        raise ToolError(
            "sim.evaluate_condition",
            f"expected a name or list of names, got {value!r}",
        )

    @staticmethod
    def _bounded(value: float, spec: dict, label: str) -> tuple[bool, dict]:
        lower, upper = spec.get("at_least"), spec.get("at_most")
        if lower is None and upper is None:
            raise ToolError(
                "sim.evaluate_condition",
                f"{label} needs at_least, at_most, or both",
            )
        passed = (
            (lower is None or value >= float(lower))
            and (upper is None or value <= float(upper))
        )
        return passed, {
            "value": float(value),
            "at_least": None if lower is None else float(lower),
            "at_most": None if upper is None else float(upper),
        }

    @staticmethod
    def _named_frame(adapter: Any, name: str) -> tuple[np.ndarray, np.ndarray]:
        getter = getattr(adapter, "named_frame_pose", None)
        reading = getter(name) if callable(getter) else None
        if reading is None:
            reading = adapter.object_pose(name)
        if reading is None:
            raise ToolError(
                "sim.evaluate_condition",
                f"no collision feature, body, site, or frame named {name!r}",
            )
        position, quat = reading
        return np.asarray(position, dtype=np.float64), np.asarray(
            quat, dtype=np.float64
        )

    def _local_offset(self, adapter: Any, body: str, frame: str) -> np.ndarray:
        body_position, _body_quat = self._named_frame(adapter, body)
        frame_position, frame_quat = self._named_frame(adapter, frame)
        return _rotmat(frame_quat).T @ (body_position - frame_position)

    def _collision_touching(self, adapter: Any, args: dict) -> tuple[bool, dict]:
        getter = getattr(adapter, "collision_feature_contact_pairs", None)
        if not callable(getter):
            raise ToolError(
                "sim.evaluate_condition",
                "scene provider has no exact collision-feature contact capability",
            )
        first = self._condition_names(args["a"])
        second = self._condition_names(args["b"])
        pairs = [tuple(str(name) for name in pair) for pair in getter()]
        matches = [
            pair for pair in pairs
            if ((pair[0] in first and pair[1] in second)
                or (pair[0] in second and pair[1] in first))
        ]
        return bool(matches), {
            "operator": "touching",
            "level": "collision_feature",
            "a": list(first),
            "b": list(second),
            "matching_pairs": matches,
        }

    def _eval_condition(
        self,
        adapter: Any,
        world: Any,
        condition: dict,
        *,
        references: tuple[str, ...],
    ) -> tuple[bool, dict]:
        if not isinstance(condition, dict) or len(condition) != 1:
            raise ToolError(
                "sim.evaluate_condition",
                "condition must contain exactly one operator",
            )
        op, args = next(iter(condition.items()))
        if op == "ref":
            name = str(args)
            definitions = self._condition_definitions()
            if name not in definitions:
                raise ToolError(
                    "sim.evaluate_condition",
                    f"no named condition {name!r} in condition_definitions",
                )
            if name in references:
                chain = " -> ".join((*references, name))
                raise ToolError(
                    "sim.evaluate_condition", f"circular condition reference: {chain}"
                )
            value, diag = self._eval_condition(
                adapter,
                world,
                definitions[name],
                references=(*references, name),
            )
            return value, {"operator": "ref", "name": name, "resolved": diag}
        if op in {"all", "any"}:
            if not isinstance(args, list):
                raise ToolError("sim.evaluate_condition", f"{op} requires a list")
            results = [
                self._eval_condition(
                    adapter, world, child, references=references,
                )
                for child in args
            ]
            values = [value for value, _diag in results]
            return (
                all(values) if op == "all" else any(values),
                {"operator": op, "children": [diag for _value, diag in results]},
            )
        if op == "not":
            value, diag = self._eval_condition(
                adapter, world, args, references=references,
            )
            return not value, {"operator": "not", "child": diag}
        if op == "flag":
            if isinstance(args, dict):
                name = str(args.get("name", ""))
                expected = args.get("equals", True)
            else:
                name, expected = str(args), True
            flags = self.get_task_flags()["flags"]
            if name not in flags:
                raise ToolError(
                    "sim.evaluate_condition",
                    f"no task flag named {name!r}; available flags are {sorted(flags)}",
                )
            actual = flags[name]
            value = actual == expected
            return value, {
                "operator": op,
                "name": name,
                "actual": actual,
                "expected": expected,
            }
        if not isinstance(args, dict):
            raise ToolError(
                "sim.evaluate_condition", f"{op} requires an argument object"
            )

        def entity(key: str) -> tuple[str, Any]:
            _provider, name = self._resolve("sim.evaluate_condition", args[key])
            return name, world.body(name)

        if op == "touching":
            if not isinstance(args.get("a"), str) or not isinstance(args.get("b"), str):
                return self._collision_touching(adapter, args)
            try:
                a_name, a = entity("a")
                b_name, _body = entity("b")
            except ToolError:
                return self._collision_touching(adapter, args)
            value = b_name in a.contacts
            return value, {
                "operator": op,
                "level": "entity",
                "a": a_name,
                "b": b_name,
                "contacts": sorted(a.contacts),
            }
        if op == "joint":
            name = str(args.get("name", ""))
            getter = getattr(adapter, "articulation_state", None)
            reading = getter(name) if callable(getter) else None
            if reading is None:
                raise ToolError(
                    "sim.evaluate_condition", f"no scalar joint named {name!r}"
                )
            value, bounds = self._bounded(
                float(reading["position"]), args, f"joint {name!r}"
            )
            return value, {"operator": op, "name": name, **bounds}
        if op in {"offset", "in_box"}:
            body, frame = str(args["body"]), str(args["frame"])
            local = self._local_offset(adapter, body, frame)
            coordinates = {
                axis: float(local[index])
                for index, axis in enumerate(("x", "y", "z"))
            }
            if op == "offset":
                axis = str(args.get("axis", "z"))
                if axis not in coordinates:
                    raise ToolError(
                        "sim.evaluate_condition", "axis must be x, y, or z"
                    )
                value, bounds = self._bounded(
                    coordinates[axis], args, f"offset {body!r} in {frame!r}"
                )
                return value, {
                    "operator": op,
                    "body": body,
                    "frame": frame,
                    "axis": axis,
                    "local": coordinates,
                    **bounds,
                }
            checks: dict[str, Any] = {}
            passed = True
            for axis in ("x", "y", "z"):
                bounds = args.get(axis)
                if bounds is None:
                    continue
                lower, upper = float(bounds[0]), float(bounds[1])
                actual = coordinates[axis]
                axis_passed = lower <= actual <= upper
                passed = passed and axis_passed
                checks[axis] = {
                    "actual": actual,
                    "allowed": [lower, upper],
                    "passed": axis_passed,
                }
            return passed, {
                "operator": op,
                "body": body,
                "frame": frame,
                "local": coordinates,
                "axes": checks,
            }
        if op == "holding":
            name, body = entity("entity")
            value = body.is_grasped()
            return value, {
                "operator": op, "entity": name, "grasped": value,
            }
        if op == "inside":
            name, body = entity("entity")
            container_name, container = entity("container")
            value = body.is_in(
                container,
                tol_m=float(args.get("tolerance_m", 0.0)),
                require_contact=bool(args.get("require_contact", True)),
            )
            return value, {
                "operator": op,
                "entity": name,
                "container": container_name,
                "inside": value,
            }
        if op == "on":
            name, body = entity("entity")
            support_name, support = entity("support")
            tolerance = float(args.get("tolerance_m", 0.03))
            value = (
                body.is_on_strict(support, tol_m=tolerance)
                if bool(args.get("strict", True))
                else body.is_on(support, tol_m=tolerance)
            )
            return value, {
                "operator": op,
                "entity": name,
                "support": support_name,
                "on": value,
            }
        if op == "near":
            name, body = entity("entity")
            reference_name, reference = entity("reference")
            distance = body.distance_to(reference)
            limit = float(args.get("distance_m", 0.05))
            return distance <= limit, {
                "operator": op,
                "entity": name,
                "reference": reference_name,
                "distance_m": distance,
                "limit_m": limit,
            }
        if op == "settled":
            name, body = entity("entity")
            value = body.is_settled(
                speed_thresh=float(args.get("linear_m_s", 0.02))
            )
            return value, {
                "operator": op, "entity": name, "settled": value,
            }
        if op == "articulation":
            reading = self.get_articulation(
                str(args["entity"]), str(args.get("name", ""))
            )["articulation"]
            field = "progress" if "progress" in args else "position"
            target = float(args.get(field, 0.0))
            tolerance = float(args.get("tolerance", 0.02))
            comparison = str(args.get("comparison", "at_least"))
            current = float(reading[field])
            if comparison == "at_least":
                value = current >= target - tolerance
            elif comparison == "at_most":
                value = current <= target + tolerance
            elif comparison == "near":
                value = abs(current - target) <= tolerance
            else:
                raise ToolError(
                    "sim.evaluate_condition",
                    "comparison must be at_least, at_most, or near",
                )
            return value, {
                "operator": op,
                "articulation": reading["name"],
                "field": field,
                "current": current,
                "target": target,
                "comparison": comparison,
                "tolerance": tolerance,
            }
        raise ToolError(
            "sim.evaluate_condition", f"unknown condition operator {op!r}"
        )

    # ------------------------------------------------------------------

    @staticmethod
    def _frame_convention(adapter: Any) -> str:
        naming = getattr(adapter, "reference_frame_name", None)
        return str(naming()) if callable(naming) else "connector_world"

    def _box_payload(self, tool: str, adapter: Any, name: str) -> dict:
        box = adapter.object_box(name)
        if box is None:
            raise ToolError(
                tool,
                f"{name!r} has live state but no geometry bounds; the body "
                f"carries no collidable geoms",
            )
        center, half, quat = box
        return {
            "name": name,
            "obb": {
                "center": _vec3(center),
                "extent": _vec3(half),          # HALF-extents, per gap_core.types
                "orientation": _quat_xyzw(quat),
            },
            "grasp_yaw": _grasp_yaw_deg(half, quat),
            "frame": "world",
            "frame_convention": self._frame_convention(adapter),
            "source": "privileged",
        }
