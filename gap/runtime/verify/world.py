"""World / Body / Robot snapshot vocabulary for sim-side verification.

LLM-authored checkpoint predicates are written against this vocabulary:

    def predicate(world) -> bool:

where ``world`` is a :class:`World` snapshot of one env at one timestep
(with optional history of prior snapshots via ``world.history()``).

Design notes:

- :class:`World` / :class:`Body` / :class:`Robot` are frozen dataclasses
  consumed by LLM-written code; the harness builds them from plain
  numpy/python data (e.g. a sim-backend world adapter) or via
  :func:`StubWorld` (validation only).
- Spatial helper math mirrors
  :mod:`gap.runtime.predicates.predicate_evaluators` to keep the modules
  decoupled.
- Cavity AABBs are recorded world-frame on each :class:`Body`; the
  adapter that builds the snapshot is responsible for transforming any
  body-local cavity bounds using the body's current pose.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class BodyNotFoundError(LookupError):
    """Raised by :meth:`World.body` when the requested name isn't in the scene.

    Carries the available body names so the harness can surface a useful
    diagnostic to the LLM (or to the validator's retry loop). When a scene
    spec was attached to the World, the spec's canonical ids are listed
    first (those are the names the LLM was *supposed* to use); the raw
    scene body names follow.
    """

    def __init__(
        self,
        name: str,
        available: Sequence[str],
        *,
        spec_ids: Sequence[str] | None = None,
    ) -> None:
        self.name = str(name)
        self.available = sorted(available)
        self.spec_ids: list[str] = sorted(spec_ids) if spec_ids else []
        if self.spec_ids:
            msg = (
                f"body {name!r} not found in scene; spec ids: {self.spec_ids}; "
                f"scene bodies: {self.available}"
            )
        else:
            msg = f"body {name!r} not found in scene; available: {self.available}"
        super().__init__(msg)


# ---------------------------------------------------------------------------
# Robot link prefixes (default fallback)
# ---------------------------------------------------------------------------


_DEFAULT_ROBOT_LINK_PREFIXES: tuple[str, ...] = (
    "robot", "panda_", "Robotiq", "finger_",
)


_SLUG_RX = re.compile(r"[^a-z0-9]+")


def _slugify_name(name: str) -> str:
    """Snake-case slug used for cross-module body lookup."""
    s = _SLUG_RX.sub("_", str(name).strip().lower()).strip("_")
    return s or "x"


# ---------------------------------------------------------------------------
# Body
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Body:
    """Pose, AABB, velocity, contacts, cavity at one timestep for one env_id.

    All fields are numpy arrays (shape annotated below) in the connector's
    reference frame — for sim connectors that is the **robot base frame**,
    the same frame perception clouds, OBBs, and motion targets use, so
    predicates can compare subgraph outputs against these fields directly.
    Constructed by the harness's world adapter; LLM code reads but
    does not construct ``Body`` instances.
    """

    name: str
    position: np.ndarray              # (3,) world frame
    quaternion_wxyz: np.ndarray       # (4,) world frame
    aabb_lower: np.ndarray            # (3,) world frame min
    aabb_upper: np.ndarray            # (3,) world frame max
    linear_velocity: np.ndarray       # (3,) world frame
    angular_velocity: np.ndarray      # (3,) world frame
    contacts: frozenset[str]          # body-names in contact with this body
    cavity_lower: np.ndarray | None = None
    """World-frame min of the body's interior cavity AABB; ``None`` for
    solid bodies (no interior cavity registered)."""
    cavity_upper: np.ndarray | None = None
    """World-frame max of the cavity AABB; ``None`` if no cavity."""
    is_region: bool = False
    """True for physics-free region markers (visual drop targets with
    cavity AABBs only). :meth:`is_in` skips the contact requirement when
    the container is a region (regions have no collision so contacts are
    always empty)."""
    mesh_path: str | None = None
    """Filesystem path to the body's OBJ in body-local frame, or
    ``None`` for primitives that have no mesh asset. Used by
    :meth:`bottom_footprint_xy` / :meth:`xy_coverage_over` to compute
    true mesh-projected geometry instead of AABB approximations."""
    joints: dict[str, float] | None = None
    """Interior joints of an articulated body (a drawer, a door, a knob):
    joint name -> position, in metres for slides and radians for hinges.
    ``None`` for rigid bodies."""

    # Backref to the owning world for prefix-aware predicates
    # (is_grasped needs the robot link prefixes).
    _world: World | None = None

    # ------------------------------------------------------------------
    # Component scalars (sugar for common conditions)
    # ------------------------------------------------------------------

    @property
    def x(self) -> float:
        return float(self.position[0])

    @property
    def y(self) -> float:
        return float(self.position[1])

    @property
    def z(self) -> float:
        return float(self.position[2])

    @property
    def xy(self) -> np.ndarray:
        return self.position[:2]

    @property
    def interior_lower(self) -> np.ndarray:
        """World-frame min of the interior cavity AABB, falling back to
        the body AABB when no cavity is registered (same fallback as
        :meth:`is_in`). Predicates should prefer this over raw
        ``cavity_lower``, which is ``None`` on sims whose world adapter
        does not register cavities (e.g. LIBERO)."""
        return (
            self.cavity_lower
            if self.cavity_lower is not None
            else self.aabb_lower
        )

    @property
    def interior_upper(self) -> np.ndarray:
        """World-frame max counterpart of :attr:`interior_lower`."""
        return (
            self.cavity_upper
            if self.cavity_upper is not None
            else self.aabb_upper
        )

    @property
    def top_z(self) -> float:
        return float(self.aabb_upper[2])

    @property
    def bottom_z(self) -> float:
        return float(self.aabb_lower[2])

    @property
    def left_x(self) -> float:
        return float(self.aabb_lower[0])

    @property
    def right_x(self) -> float:
        return float(self.aabb_upper[0])

    @property
    def near_y(self) -> float:
        return float(self.aabb_lower[1])

    @property
    def far_y(self) -> float:
        return float(self.aabb_upper[1])

    # ------------------------------------------------------------------
    # Spatial predicates (mirroring predicate_evaluators)
    # ------------------------------------------------------------------

    def distance_to(self, other: Body) -> float:
        """3D Euclidean distance between body centers, in meters."""
        return float(np.linalg.norm(self.position - other.position))

    def xy_distance_to(self, other: Body) -> float:
        """2D (xy-plane) Euclidean distance between body centers, in meters."""
        return float(np.linalg.norm(self.xy - other.xy))

    def is_above(
        self,
        other: Body,
        *,
        min_clearance_m: float = 0.0,
        require_xy_overlap: bool = True,
    ) -> bool:
        """True if self.bottom_z is above other.top_z by at least min_clearance_m.

        With ``require_xy_overlap=True`` (default), the two AABBs must
        also overlap in x and y. Mirrors
        :func:`gap.runtime.predicates.predicate_evaluators.above`.
        """
        z_clear = self.bottom_z - other.top_z >= float(min_clearance_m)
        if require_xy_overlap:
            xy_overlap = (
                self.right_x >= other.left_x and self.left_x <= other.right_x
                and self.far_y >= other.near_y and self.near_y <= other.far_y
            )
        else:
            xy_overlap = True
        return bool(z_clear and xy_overlap)

    def is_on(self, other: Body, *, tol_m: float = 0.05) -> bool:
        """True if self is resting on top of other.

        Self's bottom face within tol_m of other's top face, with xy
        overlap. Mirrors :func:`gap.runtime.predicates.predicate_evaluators.on`.

        WARNING: this uses AABB *overlap*, which is lenient — for an
        elongated object like a frypan, any sliver of overlap (e.g.
        only the handle hanging over the burner) returns True. Use
        :meth:`is_on_strict` when "the object body is actually resting
        on the support" is the predicate you want to verify.
        """
        dz = self.bottom_z - other.top_z
        xy_overlap = (
            self.right_x >= other.left_x and self.left_x <= other.right_x
            and self.far_y >= other.near_y and self.near_y <= other.far_y
        )
        return bool(abs(dz) <= float(tol_m) and xy_overlap)

    def is_on_strict(
        self,
        other: Body,
        *,
        tol_m: float = 0.03,
        tol_xy_m: float = 0.0,
        max_penetration_m: float = 0.005,
    ) -> bool:
        """Strict "self resting on other": Z-near (asymmetric), XY centroid
        inside other's XY AABB, AND self is not below other.

        Three checks; ALL must pass:

        1. **Self is ON TOP, not penetrating**:
           ``-max_penetration_m <= self.bottom_z - other.top_z <= tol_m``.
           Pan resting cleanly on burner → dz ≈ 0 (pass).
           Pan floating slightly → dz ≈ 0 to +tol_m (pass).
           Pan lying on the TABLE near the burner → dz ≈ -2 to -3 cm (FAIL
           — that's what the lenient ``abs(dz) <= tol_m`` used to accept,
           reporting success for a frypan with only its handle on the
           burner while the body sat on the table 2.4 cm below).
        2. **Self's XY centroid inside other's XY AABB**:
           ``other.left_x - tol_xy_m <= self.x <= other.right_x + tol_xy_m``
           and similarly for y. Default ``tol_xy_m=0`` requires strict
           containment.
        3. **(Implicit)** the Z check above also fails when self is
           clearly *above* by more than tol_m (floating in space, not
           settled).

        Defaults tightened from the prior version: ``tol_m`` 5 cm → 3 cm
        and ``max_penetration_m`` 0 → 5 mm (allow small interpenetration
        from PD overshoot but no more). Loosen on a case-by-case basis if
        a specific support is much higher than the table or the pan
        rocks visibly at rest.
        """
        dz = self.bottom_z - other.top_z
        # Asymmetric Z: -max_penetration .. +tol_m
        z_ok = -float(max_penetration_m) <= dz <= float(tol_m)
        p = self.position
        sx = float(tol_xy_m)
        sy = float(tol_xy_m)
        centroid_inside = (
            other.left_x - sx <= float(p[0]) <= other.right_x + sx
            and other.near_y - sy <= float(p[1]) <= other.far_y + sy
        )
        return bool(z_ok and centroid_inside)

    def is_in(
        self,
        container: Body,
        *,
        require_contact: bool = True,
        tol_m: float = 0.0,
        tol_xy_m: float = 0.02,
        tol_z_m: float = 0.02,
    ) -> bool:
        """True if self is inside container's cavity (or union AABB if no cavity).

        When ``require_contact=True`` (default), self must also be in
        contact with container per the contact buffer. Mirrors
        :func:`gap.runtime.predicates.predicate_evaluators.in_`'s
        ``check_contact AND check_contain`` decomposition.

        ``tol_m`` is the legacy isotropic slack added on top; ``tol_xy_m``
        / ``tol_z_m`` are the anisotropic slacks (defaults match
        robosuite's contact-success threshold).
        """
        if container.cavity_lower is not None and container.cavity_upper is not None:
            lo, hi = container.cavity_lower, container.cavity_upper
        else:
            lo, hi = container.aabb_lower, container.aabb_upper
        sx = float(tol_m) + float(tol_xy_m)
        sy = float(tol_m) + float(tol_xy_m)
        sz = float(tol_m) + float(tol_z_m)
        p = self.position
        contain = (
            lo[0] - sx <= p[0] <= hi[0] + sx
            and lo[1] - sy <= p[1] <= hi[1] + sy
            and lo[2] - sz <= p[2] <= hi[2] + sz
        )
        if not contain:
            return False
        if require_contact and not container.is_region:
            # Regions have no collision so contacts are always empty;
            # auto-skip the contact branch when checking against a region.
            return container.name in self.contacts
        return True

    def contains(
        self,
        other: Body,
        *,
        tol_m: float = 0.0,
        tol_xy_m: float = 0.02,
        tol_z_m: float = 0.02,
    ) -> bool:
        """Inverse of :meth:`is_in` (no contact requirement).

        Useful when iterating over potential contents:
        ``if basket.contains(can): ...``.
        """
        return other.is_in(
            self,
            require_contact=False,
            tol_m=tol_m,
            tol_xy_m=tol_xy_m,
            tol_z_m=tol_z_m,
        )

    def is_settled(self, *, speed_thresh: float = 0.08) -> bool:
        """True if linear velocity norm < speed_thresh m/s."""
        speed = float(np.linalg.norm(self.linear_velocity))
        return speed < float(speed_thresh)

    def is_grasped(self) -> bool:
        """True if any robot link is in self.contacts.

        The robot link prefix list is carried on the owning :class:`World`
        and defaults to ``("robot", "panda_", "Robotiq", "finger_")``.
        """
        if self._world is None:
            prefixes = _DEFAULT_ROBOT_LINK_PREFIXES
            robot_name = ""
        else:
            prefixes = self._world.robot_link_prefixes
            robot_name = self._world.robot_body_name
        for c in self.contacts:
            if robot_name and (c == robot_name or c.startswith(robot_name)):
                return True
            if any(c == p or c.startswith(p) for p in prefixes):
                return True
        return False

    def is_grasped_by(self, robot_name_or_prefix: str) -> bool:
        """Like :meth:`is_grasped` but with an explicit robot prefix."""
        prefix = str(robot_name_or_prefix)
        return any(c == prefix or c.startswith(prefix) for c in self.contacts)

    def is_grasped_by_both(
        self, prefix_a: str = "robot", prefix_b: str = "robot_1"
    ) -> bool:
        """Bimanual grasp: a contact link from *each* arm is present.

        Arm 0's articulation is ``robot`` and arm 1's is ``robot_1`` by
        convention. When the sim records contact body names without the
        articulation namespace (so the two arms are indistinguishable),
        this degrades to plain :meth:`is_grasped` rather than
        under-reporting — geometric stage checks remain the authoritative
        verdict for bimanual tasks.
        """
        a = self.is_grasped_by(prefix_a)
        b = self.is_grasped_by(prefix_b)
        if a and b:
            return True
        # Namespacing unavailable: fall back so a real two-arm grip isn't
        # reported as a failure on a contact-string technicality.
        return self.is_grasped()

    def is_axis_aligned(
        self,
        *,
        local_axis: str = "z",
        world_axis: str = "z",
        tol_rad: float = 0.20,
    ) -> bool:
        """True if the named local axis (after rotation) is within tol_rad of the named world axis.

        Mirrors :func:`gap.runtime.predicates.predicate_evaluators.axis_aligned`.
        """
        local_map = {
            "x": np.array([1.0, 0.0, 0.0]),
            "y": np.array([0.0, 1.0, 0.0]),
            "z": np.array([0.0, 0.0, 1.0]),
        }
        if local_axis not in local_map or world_axis not in local_map:
            raise ValueError(
                f"local_axis / world_axis must be one of x|y|z; "
                f"got {local_axis!r}, {world_axis!r}"
            )
        local = local_map[local_axis]
        world = local_map[world_axis]
        R = _quat_wxyz_to_rotmat(self.quaternion_wxyz)
        local_in_world = R @ local
        cos_angle = float(np.clip(np.dot(local_in_world, world), -1.0, 1.0))
        angle = math.acos(cos_angle)
        return angle <= float(tol_rad)

    # ------------------------------------------------------------------
    # Mesh-based geometry (true XY coverage)
    # ------------------------------------------------------------------

    def bottom_footprint_xy(self, *, z_slack_m: float = 0.005):
        """Return a Shapely Polygon of the convex hull of mesh vertices
        within ``z_slack_m`` of the body's world-frame minimum z.

        For a flat-bottomed object resting on a support, this is the
        body's actual XY contact footprint — the equivalent of
        projecting the bottom face onto the support's plane. Use this
        with :meth:`xy_coverage_over` for predicates where AABB overlap
        is misleading (elongated tools where a handle bloats the AABB
        but isn't part of the body resting on the support).

        Returns ``None`` when ``mesh_path`` is unset, the OBJ can't be
        loaded, or the bottom slice has fewer than 3 unique XY points.
        Lazy-imports trimesh / scipy / shapely so non-mesh predicates
        don't pay for them.
        """
        if not self.mesh_path:
            return None
        try:
            import trimesh
            from scipy.spatial import ConvexHull, QhullError
            from shapely.geometry import Polygon
        except ImportError:
            return None
        try:
            mesh = trimesh.load(self.mesh_path, force="mesh")
        except Exception:
            return None
        verts = np.asarray(mesh.vertices, dtype=np.float64)
        if verts.size == 0:
            return None
        R = _quat_wxyz_to_rotmat(self.quaternion_wxyz)
        world_verts = (R @ verts.T).T + np.asarray(self.position, dtype=np.float64)
        zmin = float(world_verts[:, 2].min())
        mask = world_verts[:, 2] <= zmin + float(z_slack_m)
        if mask.sum() < 3:
            return None
        xy = world_verts[mask, :2]
        try:
            hull = ConvexHull(xy)
        except QhullError:
            return None
        return Polygon(xy[hull.vertices])

    def xy_coverage_over(self, other: Body) -> float:
        """Fraction of ``other``'s XY AABB area covered by this body's
        mesh-projected bottom footprint.

        Range [0, 1]; 1.0 = full coverage, 0.0 = no overlap or mesh
        missing. Prefer this to ``(self_AABB ∩ other_AABB) / other_AABB``
        for elongated objects: AABB ratios always report ~100% when
        ``self``'s AABB is larger than ``other``'s in both directions,
        regardless of where the body actually sits.

        Implementation: convex hull of the bottom slice of ``self``'s
        mesh (via :meth:`bottom_footprint_xy`) intersected with
        ``other``'s axis-aligned XY rectangle, normalized by
        ``other``'s XY area. Returns ``0.0`` when either input is
        degenerate or no mesh is registered for ``self``.
        """
        footprint = self.bottom_footprint_xy()
        if footprint is None:
            return 0.0
        try:
            from shapely.geometry import Polygon
        except ImportError:
            return 0.0
        other_area = (
            (float(other.aabb_upper[0]) - float(other.aabb_lower[0]))
            * (float(other.aabb_upper[1]) - float(other.aabb_lower[1]))
        )
        if other_area <= 0:
            return 0.0
        other_poly = Polygon([
            (float(other.aabb_lower[0]), float(other.aabb_lower[1])),
            (float(other.aabb_upper[0]), float(other.aabb_lower[1])),
            (float(other.aabb_upper[0]), float(other.aabb_upper[1])),
            (float(other.aabb_lower[0]), float(other.aabb_upper[1])),
        ])
        return float(footprint.intersection(other_poly).area / other_area)


# ---------------------------------------------------------------------------
# Robot
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Robot:
    """Robot articulation view at one timestep."""

    body_name: str
    joint_pos: np.ndarray             # (n_joints,)
    joint_names: tuple[str, ...]
    ee_position: np.ndarray           # (3,) world frame
    ee_quaternion_wxyz: np.ndarray    # (4,) world frame
    gripper_open_fraction: float      # 0.0 = closed, 1.0 = fully open
    joint_vel: np.ndarray | None = None  # (n_joints,) or None if unavailable

    def gripper_is_closed(self, *, threshold: float = 0.1) -> bool:
        return self.gripper_open_fraction < float(threshold)

    def gripper_is_open(self, *, threshold: float = 0.9) -> bool:
        return self.gripper_open_fraction > float(threshold)


# ---------------------------------------------------------------------------
# World
# ---------------------------------------------------------------------------


@dataclass
class World:
    """Snapshot of the simulated scene for one env_id at one timestep.

    LLM code receives a ``World`` and uses it to:

    - Look up bodies by name: ``world.body("soup_can")`` → :class:`Body`.
    - Read robot state: ``world.robot()`` → :class:`Robot`.
    - Walk history (for temporal predicates): ``world.history()`` → list
      of prior ``World`` snapshots.
    - Use temporal helpers: ``world.eventually(fn)``, ``world.always(fn)``,
      ``world.at_end(fn)``.
    """

    env_id: int
    bodies: dict[str, Body]
    robot_view: Robot | None = None
    time_s: float = 0.0
    history_snapshots: list[World] = field(default_factory=list)
    robot_body_name: str = "robot"
    robot_link_prefixes: tuple[str, ...] = _DEFAULT_ROBOT_LINK_PREFIXES
    tabletop_body_name: str = "table_top"
    raw_contact_diagnostics: dict[str, Any] = field(default_factory=dict)
    """Diagnostic dump from the underlying contact sensors (per-sensor max
    force norms, top-N filter bodies by norm, threshold value). Populated
    by the world adapter when the backend supplies it. Useful for
    debugging "contacts came back empty" without re-running the rollout.
    Always a dict, possibly empty when the backend doesn't surface it."""
    sim_state: dict[str, Any] = field(default_factory=dict)
    """Kitchen-sink sim-state dump from the backend env: every robot
    joint pos/vel/torque, every robot-link world pose + velocity, every
    scene-body world pose + velocity, the current action targets, and
    derived grasp metrics (fingertip midpoint, fingertip-to-object
    distance). Same envelope ``raw_contact_diagnostics`` lives under as
    ``sim_state["contact_sensors"]`` — both are populated together. A
    diagnostic predicate can dump this whole dict to see exactly what
    the sim believed when the checkpoint fired."""
    scene_spec_ids: tuple[str, ...] = ()
    """Canonical spec ids (in spec order) when this World was built from a
    scene spec. Surfaced in :class:`BodyNotFoundError` so the LLM sees the
    names it was supposed to use, alongside the (possibly slugified) scene
    body names."""

    def __post_init__(self) -> None:
        # Patch Body._world backrefs so is_grasped uses the right prefixes.
        # ``Body`` is frozen so we use object.__setattr__.
        for body in self.bodies.values():
            object.__setattr__(body, "_world", self)

    def body(self, name: str) -> Body:
        resolved = self._resolve_body_name(name)
        if resolved is not None:
            return self.bodies[resolved]
        raise BodyNotFoundError(
            name, list(self.bodies.keys()),
            spec_ids=self.scene_spec_ids if self.scene_spec_ids else None,
        )

    def has_body(self, name: str) -> bool:
        return self._resolve_body_name(name) is not None

    def _resolve_body_name(self, name: str) -> str | None:
        """Four-stage name resolution for ``world.body(...)``.

        Stage 1 — exact key match.
        Stage 2 — snake-case slug match. Handles natural-language body
        refs like ``"alphabet soup"`` → ``"alphabet_soup"``.
        Stage 3 — *token-overlap* match. The LLM author often passes a
        verbose noun phrase from the task description (e.g. ``"small
        blue and white cream cheese"``) when the scene body id is just
        ``"cream_cheese"``. Stage 3 tokenizes both, finds bodies whose
        slug tokens are ALL present in the query, picks the body with
        the most tokens (longest match wins, breaking ties by id length).
        Stage 4 — *substring-token* match (only when 1-3 all fail).
        Handles scanned-asset ids whose tokens share no FULL token with
        the natural phrase (``"frying pan"`` vs ``"chefmate_8_frypan"``:
        ``"pan"`` is a substring of ``"frypan"``). Resolves only when one
        body uniquely maximises the substring-overlap count.
        Returns ``None`` when no body matches.
        """
        if name in self.bodies:
            return name
        slug = _slugify_name(name)
        if slug in self.bodies:
            return slug
        # Token-overlap match. Tokenize both sides on underscore. We
        # accept a body whose slug tokens are all present (as full
        # tokens, not substrings) in the query slug.
        query_tokens = set(slug.split("_")) - {"the", "a", "an"}
        if not query_tokens:
            return None
        best: tuple[int, int, str] | None = None
        for body_id in self.bodies:
            body_tokens = set(body_id.split("_"))
            if not body_tokens or not body_tokens.issubset(query_tokens):
                continue
            score = (len(body_tokens), -len(body_id))
            cand = (score[0], score[1], body_id)
            if best is None or cand > best:
                best = cand
        if best is not None:
            return best[2]

        # Stage 4 — substring-token match (only reached when Stages 1-3
        # all failed, i.e. this would otherwise be a hard
        # BodyNotFoundError). Scanned-asset ids carry a prefix/suffix the
        # natural task phrase never does ("frying pan" vs
        # "chefmate_8_frypan": no shared FULL token — "pan" != "frypan").
        # Score each body by how many query tokens have a substring
        # relationship with some body token (either direction), counting
        # only tokens >=3 chars and ignoring pure-digit id tokens so
        # short/numeric noise can't spuriously match. Resolve ONLY when a
        # single body uniquely maximises that score (>=1) — a strict
        # uniqueness guard keeps the blast radius minimal: ambiguous or
        # zero-overlap queries still return None exactly as before.
        # General signal-quality fix: a verbose noun phrase resolves to
        # the slugified scanned-asset id without any task-specific or
        # privileged knowledge.
        q_sub = {t for t in query_tokens if len(t) >= 3 and not t.isdigit()}
        if not q_sub:
            return None
        scored: list[tuple[int, str]] = []
        for body_id in self.bodies:
            b_sub = {
                t for t in body_id.split("_")
                if len(t) >= 3 and not t.isdigit()
            }
            overlap = sum(
                1
                for qt in q_sub
                if any(qt in bt or bt in qt for bt in b_sub)
            )
            if overlap > 0:
                scored.append((overlap, body_id))
        if not scored:
            return None
        scored.sort(key=lambda s: (-s[0], len(s[1])))
        top = scored[0][0]
        winners = [bid for sc, bid in scored if sc == top]
        if len(winners) == 1:
            return winners[0]
        return None

    def body_names(self) -> list[str]:
        return sorted(self.bodies.keys())

    def region(self, name: str) -> Body:
        """Return the body for a physics-free region marker.

        Equivalent to :meth:`body` plus an ``is_region`` check; raises
        ``BodyNotFoundError`` if the named body doesn't exist or is not
        a region.
        """
        b = self.body(name)
        if not b.is_region:
            raise BodyNotFoundError(name, list(self.region_names()))
        return b

    def region_names(self) -> list[str]:
        """Return the names of all region bodies (no-physics drop targets)."""
        return sorted(n for n, b in self.bodies.items() if b.is_region)

    def held_body(self) -> Body | None:
        """Return the body currently grasped by the robot, or ``None``.

        A body is considered held when its ``contacts`` set includes any
        registered robot link (matched by ``robot_link_prefixes`` /
        ``robot_body_name``). When more than one body satisfies the
        predicate, returns the one with the smallest distance to the
        robot's end-effector if a robot view is available, else the
        first one alphabetically.

        Postcondition primitive for ``grasp_sg``.
        """
        candidates: list[Body] = []
        for body in self.bodies.values():
            if body.name == self.robot_body_name or body.is_region:
                continue
            if body.name == self.tabletop_body_name:
                continue
            if body.is_grasped():
                candidates.append(body)
        if not candidates:
            return None
        if len(candidates) == 1:
            return candidates[0]
        if self.robot_view is not None:
            ee = self.robot_view.ee_position
            candidates.sort(key=lambda b: float(np.linalg.norm(b.position - ee)))
        else:
            candidates.sort(key=lambda b: b.name)
        return candidates[0]

    def bodies_displaced(
        self,
        *,
        min_xy_m: float = 0.05,
        min_z_m: float = 0.0,
    ) -> list[Body]:
        """Return non-static bodies that moved beyond the given thresholds.

        Compares each body's current position to its earliest snapshot
        in ``self.history_snapshots`` (the rollout's initial state).
        Returns bodies whose xy-distance moved is at least ``min_xy_m``
        AND z-distance moved is at least ``min_z_m``, sorted by xy
        displacement descending.

        Skips the robot, tabletop, and any region. Excludes containers
        (hollow physical bodies) only when they didn't actually move —
        if a hollow container itself was nudged > min_xy_m, it appears
        in the result so the LLM can detect basket displacement.

        Returns ``[]`` when history is empty (e.g. validator stub).
        """
        if not self.history_snapshots:
            return []
        initial = self.history_snapshots[0]
        out: list[tuple[float, Body]] = []
        for name, body in self.bodies.items():
            if name == self.robot_body_name or name == self.tabletop_body_name:
                continue
            if body.is_region:
                continue
            init_body = initial.bodies.get(name)
            if init_body is None:
                continue
            dxy = float(np.linalg.norm(body.position[:2] - init_body.position[:2]))
            dz = float(abs(body.position[2] - init_body.position[2]))
            if dxy < float(min_xy_m):
                continue
            if dz < float(min_z_m):
                continue
            out.append((dxy, body))
        out.sort(key=lambda pair: pair[0], reverse=True)
        return [body for _, body in out]

    def body_over(
        self,
        region: Body,
        *,
        tol_xy_m: float = 0.02,
    ) -> Body | None:
        """Return the (non-robot, non-region) body whose xy column is inside
        ``region``'s cavity AABB, regardless of z. ``None`` if no body
        qualifies. Postcondition primitive for ``transport_sg.move_above``.

        Tie-break: the body whose xy is closest to the region center.
        """
        if region.cavity_lower is None or region.cavity_upper is None:
            lo, hi = region.aabb_lower, region.aabb_upper
        else:
            lo, hi = region.cavity_lower, region.cavity_upper
        sx = float(tol_xy_m)
        sy = float(tol_xy_m)
        cx = 0.5 * (lo[0] + hi[0])
        cy = 0.5 * (lo[1] + hi[1])
        best: Body | None = None
        best_dist = float("inf")
        for name, body in self.bodies.items():
            if name == self.robot_body_name or name == self.tabletop_body_name:
                continue
            if body.is_region or name == region.name:
                continue
            p = body.position
            if (lo[0] - sx <= p[0] <= hi[0] + sx
                    and lo[1] - sy <= p[1] <= hi[1] + sy):
                dist = float((p[0] - cx) ** 2 + (p[1] - cy) ** 2)
                if dist < best_dist:
                    best_dist = dist
                    best = body
        return best

    def body_inside(
        self,
        region: Body,
        *,
        tol_xy_m: float = 0.02,
        tol_z_m: float = 0.02,
    ) -> Body | None:
        """Return the (non-robot, non-region) body whose COM lies inside
        ``region``'s cavity AABB. ``None`` if no body qualifies.

        No contact requirement (regions have no collision). Tie-break:
        body closest to cavity center.
        """
        if region.cavity_lower is None or region.cavity_upper is None:
            lo, hi = region.aabb_lower, region.aabb_upper
        else:
            lo, hi = region.cavity_lower, region.cavity_upper
        sx = float(tol_xy_m)
        sy = float(tol_xy_m)
        sz = float(tol_z_m)
        cc = 0.5 * (lo + hi)
        best: Body | None = None
        best_dist = float("inf")
        for name, body in self.bodies.items():
            if name == self.robot_body_name or name == self.tabletop_body_name:
                continue
            if body.is_region or name == region.name:
                continue
            p = body.position
            if (lo[0] - sx <= p[0] <= hi[0] + sx
                    and lo[1] - sy <= p[1] <= hi[1] + sy
                    and lo[2] - sz <= p[2] <= hi[2] + sz):
                dist = float(np.linalg.norm(p - cc))
                if dist < best_dist:
                    best_dist = dist
                    best = body
        return best

    def moved_body_inside(
        self,
        region: Body,
        *,
        min_xy_m: float = 0.05,
    ) -> Body | None:
        """Sugar: of bodies that moved during the rollout, return the one
        currently inside ``region``'s cavity AABB.

        Composes :meth:`bodies_displaced` and :meth:`body_inside` so the
        LLM doesn't have to hard-code a body name in a checkpoint
        predicate. Robust to swap-variant tasks where perception's body
        name does not match the ground-truth target.
        """
        if not self.history_snapshots:
            # Fallback: in smoke / validator mode there's no history;
            # just delegate to body_inside (no displacement filter).
            return self.body_inside(region)
        moved = self.bodies_displaced(min_xy_m=min_xy_m)
        if not moved:
            return None
        if region.cavity_lower is None or region.cavity_upper is None:
            lo, hi = region.aabb_lower, region.aabb_upper
        else:
            lo, hi = region.cavity_lower, region.cavity_upper
        sx, sy, sz = 0.02, 0.02, 0.02
        for body in moved:
            p = body.position
            if (lo[0] - sx <= p[0] <= hi[0] + sx
                    and lo[1] - sy <= p[1] <= hi[1] + sy
                    and lo[2] - sz <= p[2] <= hi[2] + sz):
                return body
        return None

    def robot(self) -> Robot:
        if self.robot_view is None:
            raise RuntimeError("World has no robot view registered")
        return self.robot_view

    def history(self) -> list[World]:
        """Return prior ``World`` snapshots in chronological order.

        The latest snapshot is ``self``; ``self.history()[-1]`` is the
        previous one. Empty list if no history was provided.
        """
        return list(self.history_snapshots)

    # ------------------------------------------------------------------
    # Temporal helpers
    # ------------------------------------------------------------------

    def eventually(self, predicate_fn: Callable[[World], bool]) -> bool:
        """True if ``predicate_fn(w)`` is True for any ``w`` in ``history() + [self]``."""
        for w in self.history_snapshots:
            if predicate_fn(w):
                return True
        return bool(predicate_fn(self))

    def always(self, predicate_fn: Callable[[World], bool]) -> bool:
        """True if ``predicate_fn(w)`` is True for every ``w`` in ``history() + [self]``."""
        for w in self.history_snapshots:
            if not predicate_fn(w):
                return False
        return bool(predicate_fn(self))

    def at_end(self, predicate_fn: Callable[[World], bool]) -> bool:
        """Equivalent to ``predicate_fn(self)``. Provided for AST-style symmetry."""
        return bool(predicate_fn(self))

    # ------------------------------------------------------------------
    # Construction (used by the harness, NOT by LLM code)
    # ------------------------------------------------------------------

    @classmethod
    def from_history(cls, snapshots: Sequence[World]) -> World:
        """Combine a sequence of ``World`` snapshots into one with history.

        Returns a copy of the LAST snapshot whose ``history_snapshots``
        carries all earlier snapshots in order.
        """
        if not snapshots:
            raise ValueError("from_history: snapshots is empty")
        last = snapshots[-1]
        return cls(
            env_id=last.env_id,
            bodies=dict(last.bodies),
            robot_view=last.robot_view,
            time_s=last.time_s,
            history_snapshots=list(snapshots[:-1]),
            robot_body_name=last.robot_body_name,
            robot_link_prefixes=last.robot_link_prefixes,
            tabletop_body_name=last.tabletop_body_name,
            raw_contact_diagnostics=dict(last.raw_contact_diagnostics),
            sim_state=dict(last.sim_state),
        )


# ---------------------------------------------------------------------------
# Free-function temporal helpers (alias World methods)
# ---------------------------------------------------------------------------


def eventually(world: World, predicate_fn: Callable[[World], bool]) -> bool:
    return world.eventually(predicate_fn)


def always(world: World, predicate_fn: Callable[[World], bool]) -> bool:
    return world.always(predicate_fn)


def at_end(world: World, predicate_fn: Callable[[World], bool]) -> bool:
    return world.at_end(predicate_fn)


# ---------------------------------------------------------------------------
# Contact-pair canonicalization
# ---------------------------------------------------------------------------


def contacts_from_pairs(
    contact_pairs: Iterable[tuple[str, str]] | None,
) -> dict[str, frozenset[str]]:
    """Canonicalize ``(body_a, body_b)`` contact pairs into per-body sets.

    Each pair is recorded symmetrically: ``a`` gains ``b`` in its contact
    set and vice versa. Returns a ``body_name -> frozenset[other_names]``
    map suitable for the ``Body.contacts`` field; bodies absent from the
    map have no contacts (use ``result.get(name, frozenset())``).
    ``None`` (no contact sensing available) maps to an empty dict — the
    evaluator then falls back to geometric ``is_above`` / ``is_in`` checks.
    """
    contacts_by_body: dict[str, set[str]] = {}
    if contact_pairs is not None:
        for a, b in contact_pairs:
            contacts_by_body.setdefault(a, set()).add(b)
            contacts_by_body.setdefault(b, set()).add(a)
    return {name: frozenset(others) for name, others in contacts_by_body.items()}


# ---------------------------------------------------------------------------
# StubWorld — for validator dry-runs (no sim backend)
# ---------------------------------------------------------------------------


def StubWorld(  # noqa: N802 — module-level factory function, capitalized for parity
    *,
    body_names: Sequence[str],
    container_names: Sequence[str] = (),
    region_names: Sequence[str] = (),
    robot_body: str = "robot",
    env_id: int = 0,
    history_depth: int = 0,
) -> World:
    """Return a minimally-populated ``World`` for validation dry-runs.

    Each named body gets a 10cm-cube AABB at the origin with zero
    velocity and empty contacts. Containers additionally get an 8cm
    interior cavity. Regions are like containers but with
    ``is_region=True`` so :meth:`Body.is_in` skips contact checks.
    The robot has its body name in the bodies dict (so
    ``world.body(robot_body)`` works) and a stub ``Robot`` view with
    7 zero joints.

    Validators use this to catch evaluator references to bodies that
    weren't built (which would raise :class:`BodyNotFoundError`).
    """
    bodies: dict[str, Body] = {}
    container_set = set(container_names)
    region_set = set(region_names)
    for nm in body_names:
        is_container = nm in container_set
        is_region = nm in region_set
        has_cavity = is_container or is_region
        bodies[nm] = Body(
            name=str(nm),
            position=np.zeros(3, dtype=np.float64),
            quaternion_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
            aabb_lower=-0.05 * np.ones(3, dtype=np.float64),
            aabb_upper=0.05 * np.ones(3, dtype=np.float64),
            linear_velocity=np.zeros(3, dtype=np.float64),
            angular_velocity=np.zeros(3, dtype=np.float64),
            contacts=frozenset(),
            cavity_lower=(
                -0.04 * np.ones(3, dtype=np.float64) if has_cavity else None
            ),
            cavity_upper=(
                0.04 * np.ones(3, dtype=np.float64) if has_cavity else None
            ),
            is_region=is_region,
        )
    robot_view = Robot(
        body_name=str(robot_body),
        joint_pos=np.zeros(7, dtype=np.float64),
        joint_names=tuple(f"panda_joint{i + 1}" for i in range(7)),
        ee_position=np.array([0.5, 0.0, 0.5]),
        ee_quaternion_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
        gripper_open_fraction=1.0,
    )
    world = World(
        env_id=int(env_id),
        bodies=bodies,
        robot_view=robot_view,
        time_s=0.0,
        history_snapshots=[],
        robot_body_name=str(robot_body),
    )
    if history_depth > 0:
        # Build a chain of `history_depth` prior stubs (cheap copies).
        prior = []
        for k in range(history_depth):
            stub_k = World(
                env_id=int(env_id),
                bodies=dict(bodies),
                robot_view=robot_view,
                time_s=float(k),
                history_snapshots=[],
                robot_body_name=str(robot_body),
            )
            prior.append(stub_k)
        world.history_snapshots = prior
    return world


def stub_world_with_history(
    base: World, history_depth: int,
) -> World:
    """Convenience wrapper used by validator dry-runs."""
    if history_depth <= 0:
        return base
    prior = []
    for k in range(history_depth):
        prior.append(World(
            env_id=base.env_id,
            bodies=dict(base.bodies),
            robot_view=base.robot_view,
            time_s=float(k),
            history_snapshots=[],
            robot_body_name=base.robot_body_name,
            robot_link_prefixes=base.robot_link_prefixes,
            tabletop_body_name=base.tabletop_body_name,
        ))
    return World(
        env_id=base.env_id,
        bodies=dict(base.bodies),
        robot_view=base.robot_view,
        time_s=base.time_s,
        history_snapshots=prior,
        robot_body_name=base.robot_body_name,
        robot_link_prefixes=base.robot_link_prefixes,
        tabletop_body_name=base.tabletop_body_name,
    )


# ---------------------------------------------------------------------------
# Quaternion math
# ---------------------------------------------------------------------------


def _quat_wxyz_to_rotmat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    n = math.sqrt(w * w + x * x + y * y + z * z)
    if n == 0.0:
        return np.eye(3)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
        [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)],
    ])


__all__ = [
    "Body",
    "BodyNotFoundError",
    "Robot",
    "StubWorld",
    "World",
    "always",
    "at_end",
    "contacts_from_pairs",
    "eventually",
    "stub_world_with_history",
]
