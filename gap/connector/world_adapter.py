"""MuJoCo scene adapters — portable privileged state from sim ground truth.

Builds :class:`gap.runtime.verify.World` snapshots for checkpoint
enforcement from the LIBERO/robosuite MuJoCo sim underneath the env:

- **object poses** from the sim's body state (``data.body_xpos`` /
  ``data.body_xquat``, wxyz) — the same ground truth the SimBridge GetState
  path surfaced; when no MuJoCo sim is reachable, the env obs dict's
  ``cube_poses`` entry is used as a fallback;
- **AABBs** precomputed once per reset from the MuJoCo *model* geoms
  (per-body local bounds from geom type/size/pos/quat, preferring the
  model's own ``geom_aabb``, falling back to the mesh's own vertices on
  builds that predate it), cached by body and re-oriented into world frame
  at snapshot time;
- **contacts** from ``sim.data.contact`` pairs, resolved to object names via
  body-subtree membership and filtered to named bodies + robot/gripper
  links, then canonicalized with :func:`verify.contacts_from_pairs`;
- **robot view** (joints, EE pose, gripper open fraction) into
  :class:`verify.Robot`.

Wired up as :meth:`gap.connector.sim.SimConnector.world_snapshot`.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from gap.runtime.verify import Articulation, Robot, World, contacts_from_pairs
from gap.runtime.verify.world import Body

logger = logging.getLogger(__name__)

# Panda link prefixes from verify.World defaults, extended with robosuite's
# "gripper0_*" namespace so finger contacts register as grasps.
_ROBOT_LINK_PREFIXES: tuple[str, ...] = (
    "robot", "panda_", "Robotiq", "finger_", "gripper",
)

# MuJoCo geom types (mjtGeom)
_GEOM_PLANE = 0
_GEOM_HFIELD = 1
_GEOM_SPHERE = 2
_GEOM_CAPSULE = 3
_GEOM_ELLIPSOID = 4
_GEOM_CYLINDER = 5
_GEOM_BOX = 6
_GEOM_MESH = 7


def find_mujoco_sim(env: Any) -> Any | None:
    """Walk the env wrapper chain looking for a MuJoCo sim handle.

    Accepts the FrankaLiberoEnv shape (``env.handle.env.sim``) and any
    nesting of ``.env`` wrappers; returns the first object exposing both
    ``.model`` and ``.data``.
    """
    seen: set[int] = set()
    frontier = [env]
    for _ in range(8):
        nxt = []
        for obj in frontier:
            if obj is None or id(obj) in seen:
                continue
            seen.add(id(obj))
            sim = getattr(obj, "sim", None)
            if sim is not None and hasattr(sim, "model") and hasattr(sim, "data"):
                return sim
            for attr in ("handle", "env"):
                child = getattr(obj, attr, None)
                if child is not None:
                    nxt.append(child)
        if not nxt:
            break
        frontier = nxt
    return None


def _find_object_map(env: Any) -> dict[str, int] | None:
    """Find the env's ``{object_name: root_body_id}`` registry, if any."""
    seen: set[int] = set()
    frontier = [env]
    for _ in range(8):
        nxt = []
        for obj in frontier:
            if obj is None or id(obj) in seen:
                continue
            seen.add(id(obj))
            for attr in ("obj_body_id", "_obj_body_id"):
                mapping = getattr(obj, attr, None)
                if isinstance(mapping, dict) and mapping:
                    return {str(k): int(v) for k, v in mapping.items()}
            for attr in ("handle", "env"):
                child = getattr(obj, attr, None)
                if child is not None:
                    nxt.append(child)
        if not nxt:
            break
        frontier = nxt
    return None


def _quat_wxyz_to_rotmat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = (float(v) for v in q[:4])
    n = (w * w + x * x + y * y + z * z) ** 0.5
    if n == 0.0:
        return np.eye(3)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
        [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)],
    ])


def _mesh_vertices(model: Any, gid: int, cache: dict[int, np.ndarray | None] | None) -> np.ndarray | None:
    """Mesh geom *gid*'s vertices, in the geom's own frame.

    MuJoCo applies the mesh's scale and origin at compile time, so these need
    no further correction. Cached by *mesh* id, not geom id: a scene reuses
    one mesh across many geoms.
    """
    meshid = int(np.asarray(model.geom_dataid)[gid])
    if meshid < 0:
        return None
    if cache is not None and meshid in cache:
        return cache[meshid]

    verts: np.ndarray | None = None
    try:
        adr = int(np.asarray(model.mesh_vertadr)[meshid])
        num = int(np.asarray(model.mesh_vertnum)[meshid])
        if num > 0:
            verts = np.asarray(model.mesh_vert)[adr : adr + num].astype(np.float64)
    except (AttributeError, IndexError, ValueError):
        verts = None

    if cache is not None:
        cache[meshid] = verts
    return verts


def _geom_local_points(
    model: Any, gid: int, cache: dict[int, np.ndarray | None] | None = None
) -> np.ndarray | None:
    """Points, in geom *gid*'s own frame, whose bounds are the geom's bounds.

    A mesh contributes its **vertices**, not the corners of its own AABB. The
    caller re-bounds these in the body frame, and bounding an already-bounded
    box inflates it whenever the geom is rotated relative to the body. Across
    a 33-piece convex decomposition of a mug that inflation came to 3 cm of
    height — enough to put the box's floor below the table it stands on, and
    to make the object look too tall to fit under the drawer it goes into.
    """
    gtype = int(np.asarray(model.geom_type)[gid])
    if gtype == _GEOM_MESH:
        verts = _mesh_vertices(model, gid, cache)
        if verts is not None:
            return verts
    box = _geom_local_box(model, gid)
    return None if box is None else _box_corners(*box)


def _geom_local_box(model: Any, gid: int) -> tuple[np.ndarray, np.ndarray] | None:
    """(center, half-extents) of geom *gid* in the geom's own frame."""
    gtype = int(np.asarray(model.geom_type)[gid])
    if gtype in (_GEOM_PLANE, _GEOM_HFIELD):
        return None
    size = np.asarray(model.geom_size)[gid].astype(np.float64)
    # Prefer the model's own per-geom AABB when available (exact for meshes).
    aabb = getattr(model, "geom_aabb", None)
    if aabb is not None:
        row = np.asarray(aabb)[gid].astype(np.float64)
        if np.any(row[3:] > 0):
            return row[:3], row[3:]
    if gtype == _GEOM_SPHERE:
        half = np.array([size[0]] * 3)
    elif gtype == _GEOM_CAPSULE:
        half = np.array([size[0], size[0], size[1] + size[0]])
    elif gtype == _GEOM_CYLINDER:
        half = np.array([size[0], size[0], size[1]])
    elif gtype in (_GEOM_BOX, _GEOM_ELLIPSOID):
        half = size[:3].copy()
    else:
        # Mesh on a build with no geom_aabb (MuJoCo < 3): bound the vertices.
        # geom_size is meaningless for a mesh and geom_rbound is the bounding
        # *sphere*, which makes a mug as wide as it is tall — the difference
        # between a grasp across the body and one the hand refuses outright.
        verts = _mesh_vertices(model, gid, None)
        if verts is not None:
            lo, hi = verts.min(axis=0), verts.max(axis=0)
            return (lo + hi) * 0.5, (hi - lo) * 0.5
        # Last resort: the bounding sphere. Over-wide, but never under-covers.
        rbound = float(np.asarray(model.geom_rbound)[gid])
        half = np.array([rbound] * 3)
    return np.zeros(3), half


def _rotmat_to_quat_wxyz(R: np.ndarray) -> np.ndarray:
    """Rotation matrix -> unit quaternion (w, x, y, z)."""
    t = float(np.trace(R))
    if t > 0.0:
        s = np.sqrt(t + 1.0) * 2.0
        return np.array([
            0.25 * s,
            (R[2, 1] - R[1, 2]) / s,
            (R[0, 2] - R[2, 0]) / s,
            (R[1, 0] - R[0, 1]) / s,
        ])
    i = int(np.argmax(np.diag(R)))
    j, k = (i + 1) % 3, (i + 2) % 3
    s = np.sqrt(max(R[i, i] - R[j, j] - R[k, k] + 1.0, 0.0)) * 2.0
    q = np.empty(4)
    q[0] = (R[k, j] - R[j, k]) / s
    q[1 + i] = 0.25 * s
    q[1 + j] = (R[j, i] + R[i, j]) / s
    q[1 + k] = (R[k, i] + R[i, k]) / s
    return q


def _box_corners(center: np.ndarray, half: np.ndarray) -> np.ndarray:
    signs = np.array([
        [sx, sy, sz]
        for sx in (-1.0, 1.0)
        for sy in (-1.0, 1.0)
        for sz in (-1.0, 1.0)
    ])
    return center[None, :] + signs * half[None, :]


class MujocoSceneAdapter:
    """Build portable scene state from a robosuite-style MuJoCo environment.

    Despite its historical LIBERO-only name, this implementation only relies
    on the MuJoCo / robosuite object model. It is therefore also the shared
    privileged provider for MimicGen and other robosuite benchmarks.
    """

    def __init__(
        self,
        env: Any,
        *,
        robot_link_prefixes: tuple[str, ...] = _ROBOT_LINK_PREFIXES,
        arm_dof: int = 7,
    ) -> None:
        self.env = env
        self.robot_link_prefixes = tuple(robot_link_prefixes)
        self.arm_dof = int(arm_dof)
        self._sim: Any | None = None
        self._objects: dict[str, int] = {}            # object name -> root body id
        self._body_owner: dict[int, str] = {}          # any body id -> contact name
        self._local_aabbs: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self._body_aabbs: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        # Per-body boxes and ids for the *parts* of an object — the model
        # bodies inside a root's subtree. An articulated object's moving piece
        # (a drawer's sliding link, a door's leaf) is the thing a grasp
        # actually targets, and its box is nothing like the whole cabinet's.
        self._part_aabbs: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self._part_bids: dict[str, tuple[str, int]] = {}   # part -> (root, bid)
        self._feature_geoms: dict[str, tuple[str, int, np.ndarray, np.ndarray]] = {}
        self._articulations: dict[str, tuple[str, int]] = {}
        self._movable: set[str] = set()                    # free-jointed objects
        self._tabletop_name = "table"
        self._prepared = False
        self.refresh()

    # ------------------------------------------------------------------
    # Reset-time precomputation (cached by body)
    # ------------------------------------------------------------------

    def refresh(self) -> None:
        """(Re)build body registries + local AABBs. Call after env reset —
        a robosuite hard reset rebuilds the MjModel."""
        self._sim = find_mujoco_sim(self.env)
        self._objects = {}
        self._body_owner = {}
        self._local_aabbs = {}
        self._body_aabbs = {}
        self._part_aabbs = {}
        self._part_bids = {}
        self._feature_geoms = {}
        self._articulations = {}
        self._movable = set()
        self._base_bid: int | None = None
        self._prepared = False
        if self._sim is None:
            return
        model = self._sim.model

        # Robot base body: snapshots are expressed in this body's frame so
        # privileged truth lands in the SAME frame as perception clouds and
        # motion targets (both robot-base-framed). Without this, generated
        # checkpoints comparing subgraph outputs against w.body(...) fail
        # by the base offset (~0.6 m x in LIBERO) on every run.
        for base_name in ("robot0_base", "robot0_link0"):
            bid = self._body_id(model, base_name)
            if bid is not None:
                self._base_bid = bid
                break

        objects = _find_object_map(self.env) or self._free_joint_bodies(model)
        # Tabletop body, when present.
        table_id = self._body_id(model, "table")
        if table_id is None:
            table_id = self._body_id(model, "table_top")
        nbody = int(getattr(model, "nbody", 0))
        parentid = np.asarray(model.body_parentid)

        roots = dict(objects)
        if table_id is not None:
            roots[self._tabletop_name] = table_id

        # Subtree membership: each model body maps to the object/table whose
        # root it descends from; robot links keep their raw names.
        root_of = {bid: name for name, bid in roots.items()}
        for b in range(nbody):
            cur = b
            for _ in range(nbody):
                if cur in root_of:
                    self._body_owner[b] = root_of[cur]
                    break
                parent = int(parentid[cur])
                if parent == cur or parent < 0:
                    break
                cur = parent
            if b not in self._body_owner:
                name = self._body_name(model, b)
                if name and name.startswith(self.robot_link_prefixes):
                    self._body_owner[b] = name

        self._objects = objects
        # Which of those can actually be picked up. An object hung on a free
        # joint moves as a whole when a hand lifts it; one hung on a slide or
        # hinge (a drawer, a door) is bolted to the scene and only articulates.
        # Both are "objects" to the task; only the first is a grasp target.
        free_bids = set(self._free_joint_bodies(model).values())
        self._movable = {n for n, bid in objects.items() if bid in free_bids}

        # Per-body local AABBs from model geoms, accumulated over the subtree.
        # One mesh-vertex cache for the whole sweep: scenes reuse meshes
        # heavily and this loop visits every geom of every named body.
        mesh_cache: dict[int, np.ndarray | None] = {}
        for name, root in roots.items():
            lo = np.full(3, np.inf)
            hi = np.full(3, -np.inf)
            members = [b for b in range(nbody) if self._body_owner.get(b) == name]
            for b in members:
                offset_pos, offset_rot = self._subtree_transform(model, root, b)
                gids = np.where(np.asarray(model.geom_bodyid) == b)[0]
                # This body's own box falls out of the same points, one
                # transform earlier, so parts cost nothing beyond the dict.
                blo = np.full(3, np.inf)
                bhi = np.full(3, -np.inf)
                for gid in gids:
                    pts = _geom_local_points(model, int(gid), mesh_cache)
                    if pts is None:
                        continue
                    gpos = np.asarray(model.geom_pos)[gid].astype(np.float64)
                    grot = _quat_wxyz_to_rotmat(np.asarray(model.geom_quat)[gid])
                    feature_name = self._geom_name(model, int(gid))
                    if feature_name:
                        flo, fhi = pts.min(axis=0), pts.max(axis=0)
                        self._feature_geoms[feature_name] = (
                            name, int(gid),
                            (flo + fhi) / 2.0,
                            (fhi - flo) / 2.0,
                        )
                    pts = (grot @ pts.T).T + gpos                # body frame
                    blo = np.minimum(blo, pts.min(axis=0))
                    bhi = np.maximum(bhi, pts.max(axis=0))
                    pts = (offset_rot @ pts.T).T + offset_pos    # root frame
                    lo = np.minimum(lo, pts.min(axis=0))
                    hi = np.maximum(hi, pts.max(axis=0))
                part = self._body_name(model, b)
                if np.all(np.isfinite(blo)):
                    self._body_aabbs[b] = (blo, bhi)
                if part and b != root and np.all(np.isfinite(blo)):
                    self._part_aabbs[part] = (blo, bhi)
                    self._part_bids[part] = (name, b)
            if np.all(np.isfinite(lo)):
                self._local_aabbs[name] = (lo, hi)
            else:
                self._local_aabbs[name] = (
                    -0.05 * np.ones(3), 0.05 * np.ones(3),
                )
        for jid in range(int(getattr(model, "njnt", 0))):
            try:
                jtype = int(np.asarray(model.jnt_type)[jid])
                bid = int(np.asarray(model.jnt_bodyid)[jid])
            except Exception:
                continue
            if jtype not in (2, 3):  # slide / hinge
                continue
            root = self._body_owner.get(bid)
            if root is None or root.startswith(self.robot_link_prefixes):
                continue
            name = self._joint_name(model, jid) or f"{root}:dof{jid}"
            self._articulations[name] = (root, jid)
        self._prepared = True

    @staticmethod
    def _body_id(model: Any, name: str) -> int | None:
        try:
            return int(model.body_name2id(name))
        except Exception:
            return None

    @staticmethod
    def _body_name(model: Any, bid: int) -> str | None:
        try:
            return model.body_id2name(bid)
        except Exception:
            return None

    @staticmethod
    def _geom_name(model: Any, gid: int) -> str | None:
        try:
            return model.geom_id2name(gid)
        except Exception:
            try:
                return model.id2name(gid, "geom")
            except Exception:
                return None

    @staticmethod
    def _joint_name(model: Any, jid: int) -> str | None:
        try:
            return model.joint_id2name(jid)
        except Exception:
            try:
                return model.jnt_id2name(jid)
            except Exception:
                return None

    def _free_joint_bodies(self, model: Any) -> dict[str, int]:
        """Fallback object discovery: bodies hung on a free joint."""
        objects: dict[str, int] = {}
        try:
            jnt_type = np.asarray(model.jnt_type)
            jnt_bodyid = np.asarray(model.jnt_bodyid)
        except Exception:
            return objects
        for j in range(len(jnt_type)):
            if int(jnt_type[j]) != 0:  # mjJNT_FREE
                continue
            bid = int(jnt_bodyid[j])
            name = self._body_name(model, bid)
            if not name or name.startswith(self.robot_link_prefixes):
                continue
            key = name[:-5] if name.endswith("_main") else name
            objects[key] = bid
        return objects

    def _subtree_transform(
        self, model: Any, root: int, body: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Static transform of *body* in *root*'s frame (model rest pose).

        Interior joints inside an object subtree are rare in LIBERO scenes;
        accumulated ``body_pos``/``body_quat`` chains are sufficient.
        """
        pos = np.zeros(3)
        rot = np.eye(3)
        cur = body
        parentid = np.asarray(model.body_parentid)
        body_pos = np.asarray(model.body_pos)
        body_quat = np.asarray(model.body_quat)
        for _ in range(int(getattr(model, "nbody", 0))):
            if cur == root:
                return pos, rot
            p = body_pos[cur].astype(np.float64)
            r = _quat_wxyz_to_rotmat(body_quat[cur])
            pos = r @ pos + p
            rot = r @ rot
            parent = int(parentid[cur])
            if parent == cur or parent < 0:
                break
            cur = parent
        return pos, rot

    # ------------------------------------------------------------------
    # Snapshot
    # ------------------------------------------------------------------

    def snapshot(self, env_id: int = 0) -> World:
        if self._sim is None or not self._prepared:
            self.refresh()
        if self._sim is None:
            return self._snapshot_from_obs(env_id)

        sim = self._sim
        model, data = sim.model, sim.data

        contacts = contacts_from_pairs(self._contact_pairs(model, data))

        # World -> robot-base transform (p_base = R_b^T @ (p_world - t_b)):
        # keeps privileged truth in the frame all workflow outputs use.
        base_t, base_Rt = self._reference_transform()

        bodies: dict[str, Body] = {}
        names = dict(self._objects)
        if self._tabletop_name in self._local_aabbs:
            table_id = self._body_id(model, "table") or self._body_id(model, "table_top")
            if table_id is not None:
                names[self._tabletop_name] = table_id
        for name, bid in names.items():
            pos = np.asarray(data.body_xpos)[bid].astype(np.float64).copy()
            quat = np.asarray(data.body_xquat)[bid].astype(np.float64).copy()  # wxyz
            live_box = self._object_box_world(name)
            if live_box is None:
                box_center = pos
                half = 0.05 * np.ones(3)
                box_quat = quat
            else:
                box_center, half, box_quat = live_box
            pos = base_Rt @ (pos - base_t)
            box_center = base_Rt @ (box_center - base_t)
            R = base_Rt @ _quat_wxyz_to_rotmat(box_quat)
            quat = _rotmat_to_quat_wxyz(
                base_Rt @ _quat_wxyz_to_rotmat(quat)
            )
            corners = (R @ _box_corners(np.zeros(3), half).T).T + box_center
            lin, ang = self._body_velocity(data, bid)
            bodies[name] = Body(
                name=name,
                position=pos,
                quaternion_wxyz=quat,
                aabb_lower=corners.min(axis=0),
                aabb_upper=corners.max(axis=0),
                linear_velocity=base_Rt @ lin,
                angular_velocity=base_Rt @ ang,
                contacts=contacts.get(name, frozenset()),
            )

        articulations: dict[str, Articulation] = {}
        for articulation_name in self.articulation_names():
            state = self.articulation_state(articulation_name)
            if state is None:
                continue
            articulations[articulation_name] = Articulation(
                name=state["name"], parent=state["parent"], child=state["child"],
                kind=state["kind"], position=float(state["position"]),
                lower=float(state["lower"]), upper=float(state["upper"]),
                progress=float(state["progress"]),
                axis=np.asarray(state["axis"], dtype=np.float64).copy(),
                pivot=np.asarray(state["pivot"], dtype=np.float64).copy(),
            )

        robot_view = self._robot_view(model, data, base_t=base_t, base_Rt=base_Rt)
        time_s = 0.0
        if hasattr(self.env, "get_current_time_s"):
            try:
                time_s = float(self.env.get_current_time_s())
            except Exception:
                time_s = 0.0

        return World(
            env_id=int(env_id),
            bodies=bodies,
            articulations=articulations,
            robot_view=robot_view,
            time_s=time_s,
            robot_link_prefixes=self.robot_link_prefixes,
            tabletop_body_name=self._tabletop_name,
        )

    # Alias so SimConnector.world_snapshot can pass straight through.
    __call__ = snapshot

    # ------------------------------------------------------------------
    # Connector-reference ground truth (backs the privileged sim.* tools)
    # ------------------------------------------------------------------
    #
    # ``snapshot()``, perception clouds, motion targets, and these accessors
    # must share one frame. In the sim connector that reference is the robot
    # base frame; raw MuJoCo-world coordinates create a constant base-offset
    # error in every privileged grasp.

    def object_names(self) -> list[str]:
        """Every body the privileged accessors below can answer about."""
        if self._sim is None or not self._prepared:
            self.refresh()
        names = list(self._objects)
        if self._tabletop_name in self._local_aabbs and self._tabletop_name not in names:
            names.append(self._tabletop_name)
        return names

    def task_object_names(self) -> list[str]:
        """The subset of :meth:`object_names` the *task* declares.

        Everything else in the list is scene furniture the adapter adds for
        reference — today just the tabletop. The distinction matters to a
        caller trying to turn a natural noun phrase into a body name: LIBERO
        habitually calls the one manipulable body ``object``, so "the white
        mug" matches nothing by text and everything by elimination.
        """
        return [n for n in self.object_names() if n in self._objects]

    def movable_object_names(self) -> list[str]:
        """The subset of :meth:`task_object_names` a hand could carry away.

        Free-jointed bodies, as opposed to objects bolted to the scene that
        merely articulate — a mug rather than the drawer it goes into.
        """
        return [n for n in self.object_names() if n in self._movable]

    def workspace_surface_box(self):
        """Primary support surface in connector coordinates, when present."""
        return self.object_box(self._tabletop_name)

    def part_names(self, root: str) -> list[str]:
        """The model bodies inside *root*'s subtree, excluding *root* itself.

        These are the object's moving or distinguishable pieces — a drawer's
        sliding link, a door's leaf — addressable by the same accessors as a
        whole object. Names are the model's own body names, which is what the
        caller has to quote: there is no noun-phrase layer down here.
        """
        if self._sim is None or not self._prepared:
            self.refresh()
        return sorted(p for p, (r, _bid) in self._part_bids.items() if r == root)

    def feature_names(self, root: str) -> list[str]:
        """Named contact geometry and body parts owned by *root*."""
        if self._sim is None or not self._prepared:
            self.refresh()
        geoms = [
            name for name, (owner, *_rest) in self._feature_geoms.items()
            if owner == root
        ]
        return sorted(set(geoms + self.part_names(root)))

    def feature_box(
        self, root: str, feature: str,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        """Live feature OBB in connector coordinates."""
        if feature in self._part_bids and self._part_bids[feature][0] == root:
            return self.object_box(feature)
        record = self._feature_geoms.get(feature)
        if record is None or record[0] != root or self._sim is None:
            return None
        _owner, gid, local_center, half = record
        data = self._sim.data
        try:
            world_pos = np.asarray(data.geom_xpos)[gid].astype(np.float64)
            world_R = np.asarray(data.geom_xmat)[gid].astype(np.float64).reshape(3, 3)
        except Exception:
            return None
        center = world_pos + world_R @ local_center
        base_t, base_Rt = self._reference_transform()
        R = base_Rt @ world_R
        return (
            base_Rt @ (center - base_t),
            half.copy(),
            _rotmat_to_quat_wxyz(R),
        )

    def articulation_names(self, root: str | None = None) -> list[str]:
        """Every scalar prismatic/revolute DOF, optionally scoped to entity."""
        if self._sim is None or not self._prepared:
            self.refresh()
        return sorted(
            name for name, (owner, _jid) in self._articulations.items()
            if root is None or owner == root
        )

    def articulation_state(self, name: str) -> dict[str, Any] | None:
        """Backend-neutral live joint state in connector coordinates."""
        record = self._articulations.get(name)
        if record is None or self._sim is None:
            return None
        root, jid = record
        model, data = self._sim.model, self._sim.data
        jtype = int(np.asarray(model.jnt_type)[jid])
        bid = int(np.asarray(model.jnt_bodyid)[jid])
        child = self._body_name(model, bid) or root
        try:
            addr = int(np.asarray(model.jnt_qposadr)[jid])
        except Exception:
            try:
                addr = model.get_joint_qpos_addr(name)
                if isinstance(addr, tuple):
                    addr = addr[0]
                addr = int(addr)
            except Exception:
                return None
        position = float(np.asarray(data.qpos)[addr])
        try:
            limited = bool(np.asarray(model.jnt_limited)[jid])
            limits = np.asarray(model.jnt_range)[jid].astype(np.float64)
            lower, upper = (
                (float(limits[0]), float(limits[1]))
                if limited else (-np.inf, np.inf)
            )
        except Exception:
            lower, upper = -np.inf, np.inf
        progress = (
            (position - lower) / (upper - lower)
            if np.isfinite(lower) and np.isfinite(upper) and upper > lower
            else 0.0
        )
        axis_local = np.asarray(model.jnt_axis)[jid].astype(np.float64)
        body_R = _quat_wxyz_to_rotmat(
            np.asarray(data.body_xquat)[bid].astype(np.float64)
        )
        _base_t, base_Rt = self._reference_transform()
        axis = base_Rt @ body_R @ axis_local
        joint_local = np.asarray(model.jnt_pos)[jid].astype(np.float64)
        body_position = np.asarray(data.body_xpos)[bid].astype(np.float64)
        pivot_world = body_position + body_R @ joint_local
        pivot = base_Rt @ (pivot_world - _base_t)
        norm = float(np.linalg.norm(axis))
        if norm > 1e-12:
            axis /= norm
        return {
            "name": name,
            "parent": root,
            "child": child,
            "kind": "prismatic" if jtype == 2 else "revolute",
            "axis": axis,
            "pivot": pivot,
            "position": position,
            "lower": lower,
            "upper": upper,
            "progress": float(progress),
        }

    def contact_pairs(self) -> list[tuple[str, str]]:
        """Live canonical contact pairs with backend geom ids removed."""
        if self._sim is None or not self._prepared:
            self.refresh()
        if self._sim is None:
            return []
        return self._contact_pairs(self._sim.model, self._sim.data)

    def collision_feature_contact_pairs(self) -> list[tuple[str, str]]:
        """Exact live collision-shape pairs, named without exposing numeric ids."""
        if self._sim is None or not self._prepared:
            self.refresh()
        if self._sim is None:
            return []
        model, data = self._sim.model, self._sim.data
        pairs: list[tuple[str, str]] = []
        try:
            count = int(data.ncon)
        except Exception:
            return pairs
        for index in range(count):
            contact = data.contact[index]
            first_id, second_id = int(contact.geom1), int(contact.geom2)
            first = self._geom_name(model, first_id) or f"geom_{first_id}"
            second = self._geom_name(model, second_id) or f"geom_{second_id}"
            pairs.append((first, second))
        return pairs

    def named_frame_pose(
        self, name: str,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """Exact named collision/body/site frame in connector coordinates.

        Resolution order matches Agent2Policy's normalized state contract:
        collision feature, body, then site. Sites are points there, so their
        orientation is the public-world basis rather than a simulator-only
        site orientation.
        """
        if self._sim is None or not self._prepared:
            self.refresh()
        if self._sim is None:
            return None
        if name == "world":
            return np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0])
        model, data = self._sim.model, self._sim.data
        position: np.ndarray | None = None
        rotation: np.ndarray | None = None
        for kind, pos_attr, mat_attr in (
            ("geom", "geom_xpos", "geom_xmat"),
            ("body", "body_xpos", None),
            ("site", "site_xpos", None),
        ):
            try:
                lookup = getattr(model, f"{kind}_name2id")
                index = int(lookup(name))
            except Exception:
                continue
            position = np.asarray(getattr(data, pos_attr))[index].astype(np.float64)
            if kind == "body":
                rotation = _quat_wxyz_to_rotmat(
                    np.asarray(data.body_xquat)[index].astype(np.float64)
                )
            elif mat_attr is not None:
                rotation = np.asarray(getattr(data, mat_attr))[index].astype(
                    np.float64
                ).reshape(3, 3)
            else:
                rotation = np.eye(3)
            break
        if position is None or rotation is None:
            return None
        base_t, base_Rt = self._reference_transform()
        return (
            base_Rt @ (position - base_t),
            _rotmat_to_quat_wxyz(base_Rt @ rotation),
        )

    def _live_body(self, name: str) -> tuple[int, np.ndarray, np.ndarray] | None:
        """``(body_id, world_position, world_quat_wxyz)`` of the body origin."""
        if self._sim is None or not self._prepared:
            self.refresh()
        if self._sim is None:
            return None
        bid = self._objects.get(name)
        if bid is None and name == self._tabletop_name:
            model = self._sim.model
            bid = self._body_id(model, "table") or self._body_id(model, "table_top")
        if bid is None and name in self._part_bids:
            bid = self._part_bids[name][1]
        if bid is None:
            return None
        data = self._sim.data
        return (
            int(bid),
            np.asarray(data.body_xpos)[bid].astype(np.float64).copy(),
            np.asarray(data.body_xquat)[bid].astype(np.float64).copy(),
        )

    def reference_frame_name(self) -> str:
        """Concrete basis behind the connector public-world label."""
        if self._sim is None or not self._prepared:
            self.refresh()
        return "robot_base" if self._base_bid is not None else "mujoco_world"

    def object_pose(self, name: str) -> tuple[np.ndarray, np.ndarray] | None:
        """Body pose in the connector reference frame (robot-base in sim)."""
        live = self._live_body(name)
        if live is None:
            return None
        _bid, pos, quat = live
        base_t, base_Rt = self._reference_transform()
        return (
            base_Rt @ (pos - base_t),
            _rotmat_to_quat_wxyz(base_Rt @ _quat_wxyz_to_rotmat(quat)),
        )

    def _reference_transform(self) -> tuple[np.ndarray, np.ndarray]:
        """MuJoCo-world to connector-reference translation and rotation."""
        if self._sim is None or self._base_bid is None:
            return np.zeros(3), np.eye(3)
        data = self._sim.data
        base_t = np.asarray(data.body_xpos)[self._base_bid].astype(np.float64)
        base_Rt = _quat_wxyz_to_rotmat(
            np.asarray(data.body_xquat)[self._base_bid].astype(np.float64)
        ).T
        return base_t, base_Rt

    def _object_box_world(
        self, name: str,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        """Live box in MuJoCo world coordinates before reference conversion.

        For an articulated root, every child's cached local geometry is
        carried by that child's live pose and re-bounded in the root frame.
        A drawer or door therefore stays correct after its joint moves.
        """
        live = self._live_body(name)
        if live is None or self._sim is None:
            return None
        _bid, pos, quat = live

        if name in self._part_bids:
            bounds = self._part_aabbs.get(name)
            if bounds is None:
                return None
            lo, hi = bounds
            R = _quat_wxyz_to_rotmat(quat)
            return pos + R @ ((lo + hi) / 2.0), (hi - lo) / 2.0, quat

        members = [
            b for b, owner in self._body_owner.items()
            if owner == name and b in self._body_aabbs
        ]
        if not members:
            bounds = self._local_aabbs.get(name)
            if bounds is None:
                return None
            lo, hi = bounds
            R = _quat_wxyz_to_rotmat(quat)
            return pos + R @ ((lo + hi) / 2.0), (hi - lo) / 2.0, quat

        data = self._sim.data
        root_R = _quat_wxyz_to_rotmat(quat)
        points_root: list[np.ndarray] = []
        for member in members:
            lo, hi = self._body_aabbs[member]
            member_pos = np.asarray(data.body_xpos)[member].astype(np.float64)
            member_R = _quat_wxyz_to_rotmat(
                np.asarray(data.body_xquat)[member].astype(np.float64)
            )
            world_pts = (
                member_R
                @ _box_corners((lo + hi) / 2.0, (hi - lo) / 2.0).T
            ).T + member_pos
            points_root.append((root_R.T @ (world_pts - pos).T).T)
        pts = np.concatenate(points_root, axis=0)
        lo, hi = pts.min(axis=0), pts.max(axis=0)
        return pos + root_R @ ((lo + hi) / 2.0), (hi - lo) / 2.0, quat

    def object_box(self, name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        """Live oriented box in the connector reference frame.

        ``center`` is the geometry center, not necessarily the body origin.
        Object roots aggregate their live subtree; named parts use their own
        body-local geometry.
        """
        box = self._object_box_world(name)
        if box is None:
            return None
        center, half, quat = box
        base_t, base_Rt = self._reference_transform()
        return (
            base_Rt @ (center - base_t),
            half,
            _rotmat_to_quat_wxyz(base_Rt @ _quat_wxyz_to_rotmat(quat)),
        )

    def _contact_pairs(self, model: Any, data: Any) -> list[tuple[str, str]]:
        pairs: list[tuple[str, str]] = []
        try:
            ncon = int(data.ncon)
        except Exception:
            return pairs
        geom_bodyid = np.asarray(model.geom_bodyid)
        for i in range(ncon):
            con = data.contact[i]
            b1 = int(geom_bodyid[int(con.geom1)])
            b2 = int(geom_bodyid[int(con.geom2)])
            n1 = self._body_owner.get(b1)
            n2 = self._body_owner.get(b2)
            # Filtered to named bodies + robot/gripper links only.
            if n1 is None or n2 is None or n1 == n2:
                continue
            pairs.append((n1, n2))
        return pairs

    def _body_velocity(self, data: Any, bid: int) -> tuple[np.ndarray, np.ndarray]:
        try:
            cvel = np.asarray(data.cvel)[bid].astype(np.float64)
            return cvel[3:6].copy(), cvel[0:3].copy()
        except Exception:
            return np.zeros(3), np.zeros(3)

    def _robot_view(
        self,
        model: Any,
        data: Any,
        *,
        base_t: np.ndarray | None = None,
        base_Rt: np.ndarray | None = None,
    ) -> Robot | None:
        joint_names: list[str] = []
        joint_pos: list[float] = []
        qpos = np.asarray(data.qpos)
        for i in range(1, self.arm_dof + 1):
            jn = f"robot0_joint{i}"
            try:
                addr = model.get_joint_qpos_addr(jn)
            except Exception:
                addr = None
            if addr is None:
                continue
            if isinstance(addr, tuple):
                addr = addr[0]
            joint_names.append(jn)
            joint_pos.append(float(qpos[int(addr)]))
        if not joint_pos:
            return None

        ee_pos = np.array([0.5, 0.0, 0.5])
        ee_quat = np.array([1.0, 0.0, 0.0, 0.0])
        eef_id = self._body_id(model, "gripper0_eef")
        if eef_id is not None:
            ee_pos = np.asarray(data.body_xpos)[eef_id].astype(np.float64).copy()
            ee_quat = np.asarray(data.body_xquat)[eef_id].astype(np.float64).copy()
            if base_t is not None and base_Rt is not None:
                ee_pos = base_Rt @ (ee_pos - base_t)
                ee_quat = _rotmat_to_quat_wxyz(
                    base_Rt @ _quat_wxyz_to_rotmat(ee_quat)
                )

        fraction = float(getattr(self.env, "_gripper_fraction", 1.0))
        try:
            addr = model.get_joint_qpos_addr("gripper0_finger_joint1")
            if isinstance(addr, tuple):
                addr = addr[0]
            fraction = float(np.clip(qpos[int(addr)] / 0.04, 0.0, 1.0))
        except Exception:
            pass

        return Robot(
            body_name="robot",
            joint_pos=np.asarray(joint_pos, dtype=np.float64),
            joint_names=tuple(joint_names),
            ee_position=ee_pos,
            ee_quaternion_wxyz=ee_quat,
            gripper_open_fraction=fraction,
        )

    def _snapshot_from_obs(self, env_id: int) -> World:
        """No-MuJoCo fallback: object poses from the obs dict's cube_poses
        (the SimBridge GetState source) with nominal 10 cm AABBs."""
        obs = self.env.get_observation()
        bodies: dict[str, Body] = {}
        for name, pose_data in obs.get("cube_poses", {}).items():
            arr = np.asarray(pose_data, dtype=np.float64)
            pos, quat = arr[:3], arr[3:7]
            bodies[str(name)] = Body(
                name=str(name),
                position=pos.copy(),
                quaternion_wxyz=quat.copy(),
                aabb_lower=pos - 0.05,
                aabb_upper=pos + 0.05,
                linear_velocity=np.zeros(3),
                angular_velocity=np.zeros(3),
                contacts=frozenset(),
            )
        return World(
            env_id=int(env_id),
            bodies=bodies,
            robot_view=None,
            robot_link_prefixes=self.robot_link_prefixes,
        )


class LiberoWorldAdapter(MujocoSceneAdapter):
    """Backward-compatible LIBERO spelling of :class:`MujocoSceneAdapter`."""


__all__ = ["LiberoWorldAdapter", "MujocoSceneAdapter", "find_mujoco_sim"]
