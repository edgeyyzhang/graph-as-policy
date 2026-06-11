"""Report the 3D location and plane orientation of the detected white tape."""

import json
import logging
from pathlib import Path
from typing import TypedDict

import numpy as np
from scipy.spatial.transform import Rotation

from gap import NodeContext
from gap.types import CameraFrame, OrientedBoundingBox, PointCloud

logger = logging.getLogger(__name__)


class Output(TypedDict):
    position_x: float
    position_y: float
    position_z: float
    camera_position_x: float
    camera_position_y: float
    camera_position_z: float
    normal_x: float
    normal_y: float
    normal_z: float
    orientation_quat: list[float]
    orientation_euler_deg: list[float]


def ransac_plane(
    pts: np.ndarray,
    n_iterations: int = 1000,
    distance_threshold: float = 0.005,
) -> tuple[np.ndarray, float, np.ndarray]:
    """Fit a plane (ax+by+cz+d=0) via RANSAC.

    Returns (normal, d, inlier_mask).
    """
    best_inliers = np.zeros(len(pts), dtype=bool)
    best_normal = np.array([0.0, 0.0, 1.0])
    best_d = 0.0
    rng = np.random.default_rng(42)

    for _ in range(n_iterations):
        idx = rng.choice(len(pts), size=3, replace=False)
        p0, p1, p2 = pts[idx]
        normal = np.cross(p1 - p0, p2 - p0)
        norm = np.linalg.norm(normal)
        if norm < 1e-12:
            continue
        normal /= norm
        d = -normal.dot(p0)
        dists = np.abs(pts.dot(normal) + d)
        inliers = dists < distance_threshold
        if inliers.sum() > best_inliers.sum():
            best_inliers = inliers
            best_normal = normal
            best_d = d

    # Orient normal to point towards +z (camera convention)
    if best_normal[2] < 0:
        best_normal = -best_normal
        best_d = -best_d

    return best_normal, best_d, best_inliers


def normal_to_quaternion(normal: np.ndarray) -> np.ndarray:
    """Rotation that maps +z to the given normal, returned as [x, y, z, w]."""
    z = np.array([0.0, 0.0, 1.0])
    n = normal / np.linalg.norm(normal)
    axis = np.cross(z, n)
    sin_angle = np.linalg.norm(axis)
    cos_angle = np.dot(z, n)
    if sin_angle < 1e-8:
        if cos_angle > 0:
            return np.array([0.0, 0.0, 0.0, 1.0])
        return np.array([1.0, 0.0, 0.0, 0.0])
    axis /= sin_angle
    angle = np.arctan2(sin_angle, cos_angle)
    return Rotation.from_rotvec(axis * angle).as_quat()


def points_in_obb(
    pts: np.ndarray, obb: OrientedBoundingBox
) -> np.ndarray:
    """Return the subset of pts that lie inside the oriented bounding box."""
    c = obb["center"]
    e = obb["extent"]
    center = np.array([c["x"], c["y"], c["z"]])
    half = np.array([e["x"], e["y"], e["z"]])
    q = obb["orientation"]
    rot_inv = Rotation.from_quat([q["x"], q["y"], q["z"], q["w"]]).inv()
    local = rot_inv.apply(pts - center)
    mask = np.all(np.abs(local) <= half, axis=1)
    return pts[mask]


def world_to_camera(point_world: np.ndarray, camera: CameraFrame) -> np.ndarray:
    """Transform a world-frame point into camera coordinates."""
    pos = camera["pose"]["position"]
    rot = camera["pose"]["rotation"]
    cam_pos_world = np.array([pos["x"], pos["y"], pos["z"]], dtype=np.float64)
    cam_rot_world = Rotation.from_quat([rot["x"], rot["y"], rot["z"], rot["w"]])
    return cam_rot_world.inv().apply(point_world - cam_pos_world)


def run(
    ctx: NodeContext,
    obb: OrientedBoundingBox,
    cloud: PointCloud,
    camera: CameraFrame,
) -> Output:
    x = obb["center"]["x"]
    y = obb["center"]["y"]
    z = obb["center"]["z"]
    point_world = np.array([x, y, z], dtype=np.float64)
    point_camera = world_to_camera(point_world, camera)

    all_pts = np.asarray(cloud["points"], dtype=np.float64).reshape(-1, 3)
    pts = points_in_obb(all_pts, obb)
    if len(pts) < 3:
        logger.warning("Only %d points inside OBB, falling back to all %d points",
                       len(pts), len(all_pts))
        pts = all_pts

    normal, d, inlier_mask = ransac_plane(pts)
    n_inliers = int(inlier_mask.sum())
    quat = normal_to_quaternion(normal)
    euler_deg = Rotation.from_quat(quat).as_euler("xyz", degrees=True).tolist()

    cam_name = camera.get("name") or "camera"
    logger.info("=" * 60)
    logger.info("WHITE TAPE 3D LOCATION (robot base frame)")
    logger.info("  Position: [%.4f, %.4f, %.4f] m", x, y, z)
    logger.info("WHITE TAPE 3D LOCATION (camera frame: %s)", cam_name)
    logger.info("  Position: [%.4f, %.4f, %.4f] m",
                point_camera[0], point_camera[1], point_camera[2])
    logger.info("  OBB extent: [%.4f, %.4f, %.4f] m",
                obb["extent"]["x"], obb["extent"]["y"], obb["extent"]["z"])
    logger.info("  Point cloud: %d/%d points in OBB (%d inliers)",
                len(pts), len(all_pts), n_inliers)
    logger.info("  Plane normal: [%.4f, %.4f, %.4f], d=%.4f",
                normal[0], normal[1], normal[2], d)
    logger.info("  Orientation (quat xyzw): [%.4f, %.4f, %.4f, %.4f]",
                quat[0], quat[1], quat[2], quat[3])
    logger.info("  Orientation (euler xyz°): [%.2f, %.2f, %.2f]",
                euler_deg[0], euler_deg[1], euler_deg[2])
    logger.info("=" * 60)

    print(f"\nWhite tape location: [{x:.4f}, {y:.4f}, {z:.4f}] m")
    print(f"Camera-frame location: [{point_camera[0]:.4f}, {point_camera[1]:.4f}, {point_camera[2]:.4f}] m")
    print(f"Plane normal:        [{normal[0]:.4f}, {normal[1]:.4f}, {normal[2]:.4f}]")
    print(f"Orientation (euler):  [{euler_deg[0]:.2f}, {euler_deg[1]:.2f}, {euler_deg[2]:.2f}] deg")

    result = {
        "position": {"x": x, "y": y, "z": z},
        "camera_frame_position": {
            "x": float(point_camera[0]),
            "y": float(point_camera[1]),
            "z": float(point_camera[2]),
            "camera_name": cam_name,
        },
        "bounding_box": {
            "center": {"x": x, "y": y, "z": z},
            "extent": {
                "x": obb["extent"]["x"],
                "y": obb["extent"]["y"],
                "z": obb["extent"]["z"],
            },
        },
        "plane": {
            "normal": {"x": float(normal[0]), "y": float(normal[1]), "z": float(normal[2])},
            "d": float(d),
            "n_inliers": n_inliers,
            "n_points": len(pts),
        },
        "orientation": {
            "quaternion_xyzw": quat.tolist(),
            "euler_xyz_deg": euler_deg,
        },
    }

    out_path = Path("/tmp/white_tape_location.json")
    out_path.write_text(json.dumps(result, indent=2))
    logger.info("Location data saved to %s", out_path)

    return {
        "position_x": x,
        "position_y": y,
        "position_z": z,
        "camera_position_x": float(point_camera[0]),
        "camera_position_y": float(point_camera[1]),
        "camera_position_z": float(point_camera[2]),
        "normal_x": float(normal[0]),
        "normal_y": float(normal[1]),
        "normal_z": float(normal[2]),
        "orientation_quat": quat.tolist(),
        "orientation_euler_deg": euler_deg,
    }
