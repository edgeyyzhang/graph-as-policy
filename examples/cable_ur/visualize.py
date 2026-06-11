"""Visualize cable perception results in viser.

Shows the UR robot URDF at the detected joint state, the ZED camera
frustum positioned via the hand-eye calibration, the perception point
cloud, and the oriented bounding box around the detected white tape —
all loaded from a `gap run` trace directory.

Usage:
    python examples/cable_ur/visualize.py [--trace outputs/run_YYYYmmdd_HHMMSS]
    # Then open the URL printed in the terminal

Without --trace, the most recent ``outputs/run_*`` directory is used.
The URDF resolves from $GAP_UR_URDF or robot_descriptions'
ur5e_description; the optional 4x4 camera→wrist calibration .npy comes
from --calib or $GAP_UR_ZED_CALIB (matching gap.envs.ur_zed_env).
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import trimesh
import viser
from scipy.spatial.transform import Rotation


def _swap_mount_xy(T: np.ndarray) -> np.ndarray:
    """Match URZedEnv's wrist-frame camera offset correction."""
    out = np.array(T, copy=True)
    out[0, 3], out[1, 3] = -T[1, 3], T[0, 3]
    return out


def _mat_to_pos_wxyz(T: np.ndarray):
    pos = T[:3, 3].astype(np.float32)
    q = Rotation.from_matrix(T[:3, :3]).as_quat()  # xyzw
    wxyz = np.array([q[3], q[0], q[1], q[2]], dtype=np.float32)
    return pos, wxyz


# ---------------------------------------------------------------------------
# Load trace data
# ---------------------------------------------------------------------------


def _latest_trace() -> Path:
    runs = sorted(Path("outputs").glob("run_*"), key=lambda p: p.stat().st_mtime)
    if not runs:
        raise SystemExit(
            "No outputs/run_* trace found — run the graph first:\n"
            "  gap run examples/cable_ur/graph --real ur_zed"
        )
    return runs[-1]


def _node_dir(trace: Path, suffix: str) -> Path:
    """Find node_data/<...>.<suffix> regardless of the subgraph node name."""
    node_data = trace / "node_data"
    matches = [d for d in node_data.iterdir() if d.name.endswith(suffix)]
    if not matches:
        raise SystemExit(f"No node_data/*{suffix} in {trace}")
    return matches[0]


def _load_trace(trace: Path) -> dict:
    observe_dir = _node_dir(trace, ".observe")
    report_dir = _node_dir(trace, ".report")

    obs = json.loads((observe_dir / "output.json").read_text())
    report_in = json.loads((report_dir / "resolved_inputs.json").read_text())
    report_out = json.loads((report_dir / "output.json").read_text())

    arm = obs["arms"][0]
    joints = list(arm["joint_state"]["positions"])[:6]

    cam = obs["cameras"][0]
    K = np.asarray(cam["intrinsics"], dtype=np.float64).reshape(3, 3)
    cam_pos = cam["pose"]["position"]
    cam_rot = cam["pose"]["rotation"]
    cam_position = np.array([cam_pos["x"], cam_pos["y"], cam_pos["z"]])
    cam_wxyz = np.array([cam_rot["w"], cam_rot["x"], cam_rot["y"], cam_rot["z"]])

    obb = report_in["obb"]
    obb_center = np.array([obb["center"]["x"], obb["center"]["y"], obb["center"]["z"]])
    obb_extent = np.array([obb["extent"]["x"], obb["extent"]["y"], obb["extent"]["z"]])
    q = obb.get("orientation", {})
    obb_wxyz = np.array([
        q.get("w", 1.0), q.get("x", 0.0), q.get("y", 0.0), q.get("z", 0.0),
    ])

    # Point cloud: large arrays land in assets/<...>_cloud.npz, not in JSON.
    pts = np.zeros((0, 3), dtype=np.float64)
    colors = np.zeros((0, 3), dtype=np.uint8)
    clouds = sorted((report_dir / "assets").glob("*_cloud.npz"))
    if clouds:
        npz = np.load(clouds[0])
        pts = np.asarray(npz["positions"], dtype=np.float64).reshape(-1, 3)
        if "colors" in npz:
            colors = np.asarray(npz["colors"], dtype=np.uint8).reshape(-1, 3)
    if len(colors) != len(pts):
        colors = np.full((len(pts), 3), [0, 200, 0], dtype=np.uint8)

    plane_normal = np.array([
        report_out["normal_x"], report_out["normal_y"], report_out["normal_z"],
    ])
    plane_quat_xyzw = np.array(report_out["orientation_quat"])
    plane_wxyz = np.array([
        plane_quat_xyzw[3], plane_quat_xyzw[0],
        plane_quat_xyzw[1], plane_quat_xyzw[2],
    ])

    rgb_assets = sorted((observe_dir / "assets").glob("*_rgb.png"))

    return {
        "joints": joints,
        "cam_position": cam_position,
        "cam_wxyz": cam_wxyz,
        "intrinsics": K,
        "obb_center": obb_center,
        "obb_extent": obb_extent,
        "obb_wxyz": obb_wxyz,
        "points": pts,
        "colors": colors,
        "plane_normal": plane_normal,
        "plane_wxyz": plane_wxyz,
        "rgb_path": rgb_assets[0] if rgb_assets else None,
    }


# ---------------------------------------------------------------------------
# URDF rendering
# ---------------------------------------------------------------------------

_UR_JOINT_NAMES = [
    "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
    "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
]


def _load_urdf():
    urdf_path = os.environ.get("GAP_UR_URDF")
    if urdf_path:
        import yourdfpy

        return yourdfpy.URDF.load(urdf_path), Path(urdf_path).parent
    from robot_descriptions.loaders.yourdfpy import load_robot_description

    return load_robot_description("ur5e_description"), None


def _add_urdf_to_scene(server: viser.ViserServer, joints: list[float]):
    robot, mesh_root = _load_urdf()
    cfg = dict(zip(_UR_JOINT_NAMES, joints, strict=True))
    robot.update_cfg(cfg)

    for link_name in robot.link_map:
        link = robot.link_map[link_name]
        T = robot.get_transform(link_name)
        if T is None:
            continue

        pos, wxyz = _mat_to_pos_wxyz(T)
        server.scene.add_frame(
            f"/robot/{link_name}",
            position=pos,
            wxyz=wxyz,
            axes_length=0.0,
            axes_radius=0.0,
        )

        for vi, visual in enumerate(link.visuals):
            if visual.geometry is None or visual.geometry.mesh is None:
                continue
            mesh_path = visual.geometry.mesh.filename
            if not Path(mesh_path).is_absolute() and mesh_root is not None:
                mesh_path = str(mesh_root / mesh_path)
            try:
                mesh = trimesh.load(mesh_path, force="mesh")
            except Exception:
                continue

            if visual.geometry.mesh.scale is not None:
                mesh.apply_scale(visual.geometry.mesh.scale)

            if visual.origin is not None:
                vp, vwxyz = _mat_to_pos_wxyz(visual.origin)
            else:
                vp = np.zeros(3, dtype=np.float32)
                vwxyz = np.array([1, 0, 0, 0], dtype=np.float32)

            server.scene.add_mesh_trimesh(
                f"/robot/{link_name}/visual_{vi}",
                mesh=mesh,
                position=vp,
                wxyz=vwxyz,
            )

    return robot


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", default=None, help="gap run trace directory")
    parser.add_argument(
        "--calib", default=os.environ.get("GAP_UR_ZED_CALIB"),
        help="4x4 camera-to-wrist calibration .npy (optional)",
    )
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    trace = Path(args.trace) if args.trace else _latest_trace()
    print(f"Loading trace from {trace}")
    data = _load_trace(trace)
    server = viser.ViserServer(host="0.0.0.0", port=args.port)

    # --- Robot URDF at recorded joint config ---
    robot = _add_urdf_to_scene(server, data["joints"])

    # --- EE link frame ---
    ee_link = "ee_link" if "ee_link" in robot.link_map else "wrist_3_link"
    T_ee = robot.get_transform(ee_link)
    if T_ee is not None:
        ee_pos, ee_wxyz = _mat_to_pos_wxyz(T_ee)
        server.scene.add_frame(
            "/robot/ee",
            position=ee_pos,
            wxyz=ee_wxyz,
            axes_length=0.06,
            axes_radius=0.003,
        )
        server.scene.add_label("/robot/ee/label", text="EE (URDF FK)")

    # --- Camera frustum: calibration when available, else the trace pose ---
    K = data["intrinsics"]
    fy = K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    img_w, img_h = int(cx * 2), int(cy * 2)
    frustum_kwargs = dict(
        fov=float(2 * np.arctan(img_h / (2 * fy))),
        aspect=float(img_w / max(img_h, 1)),
        scale=0.12,
        color=(100, 150, 255),
    )
    if data["rgb_path"] is not None:
        import imageio.v3 as iio

        frustum_kwargs["image"] = iio.imread(str(data["rgb_path"]))

    T_wrist = robot.get_transform("wrist_3_link")
    if args.calib and T_wrist is not None:
        # Camera pose in base = T_base_wrist @ inv(T_cam_to_wrist):
        # T_cam_wrist maps camera frame → wrist frame, so inverting it puts
        # the camera origin in wrist coords; compose with the FK wrist pose.
        T_cam_wrist = np.load(args.calib)  # 4x4 camera→wrist
        T_cam_in_world = T_wrist @ _swap_mount_xy(np.linalg.inv(T_cam_wrist))
        cam_pos_calib, cam_wxyz_calib = _mat_to_pos_wxyz(T_cam_in_world)

        server.scene.add_frame(
            "/camera_calib",
            position=cam_pos_calib,
            wxyz=cam_wxyz_calib,
            axes_length=0.06,
            axes_radius=0.003,
        )
        server.scene.add_label("/camera_calib/label", text="ZED (from calibration)")
        server.scene.add_camera_frustum("/camera_calib/frustum", **frustum_kwargs)
    else:
        server.scene.add_camera_frustum(
            "/camera_trace/frustum",
            position=data["cam_position"].astype(np.float32),
            wxyz=data["cam_wxyz"].astype(np.float32),
            **frustum_kwargs,
        )

    # --- Camera pose from trace (for comparison) ---
    server.scene.add_frame(
        "/camera_trace",
        position=data["cam_position"].astype(np.float32),
        wxyz=data["cam_wxyz"].astype(np.float32),
        axes_length=0.04,
        axes_radius=0.002,
    )
    server.scene.add_label("/camera_trace/label", text="cam (trace)")

    # --- Point cloud ---
    if len(data["points"]):
        server.scene.add_point_cloud(
            "/perception/points",
            points=data["points"].astype(np.float32),
            colors=data["colors"],
            point_size=0.003,
            point_shape="circle",
        )

    # --- OBB wireframe ---
    obb_center = data["obb_center"]
    obb_extent = data["obb_extent"]
    obb_wxyz = data["obb_wxyz"]

    corners_local = np.array([
        [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
        [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1],
    ], dtype=np.float64) * obb_extent

    R_obb = Rotation.from_quat(
        [obb_wxyz[1], obb_wxyz[2], obb_wxyz[3], obb_wxyz[0]]
    ).as_matrix()
    corners_world = (R_obb @ corners_local.T).T + obb_center

    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]
    seg_pts = np.array(
        [[corners_world[a], corners_world[b]] for a, b in edges],
        dtype=np.float32,
    )
    server.scene.add_line_segments(
        "/perception/obb/edges",
        points=seg_pts,
        colors=(255, 50, 50),
        line_width=3.0,
    )

    server.scene.add_frame(
        "/perception/obb/center",
        position=obb_center.astype(np.float32),
        wxyz=obb_wxyz.astype(np.float32),
        axes_length=0.02,
        axes_radius=0.001,
    )
    server.scene.add_label(
        "/perception/obb/label",
        text=(
            f"White tape\n[{obb_center[0]:.3f}, {obb_center[1]:.3f}, "
            f"{obb_center[2]:.3f}] m"
        ),
        position=(obb_center + np.array([0, 0, 0.03])).astype(np.float32),
    )

    # --- Fitted plane (semi-transparent quad at OBB center) ---
    plane_normal = data["plane_normal"]
    plane_wxyz = data["plane_wxyz"]
    plane_size = max(obb_extent[0], obb_extent[1], obb_extent[2]) * 3.0

    plane_mesh = trimesh.creation.box(extents=[plane_size, plane_size, 0.0005])
    plane_mesh.visual.face_colors = [50, 150, 255, 100]
    server.scene.add_mesh_trimesh(
        "/perception/plane",
        mesh=plane_mesh,
        position=obb_center.astype(np.float32),
        wxyz=plane_wxyz.astype(np.float32),
    )

    normal_end = obb_center + plane_normal * 0.08
    server.scene.add_line_segments(
        "/perception/plane/normal_arrow",
        points=np.array([[obb_center, normal_end]], dtype=np.float32),
        colors=(50, 150, 255),
        line_width=3.0,
    )
    server.scene.add_label(
        "/perception/plane/label",
        text="fitted plane",
        position=normal_end.astype(np.float32),
    )

    # --- World origin + grid ---
    server.scene.add_frame(
        "/world",
        position=(0, 0, 0),
        axes_length=0.15,
        axes_radius=0.003,
    )
    server.scene.add_label("/world/label", text="robot base", position=(0.06, 0, 0))

    server.scene.add_grid(
        "/grid",
        width=1.5,
        height=1.5,
        cell_size=0.1,
        position=(0, 0, -0.22),
    )

    print("\nOpen the URL above in your browser.")
    print("You should see:")
    print("  - UR robot at the recorded joint config")
    print("  - Camera frustum (calibration when --calib given, else trace pose)")
    print("  - Point cloud + red OBB for the white tape")
    print("  - Blue fitted plane with normal arrow")
    print("Press Ctrl+C to exit.\n")

    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
