"""Geometric-consistency tests for make_test_observation.

The contract under test: unprojecting masked depth pixels through the
returned intrinsics + camera pose must land on the ground-truth object's
world-space surface. This is what makes the fixture usable for real
perception-math tests (mask → points → OBB).
"""

import numpy as np
from gap_core.types import pose_to_matrix

from gap.testing import make_test_observation


def _unproject(frame, mask):
    k = frame["intrinsics"]
    depth = frame["depth"].astype(np.float64)
    vs, us = np.nonzero(mask)
    zs = depth[vs, us]
    xs = (us + 0.5 - k[0, 2]) / k[0, 0] * zs
    ys = (vs + 0.5 - k[1, 2]) / k[1, 1] * zs
    pts_cam = np.stack([xs, ys, zs], axis=1)
    cam_mat = pose_to_matrix(frame["pose"])
    return pts_cam @ cam_mat[:3, :3].T + cam_mat[:3, 3]


def test_masked_pixels_reproject_onto_box_surface():
    center, size = (0.05, -0.02, 0.04), (0.06, 0.08, 0.08)
    obs, gt = make_test_observation([("box", center, size)])
    frame = obs["cameras"][0]
    mask = gt["box"]["mask"]
    assert mask.sum() > 50, "object should be visible"

    pts = _unproject(frame, mask)
    lo = np.asarray(center) - np.asarray(size) / 2 - 1e-6
    hi = np.asarray(center) + np.asarray(size) / 2 + 1e-6
    inside = ((pts >= lo - 1e-4) & (pts <= hi + 1e-4)).all(axis=1)
    assert inside.mean() > 0.99, "unprojected points must lie on the box"

    # On the *surface*: each point touches at least one face plane.
    face_dist = np.minimum(np.abs(pts - lo), np.abs(pts - hi)).min(axis=1)
    assert np.quantile(face_dist, 0.99) < 1e-3


def test_visible_extent_matches_ground_truth():
    center, size = (0.0, 0.0, 0.05), (0.10, 0.06, 0.10)
    obs, gt = make_test_observation([("box", center, size)])
    pts = _unproject(obs["cameras"][0], gt["box"]["mask"])
    # The visible surface spans the full footprint in x/y (top face visible).
    span = pts.max(axis=0) - pts.min(axis=0)
    assert abs(span[0] - size[0]) < 0.01
    assert abs(span[1] - size[1]) < 0.01
    # Top face height matches.
    assert abs(pts[:, 2].max() - (center[2] + size[2] / 2)) < 1e-3


def test_two_objects_distinct_colors_and_masks():
    obs, gt = make_test_observation(
        [
            ("left", (-0.08, 0.0, 0.03), (0.05, 0.05, 0.06)),
            ("right", (0.08, 0.0, 0.03), (0.05, 0.05, 0.06)),
        ]
    )
    rgb = obs["cameras"][0]["rgb"]
    m_l, m_r = gt["left"]["mask"], gt["right"]["mask"]
    assert m_l.any() and m_r.any() and not (m_l & m_r).any()
    assert tuple(rgb[m_l][0]) == gt["left"]["color"]
    assert tuple(rgb[m_r][0]) == gt["right"]["color"]


def test_observation_shape_contract():
    obs, _ = make_test_observation(image_hw=(60, 80), arm_joints=6)
    frame = obs["cameras"][0]
    assert frame["rgb"].dtype == np.uint8 and frame["rgb"].shape == (60, 80, 3)
    assert frame["depth"].dtype == np.float32 and frame["depth"].shape == (60, 80)
    assert frame["intrinsics"].shape == (3, 3)
    assert obs["arms"][0]["joint_state"]["positions"].shape == (6,)
    assert 0.0 <= obs["arms"][0]["gripper_fraction"] <= 1.0
