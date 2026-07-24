"""Classical-CV RGB-D localization core — no learned models, no tool bundles.

DINO+SAM is overkill for this scene: the objects are uniformly coloured rigid
bodies resting on a table plane the depth camera measures directly. This core
segments by COLOUR + HEIGHT instead, entirely in-process (numpy + cv2 — zero
tool calls, zero model servers, nothing to time out or mis-score):

  1. back-project the FULL depth image to per-pixel world height,
  2. table plane = the modal 5 mm height bin of the scene (the dominant flat
     surface in the agentview) — estimated per call, no scene constant,
  3. height mask: pixels between ``CV_MIN_ABOVE_TABLE`` and
     ``CV_MAX_ABOVE_TABLE`` above that plane (kills the table, its shadows,
     and everything tall),
  4. colour mask from the query's colour word — chromatic colours are HSV
     hue bands; "gray"/"white" queries use a low-saturation (achromatic)
     band. A query with no known colour word falls back to height alone,
  5. AND the two, clean up with morphology, then connected components:
     drop specks, PREFER blobs that do not touch the image border (the
     robot's arms always enter the frame from a border; free-standing
     tabletop objects are interior blobs), keep the largest,
  6. finish on the SAME robust top-face estimators as the learned path
     (``_perceive.top_face_from_mask``), so the two pipelines emit
     identical-in-kind geometry and cannot drift apart downstream.

Failure modes are loud: an empty mask raises (or returns ``None`` under
``raise_if_missing=False``) — there is no low-confidence half-answer.
"""

from __future__ import annotations

import cv2
import numpy as np

from gap import NodeContext
from gap_core.types import CameraFrame, pose_to_matrix

from ._perceive import top_face_from_mask
from .constants import (
    CV_ACHROMATIC_MAX_SAT,
    CV_ACHROMATIC_MIN_VAL,
    CV_MAX_ABOVE_TABLE,
    CV_MIN_ABOVE_TABLE,
    CV_MIN_BLOB_PX,
    CV_TABLE_BIN,
    PERCEIVE_TOP_SLAB,
)

# HSV bands per colour word (OpenCV ranges: H 0-179, S/V 0-255). Multiple
# bands per colour support hue wrap-around (red). Matched against the words
# of the object query — "yellow tape" -> yellow, "red spool" -> red.
COLOR_HSV_BANDS: dict[str, list[tuple[tuple, tuple]]] = {
    "yellow": [((20, 70, 70), (40, 255, 255))],
    "orange": [((10, 70, 70), (20, 255, 255))],
    "red":    [((0, 70, 70), (10, 255, 255)),
               ((170, 70, 70), (179, 255, 255))],
    "green":  [((40, 60, 60), (85, 255, 255))],
    "blue":   [((95, 60, 60), (130, 255, 255))],
}
# Colour words that mean "no strong colour" — matched with a low-saturation
# band instead of a hue band (the grey/white duct).
ACHROMATIC_WORDS = frozenset({"gray", "grey", "white", "silver"})


def world_height_map(cam: CameraFrame) -> np.ndarray:
    """Per-pixel world Z (m) from the depth image; NaN where depth is invalid."""
    c2w = pose_to_matrix(cam["pose"])
    K = np.asarray(cam["intrinsics"], dtype=float)
    depth = np.asarray(cam["depth"], dtype=float)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    h, w = depth.shape
    us, vs = np.meshgrid(np.arange(w), np.arange(h))
    xc = (us - cx) * depth / fx
    yc = (vs - cy) * depth / fy
    # Only the world-Z row of the transform is needed for the height map.
    wz = c2w[2, 0] * xc + c2w[2, 1] * yc + c2w[2, 2] * depth + c2w[2, 3]
    wz[~(depth > 0)] = np.nan
    return wz


def estimate_table_z(wz: np.ndarray) -> float:
    """Table height = the modal ``CV_TABLE_BIN`` bin of the scene's heights.

    The table is the dominant flat surface in the agentview, so its plane is
    the tallest histogram peak — no scene constant, and robust to the exact
    camera pose.
    """
    z = wz[np.isfinite(wz)]
    if z.size == 0:
        raise RuntimeError("perceive_cv: depth image has no valid pixels")
    lo, hi = float(z.min()), float(z.max())
    nbins = max(int(np.ceil((hi - lo) / CV_TABLE_BIN)), 1)
    hist, edges = np.histogram(z, bins=nbins, range=(lo, hi))
    k = int(np.argmax(hist))
    return float((edges[k] + edges[k + 1]) / 2.0)


def _color_mask(rgb: np.ndarray, query: str) -> np.ndarray | None:
    """Pixel mask for the query's colour word, or ``None`` (no colour word)."""
    words = query.lower().split()
    hsv = cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2HSV)
    for word in words:
        if word in COLOR_HSV_BANDS:
            mask = np.zeros(rgb.shape[:2], dtype=np.uint8)
            for lo, hi in COLOR_HSV_BANDS[word]:
                mask |= cv2.inRange(hsv, np.array(lo, np.uint8),
                                    np.array(hi, np.uint8))
            return mask > 0
        if word in ACHROMATIC_WORDS:
            return ((hsv[..., 1] <= CV_ACHROMATIC_MAX_SAT)
                    & (hsv[..., 2] >= CV_ACHROMATIC_MIN_VAL))
    return None


def segment_query(cam: CameraFrame, query: str):
    """Segment ``query`` in one RGB-D frame. Returns ``(mask, table_z)``.

    ``mask`` is the selected object's boolean pixel mask (all-False when
    nothing matches); ``table_z`` the estimated table height (m, world).
    """
    wz = world_height_map(cam)
    table_z = estimate_table_z(wz)
    height = ((wz > table_z + CV_MIN_ABOVE_TABLE)
              & (wz < table_z + CV_MAX_ABOVE_TABLE))
    color = _color_mask(np.asarray(cam["rgb"]), query)
    mask = height if color is None else (height & color)

    # Morphology: open kills speckle, close bridges shading/specular gaps.
    m8 = mask.astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    m8 = cv2.morphologyEx(m8, cv2.MORPH_OPEN, kernel)
    m8 = cv2.morphologyEx(m8, cv2.MORPH_CLOSE, kernel)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(m8, connectivity=8)
    h, w = m8.shape
    candidates = []
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if area < CV_MIN_BLOB_PX:
            continue
        touches_border = x == 0 or y == 0 or x + bw >= w or y + bh >= h
        candidates.append((i, int(area), touches_border))
    if not candidates:
        return np.zeros((h, w), dtype=bool), table_z
    # The arms always reach in from a border; a free-standing tabletop object
    # is an interior blob. Prefer interior, then take the largest.
    interior = [c for c in candidates if not c[2]]
    pool = interior or candidates
    best = max(pool, key=lambda c: c[1])[0]
    return labels == best, table_z


def perceive_top_face_cv(ctx: NodeContext, cameras: list[CameraFrame],
                         query: str, top_slab: float = PERCEIVE_TOP_SLAB, *,
                         raise_if_missing: bool = True):
    """CV drop-in for ``_perceive.perceive_top_face``: ``(x, y, top_z, pts)``.

    Same agentview selection, same return contract, same robust estimators —
    only the segmentation differs (colour + height instead of DINO+SAM).
    ``ctx`` is unused (no tool calls); kept for signature parity.
    """
    cam = next((c for c in cameras if c["name"] == "agentview"), None)
    if cam is None:
        raise RuntimeError(
            f"perceive_cv: no 'agentview' camera in {[c.get('name') for c in cameras]}")
    mask, table_z = segment_query(cam, query)
    n_px = int(mask.sum())
    if n_px == 0:
        if not raise_if_missing:
            return None
        raise RuntimeError(
            f"perceive_cv: no '{query}' blob above the table "
            f"(table_z={table_z:.3f})")
    print(f"[perceive_cv] '{query}': {n_px}px blob, table_z={table_z:.4f}",
          flush=True)
    # floor_trim=False: the height mask already excluded the table, so the
    # percentile trim would only chop real low side points (biasing half_z).
    return top_face_from_mask(cam, mask, top_slab, label=query,
                              raise_if_missing=raise_if_missing,
                              floor_trim=False)
