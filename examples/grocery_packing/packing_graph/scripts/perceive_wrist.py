"""Top-down re-perception from the wrist (eye-in-hand) camera.

After ``move_on_top`` the wrist camera looks straight down on the object, so its
RGB-D sees the object's TOP face. We GroundingDINO-detect, take the detection
nearest the image centre (the object directly below), SAM-segment it, and fuse
into a world-frame cloud. The OBB of that cloud has the true footprint centre —
unbiased by the single front-view depth that pulled the grip toward the camera.
This is the per-object analogue of the multi-view perceive skill: front view to
identify + approach, wrist view to localize precisely.
"""

from typing import Any, TypedDict

import numpy as np

from gap import NodeContext
from gap_core.types import CameraFrame, Mask, PointCloud


class Output(TypedDict):
    found: bool
    cloud: PointCloud
    mask: Mask


def _empty() -> Output:
    return {"found": False,
            "cloud": {"points": np.zeros((0, 3), dtype=np.float32)},
            "mask": np.zeros((0, 0), dtype=np.uint8)}


def run(ctx: NodeContext, cameras: list[CameraFrame],
        object_name: str = "grocery item") -> Output:
    wrist = [c for c in cameras if "eye_in_hand" in (c.get("name") or "")]
    cam: Any = (wrist or cameras)[0]
    rgb = cam["rgb"]
    h, w = rgb.shape[:2]
    cx, cy = w / 2.0, h / 2.0

    # GroundingDINO; keep the detection whose box centre is closest to the image
    # centre — that is the object we are hovering directly above.
    box = None
    try:
        dets = ctx.tool("grounding-dino.detect", image=rgb, query="object.",
                        box_threshold=0.20, text_threshold=0.20)["detections"]
        if dets:
            def _d(det: Any) -> float:
                b = det["box"]
                return ((b["x1"] + b["x2"]) / 2 - cx) ** 2 + ((b["y1"] + b["y2"]) / 2 - cy) ** 2
            box = min(dets, key=_d)["box"]
    except Exception:
        box = None

    if box is not None:
        seg = ctx.tool("sam3.segment_box", image=rgb, box=box)
    else:
        seg = ctx.tool("sam3.segment_text", image=rgb, query=object_name)
    if not seg.get("masks"):
        return _empty()

    mask = seg["masks"][0]
    cloud = ctx.tool("geometry.mask_to_world_points", mask=mask, depth=cam["depth"],
                     intrinsics=cam["intrinsics"], camera_pose=cam["pose"])["points"]
    if len(cloud["points"]) < 10:
        return _empty()
    return {"found": True, "cloud": cloud, "mask": mask}
