"""What the eye-in-hand RGB-D view offers a vision-language model, and the gripper's own geometry.

Used by the LunarBot manipulation rSkills: the view split into surfaces at depth discontinuities and marked with
numbers, so the model can say *which* surface is the described item without ever emitting coordinates
(``find_regions`` / ``locate_region``), a pixel answer for a target that is a patch on a larger surface
(``locate_point``), and ``GripperGeometry``. Where the item's lifting interface is, to the millimetre, is not asked of
the model: ``interface_fit`` measures it against the catalogue's declared shape.
"""

from __future__ import annotations

import base64
import json
import math
import re
from dataclasses import asdict, dataclass

import cv2
import numpy as np

__all__ = ["GripperGeometry", "upright", "Region", "find_regions", "render_regions", "locate_region", "locate_point", "depth_at"]


@dataclass(frozen=True)
class GripperGeometry:
    """The gripper's own dimensions (EG2-4C2 on the RM-75: ~66 mm opening, research repo F44)."""

    contact_links: tuple[str, ...] = ("eg2_link5", "eg2_link6")  # distal jaw bodies; not drive linkages
    open_width_m: float = 0.066
    clearance_m: float = 0.010  # per side, between a finger and the grasped part when the jaws are open
    finger_width_m: float = 0.016  # finger thickness along the closing axis
    pad_height_m: float = 0.014  # minimum extent of the part along the pads (perpendicular to closing and approach)
    min_part_width_m: float = 0.005
    #: where along the approach a part is actually clamped, past the tool origin -- from the gripper's own collision
    #: meshes at a closed jaw: the fingertips reach 6 mm, the pads' clamping ridges run back to 30 mm, and the jaw
    #: body closes the pocket at 43 mm. A part short of the near edge is pinched by the fingertip alone: ga1 closed
    #: with the ORU's neck 3.2 mm in, held it, and lost it out of the front of the jaws during the drive (F81).
    pad_band_m: tuple[float, float] = (0.006, 0.030)
    depth_step_m: float = 0.030  # how far both sides must fall away behind the part's front for the fingers to pass
    self_depth_m: float = 0.25  # nearer than this along the optical axis is the gripper itself in the wrist view
    min_part_depth_m: float = 0.28  # a part nearer than this is already at the fingers (the TCP is 0.22 m ahead of the camera)
    max_range_m: float = 1.2

    @property
    def seat_depth_m(self) -> float:
        """How far past the tool origin to drive the contact before closing: the middle of the pads."""
        return (self.pad_band_m[0] + self.pad_band_m[1]) / 2

    @property
    def seat_tolerance_m(self) -> float:
        """How far off that seat the contact may still be and remain on the pads: half their length.

        Aiming at the middle of the pads and accepting half their length are two different quantities from one
        measurement. One number serving as both accepted a contact that had not entered the jaws at all."""
        return (self.pad_band_m[1] - self.pad_band_m[0]) / 2


def upright_turn(up_cam: np.ndarray):
    """The quarter turn that shows the image gravity-up, from the world up direction in the camera frame."""
    return {0: None, 1: cv2.ROTATE_90_COUNTERCLOCKWISE, 2: cv2.ROTATE_180, 3: cv2.ROTATE_90_CLOCKWISE}[
        round(math.atan2(up_cam[0], -up_cam[1]) / (math.pi / 2)) % 4]


def _turn_point(u, v, w, h, turn):
    return {None: (u, v), cv2.ROTATE_90_COUNTERCLOCKWISE: (v, w - 1 - u), cv2.ROTATE_180: (w - 1 - u, h - 1 - v),
            cv2.ROTATE_90_CLOCKWISE: (h - 1 - v, u)}[turn]


def _unturn_point(u, v, w, h, turn):
    """Where a pixel read off the gravity-up view falls in the camera's own image (`w`, `h` are the camera's)."""
    return {None: (u, v), cv2.ROTATE_90_COUNTERCLOCKWISE: (w - 1 - v, u), cv2.ROTATE_180: (w - 1 - u, h - 1 - v),
            cv2.ROTATE_90_CLOCKWISE: (v, h - 1 - u)}[turn]


def upright(bgr: np.ndarray, up_cam: np.ndarray | None) -> np.ndarray:
    turn = upright_turn(up_cam) if up_cam is not None else None
    return bgr if turn is None else cv2.rotate(bgr, turn)


@dataclass
class Region:
    id: int
    u: float
    v: float
    area_px: int
    depth_m: float
    p_cam: list[float]

    def as_dict(self) -> dict:
        return asdict(self)


def find_regions(depth: np.ndarray, K: np.ndarray, geom: GripperGeometry = GripperGeometry(), max_regions: int = 15,
                 min_area_px: int = 150) -> list[Region]:
    """Surfaces in view: connected areas of the depth image split at depth discontinuities, each marked at its most
    interior pixel with the 3-D point there. Region marks let the vision model say *where* the described part is without
    giving coordinates."""
    z = depth.astype(np.float32)
    valid = np.isfinite(z) & (z > geom.self_depth_m) & (z < geom.max_range_m)
    zz = np.where(valid, z, 100.0)
    edge = np.zeros_like(valid)
    edge[:, 1:] |= np.abs(np.diff(zz, axis=1)) > 0.02
    edge[1:, :] |= np.abs(np.diff(zz, axis=0)) > 0.02
    edge = cv2.dilate(edge.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    n, lab, stats, _ = cv2.connectedComponentsWithStats((valid & ~edge).astype(np.uint8), connectivity=4)
    order = sorted(range(1, n), key=lambda i: -stats[i, cv2.CC_STAT_AREA])
    out = []
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    for i in order:
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_area_px or len(out) >= max_regions:
            break
        m = (lab == i).astype(np.uint8)
        dist = cv2.distanceTransform(m, cv2.DIST_L2, 3)
        v, u = np.unravel_index(int(np.argmax(dist)), dist.shape)
        d = float(z[v, u])
        out.append(Region(id=len(out) + 1, u=float(u), v=float(v), area_px=area, depth_m=round(d, 4),
                          p_cam=[round(float((u - cx) / fx * d), 4), round(float((v - cy) / fy * d), 4), round(d, 4)]))
    return out


def render_regions(bgr: np.ndarray, regions: list[Region], up_cam: np.ndarray | None = None) -> np.ndarray:
    """Numbered dots on the surfaces, on the image turned gravity-up."""
    img = bgr.copy()
    h, w = img.shape[:2]
    for r in regions:
        cv2.circle(img, (int(r.u), int(r.v)), 6, (0, 0, 0), -1)
        cv2.circle(img, (int(r.u), int(r.v)), 4, (0, 255, 255), -1)
    turn = upright_turn(up_cam) if up_cam is not None else None
    out = img if turn is None else cv2.rotate(img, turn)
    for r in regions:
        x, y = _turn_point(r.u, r.v, w, h, turn)
        label = str(r.id)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        org = (int(x) + 8, int(y) + th // 2)
        cv2.rectangle(out, (org[0] - 2, org[1] - th - 3), (org[0] + tw + 2, org[1] + 3), (0, 0, 0), -1)
        cv2.putText(out, label, org, cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA)
    return out


LOCATE_PROMPT = """This image is from the camera on a robot's gripper, looking over its work area. Numbered yellow dots mark separate surfaces in view.

Task: pick up {target}, holding it by {part}.

Is the described item in view? If so, which numbered dot lies on the body of that item? This view is from above and is used only to find where the item stands -- the part to grasp is looked for later, from the side, so choose a dot on the item itself rather than one you think is on {part}.

Reply with JSON only: {{"item_visible": true or false, "what_is_visible": "<short description of the items you see>", "choice": <dot number or null>, "reason": "<one sentence>"}}"""


def ask_json(client, model: str, prompt: str, images: list[np.ndarray]) -> tuple[dict, str]:
    """One vision-language question with images; the parsed JSON object (empty when the reply is not the JSON asked
    for) and the raw reply.

    The request goes out over plain HTTP rather than through the vendor SDK: the skill runs inside the ROS runner,
    whose interpreter carries openai 1.60, and that version left image requests hanging until the deadline while the
    same request by hand answered in seconds (research repo F57/F58). One POST is all this needs. `client` is still the
    caller's configured client -- its endpoint, key and deadline are read from it."""
    import httpx

    urls = [f"data:image/jpeg;base64,{base64.b64encode(cv2.imencode('.jpg', im, [cv2.IMWRITE_JPEG_QUALITY, 92])[1].tobytes()).decode()}"
            for im in images]
    body = {"model": model, "messages": [{"role": "user", "content": [
        {"type": "text", "text": prompt}, *({"type": "image_url", "image_url": {"url": u}} for u in urls)]}]}
    timeout = getattr(client, "timeout", None)
    r = httpx.post(f"{str(client.base_url).rstrip('/')}/chat/completions", json=body,
                   headers={"Authorization": f"Bearer {client.api_key}"},
                   timeout=float(timeout if isinstance(timeout, (int, float)) else 300.0))
    r.raise_for_status()
    reply = (r.json()["choices"][0]["message"].get("content") or "").strip()
    found = re.search(r"\{.*\}", reply, re.S)
    try:
        return (json.loads(found.group(0)) if found else {}), reply
    except json.JSONDecodeError:  # a reply that is not the JSON asked for is no answer; the raw reply stays in the record
        return {}, reply


POINT_PROMPT = """This image is from the camera on a robot's gripper, looking out over its work area. It is shown the way up a person would see it.

Find {target} in this {w}x{h} image.

Reply with JSON only: {{"found": true or false, "point": [x, y], "reason": "<one short sentence>"}} -- pixel coordinates, origin at the top-left, the point on the thing itself. found=false if it is not clearly visible."""


def locate_point(client, model: str, bgr: np.ndarray, target: str, up_cam: np.ndarray | None = None):
    """Where a described thing is in this camera image: its pixel, or None with why not.

    For a target that is a patch on a larger surface -- a button on a panel -- rather than a structure standing clear
    of its surroundings. `find_regions` splits the view at depth discontinuities, so a button shares its region with
    the panel it is on and set-of-mark selection has no mark to offer; this asks for the pixel itself. The model is
    shown the gravity-up view, as everywhere else (research repo F57), and the answer is mapped back to the camera's
    own pixels here, so callers work in the frame their depth is in."""
    h, w = bgr.shape[:2]
    turn = upright_turn(up_cam) if up_cam is not None else None
    shown = bgr if turn is None else cv2.rotate(bgr, turn)
    sh, sw = shown.shape[:2]
    answer, reply = ask_json(client, model, POINT_PROMPT.format(target=target, w=sw, h=sh), [shown])
    rec = {"target": target, "reason": answer.get("reason", ""), "raw": reply, "model": model}
    pt = answer.get("point") or []
    if not answer.get("found") or len(pt) != 2:
        return None, {**rec, "found": False}
    us, vs = float(pt[0]), float(pt[1])
    if not (0 <= us < sw and 0 <= vs < sh):
        return None, {**rec, "found": False, "reason": f"the pixel {pt} is outside the {sw}x{sh} image"}
    u, v = _unturn_point(us, vs, w, h, turn)
    return (float(u), float(v)), {**rec, "found": True, "shown_px": [round(us, 1), round(vs, 1)],
                                  "point_px": [round(float(u), 1), round(float(v), 1)]}


def locate_region(client, model: str, marked_bgr: np.ndarray, target: str, part: str, regions: list[Region]) -> dict:
    answer, reply = ask_json(client, model, LOCATE_PROMPT.format(target=target, part=part), [marked_bgr])
    ids = {r.id for r in regions}
    choice = answer.get("choice")
    return dict(item_visible=bool(answer.get("item_visible")), what_is_visible=answer.get("what_is_visible", ""),
                choice=choice if choice in ids else None, reason=answer.get("reason", ""), raw=reply, model=model)


def depth_at(depth: np.ndarray, u: float, v: float, half_px: int = 3) -> float | None:
    """The depth of the surface at a pixel: the near quartile of a small patch, so a thin part in front of a far
    background reads as the part, not as the background. None where nothing was measured."""
    patch = depth[max(int(v) - half_px, 0):int(v) + half_px + 1, max(int(u) - half_px, 0):int(u) + half_px + 1]
    patch = patch[np.isfinite(patch) & (patch > 0.05)]
    return float(np.percentile(patch, 25)) if patch.size else None


