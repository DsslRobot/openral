"""Class-agnostic grasp perception on one eye-in-hand RGB-D view (no object models).

Used inside the LunarBot manipulation rSkills (``procedural_pick``): the stages *find* (antipodal grasp candidates from
depth discontinuities), *select* (a vision-language model chooses among the numbered candidates drawn on the image for
a target and part described in words -- it never outputs coordinates) and the per-frame re-association that keeps a
chosen candidate locked while the camera moves.

A candidate is a planar parallel-jaw grasp seen along the camera's viewing ray: a foreground segment whose two sides
drop away in depth far enough for the fingers to pass, whose metric width fits inside the open jaws with clearance,
and which continues along the jaw pads' height. It carries its centre pixel and 3-D point in the camera frame, the
closing axis (image angle and 3-D direction), width, depth step and score. Everything is measured in the current
image; the only geometry assumed is the gripper's own (jaw opening, finger width, pad height).
"""

from __future__ import annotations

import base64
import json
import math
import re
from dataclasses import asdict, dataclass, field, replace

import cv2
import numpy as np

__all__ = ["Candidate", "GripperGeometry", "find_candidates", "refine_contact", "render_candidates", "select_candidate", "associate", "upright", "PartTracker", "Region", "find_regions", "render_regions", "locate_region"]


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
        measurement. Using a single number for both (the old 12 mm `pad_offset_m`) accepted a contact that had not
        entered the jaws at all."""
        return (self.pad_band_m[1] - self.pad_band_m[0]) / 2


@dataclass
class Candidate:
    id: int
    u: float
    v: float
    angle_deg: float  # closing axis in the image, degrees from +u towards +v
    width_m: float
    depth_m: float
    length_m: float
    step_m: float
    score: float
    p_cam: list[float]
    axis_cam: list[float]
    part_axis_cam: list[float]  # the part's extent along the pads (perpendicular to the closing axis)
    ends_px: list[list[float]] = field(default_factory=list)  # the two edge points across the part

    def as_dict(self) -> dict:
        return asdict(self)


def _free(z: np.ndarray, front: float, step: float) -> np.ndarray:
    """Pixels behind which a finger can pass: invalid returns (sky, out of range) or deeper than the part by `step`. The
    gripper's own pixels (masked to 0) are not free."""
    return ~np.isfinite(z) | (z >= front + step)


def refine_contact(depth: np.ndarray, K: np.ndarray, candidate: Candidate,
                   geom: GripperGeometry) -> tuple[Candidate, dict]:
    """Measure the selected contact's tangent frame from its visible depth surface.

    The image-plane closing axis is only a proposal. At an oblique view, assigning
    the same depth to both edges rotates the jaws into the surface. Fit the local
    surface inside the jaw span, then intersect the part's image direction with it.
    This refines the same selected contact before commitment, never selects another.
    """
    center = np.array([candidate.u, candidate.v])
    edge = np.diff(np.asarray(candidate.ends_px), axis=0)[0]
    span = np.linalg.norm(edge)
    closing_px = np.asarray(candidate.axis_cam[:2])
    closing_px = closing_px / np.linalg.norm(closing_px)
    along_px = np.array([-closing_px[1], closing_px[0]])
    radius = int(math.ceil(max(span, geom.pad_height_m * K[1, 1] / candidate.depth_m)))
    h, w = depth.shape
    x0, x1 = max(0, int(candidate.u) - radius), min(w, int(candidate.u) + radius + 1)
    y0, y1 = max(0, int(candidate.v) - radius), min(h, int(candidate.v) + radius + 1)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    delta = np.stack([xx - center[0], yy - center[1]], axis=-1)
    z = depth[y0:y1, x0:x1]
    inside = (abs(delta @ closing_px) < span / 2 - 1) & (
        abs(delta @ along_px) < geom.pad_height_m * K[1, 1] / candidate.depth_m / 2)
    inside &= np.isfinite(z) & (z > 0) & (abs(z - candidate.depth_m) < geom.depth_step_m)
    z = z[inside]
    points = np.stack([(xx[inside] - K[0, 2]) * z / K[0, 0],
                       (yy[inside] - K[1, 2]) * z / K[1, 1], z], axis=1)
    if len(points) < 3:
        raise ValueError("selected contact has insufficient depth for a surface direction")
    mean = points.mean(axis=0)
    _, _, vectors = np.linalg.svd(points - mean, full_matrices=False)
    normal = vectors[-1]
    normal *= np.sign(normal[2])
    rays = np.array([[(p[0] - K[0, 2]) / K[0, 0],
                      (p[1] - K[1, 2]) / K[1, 1], 1.0]
                     for p in (center - along_px, center + along_px)])
    surface_points = rays * (np.dot(normal, mean) / (rays @ normal))[:, None]
    part = surface_points[1] - surface_points[0]
    part /= np.linalg.norm(part)
    if np.dot(part, candidate.part_axis_cam) < 0:
        part = -part
    closing = np.cross(part, normal)
    closing /= np.linalg.norm(closing)
    if np.dot(closing, candidate.axis_cam) < 0:
        closing = -closing
    evidence = {"points": len(points), "normal_cam": normal.tolist(),
                "plane_rms_m": float(np.sqrt(np.mean(((points - mean) @ normal) ** 2))),
                "closing_correction_deg": math.degrees(math.acos(float(np.clip(
                    np.dot(closing, candidate.axis_cam) / np.linalg.norm(candidate.axis_cam), -1, 1))))}
    return replace(candidate, axis_cam=closing.tolist(), part_axis_cam=part.tolist()), evidence


def find_candidates(depth: np.ndarray, K: np.ndarray, geom: GripperGeometry = GripperGeometry(),
                    angles_deg: tuple[float, ...] = tuple(range(0, 180, 15)), max_candidates: int = 12,
                    border_px: int = 28, contact_width_m: float | None = None) -> list[Candidate]:
    """Antipodal parallel-jaw candidates on a depth image (metres along the optical axis).

    With `contact_width_m` -- the declared width of the item's contact interface -- only places the jaws would close
    across that width are offered. An interface says what may be held: for a lifting eye that is the neck, not the
    head, the pedestal, or the neck's other cross-section. Places that do not measure as that contact are not grasps
    of the declared interface, so they are never offered for selection (g9f carried a 30 mm cut of the same neck and
    the drive pulled it out of the jaws).
    """
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    h, w = depth.shape
    z_all = depth.astype(np.float32).copy()
    self_mask = np.isfinite(z_all) & (z_all > 0) & (z_all < geom.self_depth_m)
    z_all[~np.isfinite(z_all) | (z_all <= 0) | (z_all > geom.max_range_m)] = np.inf
    z_all[self_mask] = 0.0  # the gripper: neither a part nor free space
    centre = (w / 2.0, h / 2.0)
    raw = []
    for ang in angles_deg:
        # rotate so the closing axis is the image row; nearest keeps depth values honest
        M = cv2.getRotationMatrix2D(centre, ang, 1.0)
        diag = int(math.ceil(math.hypot(w, h)))
        M[0, 2] += (diag - w) / 2.0
        M[1, 2] += (diag - h) / 2.0
        zr = cv2.warpAffine(z_all, M, (diag, diag), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=float("nan"))
        Minv = cv2.invertAffineTransform(M)
        runs = []  # (row, c0, c1, front)
        for r in range(0, diag, 2):
            row = zr[r]
            known = ~np.isnan(row)
            if known.sum() < 10:
                continue
            fin = np.where(np.isfinite(row) & (row > 0), row, np.inf)
            jump = np.abs(np.diff(np.where(np.isinf(fin), 50.0, fin)))
            edges = np.nonzero((jump > geom.depth_step_m) & known[:-1] & known[1:])[0]
            for a, b in zip(edges[:-1], edges[1:]):
                seg = fin[a + 1:b + 1]
                if not np.all(np.isfinite(seg)) or seg.size < 2:
                    continue
                front = float(np.median(seg))
                if np.ptp(seg) > geom.depth_step_m or front < geom.min_part_depth_m:
                    continue
                px_per_m = fx / front
                width = seg.size / px_per_m
                if not geom.min_part_width_m <= width <= geom.open_width_m - 2 * geom.clearance_m:
                    continue
                # both fingers fit: free space for clearance + finger width beyond each edge
                reach = int(math.ceil((geom.clearance_m + geom.finger_width_m) * px_per_m))
                left, right = row[max(a + 1 - reach, 0):a + 1], row[b + 1:b + 1 + reach]
                if left.size < reach or right.size < reach or np.isnan(left).any() or np.isnan(right).any():
                    continue
                if not (_free(left, front, geom.depth_step_m).all() and _free(right, front, geom.depth_step_m).all()):
                    continue
                side = np.concatenate([left, right])
                step = float(np.min(np.where(np.isfinite(side), side, front + 1.0)) - front)
                runs.append((r, a + 1, b + 1, front, step))
        # chain runs on neighbouring rows into parts that extend along the pads
        runs.sort()
        used = [False] * len(runs)
        for i, (r0, c0, c1, f0, s0) in enumerate(runs):
            if used[i]:
                continue
            chain, last = [i], i
            for j in range(i + 1, len(runs)):
                rj, cj0, cj1, fj, _ = runs[j]
                rl, cl0, cl1, fl, _ = runs[last]
                if rj - rl > 2:
                    if rj - rl > 4:
                        break
                    continue
                # Follow both boundaries, not only the centreline: a narrow part
                # and its wider mounting block can have the same centre. Joining
                # them corrupts the measured contact span and can hide the narrow
                # part from the candidates. Keep the existing contour tolerance.
                if rj > rl and max(abs(cj0 - cl0), abs(cj1 - cl1)) <= 3 and abs(fj - fl) < 0.015:
                    chain.append(j)
                    last = j
            for j in chain:
                used[j] = True
            rows = [runs[j] for j in chain]
            front = float(np.median([x[3] for x in rows]))
            length = (rows[-1][0] - rows[0][0] + 2) * front / fy
            if length < geom.pad_height_m:
                continue
            # one candidate per pad height along a long part
            top, bot = rows[0], rows[-1]
            part_ends = [Minv @ np.array([(top[1] + top[2]) / 2 - 0.5, top[0], 1.0]), Minv @ np.array([(bot[1] + bot[2]) / 2 - 0.5, bot[0], 1.0])]
            n_seg = max(1, int(length // (2 * geom.pad_height_m)))
            for k in range(n_seg):
                sub = rows[int(k * len(rows) / n_seg):int((k + 1) * len(rows) / n_seg)]
                rc = float(np.mean([x[0] for x in sub]))
                c0 = float(np.mean([x[1] for x in sub]))
                c1 = float(np.mean([x[2] for x in sub]))
                fr = float(np.median([x[3] for x in sub]))
                step = float(np.min([x[4] for x in sub]))
                ends = [Minv @ np.array([c0 - 0.5, rc, 1.0]), Minv @ np.array([c1 - 0.5, rc, 1.0])]
                uc, vc = (ends[0] + ends[1]) / 2
                width = (c1 - c0) * fr / fx
                raw.append(dict(ang=ang, u=float(uc), v=float(vc), front=fr, width=width, length=length, step=step,
                                ends=[e.tolist() for e in ends], part_ends=[e.tolist() for e in part_ends],
                                part_depths=[top[3], bot[3]]))
    # 3-D, score, non-maximum suppression across angles and positions
    cands = []
    for c in raw:
        # antipodal refinement: the fingers close perpendicular to the part's own long axis (measured along the chain),
        # not along the scan direction that found it; the width shrinks by the angle between the two
        e0, e1 = np.array(c["ends"][0]), np.array(c["ends"][1])
        pd = np.array(c["part_ends"][1]) - np.array(c["part_ends"][0])
        if np.linalg.norm(pd) > 3.0:
            pn = np.array([-pd[1], pd[0]]) / np.linalg.norm(pd)
            cut = e1 - e0
            cosang = abs(float(np.dot(cut / max(np.linalg.norm(cut), 1e-9), pn)))
            half = np.linalg.norm(cut) * cosang / 2
            mid = (e0 + e1) / 2
            pn = pn if np.dot(pn, cut) >= 0 else -pn
            c["ends"] = [(mid - pn * half).tolist(), (mid + pn * half).tolist()]
            c["width"] = c["width"] * cosang
            c["ang"] = round(math.degrees(math.atan2(pn[1], pn[0])) % 180.0, 1)
        if not geom.min_part_width_m <= c["width"]:
            continue
        p = np.array([(c["u"] - cx) / fx, (c["v"] - cy) / fy, 1.0]) * (c["front"] + c["width"] / 2)  # the part's centre, not its face
        e0 = np.array([(c["ends"][0][0] - cx) / fx, (c["ends"][0][1] - cy) / fy, 1.0]) * c["front"]
        e1 = np.array([(c["ends"][1][0] - cx) / fx, (c["ends"][1][1] - cy) / fy, 1.0]) * c["front"]
        axis = (e1 - e0) / max(np.linalg.norm(e1 - e0), 1e-9)
        # each end of the part at its own depth: a part leaning towards the camera is not in the image plane
        a0 = np.array([(c["part_ends"][0][0] - cx) / fx, (c["part_ends"][0][1] - cy) / fy, 1.0]) * c["part_depths"][0]
        a1 = np.array([(c["part_ends"][1][0] - cx) / fx, (c["part_ends"][1][1] - cy) / fy, 1.0]) * c["part_depths"][1]
        part_axis = a1 - a0 - axis * np.dot(a1 - a0, axis)
        part_axis = part_axis / max(np.linalg.norm(part_axis), 1e-9)
        # what this view cannot measure well is not a grasp: a part whose extent along the pads runs into the viewing
        # ray is seen end-on, so its direction -- and with it the jaw axis -- comes out of the depth noise, and a place
        # within a template's half-width of the border has no room for the tracker to follow it (research repo F57: a
        # corner candidate with its axis along the ray read 0.157 rad in the jaws and slipped out under the lift)
        if abs(float(part_axis[2])) > 0.8:
            continue
        if not (border_px <= c["u"] < w - border_px and border_px <= c["v"] < h - border_px):
            continue
        # a cut across a part at an angle is wider than the perpendicular one: the true antipodal closing axis is the
        # narrowest cut at a place, so the narrowest wins there; places are then ranked by depth step and extent
        score = min(c["step"], 0.3) / 0.3 + min(c["length"], 0.06) / 0.06
        cands.append(dict(c, p=p, axis=axis, part_axis=part_axis, score=score))
    if contact_width_m is not None:
        # the same two pixels of doubt at each measured edge the harness reconciles with
        cands = [c for c in cands if abs(c["width"] - contact_width_m) <= 4 * c["front"] / fx]
    cands.sort(key=lambda c: c["width"])
    kept = []
    for c in cands:
        if all(np.linalg.norm(c["p"] - k["p"]) > 0.02 for k in kept):
            kept.append(c)
    kept = sorted(kept, key=lambda c: -c["score"])[:max_candidates]
    return [Candidate(id=i + 1, u=round(c["u"], 1), v=round(c["v"], 1), angle_deg=float(c["ang"]), width_m=round(c["width"], 4),
                      depth_m=round(c["front"], 4), length_m=round(c["length"], 4), step_m=round(min(c["step"], 9.9), 3),
                      score=round(c["score"], 3), p_cam=[round(float(v), 4) for v in c["p"]],
                      axis_cam=[round(float(v), 4) for v in c["axis"]],
                      part_axis_cam=[round(float(v), 4) for v in c["part_axis"]], ends_px=[[round(v, 1) for v in e] for e in c["ends"]])
            for i, c in enumerate(kept)]


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


def render_candidates(bgr: np.ndarray, cands: list[Candidate], up_cam: np.ndarray | None = None, zoom_px: int = 640) -> np.ndarray:
    """Numbered marks for set-of-mark selection on a magnified crop around the candidates, turned gravity-up: each
    candidate is drawn as the jaw span across the part (a bar with two finger ticks) and its number. Marks are drawn
    after magnification so they stay small against the part and the digits stay legible."""
    h, w = bgr.shape[:2]
    us = [e[0] for c in cands for e in c.ends_px] + [c.u for c in cands]
    vs = [e[1] for c in cands for e in c.ends_px] + [c.v for c in cands]
    half = max(max(us) - min(us), max(vs) - min(vs)) / 2 + 50
    cu, cv_ = (max(us) + min(us)) / 2, (max(vs) + min(vs)) / 2
    x0, y0 = int(max(cu - half, 0)), int(max(cv_ - half, 0))
    x1, y1 = int(min(cu + half, w)), int(min(cv_ + half, h))
    scale = zoom_px / max(x1 - x0, y1 - y0)
    img = cv2.resize(bgr[y0:y1, x0:x1], None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    zh, zw = img.shape[:2]
    to_z = lambda u, v: ((u - x0) * scale, (v - y0) * scale)  # noqa: E731
    for c in cands:
        (u0, v0), (u1, v1) = (to_z(*e) for e in c.ends_px)
        d = np.array([u1 - u0, v1 - v0])
        d = d / max(np.linalg.norm(d), 1e-9)
        n = np.array([-d[1], d[0]])
        a, b = np.array([u0, v0]) - d * 10, np.array([u1, v1]) + d * 10
        cv2.line(img, tuple(map(int, a)), tuple(map(int, b)), (0, 255, 255), 2, cv2.LINE_AA)
        for q in (a, b):
            cv2.line(img, tuple(map(int, q - n * 7)), tuple(map(int, q + n * 7)), (0, 255, 255), 2, cv2.LINE_AA)
    turn = upright_turn(up_cam) if up_cam is not None else None
    out = img if turn is None else cv2.rotate(img, turn)
    for c in cands:  # labels after the turn so the digits read upright; placed off the jaw span
        (u0, v0), (u1, v1) = (to_z(*e) for e in c.ends_px)
        x, y = _turn_point(u1, v1, zw, zh, turn)
        label = str(c.id)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        org = (int(x) + 10, int(y) + th // 2)
        cv2.rectangle(out, (org[0] - 3, org[1] - th - 3), (org[0] + tw + 3, org[1] + 4), (0, 0, 0), -1)
        cv2.putText(out, label, org, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
    return out


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


SELECT_PROMPT = """{n} views from the camera on a robot's gripper, of the same work area from different heights and sides. The numbered yellow bars span parts the two fingers could close on; the numbering runs across all the views, so each number appears in exactly one of them.

Task: pick up {target}, holding it by {part}.

Choose the numbered mark that lies on that part of that item. Judge every view: a view from lower down sees the sides of a part that a view from above cannot, and a mark on another object, on another part of the item, or on the structure it stands on is not a choice. If no mark lies on the described part of the described item, the choice is null.

Reply with JSON only: {{"choice": <mark number or null>, "reason": "<one short sentence>"}}"""
# Every view in one question. The single-view question this replaced was a concession to a gateway that timed out on
# anything larger (F58): with the model on DeepSeek's own endpoint a marked view is read in 1-6 s, so the views can be
# judged against each other instead of one at a time. Asking one at a time and stopping at the first view that answers
# made the first view decide the grasp -- and this model essentially always answers, so the lower views that actually
# see the part were never looked at (run g4q chose a mark on the depot stand 42 cm behind the ORU's handle, from the
# one view taken from above, and reported it as "the narrow neck under the T-shaped head").


def select_candidate(client, model: str, marked_views: list[np.ndarray], target: str, part: str,
                     ids: set[int]) -> dict:
    """VLM set-of-mark choice over every marked view at once. Returns the parsed answer plus the raw reply."""
    prompt = SELECT_PROMPT.format(n=len(marked_views), target=target, part=part)
    answer, reply = ask_json(client, model, prompt, marked_views)
    choice = answer.get("choice")
    choice = choice if choice in ids else None
    # a choice is the answer to "is the item there": the model is not asked for a separate flag it would have to
    # reason about and write out (F57 -- what it is asked to produce is what it spends its time on)
    return dict(item_visible=choice is not None, what_is_visible=answer.get("reason", ""), choice=choice,
                runner_up=None, reason=answer.get("reason", ""), raw=reply, model=model)


def depth_at(depth: np.ndarray, u: float, v: float, half_px: int = 3) -> float | None:
    """The depth of the surface at a pixel: the near quartile of a small patch, so a thin part in front of a far
    background reads as the part, not as the background. None where nothing was measured."""
    patch = depth[max(int(v) - half_px, 0):int(v) + half_px + 1, max(int(u) - half_px, 0):int(u) + half_px + 1]
    patch = patch[np.isfinite(patch) & (patch > 0.05)]
    return float(np.percentile(patch, 25)) if patch.size else None


class PartTracker:
    """Follows the chosen grasp in the wrist image while the tool moves towards it.

    Template matching around where the camera's own motion says the part must now be, with the template rescaled by the
    change in depth. The reference template remains fixed. A match is visibility evidence;
    its score alone does not establish a new spatial contact point."""

    def __init__(self, frame_bgr: np.ndarray, depth: np.ndarray, K: np.ndarray, cand: Candidate, half_px: int = 28,
                 R_base_cam: np.ndarray | None = None) -> None:
        self.K, self.half, self.R_ref = K, half_px, R_base_cam
        u, v = int(round(cand.u)), int(round(cand.v))
        h, w = depth.shape
        self.u0, self.v0 = min(max(u, half_px), w - half_px - 1), min(max(v, half_px), h - half_px - 1)
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        self.template = gray[self.v0 - half_px:self.v0 + half_px + 1, self.u0 - half_px:self.u0 + half_px + 1]
        self.depth = float(cand.depth_m)
        self.reference_depth = self.depth
        self.width_m, self.axis_cam = cand.width_m, np.array(cand.axis_cam, float)
        self.part_axis_cam = np.array(cand.part_axis_cam, float)
        self.score, self.roll_deg, self.reject, self.covered = 1.0, 0.0, "", False
        self.last_px, self.last_score = (float(self.u0), float(self.v0)), 1.0

    def measure(self, frame_bgr: np.ndarray, depth: np.ndarray, predicted_cam: np.ndarray, search_px: int = 60,
                min_score: float = 0.6, max_depth_jump_m: float = 0.03,
                R_base_cam: np.ndarray | None = None, self_depth_m: float = 0.0,
                predict_depth_min_score: float = 0.9):
        """Where the part is in this frame: (pixel, depth, score) or None. `predicted_cam` is its 3-D point in this
        camera as the arm's own motion predicts it, `R_base_cam` its orientation in the same fixed
        reference frame used at construction (odom for pick).

        The template is corrected for the camera's own motion since it was taken: bigger by the change in depth, and
        turned by however much the camera has rolled about its viewing direction -- a grasp is approached with the tool
        rolled to the side the part is free on, which can be half a turn from the view it was chosen in, and no image of
        the part survives that untouched. The committed template and its reference distance stay fixed:
        repeated high-score matches on adjacent uniform surfaces must not replace the selected feature.

        When the depth where the part should be reads the gripper's own body instead (nearer than `self_depth_m`, the
        robot's own reach in front of this camera), the part is not lost but covered by the hand that is reaching for
        it: `covered` says so, and the caller closes the last stretch on its last measurement. A blocker further away
        than the tool is something else, and counts as losing sight of the part."""
        self.covered = self.depth_predicted = False
        if predicted_cam[2] <= 0.05:
            return None
        pu = self.K[0, 0] * predicted_cam[0] / predicted_cam[2] + self.K[0, 2]
        pv = self.K[1, 1] * predicted_cam[1] / predicted_cam[2] + self.K[1, 2]
        # Association stays inside the selected contact's projected extent. A
        # fixed pixel search window can include an adjacent mounting block as
        # the camera approaches; a high correlation there is not this contact.
        search_px = min(search_px, self.width_m * self.K[0, 0] / (2 * float(predicted_cam[2])))
        here = depth_at(depth, pu, pv)
        if self_depth_m and here is not None and here < min(self_depth_m, float(predicted_cam[2]) - max_depth_jump_m):
            self.covered, self.reject = True, f"the gripper is in front of the part ({here:.2f} m)"
            self.last_px = (float(pu), float(pv))
            return None
        scale = self.reference_depth / max(float(predicted_cam[2]), 1e-6)
        roll_deg = 0.0
        if R_base_cam is not None and self.R_ref is not None:
            M = np.asarray(R_base_cam, float).T @ np.asarray(self.R_ref, float)  # the template's camera seen from this one
            roll_deg = math.degrees(math.atan2(M[1, 0], M[0, 0]))
        if abs(roll_deg) > 3.0:
            side = self.template.shape[0]
            c = (side - 1) / 2.0
            W = cv2.getRotationMatrix2D((c, c), -roll_deg, scale)
            k = int(side * 0.35)
            tpl = cv2.warpAffine(self.template, W, (side, side), flags=cv2.INTER_LINEAR)[
                int(c) - k:int(c) + k + 1, int(c) - k:int(c) + k + 1]
        else:
            tpl = self.template if abs(scale - 1.0) < 0.05 else cv2.resize(self.template, None, fx=scale, fy=scale,
                                                                           interpolation=cv2.INTER_LINEAR)
        th, tw = tpl.shape
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        x0, y0 = int(max(pu - search_px - tw // 2, 0)), int(max(pv - search_px - th // 2, 0))
        x1, y1 = int(min(pu + search_px + tw // 2, w)), int(min(pv + search_px + th // 2, h))
        window = gray[y0:y1, x0:x1]
        if window.shape[0] <= th or window.shape[1] <= tw:
            return None
        res = cv2.matchTemplate(window, tpl, cv2.TM_CCOEFF_NORMED)
        self.roll_deg, self.reject, hit = roll_deg, "", None
        # the best match *at the part's distance*: the narrowing view shows the fingers and the structures behind the
        # part, which can match the template better than the part itself does, so the depth decides between the peaks
        for idx in np.argsort(res.ravel())[::-1][:40]:
            score = float(res.ravel()[idx])
            if score < min_score:
                self.reject = self.reject or f"nothing matches above {min_score:.2f}"
                break
            yy, xx = divmod(int(idx), res.shape[1])
            u, v = x0 + xx + tw / 2.0, y0 + yy + th / 2.0
            self.last_px, self.last_score = (u, v), score
            d = depth_at(depth, u, v)
            if d is None:
                # A grasp that holds is a thin one: the bail's neck is 12 mm across, some 20 px at the distance the
                # close-in starts, and what stands behind it is sky, which returns no range at all. So the depth
                # image has no value at a pixel the template is certain of -- g6b matched at 1.00 and was thrown
                # away eight frames running for want of a number. The pixel is the strong measurement here; the
                # range is the weak one, and the arm's own motion already implies it (`predicted_cam`). Taking it
                # does not feed the loop: that prediction is the tracked point carried through the camera's
                # measured motion, so the distance still comes from the arm, never from the match (F64 §5).
                if score < predict_depth_min_score:
                    self.reject = "no depth where it matches"
                    continue
                # ...and the range is not invented either. A first attempt at this returned the predicted distance
                # as if it had been measured, and the caller rebuilt the part's point from it: the prediction comes
                # from that same point, so the point walked away down the line of sight -- 10.0 cm to 36.4 cm in
                # nineteen seconds (g6c). The part is a static rigid body. Its place is already known; this frame
                # only has to say it is still there and still matched, so the range comes back as None and the
                # caller keeps the point it has.
                self.depth_predicted, hit = True, (u, v, None, score)
                break
            off = d - float(predicted_cam[2])
            if abs(off) > max_depth_jump_m:
                self.reject = f"every match is off the part's distance (best {off * 100:+.0f} cm)"
                continue
            self.depth_predicted, hit = False, (u, v, d, score)
            break
        if hit is None:
            return None
        u, v, d, score = hit
        self.reject, self.score = "", score
        if d is not None:  # an unranged frame keeps the last distance for the record; it returns None to the caller
            self.depth = d
        return np.array([u, v]), d, float(score)

    def point_cam(self, pixel: np.ndarray, d: float) -> np.ndarray:
        return np.array([(pixel[0] - self.K[0, 2]) / self.K[0, 0] * d, (pixel[1] - self.K[1, 2]) / self.K[1, 1] * d, d])


def associate(prev: Candidate, prev_p_now: np.ndarray, cands: list[Candidate], max_dist_m: float = 0.02) -> Candidate | None:  # noqa: D401
    """The candidate in the current view that is the locked one: its 3-D point predicted into the current camera frame
    (`prev_p_now`, from the camera's own motion) and matched by position, width and closing axis."""
    best, best_d = None, max_dist_m
    for c in cands:
        d = float(np.linalg.norm(np.array(c.p_cam) - prev_p_now))
        same_axis = abs(float(np.dot(c.axis_cam, prev.axis_cam))) > 0.8
        if d < best_d and abs(c.width_m - prev.width_m) < 0.008 and same_axis:
            best, best_d = c, d
    return best
