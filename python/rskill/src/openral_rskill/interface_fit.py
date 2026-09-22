"""Where a declared lifting interface is, from one wrist depth frame.

The equipment catalogue says what the interface *is* -- a neck between a head and a collar, with every dimension --
and the site survey says which way it faces. What is left to measure is where it is, and that is answered here by
fitting the declared shape to the depth image: three boxes and the pedestal under them, over position alone -- which
way it faces is declared too, by the support it stands in. Nothing is thresholded. The answer carries two numbers that say how good it is -- what
fraction of the interface's own projected area the depth agrees with, and the mean residual on it -- and a bad fit
is a low fraction, not an empty list (research repo F91 and `docs/grasp_target_selection_proposal.md`).

Plain numpy; a frame costs a few milliseconds on one core.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

__all__ = ["DiscFit", "Fit", "boxes", "fit_wall", "locate", "locate_disc", "track", "yaw_of"]

SIGMA_M = 0.003  # a depth pixel that agrees: within one pixel's worth of range at the working distance


@dataclass
class Fit:
    xyz_cam: np.ndarray   # the slot centre (the handle frame) in the camera
    yaw: float            # the declared turn about the support normal, in the frame `frame_of(up)` builds
    agreement: float      # fraction of the interface's projected pixels the depth agrees with
    residual_m: float     # mean |dz| over those
    points: int

    def as_dict(self) -> dict:
        return {"xyz_cam": [round(float(v), 4) for v in self.xyz_cam], "yaw_rad": round(self.yaw, 4),
                "agreement": round(self.agreement, 3), "residual_m": round(self.residual_m, 4), "points": self.points}


def boxes(d: dict) -> np.ndarray:
    """The interface as boxes in its own frame (z up, x across the jaws, origin at the slot centre): neck, head,
    collar, and the pedestal down into the item. The pedestal is not what the fingers take, but it is what the
    camera sees under the collar, and without it the fit slides along the bars (F91 experiment)."""
    nt, nw, nh = d["neck_thickness"], d["neck_width"], d["neck_height"]
    hw, hd, ct = d["head_width"], d["head_depth"], d["collar_thickness"]
    ped = d["standoff"] - nh / 2 - ct + 0.02  # the column stands on the lid, sunk 20 mm into it (blender_process)
    return np.array([[0, 0, 0.0, nt / 2, nw / 2, (nh + ct) / 2],
                     [0, 0, nh / 2 + ct / 2, hw / 2, hd / 2, ct / 2],
                     [0, 0, -nh / 2 - ct / 2, d["collar_width"] / 2, d["collar_depth"] / 2, ct / 2],
                     [0, 0, -nh / 2 - ct - ped / 2, nt * 2.2 / 2, nw * 1.4 / 2, ped / 2]], float)


def frame_of(up: np.ndarray, yaw: float) -> np.ndarray:
    """Rotation of the interface frame in the camera: its z is `up`, turned by `yaw` about it."""
    x0 = np.cross([0.0, 0.0, 1.0], up)
    x0 /= np.linalg.norm(x0)
    R0 = np.stack([x0, np.cross(up, x0), up], axis=1)
    c, s = math.cos(yaw), math.sin(yaw)
    K = np.array([[0, -up[2], up[1]], [up[2], 0, -up[0]], [-up[1], up[0], 0]])
    return (np.eye(3) + s * K + (1 - c) * (K @ K)) @ R0


def yaw_of(up: np.ndarray, axis_cam: np.ndarray) -> float:
    """The yaw at which `frame_of` puts the interface's x axis along `axis_cam` (both in the camera)."""
    R0 = frame_of(up, 0.0)
    a = axis_cam - up * np.dot(axis_cam, up)
    return math.atan2(float(np.dot(a, R0[:, 1])), float(np.dot(a, R0[:, 0])))


def _predict(rays: np.ndarray, R: np.ndarray, t: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Optical depth at which each ray first enters the interface; inf where it misses (slab test per box)."""
    o = R.T @ -t
    d = rays @ R
    inv = 1.0 / np.where(np.abs(d) < 1e-7, 1e-7, d)
    best = np.full(len(rays), np.inf)
    for b in B:
        lo, hi = (b[:3] - b[3:] - o) * inv, (b[:3] + b[3:] - o) * inv
        far, near = np.maximum(lo, hi).min(-1), np.minimum(lo, hi).max(-1)
        ok = (far >= np.maximum(near, 0)) & (far > 0)
        best = np.where(ok, np.minimum(best, np.maximum(near, 0)), best)
    return best


def _fit(rays, zm, up, B, p, iters=10, eps=3e-4):
    """Gauss-Newton on the depth residual over position, with a Huber weight so the step shrinks with the residual
    and settles where a fixed-step optimiser walks past. The orientation is not fitted: which way the interface
    faces is declared (an item in its slot faces the slot's normal), and left free it wandered 65 deg while sliding
    onto a larger surface that explained more pixels."""
    p = np.asarray(p, float).copy()
    R = frame_of(up, p[3])
    for _ in range(iters):
        z0 = _predict(rays, R, p[:3], B)
        hit = np.isfinite(z0)
        if hit.sum() < 20:
            break
        r = zm[hit] - z0[hit]
        J = np.empty((int(hit.sum()), 3))
        for k in range(3):
            q = p[:3].copy()
            q[k] += eps
            zk = _predict(rays[hit], R, q, B)
            J[:, k] = np.where(np.isfinite(zk), (zk - z0[hit]) / eps, 0.0)
        w = np.minimum(1.0, 0.004 / np.maximum(np.abs(r), 1e-6))
        p[:3] += np.linalg.solve(J.T @ (J * w[:, None]) + 1e-6 * np.eye(3), J.T @ (w * r))
    return p


def _crop(depth, K, centre, radius_m, n_pts):
    """The measured points within `radius_m` of a point, subsampled to `n_pts`, as rays and ranges."""
    h, w = depth.shape
    v, u = np.mgrid[0:h, 0:w]
    z = np.where(np.isfinite(depth) & (depth > 0), depth, 9.0)
    pc = np.stack([(u - K[0, 2]) / K[0, 0] * z, (v - K[1, 2]) / K[1, 1] * z, z], -1)
    idx = np.flatnonzero(((z < 1.2) & (np.linalg.norm(pc - centre, axis=-1) < radius_m)).ravel())
    if idx.size > n_pts:
        idx = idx[np.linspace(0, idx.size - 1, n_pts).astype(int)]
    rays = np.stack([(u.ravel()[idx] - K[0, 2]) / K[0, 0], (v.ravel()[idx] - K[1, 2]) / K[1, 1], np.ones(idx.size)], -1)
    return rays, z.ravel()[idx].astype(float)


def _judge(depth, K, up, B, p) -> Fit:
    """The answer and its evidence, judged over everything the shape projects to -- every pixel in the window
    round it, sky included. A pixel the interface should occupy but the camera saw nothing at is a disagreement;
    judged over measured points only, a distant structure agreed 0.79 with the shape hanging half over the void
    (gui_verified replay)."""
    R, t = frame_of(up, p[3]), p[:3]
    h, w = depth.shape
    half = int(0.15 * K[0, 0] / max(t[2], 0.1))
    cu, cv = int(round(K[0, 0] * t[0] / t[2] + K[0, 2])), int(round(K[1, 1] * t[1] / t[2] + K[1, 2]))
    v0, v1, u0, u1 = max(cv - half, 0), min(cv + half + 1, h), max(cu - half, 0), min(cu + half + 1, w)
    if v1 <= v0 or u1 <= u0 or t[2] <= 0.05:  # the fit ran out of the picture: nothing to judge it on
        return Fit(t.copy(), float(p[3]), 0.0, 9.9, 0)
    step = max(1, int(np.sqrt((v1 - v0) * (u1 - u0) / 6000)))  # a fraction does not need every pixel
    v, u = np.mgrid[v0:v1:step, u0:u1:step]
    rays = np.stack([(u.ravel() - K[0, 2]) / K[0, 0], (v.ravel() - K[1, 2]) / K[1, 1], np.ones(u.size)], -1)
    z = _predict(rays, R, t, B)
    hit = np.isfinite(z)
    zm = depth[v.ravel(), u.ravel()][hit]
    r = np.abs(np.where(np.isfinite(zm), zm, 9.0) - z[hit])
    good = r < SIGMA_M
    n = int(hit.sum())
    return Fit(t.copy(), float(p[3]), float(good.sum() / max(n, 1)), float(r[good].mean()) if good.any() else 9.9, n)


def _explained(depth, K, up, B, p) -> float:
    """How well a hypothesis accounts for the picture: the points it explains, times the fraction of itself that
    is explained. The count alone lets a bigger surface win with half its area wrong; the fraction alone lets a
    model slid half off the bars win, agreeing with everything it still covers."""
    rays, zm = _crop(depth, K, p[:3], 0.12, 3000)
    z = _predict(rays, frame_of(up, p[3]), p[:3], B)
    hit = np.isfinite(z)
    good = int((hit & (np.abs(zm - np.where(hit, z, 0.0)) < SIGMA_M)).sum())
    return good * good / max(int(hit.sum()), 1)


def _seeds(depth, K, up, d: dict, self_depth_m: float, near_m: float, hint: np.ndarray | None):
    """Where to start: each connected lump of surface beyond the gripper's own reach in its camera, its topmost
    band along the support normal (the head bar's top face), and the slot centre a declared distance below that.
    Nearest to `hint` first when one is given."""
    m = (np.isfinite(depth) & (depth > self_depth_m) & (depth < near_m)).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    h, w = depth.shape
    v, u = np.mgrid[0:h, 0:w]
    drop = d["neck_height"] / 2 + d["collar_thickness"]
    out = []
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < 300:
            continue
        sel = lab == i
        z = depth[sel]
        p = np.stack([(u[sel] - K[0, 2]) / K[0, 0] * z, (v[sel] - K[1, 2]) / K[1, 1] * z, z], -1)
        hgt = p @ up
        top = p[hgt > hgt.max() - 0.010].mean(0)
        out.append((top - up * drop, int(stats[i, cv2.CC_STAT_AREA])))
    key = (lambda s: float(np.linalg.norm(s[0] - hint))) if hint is not None else (lambda s: -s[1])
    return [s for s, _ in sorted(out, key=key)]


def locate(depth: np.ndarray, K: np.ndarray, dims: dict, up_cam: np.ndarray, yaw: float,
           self_depth_m: float, hint: np.ndarray | None = None, tries: int = 3) -> Fit:
    """First frame. `yaw` is the declared orientation (which way the interface faces, from the site survey),
    `hint` where the caller believes the item is; the picture decides between the lumps nearest that."""
    B, up = boxes(dims), np.asarray(up_cam, float) / np.linalg.norm(up_cam)
    best, best_n = None, -1
    for seed in _seeds(depth, K, up, dims, self_depth_m, 1.2, hint)[:tries]:
        rays, zm = _crop(depth, K, seed, 0.12, 800)
        p = _fit(rays, zm, up, B, [*seed, yaw], iters=8)
        rays, zm = _crop(depth, K, p[:3], 0.08, 2000)
        p = _fit(rays, zm, up, B, p, iters=8)
        n = _explained(depth, K, up, B, p)
        if n > best_n:
            best, best_n = p, n
    if best is None:
        return Fit(np.full(3, np.nan), yaw, 0.0, 9.9, 0)
    return _judge(depth, K, up, B, best)


def track(depth: np.ndarray, K: np.ndarray, dims: dict, up_cam: np.ndarray, xyz_cam: np.ndarray, yaw: float) -> Fit:
    """Every frame after: from where the interface was, a few Gauss-Newton steps."""
    B, up = boxes(dims), np.asarray(up_cam, float) / np.linalg.norm(up_cam)
    rays, zm = _crop(depth, K, xyz_cam, 0.08, 800)
    if len(zm) < 20:
        return Fit(np.asarray(xyz_cam, float), yaw, 0.0, 9.9, 0)
    return _judge(depth, K, up, B, _fit(rays, zm, up, B, [*xyz_cam, yaw], iters=4))


@dataclass
class DiscFit:
    tip_cam: np.ndarray     # the button's front-most point, on its axis, in the camera
    normal_cam: np.ndarray  # the plate's normal, towards the camera
    agreement: float        # fraction of the button's own projected pixels the depth agrees with
    residual_m: float
    points: int

    def as_dict(self) -> dict:
        return {"tip_cam": [round(float(v), 4) for v in self.tip_cam], "normal_cam": [round(float(v), 4) for v in self.normal_cam],
                "agreement": round(self.agreement, 3), "residual_m": round(self.residual_m, 4), "points": self.points}


def _cap_profile(rho: np.ndarray, d: dict) -> np.ndarray:
    """Height above the plate of the declared button at distance `rho` from its axis: the dome, then the collar
    round its foot, then the plate."""
    h = np.zeros_like(rho)
    h[rho <= d["collar_radius_m"]] = d["collar_top_height_m"]
    dome = rho <= d["cap_radius_m"]
    h[dome] = d["cap_centre_height_m"] + np.sqrt(np.maximum(d["cap_radius_m"] ** 2 - rho[dome] ** 2, 0.0))
    return h


def _plate(P: np.ndarray, iters: int = 80) -> tuple[np.ndarray, np.ndarray]:
    """The plane most of `P` lies in (points within two depth pixels' worth of it), normal towards the camera."""
    rng = np.random.default_rng(0)
    best, best_n = None, -1
    for _ in range(iters):
        s = P[rng.choice(len(P), 3, replace=False)]
        n = np.cross(s[1] - s[0], s[2] - s[0])
        if np.linalg.norm(n) < 1e-9:
            continue
        n /= np.linalg.norm(n)
        k = int((np.abs((P - s[0]) @ n) < 2 * SIGMA_M).sum())
        if k > best_n:
            best, best_n = (n, s[0]), k
    n, p0 = best
    inl = np.abs((P - p0) @ n) < 2 * SIGMA_M
    c = P[inl].mean(0)
    n = np.linalg.svd(P[inl] - c)[2][2]
    return (-n if n[2] > 0 else n), c


def locate_disc(depth: np.ndarray, K: np.ndarray, dims: dict, hint_px: tuple[float, float]) -> DiscFit | None:
    """Where a declared push button is, from one wrist depth frame, as `locate` does for the lifting interface.

    The catalogue says what the button is -- a dome on a collar on a plate, with its dimensions. The plate is what
    most of the picture around the hint lies in; the button is then the position on that plate at which the declared
    shape explains the depth better than the bare plate does. The hint is the vision model's pixel and only says
    which feature is meant: it may be centimetres off (mc1: 5.7 cm at the stand-off view) and the answer does not
    depend on it, provided the button is within reach of the window. Nothing is thresholded; the answer says how well
    the shape agrees over its own footprint, and a hint on something that is not this button agrees badly."""
    fx, u0, v0 = float(K[0, 0]), int(round(hint_px[0])), int(round(hint_px[1]))
    z0 = np.nanmedian(depth[max(v0 - 3, 0):v0 + 4, max(u0 - 3, 0):u0 + 4])
    if not np.isfinite(z0) or z0 < 0.05:
        return None
    Rc = float(dims["collar_radius_m"])
    half = int(3.2 * float(dims["cap_radius_m"]) * fx / z0)
    v, u = np.mgrid[max(v0 - half, 0):min(v0 + half + 1, depth.shape[0]), max(u0 - half, 0):min(u0 + half + 1, depth.shape[1])]
    step = max(1, int(np.sqrt(v.size / 9000)))
    v, u = v[::step, ::step].ravel(), u[::step, ::step].ravel()
    z = depth[v, u]
    ok = np.isfinite(z) & (z > 0.05)
    v, u, z = v[ok], u[ok], z[ok]
    P = np.stack([(u - K[0, 2]) * z / K[0, 0], (v - K[1, 2]) * z / K[1, 1], z], -1)
    c0 = np.array([(u0 - K[0, 2]) * z0 / K[0, 0], (v0 - K[1, 2]) * z0 / K[1, 1]])
    rho = np.linalg.norm(P[:, :2] - c0, axis=1)
    ring = (rho > 1.7 * Rc) & (rho < 3.0 * Rc)
    if ring.sum() < 200:
        return None
    n, p0 = _plate(P[ring])
    e1 = np.cross(n, [0.0, 1.0, 0.0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(n, e1)
    a, b, h = (P - p0) @ e1, (P - p0) @ e2, (P - p0) @ n
    a0, b0 = (np.array([c0[0], c0[1], z0]) - p0) @ e1, (np.array([c0[0], c0[1], z0]) - p0) @ e2
    w = 2 * SIGMA_M

    def gain(ca: np.ndarray, cb: np.ndarray) -> np.ndarray:
        """How much better the button at (ca, cb) explains the depth than the bare plate: the sum over the picture of
        (agreement with the shape) - (agreement with the plate)."""
        r = np.hypot(a[None, :] - ca[:, None], b[None, :] - cb[:, None])
        return (np.exp(-((h[None, :] - _cap_profile(r, dims)) / w) ** 2) - np.exp(-((h / w) ** 2))[None, :]).sum(1)

    reach = 1.5 * float(dims["cap_radius_m"])
    gs, step_m = np.meshgrid(np.arange(-reach, reach + 1e-9, 0.003), np.arange(-reach, reach + 1e-9, 0.003))
    cand = np.stack([a0 + gs.ravel(), b0 + step_m.ravel()], 1)
    best = cand[int(np.argmax(gain(cand[:, 0], cand[:, 1])))]
    fine = np.stack(np.meshgrid(best[0] + np.arange(-0.003, 0.0031, 0.0005), best[1] + np.arange(-0.003, 0.0031, 0.0005)), -1).reshape(-1, 2)
    ca, cb = fine[int(np.argmax(gain(fine[:, 0], fine[:, 1])))]
    r = np.hypot(a - ca, b - cb)
    inside = r <= Rc
    if inside.sum() < 30:
        return DiscFit(p0, n, 0.0, 9.9, int(inside.sum()))
    res = np.abs(h[inside] - _cap_profile(r[inside], dims))
    good = res < 2 * SIGMA_M
    tip = p0 + e1 * ca + e2 * cb + n * (dims["cap_centre_height_m"] + dims["cap_radius_m"])
    return DiscFit(tip, n, float(good.mean()), float(res[good].mean()) if good.any() else 9.9, int(inside.sum()))


def fit_wall(depth: np.ndarray, K: np.ndarray, normal_cam: np.ndarray, tilt_rad: float = 0.35) -> tuple[np.ndarray, np.ndarray, float] | None:
    """The plane most of the depth image lies in, among those facing the way the survey says the wall behind a support
    faces: (unit normal towards the camera, a point on it, the fraction of the image it explains). A wall is the
    biggest flat thing in view, and the survey's direction is only there to keep the answer from being a floor or a
    ceiling; None when nothing flat faces that way."""
    v, u = np.mgrid[0:depth.shape[0]:max(1, int(np.sqrt(depth.size / 6000))), 0:depth.shape[1]:max(1, int(np.sqrt(depth.size / 6000)))]
    z = depth[v, u].ravel()
    ok = np.isfinite(z) & (z > 0.05) & (z < 4.0)
    z, v, u = z[ok], v.ravel()[ok], u.ravel()[ok]
    if len(z) < 300:
        return None
    P = np.stack([(u - K[0, 2]) * z / K[0, 0], (v - K[1, 2]) * z / K[1, 1], z], -1)
    want = np.asarray(normal_cam, float) / np.linalg.norm(normal_cam)
    rng = np.random.default_rng(1)
    best, best_n = None, 0
    for _ in range(300):
        s3 = P[rng.choice(len(P), 3, replace=False)]
        n = np.cross(s3[1] - s3[0], s3[2] - s3[0])
        if np.linalg.norm(n) < 1e-9:
            continue
        n /= np.linalg.norm(n)
        n = n if n @ want > 0 else -n
        if math.acos(min(1.0, float(n @ want))) > tilt_rad:
            continue
        k = int((np.abs((P - s3[0]) @ n) < 2 * SIGMA_M).sum())
        if k > best_n:
            best, best_n = (n, s3[0]), k
    if best is None or best_n < 0.05 * len(P):
        return None
    n, p0 = best
    inl = np.abs((P - p0) @ n) < 2 * SIGMA_M
    c = P[inl].mean(0)
    n = np.linalg.svd(P[inl] - c)[2][2]
    return (n if n @ want > 0 else -n), c, float(inl.mean())
