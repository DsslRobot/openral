"""PlaceRskill -- set the held item down on a support (one physical operation).

``kind: procedural``, LunarBot (research repo plan v4 §4). The caller names the support -- a surveyed place of the fixed
infrastructure (e.g. ``pdu_1_shelf/shelf``), which the rover's localization puts in the rover frame; the held item
itself is never localized: it hangs from the jaws and is lowered until the support takes its weight.

Stages:

1. over    -- carry the tool at its current height to above the support point, tool orientation kept.
2. lower   -- straight down at a slow speed until contact: the tool stops descending while still commanded down (the
              support takes the load); a floor just above the support ends the stage as "no contact".
3. release -- open the jaws.
4. retreat -- straight back out along the tool axis, then up, clear of the item.
5. settle  -- two wrist images a moment apart after the retreat: the item stays where it was set (no image motion on
              the near foreground), and it is no longer in the jaws (jaw open).

Failures return the stage and evidence; nothing here chooses another support.
"""

from __future__ import annotations

import cv2
import numpy as np

from openral_rskill._eye_in_hand import JAW_OPEN_MIN_RAD, EyeInHandSkill, StageFailure, mat_to_rotvec
from openral_rskill.procedural_pick import tool_in_view

__all__ = ["PlaceRskill"]


class PlaceRskill(EyeInHandSkill):
    def procedure(self) -> None:
        g = self.goal
        self._evidence.update(support=g["support"])
        self.stage("over")
        # the gripper's own place in its camera, from the mount: what is nearer than this is the tool, not the scene
        f0 = self.frame(after=self._clock() - 0.05)
        self._tool_view = tool_in_view(f0.K, np.linalg.inv(self.T("tcp_frame", f0.frame_id)), *f0.depth.shape)
        self._evidence["tool_in_view"] = {k: v for k, v in self._tool_view.items() if k != "mask"}
        T_bm = self.T("chassis_base_link", "map")
        s_map = np.array(g["support_xyz_map"], float)
        s = T_bm[:3, :3] @ s_map + T_bm[:3, 3]
        p0, R0 = self.tcp()
        over = np.array([s[0], s[1], p0[2]])
        self.servo(lambda: (over, R0), "over", tol_m=0.01, tol_rad=0.05, max_joint_rate_rad_s=0.2, timeout_s=60.0)
        self._evidence["over"] = {"support_rover": [round(float(v), 3) for v in s], "tcp": [round(float(v), 3) for v in self.tcp()[0]]}

        self.stage("lower")
        floor = s[2] + float(g["min_tool_above_support_m"])
        v_down = float(g["lower_speed_m_s"])
        t_start = t_prev = self._clock()
        z_prev, slow_since, contact = p0[2], None, False
        q_cmd = np.array(self.arm_q())
        while not contact:
            p, _ = self.tcp()
            if p[2] <= floor:
                raise StageFailure("lower", f"no contact before the tool was {g['min_tool_above_support_m']} m above the support point")
            now = self._clock()
            dt = max(now - t_prev, 1e-3)
            # a waypoint half a second below the tool, over the support point, orientation kept
            _, R = self.tcp()
            dx = np.array([over[0] - p[0], over[1] - p[1], -v_down * 0.5])
            q_cmd = self.resolved_rate_step(q_cmd, dx, mat_to_rotvec(R0 @ R.T) * 0.5, 0.3 * dt)
            with self._cmd_lock:
                self._joints = tuple(float(v) for v in q_cmd)
            self.wait(0.1)
            now2 = self._clock()
            z = self.tcp()[0][2]
            rate = (z_prev - z) / max(now2 - t_prev, 1e-3)
            t_prev, z_prev = now2, z
            if now2 - t_start > 1.5 and rate < 0.3 * v_down:  # commanded down, not descending: the support holds the item
                slow_since = slow_since or now2
                contact = now2 - slow_since > float(g["contact_confirm_s"])
            else:
                slow_since = None
        self.hold_here()
        self.wait(0.3)
        self._evidence["contact"] = {"tcp": [round(float(v), 4) for v in self.tcp()[0]], "support_rover": [round(float(v), 4) for v in s]}

        self.stage("release")
        jaw = self.set_jaw(True, "release")
        self._evidence["jaw_open_rad"] = round(jaw, 4)

        self.stage("retreat")
        p, R = self.tcp()
        back = p - R[:, 2] * float(g["retreat_m"])
        self.servo(lambda: (back, R), "retreat", tol_m=0.015, tol_rad=0.06, timeout_s=40.0)
        up = back + np.array([0.0, 0.0, float(g["retreat_up_m"])])
        self.servo(lambda: (up, R), "retreat", tol_m=0.015, tol_rad=0.06, timeout_s=40.0)

        self.stage("settle")
        a = self.frame(after=self._clock() - 0.05)
        self.wait(float(g["settle_s"]))
        b = self.frame(after=self._clock() - 0.05)
        flow = scene_still(a, b, float(self._tool_view["self_depth_m"]))
        jaw = self.jaw()
        np.save(self.evidence_dir / "settle_a_depth.npy", a.depth)
        self._evidence["settle"] = {**flow, "jaw_rad": round(jaw, 4), "before": self.save("settle_a.png", a.bgr), "after": self.save("settle_b.png", b.bgr),
                                    "before_depth": str(self.evidence_dir / "settle_a_depth.npy")}
        if jaw < JAW_OPEN_MIN_RAD:
            raise StageFailure("settle", f"the jaws are not open (jaw {jaw:.3f} rad)")
        if not flow["still"]:
            raise StageFailure("settle", f"the item is still moving after release (image motion {flow['median_flow_px']} px)")
        self._evidence["outcome"] = "placed"


def scene_still(a, b, self_depth_m: float) -> dict:
    g0, g1 = (cv2.cvtColor(f.bgr, cv2.COLOR_BGR2GRAY) for f in (a, b))
    flow = cv2.calcOpticalFlowFarneback(g0, g1, None, 0.5, 4, 31, 5, 7, 1.5, 0)
    z = a.depth
    near = np.isfinite(z) & (z > self_depth_m) & (z < 1.0)  # beyond the gripper itself: the scene it just let go of
    if near.sum() < 200:
        return {"still": True, "median_flow_px": None, "near_pixels": int(near.sum())}
    med = float(np.median(np.linalg.norm(flow[near], axis=1)))
    return {"still": med < 1.5, "median_flow_px": round(med, 2), "near_pixels": int(near.sum())}

