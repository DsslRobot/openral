"""PlaceRskill -- set the held item down on a support (one physical operation).

``kind: procedural``, LunarBot (research repo plan v4 §4). The caller names the support -- a surveyed place of the fixed
infrastructure (e.g. ``pdu_1_shelf/shelf``), which the rover's localization puts in the rover frame; the held item
itself is never localized: it hangs from the jaws and is lowered until the support takes its weight.

Stages:

1. over    -- carry the tool at its current height to above the support point, tool orientation kept.
2. lower   -- straight down at a slow speed until contact: the tool stops descending while still commanded down (the
              support takes the load); a floor just above the support ends the stage as "no contact".
3. release -- open the jaws.
4. retreat -- withdraw under the interface's release constraint, then up, clear of the item.
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
        try:
            self.put_it_down()
        finally:
            self.working_on(None)  # the world model measures this space as it finds it again

    def put_it_down(self) -> None:
        g = self.goal
        self._evidence.update(support=g["support"])
        self.stage("over")
        # the gripper's own place in its camera, from the mount: what is nearer than this is the tool, not the scene
        f0 = self.frame(after=self._clock() - 0.05)
        self._tool_view = tool_in_view(f0.K, np.linalg.inv(self.T("tcp_frame", f0.frame_id)), *f0.depth.shape)
        self._gripper_mask_path = str(self.evidence_dir / "gripper_mask.npy")
        np.save(self._gripper_mask_path, self._tool_view["mask"])
        self._evidence["tool_in_view"] = {k: v for k, v in self._tool_view.items() if k != "mask"} | {"mask": self._gripper_mask_path}
        T_bm = self.T("chassis_base_link", "map")
        s_map = np.array(g["support_xyz_map"], float)
        s = T_bm[:3, :3] @ s_map + T_bm[:3, 3]
        p0, R0 = self.tcp()
        height = p0[2]
        if g.get("held_item"):
            # Travel over the support with the hanging item's envelope above it: moving across at the carry height
            # swept the item into the shelf edge (g8s stall, g8t drop; F78). The rover's own body is checked too.
            self.carry_item(g["held_item"])
            height = max(height, s[2] + self.held_bottom_below_tcp())
        over = np.array([s[0], s[1], height])
        self._evidence["over_plan"] = {"carry_height_m": round(float(p0[2]), 3), "over_height_m": round(float(height), 3)}

        for target in ([np.array([p0[0], p0[1], height])] if height > p0[2] else []) + [over]:
            def carrying(target=target):
                # the item in the jaws, and the surface it is going onto, are what this skill is working on -- the
                # planner must not treat either as an obstacle to the tool that is holding one and reaching for the
                # other (F58)
                self.working_on(self.tcp()[0])
                return target, R0

            self.servo(carrying, "over", **g["carry_servo"])
        self._evidence["over"] = {"support_rover": [round(float(v), 3) for v in s], "tcp": [round(float(v), 3) for v in self.tcp()[0]]}

        self.stage("lower")
        floor = s[2] + float(g["min_tool_above_support_m"])
        v_down = float(g["lower_speed_m_s"])
        t_start = t_cmd = self._clock()
        z_prev, slow_since, contact = p0[2], None, False
        q_cmd = np.array(self.arm_q())
        while not contact:
            p, R = self.tcp()
            self.working_on(p)
            if p[2] <= floor:
                raise StageFailure("lower", f"no contact before the tool was {g['min_tool_above_support_m']} m above the "
                                            "support point", local_retry=False)
            now = self._clock()
            # how far the joints may move this cycle is the cycle that just elapsed -- measured from when the last
            # command went out, not from the last reading, or the step is a millisecond's worth and the arm stands still
            dt = max(now - t_cmd, 1e-3)
            dx = np.array([over[0] - p[0], over[1] - p[1], -v_down * 0.5])  # a waypoint half a second below, over the support
            q_cmd = self.resolved_rate_step(q_cmd, dx, mat_to_rotvec(R0 @ R.T) * 0.5, 0.3 * dt)
            with self._cmd_lock:
                self._check_stop()
                self._joints = tuple(float(v) for v in q_cmd)
            t_cmd = now
            self.wait(0.1)
            now2 = self._clock()
            z = self.tcp()[0][2]
            rate = (z_prev - z) / max(now2 - now, 1e-3)  # how fast it actually fell over that cycle
            z_prev = z
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
        self.carry_item(None)  # the support has the item now

        self.stage("retreat")
        p, R = self.tcp()
        withdraw = -R[:, 2].copy()
        if g["release_constraint"] == "below_overhang":
            # This operation lowers onto a horizontal support. An overhang above a side grasp must not be
            # lifted by the retreat, even if the loaded wrist tilted during transport (research repo F73).
            withdraw[2] = 0.0
            length = float(np.linalg.norm(withdraw))
            if length == 0.0:
                raise StageFailure("retreat", "below_overhang requires a side grasp; a vertical tool has no sideways withdrawal direction",
                                   local_retry=False)
            withdraw /= length
        back = p + withdraw * float(g["retreat_m"])
        self._evidence["retreat"] = {"release_constraint": g["release_constraint"], "from": p.tolist(),
                                    "withdraw_to": back.tolist()}
        # The open jaws start around the part they just released, which the live scene measures: the withdrawal stroke
        # out of it is not tested against that measurement (the lift out of a grasp is not either); the stroke up is.
        self.servo(lambda: (back, R), "retreat", tol_m=0.015, tol_rad=0.06, timeout_s=40.0, check_scene=False)
        up = back + np.array([0.0, 0.0, float(g["retreat_up_m"])])
        self.servo(lambda: (up, R), "retreat", tol_m=0.015, tol_rad=0.06, timeout_s=40.0)

        self.stage("settle")
        a = self.frame(after=self._clock() - 0.05)
        self.wait(float(g["settle_s"]))
        b = self.frame(after=self._clock() - 0.05)
        flow = scene_still(a, b, float(self._tool_view["self_depth_m"]), self._tool_view["mask"])
        jaw = self.jaw()
        np.save(self.evidence_dir / "settle_a_depth.npy", a.depth)
        self._evidence["settle"] = {**flow, "jaw_rad": round(jaw, 4), "fx": float(a.K[0, 0]),
                                    "before": self.save("settle_a.png", a.bgr), "after": self.save("settle_b.png", b.bgr),
                                    "before_depth": str(self.evidence_dir / "settle_a_depth.npy"),
                                    "gripper_mask": self._gripper_mask_path, "self_depth_m": self._tool_view["self_depth_m"]}
        if jaw < JAW_OPEN_MIN_RAD:
            raise StageFailure("settle", f"the jaws are not open (jaw {jaw:.3f} rad)")
        if not flow["still"]:
            raise StageFailure("settle", f"the item is still moving after release (image motion {flow['median_flow_px']} px)")
        self._evidence["outcome"] = "placed"


def scene_still(a, b, self_depth_m: float, tool_mask: np.ndarray, moved_mm: float = 3.0) -> dict:
    """Did what the gripper just let go of stay put? Dense optical flow between two wrist frames a moment apart, over
    what lies beyond the gripper's own body and is not the gripper itself. The threshold is a distance, not a pixel
    count: `moved_mm` at the region's own depth, through the camera's own focal length."""
    g0, g1 = (cv2.cvtColor(f.bgr, cv2.COLOR_BGR2GRAY) for f in (a, b))
    flow = cv2.calcOpticalFlowFarneback(g0, g1, None, 0.5, 4, 31, 5, 7, 1.5, 0)
    z = a.depth
    near = np.isfinite(z) & (z > self_depth_m) & (z < 1.0) & ~tool_mask
    if near.sum() < 200:
        # nothing measurable where the item was let go: that is not evidence that it stayed, and the caller is told so
        return {"still": False, "median_flow_px": None, "near_pixels": int(near.sum()), "why": "nothing in view to judge"}
    med = float(np.median(np.linalg.norm(flow[near], axis=1)))
    limit = a.K[0, 0] * (moved_mm / 1000.0) / float(np.median(z[near]))
    return {"still": med < limit, "median_flow_px": round(med, 2), "limit_px": round(float(limit), 2),
            "near_pixels": int(near.sum())}
