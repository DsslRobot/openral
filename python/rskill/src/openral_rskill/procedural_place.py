"""PlaceRskill -- set the held item down on a support (one physical operation).

``kind: procedural``, LunarBot (research repo plan v4 §4). The caller names the support -- a surveyed place of the fixed
infrastructure (e.g. ``pdu_1_shelf/shelf``), which the rover's localization puts in the rover frame; the held item
itself is never localized: it hangs from the jaws and is lowered until the support takes its weight.

Stages:

1. over    -- carry the tool at its current height to above the support point, tool orientation kept.
2. lower   -- straight down, the shared servo's reference advancing at a slow speed, until the tool stalls while still
              driven down (the support takes the load). The tool is driven to the low end of where the support should
              take the item, and reaching it without a stall is "no contact".
3. release -- open the jaws.
4. retreat -- withdraw under the interface's release constraint, then up, clear of the item.
5. settle  -- two wrist images a moment apart after the retreat: the item stays where it was set (no image motion on
              the near foreground), and it is no longer in the jaws (jaw open).

Failures return the stage and evidence; nothing here chooses another support.
"""

from __future__ import annotations

import cv2
import numpy as np

from openral_rskill._eye_in_hand import JAW_OPEN_MIN_RAD, EyeInHandSkill, StageFailure
from openral_rskill.grasp_perception import GripperGeometry
from openral_rskill.interface_fit import fit_wall
from openral_rskill.procedural_pick import tool_in_view

__all__ = ["PlaceRskill"]


class PlaceRskill(EyeInHandSkill):
    def procedure(self) -> None:
        try:
            self.put_it_down()
        finally:
            self.working_on(None)  # the world model measures this space as it finds it again

    def wall_correction(self, s: np.ndarray, n_out: np.ndarray, f0) -> float:
        """How far along the support's outward normal the wall behind it is from where the survey and the rover's
        localization put it, from the depth image: the wall is the biggest flat thing in view that faces this way. The
        declared depth of the support says how far behind its centre the wall is. Zero, with the reason recorded, when
        the wall is not measured, or is measured further off than a localization error could account for."""
        depth_m = float((self.goal.get("interface") or {}).get("depth_m", 0.0))
        rec = {"expected_wall_m": None}
        self._evidence["wall"] = rec
        if depth_m <= 0.0:
            rec["not_measured"] = "the catalogue declares no depth for this support"
            return 0.0
        R_cb = f0.T_base_cam[:3, :3].T
        got = fit_wall(f0.depth, f0.K, R_cb @ n_out)
        if got is None:
            rec["not_measured"] = "no flat surface facing the way the survey says the wall faces"
            return 0.0
        n_c, p_c, frac = got
        p_b = f0.to_base(p_c)
        measured = float(n_out @ p_b)
        expected = float(n_out @ (s - n_out * depth_m / 2))
        corr = measured - expected
        rec.update(expected_wall_m=round(expected, 4), measured_wall_m=round(measured, 4), correction_m=round(corr, 4),
                   explains=round(frac, 3), tilt_deg=round(float(np.degrees(np.arccos(min(1.0, float(n_out @ (f0.T_base_cam[:3, :3] @ n_c)))))), 2))
        if abs(corr) > depth_m / 3:
            rec["not_applied"] = "further than a localization error could account for"
            return 0.0
        return corr

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
        n_out = T_bm[:3, :3] @ np.array(g.get("support_normal_map") or [0.0, 0.0, 0.0], float)
        n_out[2] = 0.0
        if np.linalg.norm(n_out) > 1e-6:
            n_out /= np.linalg.norm(n_out)
            s = s + n_out * self.wall_correction(s, n_out, f0)
        over = np.array([s[0], s[1], height])
        # The surveyed point is the middle of the surface, and an item whose envelope is nearly as deep as the
        # surface cannot sit there: a shelf cantilevered off a wall has the wall right behind it, and in ga5 the
        # ORU's envelope (0.485 m across a 0.6 m shelf) met `site/pdu_1` 1.5 mm inside its front face and the
        # approach stalled 5.5 cm out with no reason a caller could act on. Which way is *off* the surface is the
        # support's own outward normal, surveyed with it. Step along it until the robot's own collision check
        # passes, no further than half the item's depth -- past that the item is more off the surface than on it.
        off_by = 0.0
        if g.get("held_item"):
            n = T_bm[:3, :3] @ np.array(g.get("support_normal_map") or [0.0, 0.0, 0.0], float)
            n[2] = 0.0
            if np.linalg.norm(n) > 1e-6:
                n /= np.linalg.norm(n)
                step = float(GripperGeometry().min_part_width_m)  # the smallest feature this perception resolves
                limit = float(np.max(g["held_item"]["size_m"][:2])) / 2
                tried = []
                for k in range(int(limit / step) + 1):
                    cand = over + n * (k * step)
                    q = self.ik(cand, R0, list(self.arm_q()))
                    ok = q is not None and self.state_valid(np.asarray(q))
                    tried.append({"off_support_m": round(k * step, 4), "reachable": q is not None, "clear": bool(ok)})
                    if ok:
                        over, off_by = cand, k * step
                        break
                self._evidence["over_offset"] = {"outward_normal_rover": [round(float(v), 3) for v in n],
                                                 "limit_m": round(limit, 3), "chosen_m": round(off_by, 4),
                                                 "tried": tried[:8]}
        self._evidence["over_plan"] = {"carry_height_m": round(float(p0[2]), 3), "over_height_m": round(float(height), 3),
                                       "off_support_m": round(off_by, 4)}

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
        held = g.get("held_item")
        # Where the support should take the item's weight: the surface plus how far the handle rides above the item's
        # base, give or take the neck's length, which is how far the item can sit in the jaws. The tool is driven to the
        # low end of that; a stall on the way is the support, and reaching the end without one is not.
        expected = s[2] + float(held["handle_above_base_m"]) if held else None
        goal_z = expected - float(held["neck_height_m"]) if held else s[2] + float(g["min_tool_above_support_m"])
        aim = np.array([over[0], over[1], goal_z])

        def lowering():
            self.working_on(self.tcp()[0])
            return aim, R0

        info = self.servo(lowering, "lower", tol_m=0.004, tol_rad=0.08, timeout_s=120.0, stall_s=float(g["lower_stall_s"]),
                          check_scene=False, sag_integral=False, posture_gain=0.0, gain_per_s=float(g["servo_gain_per_s"]),
                          max_joint_rate_rad_s=float(g["carry_servo"]["max_joint_rate_rad_s"]),
                          advance_m_s=float(g["lower_speed_m_s"]), advance_ramp_s=float(g["lower_ramp_s"]), stall_returns=True)
        self.hold_here()
        self.wait(0.3)
        p_stop = self.tcp()[0]
        self._evidence["contact"] = {"tcp": [round(float(v), 4) for v in p_stop], "support_rover": [round(float(v), 4) for v in s],
                                     "jaw_rad": round(self.jaw(), 4), "servo": info,
                                     **({"expected_tcp_z": round(float(expected), 4),
                                         "stopped_above_expected_m": round(float(p_stop[2] - expected), 4)} if held else {})}
        if not info.get("stalled"):
            raise StageFailure("lower", "the item was driven to where its bottom would be under the support's surface and met nothing: "
                                        "it is not over the support", local_retry=False)

        self.stage("release")
        jaw = self.set_jaw(True, "release")
        self._evidence["jaw_open_rad"] = round(jaw, 4)
        self.carry_item(None)  # the support has the item now

        self.stage("retreat")
        self.hold_here()
        self.wait(0.5)
        p, R = self.tcp()  # the orientation the tool has now the load is off it, not the one it carried: holding the carried one
        # turned the wrist against the handle the open fingers were still round (gb7 gc1: 15 deg off, joint 7 0.67 rad behind)
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
        self.servo(lambda: (back, R), "retreat", tol_m=0.015, tol_rad=0.06, timeout_s=40.0, check_scene=False,
                   gain_per_s=float(g["servo_gain_per_s"]), sag_integral=False)
        up = back + np.array([0.0, 0.0, float(g["retreat_up_m"])])
        self.servo(lambda: (up, R), "retreat", tol_m=0.015, tol_rad=0.06, timeout_s=40.0,
                   gain_per_s=float(g["servo_gain_per_s"]), sag_integral=False)

        self.stage("settle")
        self.wait_until_still()  # gb7r12: the frames were taken while the arm was still moving (0.1-0.5 rad/s), and the camera's own motion read as the item's
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
    return {"still": bool(med < limit), "median_flow_px": round(med, 2), "limit_px": round(float(limit), 2),
            "near_pixels": int(near.sum())}
