"""PickRskill -- pick up an item described in words, holding it by its declared lifting interface (one physical
operation).

``kind: procedural``, LunarBot. The caller says *what* (``target``) and *by which part* (``part``) and names the
support the item stands on; the boundary adds the equipment catalogue's description of the item's lifting interface
and the support's surveyed outward normal. Everything about *how* to take hold is then declared, not measured:

- where to hold: the slot centre of the interface (the handle frame);
- how the jaws close: across the neck, along the interface's own axis, which faces the support's normal;
- from where: along that normal, horizontally, the way the rover docked;
- how deep: the middle of the pads (the gripper's own geometry).

What is measured is where the interface is. The wrist depth image is fitted with the declared shape
(``interface_fit``), which answers with a position and two numbers saying how well the picture agrees with the
catalogue -- never a list of candidates and never a threshold on a measured dimension (research repo F91,
`docs/grasp_target_selection_proposal.md`). The vision-language model is asked one thing it is good at: which
region of the view is the described item.

Stages (each recorded with its evidence):

1. find     -- one view of the work; the model names the item's region; the interface is fitted near it.
2. approach -- the tool stands off in front of the slot in the declared orientation, then servos in, re-fitting the
               interface every frame; once the fingers cover it, the last fit carries the tool in.
3. grasp    -- close the jaws; the jaw must stop between empty and open.
4. hold     -- lift; the jaw still holds and the wrist image shows the item moving with the gripper.

A skill retries the same target only; anything that would need the base to move ends the goal with the stage
reached and its evidence.
"""

from __future__ import annotations

import json
import math
import os
import time

import cv2
import numpy as np

from openral_rskill._eye_in_hand import (BASE_FRAME_ID, READY, JAW_EMPTY_MAX_RAD, EyeInHandSkill, Frame, StageFailure,
                                          TCP_FRAME_ID, mat_to_rotvec)
from openral_rskill.grasp_perception import GripperGeometry, find_regions, locate_region, render_regions, upright
from openral_rskill.interface_fit import SIGMA_M, Fit, locate, track, yaw_of

__all__ = ["PickRskill", "tool_rotation"]


def tool_rotation(pitch_deg: float, yaw_deg: float, roll_deg: float = 0.0) -> np.ndarray:
    """Tool pointing rearward out of the rover (-x of chassis_base_link), pitched down / yawed; jaw axis horizontal.
    `roll_deg` 180 turns the tool about its own axis, which puts the wrist camera above it, looking at what stands
    above the tool axis instead of below it."""
    th, ps = math.radians(pitch_deg), math.radians(yaw_deg)
    z = np.array([-math.cos(th) * math.cos(ps), -math.cos(th) * math.sin(ps), -math.sin(th)])
    x = np.cross([0.0, 0.0, 1.0], z)
    x /= np.linalg.norm(x)
    if abs(roll_deg - 180.0) < 1e-6:
        x = -x
    return np.stack([x, np.cross(z, x), z], axis=1)


class PickRskill(EyeInHandSkill):
    def procedure(self) -> None:
        g = self.goal
        from openai import OpenAI

        # No hidden retries: the model answers within the deadline or the stage returns with what it has (F58).
        import httpx

        http = httpx.Client(limits=httpx.Limits(max_connections=8, keepalive_expiry=30.0), timeout=120.0)
        self.vlm = OpenAI(api_key=os.environ["SPACE_LLM_API_KEY"], base_url=g["vlm_endpoint"], timeout=120,
                          max_retries=0, http_client=http)
        self.geom = GripperGeometry()
        self.dims = (g.get("held_item") or {}).get("interface")
        if not self.dims:
            raise StageFailure("prepare", "the catalogue declares no lifting interface for this item, so there is "
                                          "nothing to fit the picture to and no declared way to take hold of it",
                               local_retry=False)
        self._evidence.update(target=g["target"], part=g["part"], attempts=[])
        self.stage("prepare")
        self.set_jaw(True, "prepare")
        try:
            self.attempt()
        finally:
            self.working_on(None)

    def attempt(self) -> None:
        self.evidence_dir = self.evidence_dir / "attempt_01"
        self.evidence_dir.mkdir()
        fit, f = self.find()
        self.contact_scene(True)
        p_base = f.to_base(fit.xyz_cam)
        self.working_on(p_base)
        rec = {"attempt": 1, "fit": fit.as_dict(), "slot_base_m": [round(float(v), 4) for v in p_base],
               "evidence_dir": str(self.evidence_dir), "find": self._evidence["find"]}
        self._evidence["attempts"].append(rec)
        try:
            self.approach(p_base, fit, f)
            jaw = self.grasp()
            self.contact_permission("revoke")
            self.hold(jaw)
            self.carry_item(self.goal["held_item"])
            self.bring_in()
            rec["outcome"] = "held"
            self._evidence["outcome"] = "held"
        except StageFailure as exc:
            rec["outcome"] = f"failed at {exc.stage}: {exc.why}"
            raise
        finally:
            rec["approach_track"] = self._evidence.get("approach_track", [])
            rec["take_standoff"] = self._evidence.get("take_standoff")
            (self.evidence_dir / "attempt.json").write_text(json.dumps(rec, indent=2) + "\n")

    def vlm_call(self, stage: str, fn):
        """One call to the vision-language model; when the service does not answer, the evidence says the *service*
        failed, not the grasp -- different outcomes for a study of the method."""
        import httpx
        import openai

        t0 = time.time()
        try:
            out = fn()
            self._evidence.setdefault("vlm_calls", []).append({"stage": stage, "s": round(time.time() - t0, 1), "answered": True})
            return out
        except (openai.APIError, httpx.HTTPError) as exc:
            self._evidence.setdefault("vlm_calls", []).append({"stage": stage, "s": round(time.time() - t0, 1),
                                                               "answered": False, "error": type(exc).__name__})
            self._evidence["dependency_unavailable"] = {"service": "vision_language_model", "model": self.goal["vlm_model"],
                                                        "endpoint": self.goal["vlm_endpoint"], "error": type(exc).__name__,
                                                        "stage": stage}
            raise StageFailure(stage, f"the vision-language model ({self.goal['vlm_model']}) did not answer: {type(exc).__name__}") from exc

    # ---- what is declared ---------------------------------------------------------------------------------------------
    def work_point(self) -> np.ndarray:
        """Where the work is, in the rover's own frame: the surveyed support the caller named, at the height the
        boundary computed for it -- the same height it stood the rover for."""
        g = self.goal
        if "support_xyz_map" not in g:
            raise StageFailure("find", "this pick did not say what the item stands on, so there is nothing to look "
                                       "at: name the place frame (`support`) the item is on.", local_retry=False)
        T_bm = self.T(BASE_FRAME_ID, "map")
        s = T_bm[:3, :3] @ np.array(g["support_xyz_map"], float) + T_bm[:3, 3]
        return np.array([s[0], s[1], float(g["work_z_m"]) if "work_z_m" in g else s[2]])

    def approach_dir(self) -> np.ndarray:
        """The direction the tool goes in: along the support's surveyed outward normal, horizontally -- the way the
        rover docked. An item in its slot faces the slot, so this is also the way the interface faces."""
        n = self.T(BASE_FRAME_ID, "map")[:3, :3] @ np.array(self.goal.get("support_normal_map") or [0.0, 0.0, 0.0], float)
        n[2] = 0.0
        if np.linalg.norm(n) < 1e-6:
            return np.array([-1.0, 0.0, 0.0])  # no survey: straight back out of the rover
        return -n / np.linalg.norm(n)

    def grasp_rotations(self) -> list[np.ndarray]:
        """The tool orientation of the grasp: z along the approach, jaw axis horizontal across it. The roll about
        the approach -- which side the wrist camera looks from -- is the one open choice, and both are offered:
        the camera above the tool first, which is how the view was taken."""
        z = self.approach_dir()
        x = np.cross([0.0, 0.0, 1.0], z)
        x /= np.linalg.norm(x)
        return [np.stack([-x, np.cross(z, -x), z], axis=1), np.stack([x, np.cross(z, x), z], axis=1)]

    # ---- find ----------------------------------------------------------------------------------------------------------
    def find(self) -> tuple[Fit, Frame]:
        """One view of the work, from as level as the arm can manage where the rover stands; the model names the
        item's region in it; the declared interface is fitted to the depth nearest that region."""
        g = self.goal
        self.stage("find")
        at = self.work_point()
        behind = float(np.linalg.norm(self.T(TCP_FRAME_ID, self.frame().frame_id)[:3, 3]))
        tried, q, tool = [], None, None
        for back in [float(d) - behind for d in g["view_part_depth_m"]]:
            for pitch in (0.0, 10.0, 20.0, 30.0):
                R = tool_rotation(pitch, 0.0, 180.0)
                tool = at - R[:, 2] * back
                q = self.ik(tool, R, READY)
                tried.append({"back_m": round(back, 3), "pitch_deg": pitch, "reachable": q is not None})
                if q is not None:
                    break
            if q is not None:
                break
        record: dict = {"view_tried": tried}
        self._evidence["find"] = record
        if q is None:
            raise StageFailure("find", f"the arm cannot put its camera on {g.get('support') or 'this work'} at the "
                                       f"height it is worked at ({at[2]:.2f} m) from where the rover stands: none of "
                                       f"the {len(tried)} viewing poses is in reach. Trying again from here cannot "
                                       "help; the rover has to stand somewhere else.", local_retry=False)
        self.plan_to(q, "find")
        self.wait(0.8)
        f = self.frame(after=self._clock() - 0.05)
        T_tc = self.T(TCP_FRAME_ID, f.frame_id)  # the camera in the tool frame (robot geometry)
        self._tool_view = tool_in_view(f.K, np.linalg.inv(T_tc), *f.depth.shape)
        self._gripper_mask_path = str(self.evidence_dir / "gripper_mask.npy")
        np.save(self._gripper_mask_path, self._tool_view["mask"])
        self._self_depth = float(self._tool_view["self_depth_m"])
        record["view"] = {"tool": [round(float(v), 3) for v in tool], "pitch_deg": tried[-1]["pitch_deg"],
                          "image": self.save("view.jpg", upright(f.bgr, f.up_cam))}
        record["tool_in_view"] = {k: v for k, v in self._tool_view.items() if k != "mask"} | {"mask": self._gripper_mask_path}
        np.save(self.evidence_dir / "view_depth.npy", f.depth)

        # which lump of the picture is the item: the model's job, at the granularity it is good at
        regions = find_regions(f.depth, f.K, self.geom)
        hint = None
        if regions:
            marked = render_regions(f.bgr, regions, f.up_cam)
            record["marked"] = self.save("regions.jpg", marked)
            a = self.vlm_call("select", lambda: locate_region(self.vlm, g["vlm_model"], marked, g["target"], g["part"], regions))
            record["locate"] = {k: a[k] for k in ("item_visible", "what_is_visible", "choice", "reason")}
            if a["choice"] is None:
                raise StageFailure("select", f"none of the {len(regions)} surfaces in view is the described item: "
                                             f"\"{a['what_is_visible']}\"")
            hint = np.array(next(r.p_cam for r in regions if r.id == a["choice"]), float)

        # where the interface is: the declared shape, facing the declared way, fitted to the depth near that lump
        R_g = self.grasp_rotations()[0]
        yaw = yaw_of(f.up_cam, f.T_base_cam[:3, :3].T @ R_g[:, 0])
        fit = locate(f.depth, f.K, self.dims, f.up_cam, yaw, self._self_depth, hint)
        record["fit"] = fit.as_dict()
        if fit.points:
            record["fit"]["image"] = self.save("fit.jpg", self.mark(f, fit.xyz_cam, fit.agreement))
        if fit.points == 0:
            raise StageFailure("find", "nothing beyond the gripper in the view to fit the declared interface to")
        if fit.agreement < 0.5:
            # more of the interface disagrees with the picture than agrees: what was found is not it
            raise StageFailure("find", f"the declared interface was fitted where the model pointed, but only "
                                       f"{fit.agreement:.0%} of its surface agrees with the depth image (mean residual "
                                       f"{fit.residual_m * 1000:.1f} mm over {fit.points} pixels): this is not that interface")
        self._locked_frame, self._contact_views = f, [f]
        return fit, f

    def mark(self, f: Frame, xyz_cam: np.ndarray, agreement: float, text: str = "") -> np.ndarray:
        """The view with the fitted slot centre on it."""
        img = f.bgr.copy()
        if xyz_cam[2] > 0.05:
            u = int(f.K[0, 0] * xyz_cam[0] / xyz_cam[2] + f.K[0, 2])
            v = int(f.K[1, 1] * xyz_cam[1] / xyz_cam[2] + f.K[1, 2])
            cv2.drawMarker(img, (u, v), (0, 255, 255), cv2.MARKER_CROSS, 24, 2)
        cv2.putText(img, f"agree {agreement:.2f} {text}", (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        return upright(img, f.up_cam)

    # ---- approach --------------------------------------------------------------------------------------------------------
    def approach(self, p_base: np.ndarray, fit0: Fit, f: Frame) -> None:
        """Stand off in front of the slot in the declared orientation, then servo in on the interface as re-fitted
        every frame. The slot is committed in odom so the base's own motion changes its rover-relative coordinates
        and not its identity; each accepted fit moves that commitment."""
        g = self.goal
        self.stage("approach")
        T_ob = self.T("odom", "chassis_base_link")
        state = {"p_base": p_base, "odom": T_ob[:3, :3] @ p_base + T_ob[:3, 3], "hits": 0, "carried": 0,
                 "frames": [], "trace": [], "seen": f.stamp}
        self._evidence["approach_track"] = state["trace"]
        self.working_on(p_base)
        self._evidence["target_lock"] = {"status": "committed", "frame": "odom", "position_m": state["odom"].tolist(),
                                         "source": "declared interface fitted to the wrist depth image"}

        def refresh():
            T_bo = self.T("chassis_base_link", "odom")
            state["p_base"] = T_bo[:3, :3] @ state["odom"] + T_bo[:3, 3]

        # the contact the jaws may make -- the neck, across, along the slot, to the depth the fingers pass -- is
        # declared before the stand-off is planned: the planner's endpoint check is only permitted against a
        # registered region (gb5 failed here with "no selected measured contact region"). Both grasp rotations
        # share this box: same approach, same slot, the jaw axis only mirrored.
        # The box is the neck's own volume let out by the depth's resolution on the two axes its faces lie on, so
        # the measured faces fall inside it and not on its boundary (gb7 registered the neck's exact box, it held
        # no measured triangle, nothing was exempted, and the insertion was refused on the first touch); along the
        # slot it is drawn *in* by the same amount, so the head's and collar's faces stay protected.
        region = np.eye(4)
        region[:3, 3] = state["odom"]
        region[:3, :3] = T_ob[:3, :3] @ self.grasp_rotations()[0]  # x across the neck, y along the slot, z the approach
        slack = 2 * SIGMA_M
        dims = np.array([self.dims["neck_thickness"] + slack, self.dims["neck_height"] - slack, self.dims["neck_width"] + slack])
        self._contact_region = (region, dims)
        self.contact_permission("register", region, dims, str(self.evidence_dir / "view_depth.npy"))
        R = self.take_standoff(state, [float(v) for v in np.atleast_1d(g["standoff_m"])])

        def goal_now():
            refresh()
            left = float(np.linalg.norm(state["p_base"] - self.tcp()[0]))
            if not state.get("covered") and left >= float(g["blind_ok_m"]):
                nf = self.frame(after=state["seen"], timeout_s=2.0)
                state["seen"] = nf.stamp
                R_c, t_c = nf.T_base_cam[:3, :3], nf.T_base_cam[:3, 3]
                prev = R_c.T @ (state["p_base"] - t_c)
                m = track(nf.depth, nf.K, self.dims, nf.up_cam, prev, yaw_of(nf.up_cam, R_c.T @ R[:, 0]))
                # a fit that explains at least half of what the first one did is the interface; less is the fingers
                # in front of it, and from here the last fit carries the tool in (a straight close-in)
                if m.agreement >= fit0.agreement / 2:
                    state["p_base"] = nf.to_base(m.xyz_cam)
                    T_ob2 = self.T("odom", "chassis_base_link")
                    state["odom"] = T_ob2[:3, :3] @ state["p_base"] + T_ob2[:3, 3]
                    state["hits"] += 1
                else:
                    state["covered"] = True
                    self._evidence["covered_at_m"] = round(left, 4)
                if len(state["frames"]) < 40 and (len(state["frames"]) < 8 or nf.stamp - state["frames"][-1] > 1.0):
                    path = self.save(f"servo_{len(state['frames']):02d}.jpg",
                                     self.mark(nf, m.xyz_cam, m.agreement, f"left {left * 100:.1f}cm"))
                    state["frames"].append(nf.stamp)
                    state["trace"].append({"image": path, "agreement": round(m.agreement, 3),
                                           "residual_m": round(m.residual_m, 4), "left_m": round(left, 4),
                                           "accepted": m.agreement >= fit0.agreement / 2})
            else:
                state["carried"] += 1
            self.working_on(state["p_base"])
            return state["p_base"] - R[:, 2] * state["standoff"], R

        state["standoff"] = float(state["standoff_used"])
        try:
            info = self.servo(goal_now, "approach", tol_m=0.006, tol_rad=0.04,
                              max_speed_m_s=float(g["approach_speed_m_s"]),
                              max_joint_rate_rad_s=float(g["carry_servo"]["max_joint_rate_rad_s"]),
                              timeout_s=float(g["approach_timeout_s"]), stall_s=6.0)
        except StageFailure as exc:
            # the stand-off is a place to close in from, not a pose to hit (g5g stopped 0.4 cm short of it)
            if not exc.local_retry:
                raise
            off = float(np.linalg.norm(goal_now()[0] - self.tcp()[0]))
            if off > float(g["close_in_short_ok_m"]):
                raise
            info = {"stalled_short_m": round(off, 4), "why": exc.why}
        self._evidence["standoff"] = {**info, "fits_accepted": state["hits"], "carried_frames": state["carried"],
                                      "image": self.save("standoff.jpg", upright(self.frame(after=self._clock() - 0.05).bgr, None))}
        self.contact_permission("insertion")
        state["standoff"] = -self.geom.seat_depth_m  # drive the neck to the middle of the pads
        self.stage("approach", close_in=True)
        try:
            # the slot leaves the fingers (neck_height - finger_height)/2 = 5 mm each way: the close-in has to settle
            # finer than that, or a finger lands on the collar (gb7: 5.4 mm along the slot, the collar's top face)
            info = self.servo(goal_now, "approach", tol_m=0.003, tol_rad=0.04,
                              max_speed_m_s=float(g["close_in_speed_m_s"]),
                              max_joint_rate_rad_s=float(g["carry_servo"]["max_joint_rate_rad_s"]),
                              timeout_s=60.0, stall_s=6.0, sag_integral=False, posture_gain=0.0)
        except StageFailure as exc:
            # stopping short along the slot is still on the neck; across it, or in orientation, is not (g9g)
            if not exc.local_retry or not self.on_contact(*goal_now())["on_contact"]:
                raise
            info = {"stalled_on_contact": self._evidence["closure_precondition"], "why": exc.why}
        self._closure_goal = goal_now()
        self._evidence["close_in"] = {**info, "remaining_m": round(float(np.linalg.norm(state["p_base"] - self.tcp()[0])), 4),
                                      "fits_accepted": state["hits"], "carried_frames": state["carried"], "track": state["trace"]}
        self._grasp_point = state["p_base"]

    def take_standoff(self, state: dict, standoffs_m: list[float]) -> np.ndarray:
        """Stand the tool in front of the slot, in the grasp orientation, with the planner: one reconfiguration, so
        the servo only has to close the last stretch. Both stand-off and contact must be reachable; a grasp the arm
        has no configuration for from this base pose is a fact about where the rover parked, for the caller."""
        tool_points, tool_links = self.gripper_surface_points(with_links=True)
        rec = {"tried": []}
        self._evidence["take_standoff"] = rec
        for R in self.grasp_rotations():
            p_contact = state["p_base"] + R[:, 2] * self.geom.seat_depth_m
            for standoff_m in standoffs_m:
                p_g = state["p_base"] - R[:, 2] * standoff_m
                q = self.ik(p_g, R, list(self.arm_q()))
                blocked = None if q is None else ("" if self.state_valid(np.asarray(q)) else str(self._evidence["last_state_validity"]))
                q_contact = None
                if q is not None and not blocked:
                    self.contact_permission("endpoint_check")
                    try:
                        q_contact = self.ik(p_contact, R, q)
                        if q_contact is not None and not self.state_valid(np.asarray(q_contact)):
                            blocked, q_contact = str(self._evidence["last_state_validity"]), None
                    finally:
                        self.contact_permission("revoke")
                rec["tried"].append({"jaw_axis": [round(float(v), 2) for v in R[:, 0]], "stand_off_m": round(standoff_m, 3),
                                     "reachable": q is not None, "contact_reachable": q_contact is not None, "blocked": blocked})
                if q_contact is not None:
                    self.contact_path_observed(tool_points, tool_links, p_g, p_contact, R)
                    rec["stand_off_m"] = state["standoff_used"] = round(standoff_m, 3)
                    self.plan_to(q, "approach")
                    self.wait(0.6)
                    rec["stood"] = True
                    return R
        reasons = [t["blocked"] for t in rec["tried"] if t["blocked"]]
        cause = "collision" if reasons else "ik_no_solution"
        self._evidence["decision_required"] = {"cause": cause, "scene": self._evidence.get("contact_scene"), "reasons": reasons}
        raise StageFailure("approach", f"{cause}: no reachable insertion at the slot in {len(rec['tried'])} stand-off/side "
                           "options" + ("" if reasons else ". The arm has no configuration that reaches this grasp from "
                                                             "where the rover stands, so trying again from here cannot "
                                                             "help; the rover has to stand somewhere else."), local_retry=False)

    def contact_path_observed(self, tool_points, tool_links, start, end, rotation) -> dict:
        """Record measured free rays along the gripper's insertion (evidence for the caller, not a veto)."""
        count = int(np.ceil(np.linalg.norm(end - start) / self.geom.min_part_width_m)) + 1
        local = tool_points @ rotation.T
        missing = total = permitted = 0
        region, dimensions = self._contact_region
        T_ob = self.T("odom", "chassis_base_link")
        contact_links = np.isin(tool_links, self.geom.contact_links)
        for point in np.linspace(start, end, count):
            points = local + point
            in_odom = points @ T_ob[:3, :3].T + T_ob[:3, 3]
            local_region = (in_odom - region[:3, 3]) @ region[:3, :3]
            expected_contact = contact_links & np.all(abs(local_region) <= dimensions / 2, axis=1)
            observed = expected_contact.copy()
            permitted += int(np.count_nonzero(expected_contact))
            for frame in self._contact_views:
                pc = (points - frame.T_base_cam[:3, 3]) @ frame.T_base_cam[:3, :3]
                front = pc[:, 2] > self._self_depth
                indices = np.flatnonzero(front & ~observed)
                uv = pc[indices, :2] / pc[indices, 2, None]
                uv = np.rint(uv * [frame.K[0, 0], frame.K[1, 1]] + [frame.K[0, 2], frame.K[1, 2]]).astype(int)
                h, w = frame.depth.shape
                inside = (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
                indices, uv = indices[inside], uv[inside]
                measured = frame.depth[uv[:, 1], uv[:, 0]]
                free = np.isfinite(measured) & (measured >= pc[indices, 2])
                semantics = frame.depth_semantics or {}
                if semantics.get("depth_axis") == "image_plane" and semantics.get("positive_infinity") == "beyond_far_clip":
                    free |= (np.isposinf(measured) & (pc[indices, 2] >= float(semantics["near_m"]))
                             & (pc[indices, 2] < float(semantics["far_m"])))
                observed[indices] |= free
            total += len(points)
            missing += int(np.count_nonzero(~observed))
        result = {"satisfied": missing == 0, "unobserved_samples": missing, "samples": total,
                  "views": len(self._contact_views), "step_m": self.geom.min_part_width_m,
                  "expected_contact_samples": permitted}
        self._evidence["contact_path_observation"] = result
        return result

    # ---- grasp + hold -----------------------------------------------------------------------------------------------------
    def on_contact(self, gp: np.ndarray, gR: np.ndarray) -> dict:
        """Is the tool on the slot (contract C2)? Along the slot the pads may sit anywhere it is long enough to take
        them; across the neck the part only has to lie between the open pads; along the approach the pads must be
        level with the neck within the insertion the approach offers."""
        p, R = self.tcp()
        e = gp - p
        along = float(abs(np.dot(e, gR[:, 1])))
        sideways = float(abs(np.dot(e, gR[:, 0])))
        depth = float(abs(np.dot(e, gR[:, 2])))
        side_budget = max(0.0, (self.geom.open_width_m - self.dims["neck_thickness"]) / 2 - self.geom.clearance_m)
        slack = max(0.0, (self.dims["neck_height"] - self.geom.finger_height_m) / 2)  # the fingers' own height, not the pads' minimum
        rotation_error = float(np.linalg.norm(mat_to_rotvec(gR @ R.T)))
        self._evidence["closure_precondition"] = {
            "position_error_m": float(np.linalg.norm(e)), "along_part_m": along, "along_part_budget_m": round(slack, 4),
            "sideways_m": sideways, "sideways_budget_m": round(side_budget, 4),
            "approach_m": depth, "approach_budget_m": self.geom.seat_tolerance_m,
            "rotation_error_rad": rotation_error,
            "on_contact": (along <= slack and sideways <= side_budget and depth <= self.geom.seat_tolerance_m
                           and rotation_error < 0.10),
        }
        return self._evidence["closure_precondition"]

    def grasp(self) -> float:
        if not self.on_contact(*self._closure_goal)["on_contact"]:
            raise StageFailure("grasp", "the slot was not reached; jaws remain open", local_retry=False)
        self.stage("grasp")
        jaw = self.set_jaw(False, "grasp")
        self._evidence["jaw_closed_rad"] = round(jaw, 4)
        if jaw <= JAW_EMPTY_MAX_RAD:
            raise StageFailure("grasp", f"the jaws closed on nothing (jaw {jaw:.3f} rad)")
        return jaw

    def hold(self, jaw_closed: float) -> None:
        g = self.goal
        self.stage("hold")
        before = self.lit_frame()
        p0, R0 = self.tcp()
        lift_v = np.array([0.0, 0.0, float(g["lift_m"])])

        def rising():
            self.working_on(self.tcp()[0])  # what is in the jaws travels with the tool
            return p0 + lift_v, R0

        try:
            # Joint-target servo: in the twist path the lift realised ~3 % of its commanded speed (g9a, F78). The
            # scene snapshot still holds the item where it stood; the lift carries it straight up out of it.
            self.servo(rising, "hold", tol_m=0.01, tol_rad=0.06, max_speed_m_s=float(g["lift_speed_m_s"]),
                       max_joint_rate_rad_s=float(g["carry_servo"]["max_joint_rate_rad_s"]), timeout_s=90.0, stall_s=12.0,
                       check_scene=False)
        except StageFailure as exc:
            # a held item bends the arm down (about 4.5 cm under 1 kg, F51): what counts is how far it rose
            risen = float(self.tcp()[0][2] - p0[2])
            if risen < float(g["min_lift_m"]):
                raise StageFailure("hold", f"the lift stopped after {risen * 100:.1f} cm ({exc.why}); the item may be caught "
                                           f"or too heavy") from exc
        self.wait(0.8)
        after = self.lit_frame()
        p1, _ = self.tcp()
        jaw = self.jaw()
        flow = moved_with_gripper(before, after, self._tool_view["mask"], self._self_depth)
        np.save(self.evidence_dir / "hold_before_depth.npy", before.depth)
        self._evidence["hold_check"] = {"jaw_rad": round(jaw, 4), "lift_m": round(float(p1[2] - p0[2]), 4), **flow,
                                         "before": self.save("hold_before.png", before.bgr), "after": self.save("hold_after.png", after.bgr),
                                         "before_depth": str(self.evidence_dir / "hold_before_depth.npy"), "fx": float(before.K[0, 0]),
                                         "gripper_mask": self._gripper_mask_path, "self_depth_m": self._self_depth}
        if jaw <= JAW_EMPTY_MAX_RAD:
            raise StageFailure("hold", f"the item slipped out of the jaws during the lift (jaw {jaw:.3f} rad)")
        if not flow["item_moved_with_gripper"]:
            raise StageFailure("hold", f"the item did not rise with the gripper (image shift {flow['median_flow_px']} px, "
                                       f"expected ~{flow['expected_flow_px']} px if left behind)")

    def lit_frame(self, tries: int = 5) -> Frame:
        """A wrist frame with something in it: the camera occasionally delivers an empty (black) image."""
        for _ in range(tries):
            f = self.frame(after=self._clock() - 0.05)
            if float(f.bgr.mean()) > 3.0:
                return f
            self.wait(0.3)
        raise StageFailure(self._evidence.get("stage", "?"), "the wrist camera returned only empty frames")

    def bring_in(self) -> None:
        """Carry what is now in the jaws in towards the rover before the call returns: a payload left at the far end
        of the arm's reach rides the longest lever the arm owns (F61 §6). The tool walks in along the rover's own x
        axis keeping the grasp's orientation and stops at the last pose the arm reaches and its planning scene
        allows; failing to move at all is not a failed pick."""
        self.stage("bring_in")
        p, R = self.tcp()
        self.working_on(p)  # what is in the jaws travels with them; it is the job, not an obstacle to it (F59)
        # the same search may first raise the tool, so the carried item clears the rover's own deck (g8s/g8t, F78);
        # it stops where the item would still clear after sagging one grid step (g9a)
        step = 0.05
        best, raise_only, tried = None, None, []
        for dz in [round(0.02 * k, 2) for k in range(0, 11 if self._held else 1)]:
            seed = list(self.arm_q())
            q = self.ik(p + np.array([0.0, 0.0, dz]), R, seed)
            if q is None or not self.state_valid(np.array(q), held_drop_m=step if self._held else 0.0):
                tried.append({"raise_m": dz, "reachable": q is not None, "clear": False})
                continue
            tried.append({"raise_m": dz, "reachable": True, "clear": True})
            raise_only = dz if raise_only is None else raise_only
            found, seed = None, list(q)
            for dx in [round(step * k, 2) for k in range(1, 13)]:
                q = self.ik(p + np.array([dx, 0.0, dz]), R, seed)
                if q is None or not self.state_valid(np.array(q), held_drop_m=step if self._held else 0.0):
                    break
                found, seed = dx, list(q)
            if found is not None and (best is None or found > best[0]):
                best = (found, dz)
        if best is None and raise_only is not None:
            best = (0.0, raise_only)  # out of the way of what it stood on, even when nothing can be pulled in
        self._evidence["bring_in"] = {"from": [round(float(v), 3) for v in p], "in_by_m": best[0] if best else 0.0,
                                      "raised_by_m": best[1] if best else 0.0, "tried": tried[:12]}
        if best is None or not any(best):  # nothing can be pulled in or raised: the grasp stands, the lever stays
            return
        waypoints = ([p + np.array([0.0, 0.0, best[1]])] if best[1] else []) + (
            [p + np.array([best[0], 0.0, best[1]])] if best[0] else [])
        try:
            for target in waypoints:  # the orientation-preserving, per-step collision-checked carry servo, as place
                def carrying(target=target):
                    self.working_on(self.tcp()[0])
                    return target, R

                info = self.servo(carrying, "bring_in", **self.goal["carry_servo"])
            self._evidence["bring_in"]["servo"] = info
            self._evidence["bring_in"]["tcp"] = [round(float(v), 3) for v in self.tcp()[0]]
        except StageFailure as exc:
            self._evidence["bring_in"]["why"] = exc.why
        finally:
            self.hold_here()
        jaw = self.jaw()
        self._evidence["bring_in"]["jaw_rad"] = round(jaw, 4)
        if jaw <= JAW_EMPTY_MAX_RAD:
            raise StageFailure("bring_in", f"the item slipped out during carry retraction (jaw {jaw:.3f} rad)",
                               local_retry=False)


#: the tool's own body and the outsides of its fingers, in the tool frame (x jaw axis, y up in the grip orientation,
#: z forward): what must not touch an item. The space between the fingers is left out -- that is where the part goes.
BODY_POINTS = np.array([[x, y, z] for x in (-0.055, 0.055) for y in (-0.03, 0.0, 0.03) for z in (0.01, -0.03, -0.08)]
                       + [[0.0, y, z] for y in (-0.05, 0.05) for z in (-0.03, -0.08)])

#: points of the tool in its own frame: the fingertips and pads, the jaw housing, the wrist camera
TOOL_POINTS = np.array([[x, y, z] for x in (-0.045, 0.0, 0.045) for y in (-0.03, 0.0, 0.03) for z in (0.01, -0.03, -0.08)])


def tool_in_view(K: np.ndarray, T_ct: np.ndarray, rows: int, cols: int, margin_m: float = 0.02) -> dict:
    """Where the gripper's own body is in its own camera, and how far along the optical axis it reaches: both follow
    from the mount (TF) and the tool's dimensions, and hold for every frame. Written as constants instead they tied
    the skill to one camera mount (F57)."""
    pts = (T_ct[:3, :3] @ np.vstack([BODY_POINTS, TOOL_POINTS]).T).T + T_ct[:3, 3]
    ahead = pts[pts[:, 2] > 0.01]
    self_depth = float(np.max(ahead[:, 2])) + margin_m if len(ahead) else margin_m
    v = K[1, 1] * ahead[:, 1] / ahead[:, 2] + K[1, 2]
    top = int(np.clip(np.min(v) if len(v) else rows, 0, rows))  # the image row the gripper first appears in
    mask = np.zeros((rows, cols), np.uint8)  # its silhouette: the fingers, not the space between them
    for x, y, z in ahead:
        u, vv = K[0, 0] * x / z + K[0, 2], K[1, 1] * y / z + K[1, 2]
        cv2.circle(mask, (int(round(u)), int(round(vv))), max(2, int(K[0, 0] * 0.012 / z)), 1, -1)
    return {"self_depth_m": round(self_depth, 3), "tool_top_row": top, "mask": mask.astype(bool)}


def moved_with_gripper(before: Frame, after: Frame, tool_mask: np.ndarray, self_depth_m: float,
                       beyond_m: float = 0.12) -> dict:
    """Did the item in the gripper rise with it? Dense optical flow between the wrist images before and after the
    lift, over what a held item is: near the camera, within the gripper's own reach, outside its own silhouette."""
    g0, g1 = (cv2.cvtColor(f.bgr, cv2.COLOR_BGR2GRAY) for f in (before, after))
    flow = cv2.calcOpticalFlowFarneback(g0, g1, None, 0.5, 4, 31, 5, 7, 1.5, 0)
    z = before.depth
    near = np.isfinite(z) & (z > 0.05) & (z < self_depth_m + beyond_m) & ~tool_mask
    if near.sum() < 200:
        return {"item_moved_with_gripper": False, "median_flow_px": None, "expected_flow_px": None, "near_pixels": int(near.sum())}
    mag = np.linalg.norm(flow[near], axis=1)
    lift = float(np.linalg.norm(after.T_base_cam[:3, 3] - before.T_base_cam[:3, 3]))
    expected = before.K[1, 1] * lift / float(np.median(z[near]))
    med = float(np.median(mag))
    return {"item_moved_with_gripper": bool(med < 0.35 * expected), "median_flow_px": round(float(med), 1),
            "expected_flow_px": round(float(expected), 1), "near_pixels": int(near.sum())}
