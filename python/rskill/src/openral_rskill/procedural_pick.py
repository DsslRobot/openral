"""PickRskill -- pick up an item described in words, holding it by a described part (one physical operation).

``kind: procedural``, LunarBot (docs/semantic_to_metric_manipulation_plan.md v4 §4 in the research repo). The caller
says *what* (``target``) and *by which part* (``part``); every mechanical choice is this skill's default. No object
model, no world or rover coordinates for the item: grasps come from the current wrist RGB-D view, the item lives in the
image and is re-measured every servo cycle; the only geometry used is the robot's own (arm, camera mount, gripper).

Stages (each recorded with its evidence):

1. find    -- side views of the arm's reach zone behind the rover, from the top of the grasp band downwards (arm poses
              on the working branch that centre the zone; each pose and the joint path to it in space the earlier views
              saw through), each with class-agnostic antipodal grasp candidates (``grasp_perception.find_candidates``).
              A part is graspable from the side it is seen from; a handle under an overhang is not seen from above, and
              the wrist image is half taken by the fingers, so one high overview does not cover the band.
2. select  -- in each view, a vision-language choice among the numbered candidates for target + part (the whole view
              for the item's identity); the first view with a choice ends the search.
3. approach-- the chosen candidate is locked; the tool servos to a stand-off in front of it, jaw axis across the part,
              approaching perpendicular to the part, the candidate re-associated in every new frame through the
              camera's own motion; then a guarded straight close-in along the tool axis.
4. grasp   -- close the jaws; the jaw must stop between empty and open.
5. hold    -- lift; the jaw still holds and the wrist image shows the item moving with the gripper (optical flow of
              the item region stays near zero while the scene would have shifted by the lift).

Local retries only on the same target (plan v4 locality rule): lost lock -> re-acquire from the stand-off; empty
jaws or a slipped hold -> open, back off, re-find and try again; the VLM's runner-up after the choice. Anything that
would need the base to move, another item or another operation ends the goal with the stage reached and its evidence.
"""

from __future__ import annotations

import dataclasses
import json
import math
import os
import time

import cv2
import numpy as np

from openral_rskill._eye_in_hand import (BASE_FRAME_ID, READY, JAW_EMPTY_MAX_RAD, EyeInHandSkill, Frame, StageFailure,
                                          TCP_FRAME_ID, mat_to_rotvec, rotvec_to_mat)
from openral_rskill.grasp_perception import (Candidate, GripperGeometry, PartTracker, find_candidates,
                                             render_candidates, select_candidate, upright)

__all__ = ["PickRskill", "tool_rotation"]


def tool_rotation(pitch_deg: float, yaw_deg: float, roll_deg: float = 0.0) -> np.ndarray:
    """Tool pointing rearward out of the rover (-x of chassis_base_link), pitched down / yawed; jaw axis horizontal.
    `roll_deg` 180 turns the tool about its own axis, which moves the wrist camera to the other side of it -- the side
    the fingers do not hide, so the view shows what stands above the tool axis instead of below it."""
    th, ps = math.radians(pitch_deg), math.radians(yaw_deg)
    z = np.array([-math.cos(th) * math.cos(ps), -math.cos(th) * math.sin(ps), -math.sin(th)])
    x = np.cross([0.0, 0.0, 1.0], z)
    x /= np.linalg.norm(x)
    if abs(roll_deg - 180.0) < 1e-6:
        x = -x
    return np.stack([x, np.cross(z, x), z], axis=1)


def grasp_rotation(closing_b: np.ndarray, part_b: np.ndarray, view_b: np.ndarray, R_now: np.ndarray,
                   cam_side_b: np.ndarray | None = None, cam_tool: np.ndarray | None = None) -> np.ndarray:
    """Tool rotation for a grasp: jaw axis (tool x) along the closing axis, tool z perpendicular to the part and to the
    closing axis, pointing away from the camera. The remaining choice is the 180 degree roll about the tool axis, which
    decides which side of the part the wrist camera looks from (the fingers fill the other half of its image): the roll
    that puts the camera -- at `cam_tool` in the tool frame, the mount's own geometry -- on the side `cam_side_b`.
    Without that, the nearer roll to the current tool."""
    x = closing_b / np.linalg.norm(closing_b)
    z = np.cross(part_b, x)
    if np.linalg.norm(z) < 1e-6:
        z = view_b - x * np.dot(view_b, x)
    z /= np.linalg.norm(z)
    if np.dot(z, view_b) < 0:
        z = -z
    if cam_side_b is not None and cam_tool is not None:
        # which side of the tool axis each roll puts the camera on (its mount offset across that axis): keep cam_side_b
        R = np.stack([x, np.cross(z, x), z], axis=1)
        if np.dot(R @ across_tool_axis(cam_tool), cam_side_b) < 0:
            x = -x
    elif np.dot(x, R_now[:, 0]) < 0:
        x = -x
    return np.stack([x, np.cross(z, x), z], axis=1)


def set_view_dir(state: dict, cam_base: np.ndarray) -> None:
    """Fix the direction the grasp is taken from: from this camera position towards the part, as a unit vector.

    Called when a view is *chosen* (the candidate's own frame, and again once the tool stands at the stand-off),
    never from inside the servo loop -- that is what made the goal pose depend on the tool's own pose."""
    v = np.asarray(state["p_base"]) - np.asarray(cam_base)
    state["view_base"] = np.asarray(cam_base)
    state["view_dir"] = v / max(float(np.linalg.norm(v)), 1e-9)


def across_tool_axis(v_tool: np.ndarray) -> np.ndarray:
    """The part of a vector in the tool frame that is across the tool axis -- the part a roll about that axis turns.
    Taken in the tool frame, not by projecting in the base frame: the camera looks along neither the tool axis nor the
    grasp axis, so projecting out a viewing direction cancels exactly this component (research repo F57)."""
    return np.array([v_tool[0], v_tool[1], 0.0], float)


def grasp_camera_sides(part_axis_b: np.ndarray) -> list[np.ndarray]:
    """Both antipodal jaw rolls; measured geometry decides executability.

    Visibility of one offset point cannot authorize or reject an entire tool
    posture. Each roll must pass the same IK, collision and insertion evidence.
    """
    axis = part_axis_b / max(float(np.linalg.norm(part_axis_b)), 1e-9)
    return [axis, -axis]


class PickRskill(EyeInHandSkill):
    def procedure(self) -> None:
        g = self.goal
        from openai import OpenAI

        # No hidden retries: the model answers within the deadline or the stage returns with what it has. The deadline
        # is generous against a measured answer (this model reads one marked view in 1-6 s), not against a service that
        # is not answering: a stage that waits 5 minutes for a question the robot could re-ask from a better view has
        # stopped being a robot's deadline (research repo F58).
        import httpx

        http = httpx.Client(limits=httpx.Limits(max_connections=8, keepalive_expiry=30.0), timeout=120.0)
        self.vlm = OpenAI(api_key=os.environ["SPACE_LLM_API_KEY"], base_url=g["vlm_endpoint"], timeout=120,
                          max_retries=0, http_client=http)
        self.geom = GripperGeometry()
        self._evidence.update(target=g["target"], part=g["part"], attempts=[])
        self.stage("prepare")
        self.set_jaw(True, "prepare")
        try:
            self.attempt()
        finally:
            self.working_on(None)

    def attempt(self) -> None:
        """One selected contact per call; a new observation/selection belongs to the caller."""
        self.evidence_dir = self.evidence_dir / "attempt_01"
        self.evidence_dir.mkdir()
        cand, choice = self.find_and_select()
        self.contact_scene(True)
        self.working_on(self._locked_frame.to_base(cand.p_cam))
        rec = {"attempt": 1, "candidate": cand.as_dict(), "vlm": choice, "selection_fx": float(self._locked_frame.K[0, 0]),
               "evidence_dir": str(self.evidence_dir), "find": self._evidence["find"]}
        self._evidence["attempts"].append(rec)
        try:
            self.approach(cand)
            jaw = self.grasp()
            self.contact_permission("revoke")
            self.hold(jaw)
            if self.goal.get("held_item"):
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
            rec["target_lock"] = self._evidence.get("target_lock")
            (self.evidence_dir / "attempt.json").write_text(json.dumps(rec, indent=2) + "\n")

    def vlm_call(self, stage: str, fn):
        """One call to the vision-language model. When the service does not answer, the stage fails with that reason
        and the evidence says the *service* failed, not the grasp: the two are different outcomes for a study of the
        method, and only one of them is about the robot. The skill does not retry it -- that is the caller's call."""
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

    # ---- find + select --------------------------------------------------------------------------------------------------
    @property
    def _contact_width(self) -> float | None:
        """The width the jaws close across at the item's declared contact interface (equipment catalogue), if given."""
        return (self.goal.get("held_item") or {}).get("closing_width_m")

    def work_point(self) -> np.ndarray:
        """Where the work is, in the rover's own frame: the surveyed place the caller named, at the height work
        happens there. Both come from the boundary, which computed that height to decide where the rover parks --
        so the camera and the parking place are answering the same question instead of two written-down ones."""
        g = self.goal
        if "support_xyz_map" not in g:
            raise StageFailure("find", "this pick did not say what the item stands on, so there is nothing to look "
                                       "at: name the place frame (`support`) the item is on.", local_retry=False)
        T_bm = self.T(BASE_FRAME_ID, "map")
        s = T_bm[:3, :3] @ np.array(g["support_xyz_map"], float) + T_bm[:3, 3]
        return np.array([s[0], s[1], float(g["work_z_m"]) if "work_z_m" in g else s[2]])

    def find_and_select(self) -> tuple[Candidate, dict]:
        """One view of the work, taken from straight behind the rover at the height of the part being grasped.

        A lifting handle's neck is an upright post: a level view measures the width the jaws close on, and a view
        from above cannot -- the head hides the neck (F60). So the camera goes where that measurement is made and
        takes one frame there.

        What this replaces was machinery for guessing where to point it: an overview from a pose fixed in the
        chassis frame, a 5 cm height map, the tallest cell within 0.2 m of a located blob taken for "the item's
        top", and a grid of 24 viewpoints aimed at that guess. None of it measured anything the grasp needed, and
        all of it assumed the rover parks where that fixed pose expects. It does not -- the boundary chooses the
        stand -- and when it parked 14 cm nearer, the guess landed on the ORU's body, every view framed the box,
        and no 12 mm contact was ever measured (g9p, g9q, g9r, research F79). The declared contact is what can be
        measured, so it is what is looked for, in the one view that can see it.
        """
        g = self.goal
        self.stage("find")
        # Where to look is derived, never written down. The caller names the place the item stands on; the boundary
        # sends its surveyed point and the height work happens there -- the same height it stood the rover for. The
        # camera goes level with that, far enough back that the part clears the gripper in its own view: the wrist
        # camera sits behind the tool, so the tool stands off by that much less. Nothing here is particular to this
        # item or this support, and both move with them.
        at = self.work_point()
        # how far the camera sits behind the tool, from the mount itself: one frame from wherever the arm is names
        # its own optical frame, and TF gives the rest. The part depth asked for is a property of this camera and
        # this gripper, so the tool's stand-off follows from it and nothing about the item enters here.
        behind = float(np.linalg.norm(self.T(TCP_FRAME_ID, self.frame().frame_id)[:3, 3]))
        stand_off = [float(d) - behind for d in g["view_part_depth_m"]]
        tried, q, R, tool = [], None, None, None
        for back in stand_off:
            for pitch in (0.0, 10.0, 20.0):  # level first; a little down when the arm cannot fold that flat
                R = tool_rotation(pitch, 0.0, 180.0)
                tool = at - R[:, 2] * back
                q = self.ik(tool, R, READY)
                tried.append({"back_m": back, "pitch_deg": pitch, "reachable": q is not None})
                if q is not None:
                    break
            if q is not None:
                break
        self._evidence["view_tried"] = tried
        if q is None:
            raise StageFailure("find", f"the arm cannot put its camera on {g.get('support') or 'this work'} at the "
                                       f"height it is worked at ({at[2]:.2f} m) from where the rover stands: none of "
                                       f"the {len(tried)} viewing poses is in reach. Trying again from here cannot "
                                       "help; the rover has to stand somewhere else.", local_retry=False)
        x, y, z = (float(v) for v in tool)
        pitch = float(tried[-1]["pitch_deg"])
        self.plan_to(q, "find")
        self.wait(0.8)
        f = self.frame(after=self._clock() - 0.05)
        T_tc = self.T(TCP_FRAME_ID, f.frame_id)  # the camera in the tool frame (robot geometry)
        self._cam_tool = T_tc[:3, 3].copy()
        self._tool_view = tool_in_view(f.K, np.linalg.inv(T_tc), *f.depth.shape)
        self._gripper_mask_path = str(self.evidence_dir / "gripper_mask.npy")
        np.save(self._gripper_mask_path, self._tool_view["mask"])
        self._self_depth = float(self._tool_view["self_depth_m"])
        self.geom = dataclasses.replace(self.geom, self_depth_m=self._self_depth,
                                        min_part_depth_m=self._self_depth + 0.03)
        self._heights = surface_heights([f], self._self_depth)
        np.save(self.evidence_dir / "view_depth.npy", f.depth)
        record = {"view": {"tool": [round(v, 3) for v in (x, y, z)], "pitch_deg": pitch},
                  "tool_in_view": {k: v for k, v in self._tool_view.items() if k != "mask"}
                                  | {"mask": self._gripper_mask_path}}
        self._evidence["find"] = record
        cands = find_candidates(f.depth, f.K, self.geom, max_candidates=int(g["candidates_per_view"]),
                                contact_width_m=self._contact_width)
        for i, c in enumerate(cands, 1):
            c.id = i
        record["view"]["image"] = self.save("view.jpg", upright(f.bgr, f.up_cam))
        record["candidates"] = len(cands)
        if not cands:
            raise StageFailure("find", "the view of the work measured nothing the jaws could close on"
                               + (f" across the declared {self._contact_width * 1000:.0f} mm contact"
                                  if self._contact_width else ""))
        marked = render_candidates(f.bgr, cands, f.up_cam)  # the vision model is shown the image, not its path
        record["marked"] = self.save("candidates.jpg", marked)
        self.stage("select", views=1, candidates=len(cands))
        a = self.vlm_call("select", lambda: select_candidate(
            self.vlm, g["vlm_model"], [marked], g["target"], g["part"], {c.id for c in cands}))
        record["select"] = {k: a[k] for k in ("item_visible", "what_is_visible", "choice", "reason")}
        if a["choice"] is None:
            raise StageFailure("select", f"none of the {len(cands)} marks in the view is on the described part of "
                                         "the described item")
        from openral_rskill.grasp_perception import refine_contact

        selected = next(c for c in cands if c.id == a["choice"])
        try:
            selected, geometry = refine_contact(f.depth, f.K, selected, self.geom)
        except ValueError as exc:
            raise StageFailure("select", str(exc)) from exc
        record["contact_geometry"] = geometry
        np.save(self.evidence_dir / "selected_depth.npy", f.depth)
        self.save("selected_rgb.png", f.bgr)
        self._locked_frame = f
        self._contact_views = [f]
        return selected, record["select"]

    def approach(self, cand: Candidate) -> None:
        """Approach a stationary, visually selected contact using measured robot pose feedback.

        Commit the contact in odom, so arm/base motion changes its robot-relative coordinates,
        not its identity. Image matching checks visibility; it cannot silently move the contact.
        Loss of visibility before the terminal self-occlusion region returns a failure.
        """
        g = self.goal
        f = self._locked_frame
        self.stage("approach", candidate=cand.id)
        state = {"p_base": f.to_base(cand.p_cam), "axis_base": f.T_base_cam[:3, :3] @ np.array(cand.axis_cam),
                 "part_base": f.T_base_cam[:3, :3] @ np.array(cand.part_axis_cam), "view_base": f.T_base_cam[:3, 3],
                 "seen": f.stamp, "hits": 0, "misses": 0, "frames": [], "trace": []}
        # Which side the grasp is approached from is decided once, by the view that chose it, and then held. It used
        # to be recomputed every frame from where the camera had got to -- and the camera is bolted to the tool, so
        # the goal pose was a function of the tool's own pose: `grasp_rotation` points the tool's z away from the
        # camera, the tool swung past the part, the camera changed sides, z flipped 180 degrees, and the tool swung
        # back. g6a and g5p show the ring it makes -- the error running 0.009 -> 0.020 -> 0.009 m without ever
        # closing, the part sliding out of frame, and the approach ending on a lost lock 13-15 cm out, further away
        # than it had been ten frames earlier (research repo F64). Both the position and direction are now
        # committed in odom; image matches check visibility without rewriting this contact.
        set_view_dir(state, f.T_base_cam[:3, 3])
        self._evidence["approach_track"] = state["trace"]  # the same list: a failed approach still returns its pictures
        # the camera looks from the side the part is free on, so the fingers close from the side with less material
        # (a handle on a box: from above, not through the box). The tracker carries the roll this costs, frame by frame.
        self.working_on(state["p_base"])
        state["camera_sides"] = grasp_camera_sides(state["part_base"])
        state["camera_side"] = state["camera_sides"][0]
        self._evidence["camera_side"] = [round(float(v), 2) for v in state["camera_side"]]
        T_ob = self.T("odom", "chassis_base_link")
        contact_odom = T_ob[:3, :3] @ state["p_base"] + T_ob[:3, 3]
        axes_odom = {key: T_ob[:3, :3] @ state[key]
                     for key in ("axis_base", "part_base", "view_dir", "camera_side")}
        self._evidence["target_lock"] = {
            "status": "committed", "candidate": cand.id, "frame": "odom", "position_m": contact_odom.tolist(),
            "source": "selected wrist RGB-D candidate and robot TF",
            "tracking_role": "visibility check; no contact-point replacement",
        }

        def refresh_contact_frame(camera_side=None):
            if camera_side is not None:
                axes_odom["camera_side"] = self.T("odom", "chassis_base_link")[:3, :3] @ camera_side
            T_bo = self.T("chassis_base_link", "odom")
            state["p_base"] = T_bo[:3, :3] @ contact_odom + T_bo[:3, 3]
            for key, axis in axes_odom.items():
                state[key] = T_bo[:3, :3] @ axis

        # Bound permission to the selected measured patch, never the item or work zone.
        region = np.eye(4)
        region[:3, 3] = contact_odom
        x, y = axes_odom["axis_base"], axes_odom["part_base"]
        region[:3, :3] = np.column_stack((x, y, np.cross(x, y)))
        dimensions = np.array([cand.width_m, cand.length_m, self.geom.depth_step_m])
        self._contact_region = (region, dimensions)
        self.contact_permission("register", region, dimensions, str(self.evidence_dir / "selected_depth.npy"))

        approach_from = g["standoff_m"] if isinstance(g["standoff_m"], list) else [float(g["standoff_m"])]

        # stand in front of the grasp first, in the orientation the grasp needs -- with the planner, from the view's own
        # measurement -- and only then close the loop (see take_standoff)
        tracker = self.take_standoff(cand, state, list(approach_from), refresh_contact_frame)
        # The antipodal roll is chosen before servo, with both contact and stand-off
        # IK checked. Preserve that roll in the same fixed frame as the contact.
        self._evidence["camera_side"] = state["camera_side"].tolist()
        if tracker is None:
            tracker = PartTracker(f.bgr, f.depth, f.K, cand, R_base_cam=T_ob[:3, :3] @ f.T_base_cam[:3, :3])

        def goal_now():
            """Re-express the committed contact using robot TF; check visibility without relocating it."""
            refresh_contact_frame()
            if state.get("covered") or float(np.linalg.norm(state["p_base"] - self.tcp()[0])) < float(g["blind_ok_m"]):
                R = grasp_rotation(state["axis_base"], state["part_base"], state["view_dir"],
                                   self.tcp()[1], state["camera_side"], self._cam_tool)
                self.working_on(state["p_base"])
                return state["p_base"] - R[:, 2] * state["standoff"], R
            nf = self.frame(after=state["seen"], timeout_s=2.0)
            state["seen"] = nf.stamp
            p_pred_cam = nf.T_base_cam[:3, :3].T @ (state["p_base"] - nf.T_base_cam[:3, 3])
            R_odom_cam = self.T("odom", "chassis_base_link")[:3, :3] @ nf.T_base_cam[:3, :3]
            m = tracker.measure(nf.bgr, nf.depth, p_pred_cam, R_base_cam=R_odom_cam,
                                self_depth_m=self._self_depth)
            if tracker.covered and state["hits"] > 0:
                # the fingers are in front of the part now: measuring stops here and the last measurement carries the
                # tool in, which is the same thing a straight close-in along the tool axis does. Before the first
                # measurement it means the part was never seen from here, which is a lost lock, not a covered part.
                state["covered"] = True
                self._evidence["covered_at_m"] = round(float(np.linalg.norm(state["p_base"] - self.tcp()[0])), 4)
            if m is not None:
                state["misses"] = 0
                pixel, d, score = m
                if d is not None:
                    measured = nf.to_base(tracker.point_cam(pixel, d) + np.array([0.0, 0.0, tracker.width_m / 2]))
                    self._evidence["target_lock"]["last_match_offset_m"] = float(np.linalg.norm(measured - state["p_base"]))
                    if self._evidence["target_lock"]["last_match_offset_m"] > float(g["track_gate_m"]):
                        self._evidence["target_lock"].update(status="invalidated", reason="selected contact moved or tracking disagrees")
                        raise StageFailure("approach", "current measurement disagrees with the committed contact; observation required",
                                           local_retry=False)
                state["hits"] += 1
            elif not state.get("covered"):
                state["misses"] += 1
                # lost while still far from the part: the goal must not be carried on blind -- go back and identify again
                if float(np.linalg.norm(state["p_base"] - self.tcp()[0])) > float(self.goal["blind_ok_m"]) \
                        and state["misses"] > int(self.goal["max_track_misses"]):
                    why = (f"lost sight of the chosen grasp {state['misses']} frames running "
                           f"while still {float(np.linalg.norm(state['p_base'] - self.tcp()[0])) * 100:.0f} cm away")
                    self._evidence["target_lock"].update(status="invalidated", reason=why)
                    raise StageFailure("approach", why, local_retry=False)
            # Cross: committed contact projection. Circle: current image match. Keep them distinct so a
            # visually plausible match cannot conceal drift away from the selected contact.
            if len(state["frames"]) < 40 and (len(state["frames"]) < 8 or nf.stamp - state["frames"][-1][0] > 1.0):
                mark = nf.bgr.copy()
                pc = nf.T_base_cam[:3, :3].T @ (state["p_base"] - nf.T_base_cam[:3, 3])
                if pc[2] > 0.05:
                    u = int(nf.K[0, 0] * pc[0] / pc[2] + nf.K[0, 2])
                    v = int(nf.K[1, 1] * pc[1] / pc[2] + nf.K[1, 2])
                    cv2.drawMarker(mark, (u, v), (0, 255, 255) if m is not None else (0, 0, 255), cv2.MARKER_CROSS, 24, 2)
                left = float(np.linalg.norm(state["p_base"] - self.tcp()[0]))
                cv2.circle(mark, (int(tracker.last_px[0]), int(tracker.last_px[1])), 7,
                           (0, 255, 0) if m is not None else (0, 0, 255), 2)  # where the template matched best
                cv2.putText(mark, f"{'hit' if m is not None else 'MISS'} score {tracker.last_score:.2f} "
                                  f"d {'~' if tracker.depth_predicted else ''}{tracker.depth:.3f} "
                                  f"roll {tracker.roll_deg:+.0f} left {left * 100:.1f}cm",
                            (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                if tracker.reject:
                    cv2.putText(mark, tracker.reject, (6, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                path = self.save(f"servo_{len(state['frames']):02d}.jpg", mark)
                state["frames"].append((nf.stamp, path))
                state["trace"].append({"image": path, "hit": m is not None, "score": round(tracker.last_score, 3),
                                       "depth_m": round(tracker.depth, 4), "depth_predicted": tracker.depth_predicted,
                                       "roll_deg": round(tracker.roll_deg, 1),
                                       "left_m": round(left, 4), "why": tracker.reject})
            R = grasp_rotation(state["axis_base"], state["part_base"], state["view_dir"],
                               self.tcp()[1], state["camera_side"], self._cam_tool)
            self.working_on(state["p_base"])
            return state["p_base"] - R[:, 2] * state["standoff"], R

        # close in under vision: the stand-off first, then the grasp point itself
        state["standoff"] = float(state.get("standoff_used", approach_from[0]))
        try:
            info = self.servo(goal_now, "approach", tol_m=0.006, tol_rad=0.04,
                              max_speed_m_s=float(g["approach_speed_m_s"]),
                              max_joint_rate_rad_s=float(g["carry_servo"]["max_joint_rate_rad_s"]),
                              timeout_s=float(g["approach_timeout_s"]), stall_s=6.0)
        except StageFailure as exc:
            # the stand-off is a place to close in from, not a pose to hit: the guarded close-in covers the last
            # stretch under vision anyway. Stalling a few millimetres out is arriving (g5g stopped 0.4 cm and 2 deg
            # short of it and lost the call); stalling far out is not, and still returns the stage.
            if not exc.local_retry:
                raise
            off = float(np.linalg.norm(goal_now()[0] - self.tcp()[0]))
            if off > float(g["close_in_short_ok_m"]):
                raise
            info = {"stalled_short_m": round(off, 4), "why": exc.why}
        self._evidence["standoff"] = {**info, "tracked_frames": state["hits"], "missed_frames": state["misses"],
                                      "image": self.save("standoff.jpg", upright(self.frame(after=self._clock() - 0.05).bgr, None))}
        self.contact_permission("insertion")
        state["standoff"] = -float(g["pad_offset_m"])
        self.stage("approach", close_in=True)
        self._contact_length = float(cand.length_m)
        try:
            info = self.servo(goal_now, "approach", tol_m=0.005, tol_rad=0.04,
                              max_speed_m_s=float(g["close_in_speed_m_s"]),
                              max_joint_rate_rad_s=float(g["carry_servo"]["max_joint_rate_rad_s"]),
                              timeout_s=60.0, stall_s=6.0, sag_integral=False, posture_gain=0.0)
        except StageFailure as exc:
            # Stopping short along the part's own long axis is still on the committed contact region: the pads sit a
            # little higher on the same neck. Across it, or in orientation, it is not (g9g stopped 6 mm up the neck).
            if not exc.local_retry or not self.on_contact(*goal_now())["on_contact"]:
                raise
            info = {"stalled_on_contact": self._evidence["closure_precondition"], "why": exc.why}
        self._closure_goal = goal_now()
        p1, _ = self.tcp()
        self._evidence["close_in"] = {**info, "remaining_m": round(float(np.linalg.norm(state["p_base"] - p1)), 4),
                                      "tracked_frames": state["hits"], "missed_frames": state["misses"],
                                      "track": state.get("trace", [])}
        self._grasp_point = state["p_base"]

    def take_standoff(self, cand: Candidate, state: dict, standoffs_m: list[float], refresh_contact_frame) -> PartTracker | None:
        """Stand the tool in front of the chosen grasp, in the orientation the grasp needs, with the planner -- one
        reconfiguration, not a servo motion: the view is taken from wherever frames the part, the grasp comes from the
        side the part is free on, and between the two the wrist can turn half round. Turning it where the tool stands is
        not enough either: the camera looks 17 degrees off the tool axis, so half a turn swings its view by 34 degrees
        and the part leaves the image (research repo F57). Standing at the stand-off instead puts the part back on the
        tool axis, where the camera sees it and the servo can close the last stretch.

        Both stand-off and contact must be reachable at the measured grasp direction.
        Failure returns to the caller before motion; no tilted substitute approach."""
        p, R_now = self.tcp()
        tool_points, tool_links = self.gripper_surface_points(with_links=True)
        options = [(side, grasp_rotation(state["axis_base"], state["part_base"], state["view_dir"],
                                         R_now, side, self._cam_tool)) for side in state["camera_sides"]]
        options.sort(key=lambda option: np.linalg.norm(mat_to_rotvec(option[1] @ R_now.T)))
        rec = {"tried": [], "approach_direction": "measured contact normal; no pitch substitution"}
        self._evidence["take_standoff"] = rec
        q = blocked = None
        found = False
        for side, R_t in options:
            p_contact = state["p_base"] + R_t[:, 2] * float(self.goal["pad_offset_m"])
            for standoff_m in standoffs_m:  # the stand-offs this skill offers, in order: a local retry on the same grasp
                p_g = state["p_base"] - R_t[:, 2] * standoff_m
                q = self.ik(p_g, R_t, list(self.arm_q()))
                blocked = None if q is None else ("" if self.state_valid(np.asarray(q)) else
                                                  str(self._evidence["last_state_validity"]))
                q_contact = None
                if q is not None and not blocked:
                    self.contact_permission("endpoint_check")
                    try:
                        q_contact = self.ik(p_contact, R_t, q)
                        if q_contact is not None and not self.state_valid(np.asarray(q_contact)):
                            blocked = str(self._evidence["last_state_validity"])
                            q_contact = None
                    finally:
                        self.contact_permission("revoke")
                if q_contact is not None:
                    # Evidence, not a veto: unobserved space along the insertion is recorded for the caller, while the
                    # measured-scene collision checks on every servo step still stop real contact (g8j, F78).
                    self.contact_path_observed(tool_points, tool_links, p_g, p_contact, R_t)
                rec["tried"].append({"camera_side": side.tolist(), "stand_off_m": round(standoff_m, 3),
                                     "goal": [round(float(v), 3) for v in p_g],
                                     "reachable": q is not None, "contact_reachable": q_contact is not None,
                                     "contact_goal": p_contact.tolist(), "blocked": blocked})
                if q_contact is not None:
                    rec["stand_off_m"] = state["standoff_used"] = round(standoff_m, 3)
                    rec["approach_turn_deg"] = 0.0
                    refresh_contact_frame(camera_side=side)
                    rec["turn_deg"] = round(math.degrees(float(np.linalg.norm(mat_to_rotvec(R_t @ R_now.T)))), 1)
                    R_g, found = R_t, True
                    break
            if found:
                break
        rec.update(reachable=found, blocked=blocked)
        if not found:
            q = None
        if q is None:
            # No blind servo can establish an unreachable stand-off. Return the
            # failed contact and its evidence; the caller decides the next operation.
            reasons = [trial["blocked"] for trial in rec["tried"] if trial["blocked"]]
            cause = "collision" if reasons else "ik_no_solution"
            self._evidence["decision_required"] = {
                "cause": cause, "contact_preserved": True,
                "scene": self._evidence["contact_scene"], "reasons": reasons,
                "physical_unreachable_proven": False,
            }
            # Nothing blocked the tool: the arm simply has no configuration for this grasp from this base pose. That
            # is a fact about where the rover is parked, not about the grasp, and only the caller can act on it --
            # the boundary's own stand-off test passes distances the skills then cannot work from (g9q, research F79).
            raise StageFailure("approach", f"{cause}: no reachable insertion for the selected mark {cand.id} "
                               f"({cand.width_m * 1000:.1f} mm across, {cand.depth_m:.2f} m from the camera) in "
                               f"{len(rec['tried'])} stand-off/side options"
                               + ("" if reasons else ". The arm has no configuration that reaches this grasp from "
                                  "where the rover stands, so trying again from here cannot help; the rover has to "
                                  "stand somewhere else."), local_retry=False)
        if blocked:
            raise StageFailure("approach", f"the tool cannot stand in front of this grasp without touching {blocked}")
        self.plan_to(q, "approach")
        self.wait(0.6)
        nf = self.frame(after=self._clock() - 0.05)
        refresh_contact_frame()
        state["seen"] = nf.stamp  # the stand-off is on `view_dir` by construction: standing there does not redefine it
        R_c = nf.T_base_cam[:3, :3]
        p_cam = R_c.T @ (state["p_base"] - nf.T_base_cam[:3, 3])
        if p_cam[2] < 0.1:
            return None
        u = float(nf.K[0, 0] * p_cam[0] / p_cam[2] + nf.K[0, 2])
        v = float(nf.K[1, 1] * p_cam[1] / p_cam[2] + nf.K[1, 2])
        seen = dataclasses.replace(cand, u=u, v=v, depth_m=float(p_cam[2]) - cand.width_m / 2,
                                   p_cam=[float(x) for x in p_cam], axis_cam=[float(x) for x in R_c.T @ state["axis_base"]],
                                   part_axis_cam=[float(x) for x in R_c.T @ state["part_base"]])
        mark = nf.bgr.copy()
        cv2.drawMarker(mark, (int(u), int(v)), (0, 255, 255), cv2.MARKER_CROSS, 24, 2)
        rec.update(stood=True, part_px=[round(u), round(v)], part_depth_m=round(float(p_cam[2]), 3),
                   tool=[round(float(x), 3) for x in self.tcp()[0]], image=self.save("standoff_pose.jpg", mark))
        R_odom_cam = self.T("odom", "chassis_base_link")[:3, :3] @ R_c
        return PartTracker(nf.bgr, nf.depth, nf.K, seen, R_base_cam=R_odom_cam)

    def contact_path_observed(self, tool_points, tool_links, start, end, rotation) -> dict:
        """Record measured free rays along the gripper's insertion (evidence for the caller, not a veto).

        Collision checks still test actual geometry. This separate evidence check
        treats occlusion/invalid pixels as unknown and never deletes target cells.
        Sampling resolves the smallest feature supported by the gripper perception.
        """
        count = int(math.ceil(np.linalg.norm(end - start) / self.geom.min_part_width_m)) + 1
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
                  "expected_contact_samples": permitted,
                  "depth_semantics": [frame.depth_semantics for frame in self._contact_views]}
        self._evidence["contact_path_observation"] = result
        return result

    # ---- grasp + hold -----------------------------------------------------------------------------------------------------
    def on_contact(self, gp: np.ndarray, gR: np.ndarray) -> dict:
        """Is the tool on the committed contact region (contract C2)? The three directions are not one number:

        * along the part's long axis (tool y) the pads may sit anywhere the contact is long enough to take them;
        * along the closing axis (tool x) the part only has to lie between the open pads -- closing centres it, so
          the budget is the free space each side of the contact when the jaws are open;
        * along the approach (tool z) the pads must be level with the contact, or they close on what is above or
          below it, so that budget is the insertion the approach itself offers.

        The orientation budget comes from the same geometry: at the measured contact a tenth of a radian moves the
        pad's edge less than a millimetre across the part's face, far inside it. g9k stopped 6 mm along the closing
        axis -- with the part between the pads -- and one number rejected it."""
        p, R = self.tcp()
        e = gp - p
        along = float(abs(np.dot(e, gR[:, 1])))  # the part's long axis: tool y, by construction of grasp_rotation
        sideways = float(abs(np.dot(e, gR[:, 0])))  # the closing axis: the pads centre what lies between them
        depth = float(abs(np.dot(e, gR[:, 2])))  # the approach: how level the pads are with the contact
        across = float(np.linalg.norm(e - gR[:, 1] * np.dot(e, gR[:, 1])))
        width = float(self._contact_width or self.geom.min_part_width_m)
        side_budget = max(0.0, (self.geom.open_width_m - width) / 2 - self.geom.clearance_m)
        depth_budget = float(self.goal["pad_offset_m"])
        rotation_error = float(np.linalg.norm(mat_to_rotvec(gR @ R.T)))
        # how long the contact region is along its own axis: the equipment catalogue where it says so (a lifting eye's
        # neck is as long as the catalogue's neck height), else what this view measured of it. The candidate itself is
        # one pad-height segment of a longer part, so its own extent is not that length (g9h left 0.9 mm of budget).
        length = float((self.goal.get("held_item") or {}).get("neck_height_m") or getattr(self, "_contact_length", 0.0))
        slack = max(0.0, (length - self.geom.pad_height_m) / 2)
        valid = self._evidence["target_lock"]["status"] == "committed"
        self._evidence["closure_precondition"] = {
            "position_error_m": float(np.linalg.norm(e)), "across_part_m": across,
            "along_part_m": along, "along_part_budget_m": round(slack, 4),
            "sideways_m": sideways, "sideways_budget_m": round(side_budget, 4),
            "approach_m": depth, "approach_budget_m": depth_budget,
            "rotation_error_rad": rotation_error, "target_valid": valid,
            "on_contact": (valid and along <= slack and sideways <= side_budget and depth <= depth_budget
                           and rotation_error < 0.10),
        }
        return self._evidence["closure_precondition"]

    def grasp(self) -> float:
        # Recheck the actual pose immediately before closure. A stalled servo or
        # a nonempty jaw is not evidence that the selected contact was reached.
        if not self.on_contact(*self._closure_goal)["on_contact"]:
            raise StageFailure("grasp", "selected contact pose not reached; jaws remain open", local_retry=False)
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
            # Joint-target servo, as bring_in: in the twist path the lift realised ~3 % of its commanded speed and ran
            # into the 90 s timeout with the wrist at its torque limit (g9a, research F78). The scene snapshot still
            # holds the item where it stood, and the lift carries the item with the tool straight up out of it: the
            # snapshot is not checked for this stroke, as the twist path never did (g9d stopped at 0 cm).
            self.servo(rising, "hold", tol_m=0.01, tol_rad=0.06, max_speed_m_s=float(g["lift_speed_m_s"]),
                       max_joint_rate_rad_s=float(g["carry_servo"]["max_joint_rate_rad_s"]), timeout_s=90.0, stall_s=12.0,
                       check_scene=False)
        except StageFailure as exc:
            # a held item bends the arm down (about 4.5 cm under 1 kg, research repo F51), so the tool does not reach the
            # commanded height: what counts is how far it rose, and whether the item came with it
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
        """A wrist frame with something in it: the camera occasionally delivers an empty (black) image, which would make
        an image check meaningless."""
        for _ in range(tries):
            f = self.frame(after=self._clock() - 0.05)
            if float(f.bgr.mean()) > 3.0:
                return f
            self.wait(0.3)
        raise StageFailure(self._evidence.get("stage", "?"), "the wrist camera returned only empty frames")

    def bring_in(self) -> None:
        """Carry what is now in the jaws in towards the rover before the call returns.

        A payload left where it was grasped hangs at the far end of the arm's reach: in g5k it rode 1.08 m behind the
        chassis origin for the whole carry, which is the longest lever the arm owns -- a turn at the base puts more
        acceleration on the load through that lever than driving forwards does, and a bump swings it furthest. The
        item came out of the jaws on the first metre every time (F61 §6).

        Where to stop is not written down. The tool walks in along the rover's own x axis keeping the grasp's
        orientation, so the load hangs the way it was taken, and stops at the last pose the arm reaches and its own
        planning scene allows. Failing to move at all is not a failed pick: the grasp is already made and verified,
        and the caller can still drive -- just with the lever it had before."""
        self.stage("bring_in")
        p, R = self.tcp()
        self.working_on(p)  # what is in the jaws travels with them; it is the job, not an obstacle to it (F59)
        # With the carried item in the collision checks, walking in at the grasp height can put the item into the
        # rover's own deck (g8s/g8t, F78): the same search may first raise the tool, on the same step grid. It stops
        # where the item would still clear the rover after sagging one grid step: stopping where it just clears left
        # it 2 mm over the deck edge, the wrist dipped as the servo let go, and the item wedged there (g9a, F78).
        step = 0.05
        best, raise_only, tried = None, None, []
        for dz in [round(0.02 * k, 2) for k in range(0, 11 if self._held else 1)]:
            seed = list(self.arm_q())
            q = self.ik(p + np.array([0.0, 0.0, dz]), R, seed)
            # A height the arm cannot reach, or one where the item would still be over what it stood on, is one option
            # fewer -- not the end of the search: g9j stopped at the first and left the payload over the depot stand.
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
        if best is None:
            return

        waypoints = ([p + np.array([0.0, 0.0, best[1]])] if best[1] else []) + (
            [p + np.array([best[0], 0.0, best[1]])] if best[0] else [])

        try:
            # A joint-goal plan constrains only its endpoint: it can turn the
            # held object along the route. Use the same orientation-preserving,
            # per-step collision-checked carry servo as place.
            for target in waypoints:
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

#: points of the tool in its own frame (x jaw axis, y up in the grip orientation, z forward): the fingertips and
#: pads, the jaw housing, the wrist camera
TOOL_POINTS = np.array([[x, y, z] for x in (-0.045, 0.0, 0.045) for y in (-0.03, 0.0, 0.03) for z in (0.01, -0.03, -0.08)])
def tool_points(T_base_tcp: np.ndarray, T_tcp_cam: np.ndarray) -> np.ndarray:
    return (T_base_tcp[:3, :3] @ np.vstack([TOOL_POINTS, T_tcp_cam[:3, 3]]).T).T + T_base_tcp[:3, 3]


def tool_in_view(K: np.ndarray, T_ct: np.ndarray, rows: int, cols: int, margin_m: float = 0.02) -> dict:
    """Where the gripper's own body is in its own camera, and how far along the optical axis it reaches: the camera is
    bolted to the tool, so both follow from the mount (TF) and the tool's dimensions, and hold for every frame.

    Everything the wrist views do with self-occlusion is measured against these two numbers -- which pixels are the
    gripper, beyond which depth a surface cannot be the gripper, which image band is free for the part. Writing them
    down as constants instead ties the skill to one camera mount: moving the camera silently turned the gripper's own
    pixels into "an obstacle in front of every pose", and no view of the work zone was allowed any more (F57)."""
    pts = (T_ct[:3, :3] @ np.vstack([BODY_POINTS, TOOL_POINTS]).T).T + T_ct[:3, 3]
    ahead = pts[pts[:, 2] > 0.01]
    self_depth = float(np.max(ahead[:, 2])) + margin_m if len(ahead) else margin_m
    v = K[1, 1] * ahead[:, 1] / ahead[:, 2] + K[1, 2]
    top = int(np.clip(np.min(v) if len(v) else rows, 0, rows))  # the image row the gripper first appears in
    mask = np.zeros((rows, cols), np.uint8)  # its silhouette: the fingers, not the space between them
    for x, y, z in ahead:
        u, vv = K[0, 0] * x / z + K[0, 2], K[1, 1] * y / z + K[1, 2]
        cv2.circle(mask, (int(round(u)), int(round(vv))), max(2, int(K[0, 0] * 0.012 / z)), 1, -1)
    return {"self_depth_m": round(self_depth, 3), "tool_top_row": top, "centre_row": int(top * 0.5),
            "part_row": int(top * 0.75), "mask": mask.astype(bool)}


def aim_points(heights: dict, zone_x_m: float, cell_m: float = 0.05, apart_m: float = 0.12, keep: int = 5,
               stands_out_m: float = 0.04) -> list[np.ndarray]:
    """What to point the wrist camera at, from what the views have measured: the cells that stand above their
    surroundings inside the arm's work zone behind the rover, tallest first, one per structure. The item's own place
    comes out of this; nothing here knows where it was put."""
    if not heights:
        return []
    out: list[np.ndarray] = []
    for (cx, cy), z in sorted(heights.items(), key=lambda kv: -kv[1]):
        p = np.array([(cx + 0.5) * cell_m, (cy + 0.5) * cell_m, z])
        if p[0] > zone_x_m + 0.25:  # in front of the work zone: the rover's own deck and what stands on it
            continue
        ring = [h for (dx, dy), h in heights.items() if 2 <= max(abs(dx - cx), abs(dy - cy)) <= 4]
        if ring and z - float(np.median(ring)) < stands_out_m:  # not a structure, just the support surface
            continue
        if any(float(np.linalg.norm(p - q)) < apart_m for q in out):
            continue
        out.append(p)
        if len(out) >= keep:
            break
    return out


def seen_through(view: Frame, pts: np.ndarray, self_depth_m: float, margin_m: float = 0.02) -> np.ndarray:
    """Per point: +1 the view saw through it (a surface measured behind it, or nothing measured along its ray), -1 a
    surface measured in front of it (the point may be inside what was seen), 0 not judged (outside the image or behind
    the fingers in the image)."""
    pc = (view.T_base_cam[:3, :3].T @ (pts - view.T_base_cam[:3, 3]).T).T
    K, h, w = view.K, *view.depth.shape
    out = np.zeros(len(pts), int)
    for i, (x, y, z) in enumerate(pc):
        if z <= 0.05:
            continue
        u, v = int(K[0, 0] * x / z + K[0, 2]), int(K[1, 1] * y / z + K[1, 2])
        if 0 <= u < w and 0 <= v < h:
            d = view.depth[v, u]
            if np.isfinite(d) and d < self_depth_m:  # the gripper itself in the image
                continue
            out[i] = -1 if np.isfinite(d) and d < z - margin_m else 1
    return out


def surface_heights(views: list[Frame], self_depth_m: float, cell_m: float = 0.05, behind_hull_m: float = -0.6) -> dict:
    """What stands behind the rover, as the cameras measured it: the highest surface in each ground cell (a 2.5-D height
    map in the rover frame). No object models -- only the depth the views returned."""
    out: dict[tuple[int, int], float] = {}
    for f in views:
        z = f.depth[::4, ::4]
        vs, us = np.nonzero(np.isfinite(z) & (z > self_depth_m) & (z < 1.5))
        if us.size == 0:
            continue
        d = z[vs, us]
        pc = np.stack([(us * 4 - f.K[0, 2]) / f.K[0, 0] * d, (vs * 4 - f.K[1, 2]) / f.K[1, 1] * d, d], axis=1)
        pb = (f.T_base_cam[:3, :3] @ pc.T).T + f.T_base_cam[:3, 3]
        pb = pb[(pb[:, 0] < behind_hull_m) & (np.abs(pb[:, 1]) < 0.8) & (pb[:, 2] > 0.2) & (pb[:, 2] < 1.6)]
        if f.arm_links is not None and len(pb):  # the arm's own links are in its camera's view: not surfaces to clear
            keep = np.min(np.linalg.norm(pb[:, None, :] - f.arm_links[None, :, :], axis=2), axis=1) > 0.15
            pb = pb[keep]
        for x, y, zz in pb:
            key = (int(x // cell_m), int(y // cell_m))
            if zz > out.get(key, -np.inf):
                out[key] = float(zz)
    return out


def lowest_clearance(heights: dict, pts: np.ndarray, cell_m: float = 0.05,
                     except_near: tuple[np.ndarray, float] | None = None, except_cells: set | None = None):
    """The point standing least clear of what the cameras measured under it: (clearance, point, surface height)."""
    worst = (9.9, None, None)
    for x, y, z in pts:
        if except_near is not None and math.hypot(x - except_near[0][0], y - except_near[0][1]) < except_near[1]:
            continue
        cell = (int(x // cell_m), int(y // cell_m))
        if except_cells and cell in except_cells:  # the item being picked up is not an obstacle to picking it up
            continue
        h = heights.get(cell)
        if h is not None and z - h < worst[0]:
            worst = (z - h, [round(float(v), 3) for v in (x, y, z)], round(h, 3))
    return worst


def above_surfaces(heights: dict, pts: np.ndarray, clearance_m: float, cell_m: float = 0.05,
                   except_near: tuple[np.ndarray, float] | None = None) -> bool:
    """Are all points clear of what the cameras measured standing below them? `except_near` (a point and a radius) skips
    the grasp's own surroundings -- closing on a part means entering the space it occupies."""
    for x, y, z in pts:
        if except_near is not None and math.hypot(x - except_near[0][0], y - except_near[0][1]) < except_near[1]:
            continue
        h = heights.get((int(x // cell_m), int(y // cell_m)))
        if h is not None and z < h + clearance_m:
            return False
    return True


def pose_free(views: list[Frame], heights: dict, T_base_tcp: np.ndarray, T_tcp_cam: np.ndarray, clearance_m: float,
              self_depth_m: float, arm_pts: np.ndarray | None = None) -> bool:
    """The tool (and, when given, the arm's links) at a pose stand clear of everything the cameras measured: no view saw
    a surface in front of a point, and no point is below a measured surface in its ground cell."""
    pts = tool_points(T_base_tcp, T_tcp_cam)
    if arm_pts is not None and len(arm_pts):
        pts = np.vstack([pts, arm_pts])
    if any((seen_through(v, pts, self_depth_m) == -1).any() for v in views):
        return False
    return above_surfaces(heights, pts, clearance_m)


def moved_with_gripper(before: Frame, after: Frame, tool_mask: np.ndarray, self_depth_m: float,
                       beyond_m: float = 0.12) -> dict:
    """Did the item in the gripper rise with it? Dense optical flow between the wrist images before and after the lift,
    over what a held item is: near the camera, within the gripper's own reach, and outside the gripper's own silhouette
    -- which the mount's geometry gives. (A row threshold cannot say this: where the held item falls in the frame
    depends on the camera mount, and on this one it lies in the same rows as the fingers, between them.)"""
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
