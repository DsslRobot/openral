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
import math
import os
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

from openral_rskill._eye_in_hand import (READY, JAW_EMPTY_MAX_RAD, EyeInHandSkill, Frame, StageFailure, mat_to_rotvec)
from openral_rskill.grasp_perception import (Candidate, GripperGeometry, PartTracker, find_candidates, render_candidates,
                                             select_candidate, upright)

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


def guard_or_empty(skill, p: np.ndarray, R: np.ndarray, state: dict) -> str:
    """Why the tool's own body cannot be at this pose (a surface it would touch), or "" when it can. The part being
    grasped is excluded: that is what the fingers are closing on."""
    pts = (R @ BODY_POINTS.T).T + p
    worst = lowest_clearance(skill._heights, pts, except_near=(state["p_base"], float(skill.goal["grasp_clear_radius_m"])))
    return "" if worst[0] >= float(skill.goal["clearance_m"]) else f"a surface at {worst[2]} m under {worst[1]}"


def across_tool_axis(v_tool: np.ndarray) -> np.ndarray:
    """The part of a vector in the tool frame that is across the tool axis -- the part a roll about that axis turns.
    Taken in the tool frame, not by projecting in the base frame: the camera looks along neither the tool axis nor the
    grasp axis, so projecting out a viewing direction cancels exactly this component (research repo F57)."""
    return np.array([v_tool[0], v_tool[1], 0.0], float)


def free_side(frame: Frame, point_b: np.ndarray, part_axis_b: np.ndarray, self_depth_m: float, step_m: float = 0.06) -> np.ndarray:
    """Which way along the part is clearer, as this view measured it: the side whose sample sees no surface in front of
    it. A handle on top of a box is clear above and blocked below; a bar under a ledge the other way round."""
    axis = part_axis_b / max(float(np.linalg.norm(part_axis_b)), 1e-9)
    votes = [(float(seen_through(frame, np.array([point_b + axis * step_m * s]), self_depth_m)[0]), s) for s in (1.0, -1.0)]
    votes.sort(reverse=True)
    return axis * votes[0][1]


class PickRskill(EyeInHandSkill):
    def procedure(self) -> None:
        g = self.goal
        from openai import OpenAI

        self.vlm = OpenAI(api_key=os.environ["SPACE_LLM_API_KEY"], base_url=g["vlm_endpoint"], timeout=600)
        self.geom = GripperGeometry()
        self._evidence.update(target=g["target"], part=g["part"], attempts=[])
        self.stage("prepare")
        self.set_jaw(True, "prepare")
        tried: list[np.ndarray] = []  # grasps already tried, where they were seen in the rover frame at the time
        for attempt in range(int(g["max_attempts"])):
            cand, choice = self.find_and_select(tried)
            tried.append(self._locked_frame.to_base(cand.p_cam))
            rec = {"attempt": attempt + 1, "candidate": cand.as_dict(), "vlm": choice}
            self._evidence["attempts"].append(rec)
            try:
                self.approach(cand)
                jaw = self.grasp()
                self.hold(jaw)
                rec["outcome"] = "held"
                self._evidence["outcome"] = "held"
                return
            except StageFailure as exc:
                rec["outcome"] = f"failed at {exc.stage}: {exc.why}"
                if exc.stage not in ("approach", "grasp", "hold") or attempt + 1 == int(g["max_attempts"]):
                    raise
                self.back_off()
        raise StageFailure("hold", "attempts exhausted")

    def vlm_call(self, stage: str, fn):
        import openai

        try:
            return fn()
        except openai.APIError as exc:
            raise StageFailure(stage, f"the vision-language model ({self.goal['vlm_model']}) did not answer: {type(exc).__name__}") from exc

    # ---- find + select --------------------------------------------------------------------------------------------------
    def find_and_select(self, tried: list[np.ndarray]) -> tuple[Candidate, dict]:
        """Side views of the arm's reach zone from the top of the grasp band downwards. Each view pose, and the joint path
        to it, must lie in space the views already taken saw through (the first is above the band). In each view the
        grasp candidates go to the vision-language model; the first view in which it chooses a grasp on the described
        part ends the search."""
        g = self.goal
        self.stage("find")
        # an overview from above the work area: the item's identity for the vision model (the side views show mostly
        # the part), and the first free-space evidence for the side-view paths
        x, y, z, pitch, yaw = g["overview_pose"]
        q_over = self.ik(np.array([x, y, z]), tool_rotation(pitch, yaw, 180.0), READY)  # camera above the tool axis
        if q_over is None:
            raise StageFailure("find", "the overview posture is not reachable")
        self.plan_to(q_over, "find")
        self.wait(0.8)
        over = self.frame(after=self._clock() - 0.05)
        overview = upright(over.bgr, over.up_cam)
        record = {"views": [], "overview": self.save("overview.jpg", overview)}
        self._evidence["find"] = record
        self._heights: dict = {}
        seen: list[Frame] = [over]
        T_tc = self.T("tcp_frame", over.frame_id)  # the camera in the tool frame (robot geometry)
        self._cam_tool = T_tc[:3, 3].copy()  # which side of the tool axis the camera sits on, from the mount itself
        # the gripper's own place in its camera, from that mount: its depth and the image band it leaves free
        self._tool_view = tool_in_view(over.K, np.linalg.inv(T_tc), over.depth.shape[0])
        self._self_depth = float(self._tool_view["self_depth_m"])
        self.geom = dataclasses.replace(self.geom, self_depth_m=self._self_depth,
                                        min_part_depth_m=self._self_depth + 0.03)
        record["tool_in_view"] = self._tool_view
        next_id = 1
        for h in g["side_view_heights_m"]:
            view = {"height_m": h}
            record["views"].append(view)
            P = np.array([g["reach_zone_x_m"], 0.0, h])
            pose = self.side_view_pose(P, T_tc, over.K, seen)
            if pose is None:
                view["reachable"] = False
                continue
            q, info = pose
            view.update(info)
            self.plan_to(q, "find")
            self.wait(0.8)
            f = self.frame(after=self._clock() - 0.05)
            f, framing = self.frame_part(f, q, info)  # what is in the view decides the framing, not the model of the camera
            view["framing"] = framing
            seen.append(f)
            shots = [(info, f, q)]
            rolled = self.rolled_view(info, P, T_tc, over.K, seen)  # the same place seen from the other side of the tool
            if rolled is not None:
                shots.append(rolled)
                seen.append(rolled[1])
            self._heights = surface_heights(seen, self._self_depth)
            np.save(self.evidence_dir / f"view_h{h:.2f}_depth.npy", f.depth)
            shot_cands, marked_images = [], []
            for shot_info, shot, shot_q in shots:
                cands = [c for c in find_candidates(shot.depth, shot.K, self.geom, max_candidates=int(g["candidates_per_view"]))
                         if all(np.linalg.norm(shot.to_base(c.p_cam) - t) > 0.02 for t in tried)]
                for c in cands:
                    c.id, next_id = next_id, next_id + 1
                tag = f"h{h:.2f}" + ("_rolled" if shot_info.get("roll") else "")
                self.save(f"view_{tag}.jpg", upright(shot.bgr, shot.up_cam))
                if cands:
                    marked = render_candidates(shot.bgr, cands, shot.up_cam)
                    marked_images.append(self.save(f"candidates_{tag}.jpg", marked))
                    shot_cands.append((shot, cands, marked, shot_q))
            view["candidates"] = sum(len(c) for _, c, _, _ in shot_cands)
            view["marked"] = marked_images
            if not shot_cands:
                continue
            all_cands = [c for _, cs, _, _ in shot_cands for c in cs]
            self.stage("select", height_m=h, candidates=len(all_cands))
            asks = int(g["vlm_asks"])
            with ThreadPoolExecutor(asks) as pool:
                answers = list(pool.map(lambda _: self.vlm_call("select", lambda: select_candidate(
                    self.vlm, g["vlm_model"], overview, [m for _, _, m, _ in shot_cands], g["target"], g["part"],
                    {c.id for c in all_cands}, g["handle_convention"])), range(asks)))
            f, cands, shot_q = shot_cands[0][0], shot_cands[0][1], shot_cands[0][3]
            a = majority(answers, all_cands, f)
            if a["choice"] is not None:
                f, cands, _, shot_q = next(sh for sh in shot_cands if any(c.id == a["choice"] for c in sh[1]))
            view["vlm"] = {k: a[k] for k in ("item_visible", "what_is_visible", "choice", "reason")} | {
                "answers": [{k: x[k] for k in ("item_visible", "choice", "reason")} for x in answers]}
            if a["item_visible"] and a["choice"] is not None:
                # stand where that image was taken: the tracker follows the part from the same side of the tool
                if float(np.max(np.abs(np.array(self.arm_q()) - np.array(shot_q)))) > 0.03:
                    self.plan_to(shot_q, "select")
                    self.wait(0.5)
                self._locked_frame, self._view_q = f, shot_q
                # keep the camera on the side of the tool it looked from: the tracker's template is that view's image,
                # and a roll of the tool turns that template upside down in the following frames
                self._view_R = self.fk(np.array(shot_q))[:3, :3]
                return next(c for c in cands if c.id == a["choice"]), view["vlm"]
            if any(x["item_visible"] and x["choice"] is not None for x in answers):
                # the item and a grasp on it are in this view, but the answers do not agree where: looking lower will not help
                raise StageFailure("select", "the vision model's answers did not agree on a grasp on the described part")
        saw = [v["vlm"]["what_is_visible"] for v in record["views"] if v.get("vlm")]
        raise StageFailure("select" if saw else "find", "no grasp on the described part of the described item in the views of the reach zone"
                           + (f" (the vision model saw: {saw[0]})" if saw else ""))

    def rolled_view(self, info: dict, P: np.ndarray, T_tc: np.ndarray, K: np.ndarray, seen: list[Frame]):
        """The same place seen from the other side of the tool: the fingers hide the opposite half of the image there, so
        what one view hides (an overhanging head) the other shows."""
        R = tool_rotation(info["pitch"], 0.0, 180.0 if not info.get("roll") else 0.0)
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = R, P + np.array([info["back"], 0.0, info["rise"]])
        if not pose_free(seen, self._heights, T, T_tc, float(self.goal["clearance_m"]), self._self_depth):
            return None
        q = self.ik(T[:3, 3], R, READY)
        if q is None:
            return None
        self.plan_to(q, "find")
        self.wait(0.8)
        other = dict(info, roll=0.0 if info.get("roll") else 180.0)
        f, framing = self.frame_part(self.frame(after=self._clock() - 0.05), q, other)
        return dict(other, framing=framing), f, q

    def frame_part(self, f: Frame, q: np.ndarray, info: dict) -> tuple[Frame, dict]:
        """Keep what the view shows clear of the fingers: the graspable parts must sit in the image band the fingers do
        not cover, with room above them for whatever overhangs them. When they sit too near that edge, the tool moves up
        or down by the measured offset and the view is taken again."""
        g = self.goal
        cands = find_candidates(f.depth, f.K, self.geom, max_candidates=int(g["candidates_per_view"]))
        if not cands:
            return f, {"candidates": 0}
        v = float(np.median([c.v for c in cands]))
        depth = float(np.median([c.depth_m for c in cands]))
        rows = int(g.get("image_rows_px", 480))
        rolled = bool(info.get("roll"))
        # the gripper always sits in the same place in its own camera (it is bolted to it), so the row the part should
        # sit in does not depend on the tool's roll. The roll does flip which way the image moves when the tool moves.
        target = float(self._tool_view["part_row"])
        shift = (v - target) * depth / f.K[1, 1] * (-1.0 if rolled else 1.0)
        out = {"part_row_px": round(v), "shift_m": round(shift, 3)}
        if abs(shift) < float(g["framing_tolerance_m"]):
            return f, out
        p, R = self.tcp()
        q2 = self.ik(p + np.array([0.0, 0.0, float(np.clip(shift, -0.2, 0.2))]), R, list(q))
        if q2 is None:
            return f, {**out, "moved": False}
        self.plan_to(q2, "find")
        self.wait(0.8)
        return self.frame(after=self._clock() - 0.05), {**out, "moved": True}

    def side_view_pose(self, P: np.ndarray, T_tc: np.ndarray, K: np.ndarray, seen: list[Frame]):
        """The arm pose on its working branch whose wrist camera looks at P from the rover's side, P nearest the image
        centre, with the tool -- there and along the joint path to it -- in space the earlier views saw through."""
        g = self.goal
        options = []
        rows = int(self.goal.get("image_rows_px", 480))
        for back in g["side_view_back_m"]:
            for rise in g["side_view_rise_m"]:
                for pitch in g["side_view_pitch_deg"]:
                    # the camera above the tool axis: payloads stand on their supports and are taken from above, so
                    # this is the side the fingers will come from and the side the grasp will be seen from. The
                    # opposite side is the second shot (`rolled_view`), for a part under an overhang.
                    R = tool_rotation(pitch, 0.0, 180.0)
                    T = np.eye(4)
                    T[:3, :3], T[:3, 3] = R, P + np.array([back, 0.0, rise])
                    pc = np.linalg.inv(T @ T_tc) @ np.append(P, 1.0)
                    if pc[2] < 0.25:
                        continue
                    u, v = K[0, 0] * pc[0] / pc[2] + K[0, 2], K[1, 1] * pc[1] / pc[2] + K[1, 2]
                    # the part in the middle of the image band the gripper does not hide -- the same band in every
                    # frame, whatever the tool's roll, because the camera is bolted to the gripper
                    want = float(self._tool_view["centre_row"])
                    options.append((math.hypot(u - K[0, 2], v - want), T, dict(back=back, rise=rise, pitch=pitch, roll=180.0)))
        q_from = np.array(self.arm_q())
        heights = surface_heights(seen, self._self_depth)
        clear = float(g["clearance_m"])
        for off, T, info in sorted(options, key=lambda o: o[0]):
            q = self.ik(T[:3, 3], T[:3, :3], READY)
            if q is None:
                continue
            # the pose and the joint path to it, tool and arm links, clear of every surface the views measured
            path = [(T, self.fk(q, self.ARM_LINKS)[1])]
            for k in range(1, 6):
                Tk, links = self.fk(q_from + (np.array(q) - q_from) * k / 6, self.ARM_LINKS)
                path.append((Tk, links))
            if all(pose_free(seen, heights, Tk, T_tc, clear, self._self_depth, links) for Tk, links in path):
                return q, info
        return None

    # ---- approach -----------------------------------------------------------------------------------------------------
    def approach(self, cand: Candidate) -> None:
        """Visual servo onto the chosen grasp: every camera frame measures where the part is, the goal follows it, and the
        tool closes in until the part sits between the fingers. The part is tracked by its own image template (the
        generic detector re-run per frame snaps onto other structures once the fingers cover it, research repo F57); when
        the fingers finally hide it, the last measurement holds the goal for the last centimetres."""
        g = self.goal
        f = self._locked_frame
        self.stage("approach", candidate=cand.id)
        state = {"p_base": f.to_base(cand.p_cam), "axis_base": f.T_base_cam[:3, :3] @ np.array(cand.axis_cam),
                 "part_base": f.T_base_cam[:3, :3] @ np.array(cand.part_axis_cam), "view_base": f.T_base_cam[:3, 3],
                 "seen": f.stamp, "hits": 0, "misses": 0, "frames": [], "trace": []}
        self._evidence["approach_track"] = state["trace"]  # the same list: a failed approach still returns its pictures
        # the camera looks from the side the part is free on, so the fingers close from the side with less material
        # (a handle on a box: from above, not through the box). The tracker carries the roll this costs, frame by frame.
        side = free_side(f, state["p_base"], state["part_base"], self._self_depth)
        state["camera_side"] = side / max(float(np.linalg.norm(side)), 1e-9)
        self._evidence["camera_side"] = [round(float(v), 2) for v in state["camera_side"]]
        approach_from = float(g["standoff_m"][0] if isinstance(g["standoff_m"], list) else g["standoff_m"])

        def guard_pose(p, R):
            return guard_or_empty(self, p, R, state)

        # stand in front of the grasp first, in the orientation the grasp needs -- with the planner, from the view's own
        # measurement -- and only then close the loop (see take_standoff)
        tracker = self.take_standoff(cand, state, approach_from)
        if tracker is None:
            tracker = PartTracker(f.bgr, f.depth, f.K, cand, R_base_cam=f.T_base_cam[:3, :3])

        def goal_now():
            """The tool pose that puts the tracked part between the fingers, from the newest frame. Once the gripper's
            own body covers the part -- it always does, in the last stretch, since the camera looks past the fingers --
            the goal holds its last measured place and the tool closes in straight along its axis."""
            if state.get("covered") or float(np.linalg.norm(state["p_base"] - self.tcp()[0])) < float(g["blind_ok_m"]):
                view = state["p_base"] - state["view_base"]
                R = grasp_rotation(state["axis_base"], state["part_base"], view / max(float(np.linalg.norm(view)), 1e-9),
                                   self.tcp()[1], state["camera_side"], self._cam_tool)
                return state["p_base"] - R[:, 2] * state["standoff"], R
            nf = self.frame(after=state["seen"], timeout_s=2.0)
            state["seen"] = nf.stamp
            p_pred_cam = nf.T_base_cam[:3, :3].T @ (state["p_base"] - nf.T_base_cam[:3, 3])
            m = tracker.measure(nf.bgr, nf.depth, p_pred_cam, R_base_cam=nf.T_base_cam[:3, :3],
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
                state["p_base"] = nf.to_base(tracker.point_cam(pixel, d) + np.array([0.0, 0.0, tracker.width_m / 2]))
                state["view_base"] = nf.T_base_cam[:3, 3]
                state["hits"] += 1
            elif not state.get("covered"):
                state["misses"] += 1
                # lost while still far from the part: the goal must not be carried on blind -- go back and identify again
                if float(np.linalg.norm(state["p_base"] - self.tcp()[0])) > float(self.goal["blind_ok_m"]) \
                        and state["misses"] > int(self.goal["max_track_misses"]):
                    raise StageFailure("approach", f"lost sight of the chosen grasp {state['misses']} frames running "
                                                   f"while still {float(np.linalg.norm(state['p_base'] - self.tcp()[0])) * 100:.0f} cm away")
            # the tracker's own view, so a failed approach can be read off the pictures: where the part was measured
            # (cross) or last seen, the match score, and the distance still to go
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
                                  f"d {tracker.depth:.3f} roll {tracker.roll_deg:+.0f} left {left * 100:.1f}cm",
                            (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                if tracker.reject:
                    cv2.putText(mark, tracker.reject, (6, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                path = self.save(f"servo_{len(state['frames']):02d}.jpg", mark)
                state["frames"].append((nf.stamp, path))
                state["trace"].append({"image": path, "hit": m is not None, "score": round(tracker.last_score, 3),
                                       "depth_m": round(tracker.depth, 4), "roll_deg": round(tracker.roll_deg, 1),
                                       "left_m": round(left, 4), "why": tracker.reject})
            view = state["p_base"] - state["view_base"]
            R = grasp_rotation(state["axis_base"], state["part_base"], view / max(float(np.linalg.norm(view)), 1e-9),
                               self.tcp()[1], state["camera_side"], self._cam_tool)
            return state["p_base"] - R[:, 2] * state["standoff"], R

        # close in under vision: the stand-off first, then the grasp point itself
        state["standoff"] = approach_from
        info = self.servo_twist(goal_now, "approach", tol_m=0.006, tol_rad=0.04, max_speed_m_s=float(g["approach_speed_m_s"]),
                                timeout_s=float(g["approach_timeout_s"]), stall_s=6.0, guard=guard_pose)
        self._evidence["standoff"] = {**info, "tracked_frames": state["hits"], "missed_frames": state["misses"],
                                      "image": self.save("standoff.jpg", upright(self.frame(after=self._clock() - 0.05).bgr, None))}
        state["standoff"] = -float(g["pad_offset_m"])
        self.stage("approach", close_in=True)
        try:
            self.servo_twist(goal_now, "approach", tol_m=0.005, tol_rad=0.04, max_speed_m_s=float(g["close_in_speed_m_s"]),
                             timeout_s=60.0, stall_s=6.0, guard=guard_pose)
        except StageFailure as exc:
            p1, _ = self.tcp()
            short = float(np.linalg.norm(state["p_base"] - p1))
            if short > float(g["close_in_short_ok_m"]) + float(g["pad_offset_m"]):
                raise StageFailure("approach", f"the close-in stopped {short * 100:.1f} cm from the grasp: {exc.why}") from exc
        p1, _ = self.tcp()
        self._evidence["close_in"] = {"remaining_m": round(float(np.linalg.norm(state["p_base"] - p1)), 4),
                                      "tracked_frames": state["hits"], "missed_frames": state["misses"],
                                      "track": state.get("trace", [])}
        self._grasp_point = state["p_base"]

    def take_standoff(self, cand: Candidate, state: dict, standoff_m: float) -> PartTracker | None:
        """Stand the tool in front of the chosen grasp, in the orientation the grasp needs, with the planner -- one
        reconfiguration, not a servo motion: the view is taken from wherever frames the part, the grasp comes from the
        side the part is free on, and between the two the wrist can turn half round. Turning it where the tool stands is
        not enough either: the camera looks 17 degrees off the tool axis, so half a turn swings its view by 34 degrees
        and the part leaves the image (research repo F57). Standing at the stand-off instead puts the part back on the
        tool axis, where the camera sees it and the servo can close the last stretch.

        Returns the part's tracker as seen from there, or None when the pose is not reachable or not clear, in which
        case the caller servos from where it is."""
        p, R_now = self.tcp()
        view = state["p_base"] - state["view_base"]
        R_g = grasp_rotation(state["axis_base"], state["part_base"], view / max(float(np.linalg.norm(view)), 1e-9),
                             R_now, state["camera_side"], self._cam_tool)
        p_g = state["p_base"] - R_g[:, 2] * standoff_m
        rec = {"turn_deg": round(math.degrees(float(np.linalg.norm(mat_to_rotvec(R_g @ R_now.T)))), 1),
               "stand_off_m": round(standoff_m, 3), "goal": [round(float(v), 3) for v in p_g]}
        self._evidence["take_standoff"] = rec
        q = self.ik(p_g, R_g, list(self.arm_q()))
        blocked = None if q is None else guard_or_empty(self, p_g, R_g, state)
        rec.update(reachable=q is not None, blocked=blocked)
        if q is None or blocked:
            return None  # the servo does what it can from here; the guard still holds
        self.plan_to(q, "approach")
        self.wait(0.6)
        nf = self.frame(after=self._clock() - 0.05)
        state["seen"], state["view_base"] = nf.stamp, nf.T_base_cam[:3, 3]
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
        return PartTracker(nf.bgr, nf.depth, nf.K, seen, R_base_cam=R_c)

    # ---- grasp + hold -----------------------------------------------------------------------------------------------------
    def grasp(self) -> float:
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
        try:
            self.servo_twist(lambda: (p0 + lift_v, R0), "hold", tol_m=0.01, tol_rad=0.06, max_speed_m_s=float(g["lift_speed_m_s"]),
                             timeout_s=90.0, stall_s=12.0)
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
        flow = moved_with_gripper(before, after, int(self._tool_view["tool_top_row"]))
        np.save(self.evidence_dir / "hold_before_depth.npy", before.depth)
        self._evidence["hold_check"] = {"jaw_rad": round(jaw, 4), "lift_m": round(float(p1[2] - p0[2]), 4), **flow,
                                         "before": self.save("hold_before.png", before.bgr), "after": self.save("hold_after.png", after.bgr),
                                         "before_depth": str(self.evidence_dir / "hold_before_depth.npy"), "fx": float(before.K[0, 0]),
                                         "item_rows_px": int(self._tool_view["tool_top_row"])}
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

    def back_off(self) -> None:
        """Local retry: open, a short straight retreat along the tool axis, then back to the side-view posture."""
        self.stage("back_off")
        self.set_jaw(True, "back_off")
        p, R = self.tcp()
        self.servo_twist(lambda: (p - R[:, 2] * 0.05, R), "back_off", tol_m=0.01, tol_rad=0.08, timeout_s=20.0)
        self.plan_to(self._view_q, "back_off")


#: the tool's own body and the outsides of its fingers, in the tool frame (x jaw axis, y up in the grip orientation,
#: z forward): what must not touch an item. The space between the fingers is left out -- that is where the part goes.
BODY_POINTS = np.array([[x, y, z] for x in (-0.055, 0.055) for y in (-0.03, 0.0, 0.03) for z in (0.01, -0.03, -0.08)]
                       + [[0.0, y, z] for y in (-0.05, 0.05) for z in (-0.03, -0.08)])

#: points of the tool in its own frame (x jaw axis, y up in the grip orientation, z forward): the fingertips and
#: pads, the jaw housing, the wrist camera
TOOL_POINTS = np.array([[x, y, z] for x in (-0.045, 0.0, 0.045) for y in (-0.03, 0.0, 0.03) for z in (0.01, -0.03, -0.08)])
def majority(answers: list[dict], cands: list[Candidate], f: Frame, same_place_m: float = 0.03) -> dict:
    """What most of the concurrent vision-language answers agree on. Marks less than `same_place_m` apart are the same
    place on the part (several candidates lie along one neck), so answers choosing any of them agree. The item counts as
    visible, and a place as chosen, only when more than half the answers say so; the chosen mark is the one picked most
    within the agreeing place."""
    n = len(answers)
    visible = [a for a in answers if a["item_visible"]]
    by_id = {c.id: f.to_base(c.p_cam) for c in cands}
    chosen = [a["choice"] for a in visible if a["choice"] is not None]
    best: list[int] = []
    for c in chosen:
        near = [d for d in chosen if np.linalg.norm(by_id[d] - by_id[c]) <= same_place_m]
        if len(near) > len(best):
            best = near
    if len(visible) * 2 <= n or len(best) * 2 <= n:
        base = visible[0] if visible else answers[0]
        return base | {"item_visible": len(visible) * 2 > n, "choice": None, "agreed": False}
    top = max(set(best), key=best.count)
    agree = next(a for a in visible if a["choice"] == top)
    return agree | {"agreed": True}


def tool_points(T_base_tcp: np.ndarray, T_tcp_cam: np.ndarray) -> np.ndarray:
    return (T_base_tcp[:3, :3] @ np.vstack([TOOL_POINTS, T_tcp_cam[:3, 3]]).T).T + T_base_tcp[:3, 3]


def tool_in_view(K: np.ndarray, T_ct: np.ndarray, rows: int, margin_m: float = 0.02) -> dict:
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
    return {"self_depth_m": round(self_depth, 3), "tool_top_row": top, "clear_rows": top,
            "centre_row": int(top * 0.5), "part_row": int(top * 0.75)}


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
                     except_near: tuple[np.ndarray, float] | None = None):
    """The point standing least clear of what the cameras measured under it: (clearance, point, surface height)."""
    worst = (9.9, None, None)
    for x, y, z in pts:
        if except_near is not None and math.hypot(x - except_near[0][0], y - except_near[0][1]) < except_near[1]:
            continue
        h = heights.get((int(x // cell_m), int(y // cell_m)))
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


def moved_with_gripper(before: Frame, after: Frame, item_rows_px: int = 216) -> dict:
    """Did the item in the gripper rise with it? Dense optical flow between the wrist images before and after the lift,
    over what is close in front of the camera and is not the fingers themselves (a carried item hangs right in front of
    them): a held item stays put in the image, an item left behind shifts by the lift's image displacement."""
    g0, g1 = (cv2.cvtColor(f.bgr, cv2.COLOR_BGR2GRAY) for f in (before, after))
    flow = cv2.calcOpticalFlowFarneback(g0, g1, None, 0.5, 4, 31, 5, 7, 1.5, 0)
    z = before.depth
    near = np.isfinite(z) & (z > 0.05) & (z < 0.8)
    near[item_rows_px:, :] = False  # the fingers fill the rest of the frame and never move in it: only what they hold counts
    if near.sum() < 200:
        return {"item_moved_with_gripper": False, "median_flow_px": None, "expected_flow_px": None, "near_pixels": int(near.sum())}
    mag = np.linalg.norm(flow[near], axis=1)
    lift = float(np.linalg.norm(after.T_base_cam[:3, 3] - before.T_base_cam[:3, 3]))
    expected = before.K[1, 1] * lift / float(np.median(z[near]))
    med = float(np.median(mag))
    return {"item_moved_with_gripper": bool(med < 0.35 * expected), "median_flow_px": round(float(med), 1),
            "expected_flow_px": round(float(expected), 1), "near_pixels": int(near.sum())}

