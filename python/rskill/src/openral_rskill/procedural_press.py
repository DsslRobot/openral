"""PressRskill -- press something described in words (one physical operation).

``kind: procedural``, LunarBot (research repo plan v4 §3, §4). This is the fold of Mission A's contact: what used to
be `approach_target` + `move_tool` + `grasp` called in order by the model is one call here, and the stages are the
skill's own. The caller says *what* to press; the stand-off, the speeds, how far past the face to go and how contact
is recognised are this skill's defaults, recorded in its result.

Stages (each with its own evidence):

1. locate  -- the target in the wrist camera's own view: a pixel from the vision model, its depth from the camera's
              depth image, and the point those give in the rover frame. A button is a patch on a panel, so this asks
              for the pixel rather than choosing among structures the way `pick` does (`grasp_perception.locate_point`).
2. approach-- stand off in front of it, tool facing the surface, jaw axis vertical. The target is looked at again from
              there, where it is larger and its depth better, and the stand-off corrected once.
3. press   -- a guarded straight close-in along the tool axis past the face by `press_depth_m`, stopped by contact:
              the tool stops advancing while still commanded to (the panel takes the load), which is how `place` reads
              a support taking a payload. If the whole stroke is made and nothing pushes back, that is recorded
              (`contact.resisted` false) and the operation goes on to retreat: whether the press happened is the
              independent check's to say, from where the tool went.
4. retreat -- straight back out to the stand-off.

Failures return the stage reached and its evidence. Nothing here chooses another target or moves the base.
"""

from __future__ import annotations

import numpy as np

import math

from openral_rskill._eye_in_hand import BASE_FRAME_ID, READY, TCP_FRAME_ID, EyeInHandSkill, StageFailure, mat_to_rotvec
from openral_rskill.grasp_perception import depth_at, locate_point, upright
from openral_rskill.procedural_pick import tool_in_view, tool_rotation

__all__ = ["PressRskill"]


def facing(n: np.ndarray) -> np.ndarray:
    """Tool +Z into the surface whose outward horizontal normal is `n`, jaw axis vertical (the press orientation)."""
    z = np.array([-n[0], -n[1], 0.0])
    z /= np.linalg.norm(z)
    x = np.array([0.0, 0.0, 1.0])
    return np.stack([x, np.cross(z, x), z], axis=1)


class PressRskill(EyeInHandSkill):
    def procedure(self) -> None:
        g = self.goal
        import os

        from openai import OpenAI

        import httpx

        http = httpx.Client(limits=httpx.Limits(max_connections=4, keepalive_expiry=30.0), timeout=120.0)
        self.vlm = OpenAI(api_key=os.environ["SPACE_LLM_API_KEY"], base_url=g["vlm_endpoint"], timeout=120,
                          max_retries=0, http_client=http)
        self._evidence.update(target=g["target"])
        self._tool_view = None  # where the gripper is in its own camera; read off the first frame
        try:
            self.press_it()
        finally:
            self.working_on(None)

    def face_the_work_zone(self) -> None:
        """Point the camera at the surveyed target before looking for anything.

        The caller names the site point it is pressing at; the boundary sends its surveyed position. The camera goes
        onto that point from the depth this wrist measures a part at, along the line from the rover to it, as level
        as the arm can manage where the rover stands -- the same derivation `pick` uses for its support. What this
        replaces was a written-down posture (`look_x_m` -0.55, four heights, three pitches) that the arm could not
        take at any of the press stands the boundary chose: ma5 failed `locate` twice with every one of the twelve
        out of reach, at two different stands (F94)."""
        g = self.goal
        if "feature_xyz_map" not in g:
            raise StageFailure("locate", "this press did not name the site point it is pressing at, so there is "
                                         "nothing to point the camera at: pass `feature`.", local_retry=False)
        T_bm = self.T(BASE_FRAME_ID, "map")
        at = T_bm[:3, :3] @ np.array(g["feature_xyz_map"], float) + T_bm[:3, 3]
        yaw = math.degrees(math.atan2(-at[1], -at[0]))  # the direction from the rover to the point, in tool_rotation's terms
        behind = float(np.linalg.norm(self.T(TCP_FRAME_ID, self.frame().frame_id)[:3, 3]))
        tried = []
        for back in [float(d) - behind for d in g["view_part_depth_m"]]:
            for pitch in (0.0, 10.0, 20.0, 30.0):
                R = tool_rotation(pitch, yaw, 180.0)  # camera above the tool axis, as everywhere else
                q = self.ik(at - R[:, 2] * back, R, READY)
                tried.append({"back_m": round(back, 3), "pitch": pitch, "reachable": q is not None})
                if q is None:
                    continue
                try:
                    self.plan_to(q, "locate")
                except StageFailure as exc:
                    tried[-1]["blocked"] = exc.why
                    continue
                self._evidence["faced_work_zone"] = {"at": [round(float(v), 3) for v in at], "back_m": round(back, 3),
                                                     "pitch": pitch, "tried": len(tried)}
                self.wait(0.6)
                return
        self._evidence["faced_work_zone"] = {"at": [round(float(v), 3) for v in at], "tried": tried}
        raise StageFailure("locate", f"the arm cannot put its camera on the target ({at[2]:.2f} m up, "
                                     f"{float(np.hypot(at[0], at[1])):.2f} m out) from where the rover stands: none of "
                                     f"the {len(tried)} viewing poses is in reach. The rover has to stand somewhere else.",
                           local_retry=False)

    def look_for(self, tag: str):
        """The target's point in the rover frame, as this view measures it, or a stage failure saying what was seen."""
        f = self.frame(after=self._clock() - 0.05)
        pt, rec = locate_point(self.vlm, self.goal["vlm_model"], f.bgr, self.goal["target"], f.up_cam)
        rec["image"] = self.save(f"view_{tag}.jpg", upright(f.bgr, f.up_cam))
        self._evidence.setdefault("views", []).append({"at": tag, **{k: v for k, v in rec.items() if k != "raw"}})
        if pt is None:
            raise StageFailure("locate", f"the wrist camera does not see {self.goal['target']!r}: {rec['reason']}",
                               local_retry=False)
        # The target must not be the gripper's own body in its own camera. How far out that body reaches in the camera
        # follows from the mount and the tool's dimensions (`tool_in_view`, 0.285 m, as `pick` uses it): a surface
        # measured beyond it cannot be the gripper. A depth written down instead (`min_depth_m` 0.315) refused a
        # button measured at 0.3142 m in plain view in the middle of the frame, and the gripper's silhouette mask,
        # which over-covers by design, reaches the button's lower edge (F94).
        if self._tool_view is None:
            self._tool_view = tool_in_view(f.K, np.linalg.inv(self.T(TCP_FRAME_ID, f.frame_id)), *f.depth.shape)
        d = depth_at(f.depth, *pt)
        if d is None or d <= float(self._tool_view["self_depth_m"]):
            raise StageFailure("locate", f"the pixel the vision model chose measures {d} m, within the gripper's own reach "
                                         f"in this camera ({self._tool_view['self_depth_m']} m): that is the gripper, not the target",
                               local_retry=False)
        x = (pt[0] - f.K[0, 2]) * d / f.K[0, 0]
        y = (pt[1] - f.K[1, 2]) * d / f.K[1, 1]
        self._evidence["views"][-1]["depth_m"] = round(float(d), 4)
        return f.to_base(np.array([x, y, d])), f

    def stand_in_front(self, X: np.ndarray, n: np.ndarray, R: np.ndarray, stand: float, stage: str) -> None:
        """Put the tool at the stand-off in front of the target, facing it, with the planner.

        The stand-off is offered at a few distances: the nearest one the arm can hold from where the rover is parked
        wins, and how far out that turns out to be is recorded rather than assumed."""
        tried = []
        for k in (1.0, 1.4, 0.7, 1.8):
            d = stand * k
            q = self.ik(X + n * d, R, list(self.arm_q()))
            tried.append({"stand_off_m": round(d, 3), "reachable": q is not None})
            if q is None:
                continue
            try:
                self.plan_to(q, stage)
            except StageFailure as exc:
                tried[-1]["blocked"] = exc.why
                continue
            self._evidence.setdefault("stand_offs", []).append({"used_m": round(d, 3), "tried": tried})
            self._stand = d
            self.wait(0.4)
            return
        self._evidence.setdefault("stand_offs", []).append({"used_m": None, "tried": tried})
        raise StageFailure(stage, "the arm cannot stand in front of that target from where the rover is parked",
                           local_retry=False)

    def press_it(self) -> None:
        g = self.goal
        self.stage("locate")
        self.face_the_work_zone()  # whatever posture the arm was left in, look where the rover docked first
        X, f = self.look_for("start")
        # the tool comes at the face the way the camera sees it, horizontally: a panel stands upright, and its own
        # normal is not measured -- the direction the view is taken from is what the arm can actually withdraw along
        view_from = f.T_base_cam[:3, 3]
        n = np.array([view_from[0] - X[0], view_from[1] - X[1], 0.0])
        n /= max(float(np.linalg.norm(n)), 1e-9)
        R = facing(n)
        stand = float(g["standoff_m"])

        self.stage("approach")
        self.working_on(X)
        # the planner, not a servo: a servo is a straight line from wherever the tool is, and the tool may be stowed
        # inside the hull or raised somewhere else entirely. Standing in front of the target is one reconfiguration
        # (the same reason `pick.take_standoff` gives), and it is what lets this skill be called from any posture --
        # which matters, because at a work pose close enough to reach a panel the ready posture may not fit at all,
        # and an agent that must raise the arm first has nowhere to raise it (research repo F62 §3).
        self.stand_in_front(X, n, R, stand, "approach")
        X2, _ = self.look_for("stand-off")  # nearer, the target is larger and its depth better: correct once
        moved = float(np.linalg.norm(X2 - X))
        self._evidence["correction_m"] = round(moved, 4)
        if moved > float(g["correction_done_m"]):
            X = X2
            self.working_on(X)
            self.stand_in_front(X, n, R, stand, "approach")
        self._evidence["target_rover"] = [round(float(v), 4) for v in X]

        self.stage("press")
        goal_p = X - n * float(g["press_depth_m"])
        v_in = float(g["press_speed_m_s"])
        t_start = t_cmd = self._clock()
        q_cmd = np.array(self.arm_q())
        prev, slow_since, contact, stroke_done = self.tcp()[0], None, False, False
        while not contact and not stroke_done:
            p, R_now = self.tcp()
            self.working_on(p)
            if float(np.dot(p - goal_p, -n)) >= 0.0:
                # The whole stroke, and nothing pushed back. That is a fact to record, not a failure to declare: a
                # button that gives way, or a face the simulator does not resist, ends here with the tool at its
                # full travel, and whether the press happened is for the independent check to say from where the tool
                # went and whether the lamp changed. Declaring it a miss reported "failed" for a press the evaluator
                # had counted -- PDU-2's fault was cleared and the skill said it never met the face (F94).
                stroke_done = True
                break
            now = self._clock()
            dt = max(now - t_cmd, 1e-3)
            q_cmd = self.resolved_rate_step(q_cmd, -n * v_in * 0.5, mat_to_rotvec(R @ R_now.T) * 0.5, 0.3 * dt)
            with self._cmd_lock:
                self._check_stop()
                self._joints = tuple(float(v) for v in q_cmd)
            t_cmd = now
            self.wait(0.1)
            now2 = self._clock()
            p2 = self.tcp()[0]
            rate = float(np.dot(p2 - prev, -n)) / max(now2 - now, 1e-3)  # how fast it actually advanced
            prev = p2
            if now2 - t_start > 1.5 and rate < 0.3 * v_in:  # commanded in, not advancing: the face takes the load
                slow_since = slow_since or now2
                contact = now2 - slow_since > float(g["contact_confirm_s"])
            else:
                slow_since = None
        self.hold_here()
        self.wait(0.3)
        p_contact = self.tcp()[0]
        self._evidence["contact"] = {"tcp": [round(float(v), 4) for v in p_contact], "resisted": bool(contact),
                                     "past_face_m": round(float(np.dot(X - p_contact, n)), 4),
                                     "image": self.save("contact.jpg", upright(self.frame(after=self._clock() - 0.05).bgr, None))}

        self.stage("retreat")
        # straight back out along the line it came in on: the panel's collision box stands proud of the button by more
        # than the tool went past it, so the start of this stroke is inside the box by construction (F94) -- the same
        # reason `pick`'s lift does not check the scene for the stroke that leaves what it holds
        self.servo(lambda: (X + n * self._stand, R), "retreat", tol_m=0.015, tol_rad=0.06, timeout_s=40.0, check_scene=False,
                   gain_per_s=float(g["servo_gain_per_s"]), sag_integral=False)
        self._evidence["retreated_to"] = [round(float(v), 4) for v in self.tcp()[0]]
        self._evidence["outcome"] = "pressed"
