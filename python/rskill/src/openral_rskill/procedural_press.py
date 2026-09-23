"""PressRskill -- press a push button the catalogue declares (one physical operation).

``kind: procedural``, LunarBot. The caller names the site point it is pressing at; the equipment catalogue says what
is there -- a mushroom button, a dome on a collar on a plate, with its dimensions -- and everything else is measured.
The vision model is asked once, only for which feature is meant; where the button is comes from fitting the
declared shape to the wrist depth image (`interface_fit.locate_disc`), the same way `pick` finds its handle. Every
motion is the shared servo.

1. locate   -- the button in the wrist camera's view: the fitted shape's front-most point and the plate's normal.
2. approach -- stand off in front of it, tool facing the plate, jaw axis vertical. The button is fitted again from
               there, starting from where the first fit put it and not from the vision model, and the stand-off is
               corrected once.
3. press    -- the jaws close first: a button between open fingers passes through them. Then the tool goes straight in
               along the plate's normal past the button's tip by the declared travel, the reference advancing at a
               set speed, and stops where the button takes the load or the travel is made. Whether the tool was on the
               button is recorded (how far off its axis the tool was) beside how far past its tip it went.
4. retreat  -- straight back out to the stand-off, jaws open again, and the arm folds to its travel posture.

Failures return the stage reached and its evidence. Nothing here chooses another target or moves the base.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from openral_rskill._eye_in_hand import (BASE_FRAME_ID, JAW_EMPTY_MAX_RAD, READY, STOW, TCP_FRAME_ID, EyeInHandSkill,
                                         StageFailure)
from openral_rskill.grasp_perception import locate_point, upright
from openral_rskill.interface_fit import locate_disc
from openral_rskill.procedural_pick import tool_in_view, tool_rotation

__all__ = ["PressRskill"]


#: How long the vision-language model may take to answer one view. On the paratera DeepSeek-V4.1-Flash gateway, six
#: runs at once, a pick's region choice answered in 10-115 s (median 61 s, research repo F112) and 10 of 41 requests
#: were cut by the former 120 s; the model reasons before it answers.
VLM_TIMEOUT_S = 300.0


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

        http = httpx.Client(limits=httpx.Limits(max_connections=4, keepalive_expiry=30.0), timeout=VLM_TIMEOUT_S)
        self.vlm = OpenAI(api_key=os.environ["SPACE_LLM_API_KEY"], base_url=g["vlm_endpoint"], timeout=VLM_TIMEOUT_S,
                          max_retries=0, http_client=http)
        self._evidence.update(target=g["target"])
        if not g.get("interface"):
            raise StageFailure("locate", "the catalogue declares no interface for this press point, so there is no shape to "
                                         "look for: the point needs one in the site survey", local_retry=False)
        self.iface = g["interface"]
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

    def measure(self, tag: str, near: np.ndarray | None = None):
        """The button in this view, as the declared shape fits it: its front-most point and the plate's normal in the
        rover frame. The first view asks the vision model which feature is meant; a later one starts from where the
        last measurement puts the button in this view."""
        g = self.goal
        f = self.frame(after=self._clock() - 0.05)
        if near is None:
            hint, rec = locate_point(self.vlm, g["vlm_model"], f.bgr, g["target"], f.up_cam)
            if hint is None:
                raise StageFailure("locate", f"the wrist camera does not see {g['target']!r}: {rec['reason']}", local_retry=False)
            note = {k: v for k, v in rec.items() if k != "raw"}
        else:
            p = f.T_base_cam[:3, :3].T @ (near - f.T_base_cam[:3, 3])
            hint = (f.K[0, 0] * p[0] / p[2] + f.K[0, 2], f.K[1, 1] * p[1] / p[2] + f.K[1, 2])
            note = {"hint_from": "the previous measurement, projected into this view"}
        fit = locate_disc(f.depth, f.K, self.iface, hint)
        img = f.bgr.copy()
        cv2.drawMarker(img, (int(hint[0]), int(hint[1])), (0, 0, 255), cv2.MARKER_TILTED_CROSS, 22, 2)  # where the hint was
        if fit is not None:
            u, v = f.K[0, 0] * fit.tip_cam[0] / fit.tip_cam[2] + f.K[0, 2], f.K[1, 1] * fit.tip_cam[1] / fit.tip_cam[2] + f.K[1, 2]
            cv2.drawMarker(img, (int(u), int(v)), (0, 255, 255), cv2.MARKER_CROSS, 26, 2)  # the fitted button
            cv2.putText(img, f"agree {fit.agreement:.2f}", (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        self._evidence.setdefault("views", []).append({"at": tag, "hint_px": [round(float(hint[0]), 1), round(float(hint[1]), 1)],
                                                       **note, "fit": None if fit is None else fit.as_dict(),
                                                       "image": self.save(f"view_{tag}.jpg", upright(img, f.up_cam))})
        if fit is None or fit.agreement < float(g["min_fit_agreement"]):
            raise StageFailure("locate", f"the declared button is not in the wrist view where {g['target']!r} was pointed: its shape "
                                         f"agrees with the depth over {0.0 if fit is None else fit.agreement:.0%} of its footprint",
                               local_retry=False)
        # The button must not be the gripper's own body in its own camera (how far that body reaches follows from the
        # mount and the tool's dimensions, as `pick` uses it)
        if self._tool_view is None:
            self._tool_view = tool_in_view(f.K, np.linalg.inv(self.T(TCP_FRAME_ID, f.frame_id)), *f.depth.shape)
        if fit.tip_cam[2] <= float(self._tool_view["self_depth_m"]):
            raise StageFailure("locate", f"the shape fits {fit.tip_cam[2]:.3f} m away, within the gripper's own reach in this "
                                         f"camera ({self._tool_view['self_depth_m']} m): that is the gripper", local_retry=False)
        n = f.T_base_cam[:3, :3] @ fit.normal_cam
        n[2] = 0.0  # a panel stands upright; the tool comes at it horizontally, jaw axis vertical
        if np.linalg.norm(n) < 0.5:
            raise StageFailure("locate", "the plate the button stands on faces up or down, not out: this operation presses "
                                         "horizontally", local_retry=False)
        return f.to_base(fit.tip_cam), n / np.linalg.norm(n), fit

    def stand_in_front(self, X: np.ndarray, n: np.ndarray, R: np.ndarray, stand: float, stage: str) -> None:
        """Put the tool at the stand-off in front of the target, facing it, with the planner.

        The stand-off is offered at a few distances: the nearest one the arm can hold from where the rover is parked
        wins, and how far out that turns out to be is recorded rather than assumed."""
        tried = []
        for k in (1.0, 1.4, 0.7, 1.8):
            d = stand * k
            q = self.ik(X + n * d, R, list(self.arm_q()), planned=True)
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
        X, n, _ = self.measure("start")
        R = facing(n)
        stand = float(g["standoff_m"])

        self.stage("approach")
        self.working_on(X)
        # the planner, not a servo: a servo is a straight line from wherever the tool is, and the tool may be stowed
        # inside the hull or raised somewhere else entirely; standing in front of the button is one reconfiguration
        self.stand_in_front(X, n, R, stand, "approach")
        X2, n2, _ = self.measure("stand-off", near=X)  # nearer, the shape is larger and its depth better
        moved = float(np.linalg.norm(X2 - X))
        self._evidence["correction_m"] = round(moved, 4)
        X, n, R = X2, n2, facing(n2)
        if moved > float(g["correction_done_m"]):
            self.working_on(X)
            self.stand_in_front(X, n, R, stand, "approach")
        self._evidence["target_rover"] = [round(float(v), 4) for v in X]

        # The jaws close before anything touches: with them open the button passes between the fingers and the tool
        # presses the plate beside it (mc1: jaw 0.82 rad at both presses, the tool on the plate a hand's width off).
        jaw = self.set_jaw(False, "press")
        self._evidence["jaw_closed_rad"] = round(jaw, 4)
        if jaw > JAW_EMPTY_MAX_RAD:
            raise StageFailure("press", f"the jaws did not close on nothing (jaw {jaw:.3f} rad): something is between the fingers",
                               local_retry=False)

        self.stage("press")
        goal_p = X - n * float(self.iface["travel_m"])  # the declared travel past the button's tip
        info = self.servo(lambda: (goal_p, R), "press", tol_m=0.004, tol_rad=0.05, timeout_s=60.0, stall_s=float(g["press_stall_s"]),
                          check_scene=False, sag_integral=False, posture_gain=0.0, gain_per_s=float(g["servo_gain_per_s"]),
                          advance_m_s=float(g["press_speed_m_s"]), advance_ramp_s=float(g["press_ramp_s"]), stall_returns=True)
        self.hold_here()
        self.wait(0.3)
        p, d = self.tcp()[0], self.tcp()[0] - X
        self._evidence["contact"] = {"tcp": [round(float(v), 4) for v in p], "resisted": bool(info.get("stalled")),
                                     "past_face_m": round(float(np.dot(-d, n)), 4),
                                     "off_axis_m": round(float(np.linalg.norm(d - n * np.dot(d, n))), 4),
                                     "cap_radius_m": float(self.iface["cap_radius_m"]), "jaw_rad": round(self.jaw(), 4), "servo": info,
                                     "image": self.save("contact.jpg", upright(self.frame(after=self._clock() - 0.05).bgr, None))}

        self.stage("retreat")
        # straight back out along the line it came in on; the panel's collision box stands proud of the button by more
        # than the tool went past it, so the start of this stroke is inside the box by construction (F94)
        self.servo(lambda: (X + n * stand, R), "retreat", tol_m=0.006, tol_rad=0.06, timeout_s=40.0, check_scene=False,
                   gain_per_s=float(g["servo_gain_per_s"]), sag_integral=False)
        self.set_jaw(True, "retreat")
        # the stand-off is measured from the button; a panel's collision box can stand proud of it by more than that
        # (chain_l2a_bess_r3: the BMS panel pressed and the unit restored, the arm then held 'inside the panel's
        # collision volume' at the stand-off and every later plan from there failed): back out further on the same line
        for extra in (0.03, 0.06, 0.10):
            if self.state_valid(np.array(self.arm_q())):
                break
            self.servo(lambda extra=extra: (X + n * (stand + extra), R), "retreat", tol_m=0.006, tol_rad=0.06, timeout_s=20.0,
                       check_scene=False, gain_per_s=float(g["servo_gain_per_s"]), sag_integral=False)
        self._evidence["retreated_to"] = [round(float(v), 4) for v in self.tcp()[0]]
        # An operation must not leave the arm where its own planner cannot start from (ma5r4's `stow` failed with -2 from
        # a state 5 mm inside the panel's box): the state is checked and, if it is not accepted, that is the result
        if not self.state_valid(np.array(self.arm_q())):
            raise StageFailure("retreat", "the arm is back at the stand-off but the planning scene still holds its state inside "
                                          "the panel's collision volume: it has to move further out before anything can be planned",
                               local_retry=False)
        # ...and must not leave it out at the stand-off either: the rover drives next, and the next job's planner starts
        # wherever the arm was left. chain_l2a_bess_r2 pressed the HVAC shutter button, drove to the BMS panel with the
        # arm still out at the button, and every view there was "no collision-free path" (planner -2) from that posture,
        # 3/3; chain_l2a_pdu, with the arm left at the ready posture, the same at PDU-1's panel, 3/3. Every press that
        # found its view started folded: the arm goes back to the travel posture.
        self.plan_to(list(STOW), "retreat")
        self._evidence["outcome"] = "pressed"
