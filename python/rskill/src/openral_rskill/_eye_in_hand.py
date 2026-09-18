"""Shared machinery for LunarBot's eye-in-hand manipulation rSkills (``procedural_pick``, ``procedural_place``).

A manipulation skill is one physical operation with an internal procedure. The procedure runs as plain sequential code
in a worker thread; the runner's ``step()`` only pumps the worker's current command (arm joint targets, jaw)
to the HAL at the control rate and turns the worker's end into the goal's outcome. This keeps the stages readable and
keeps slow perception (a VLM answer takes seconds) off the control tick.

Pieces here: the wrist RGB-D source on the runner node, TF helpers, a proportional TCP servo towards a goal that may be
re-measured every cycle (visual servoing), joint moves, the jaw, and the staged failure that returns control to the
caller with the stage reached and its evidence.
"""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
from openral_core.exceptions import ROSRskillGoalSatisfied, ROSRuntimeError
from openral_core.schemas import Action, ControlMode

from openral_rskill._lunar_bot_arm import (
    ARM_JOINT_NAMES,
    BASE_FRAME_ID,
    GRIPPER_JOINT_NAME,
    TCP_FRAME_ID,
    full_width_joint_row,
    joint_positions_by_name,
)
from openral_rskill.base import rSkillBase

#: READY posture of the RM-75 (the capability boundary's `arm_ready`): tool behind the rover, clear of the hull.
#: The arm's seed configuration for inverse kinematics and the posture the views start from. The wrist roll is half a
#: turn, which puts the wrist camera *above* the tool axis: payloads stand on supports and are taken from above, so the
#: fingers come from above and the camera must look from there -- the branch with the camera under the tool axis makes
#: every view and every grasp differ by half a turn of the wrist (research repo F57).
READY = (0.0, -1.0, 0.0, -1.4, 0.0, 1.2, math.pi)
#: RM-75 joint limits, joint1..joint7 (docs/lunar_bot_rm75_spec.md in the research repo)
JOINT_LIMITS_RAD = (3.107, 2.269, 3.107, 2.356, 3.107, 2.234, 6.283)
#: The RM-75's working branch (joint: centre, half-range, rad): shoulder and forearm roll near zero, wrist roll unflipped.
#: Unconstrained IK on this 7-DoF arm lands on arbitrary null-space branches (shoulder turned 90 deg, wrist rolled to its
#: stop) from which the next small motion is impossible (research repo F57); the boundary's MoveIt posture preference is
#: the same idea.
#: The wrist roll (joint7) is not in here: the tool rotation asked for determines it, and holding it near zero excluded
#: every pose with the camera above the tool axis (half a turn, 3.14 rad) and made the servo's null space unroll the
#: wrist it had just turned (research repo F57).
POSTURE = {"joint1": (0.0, 1.5), "joint2": (-0.4, 0.9), "joint3": (0.0, 1.2), "joint4": (-1.5, 0.85),
           "joint5": (0.0, 1.6)}
JAW_OPEN_MIN_RAD = 0.78
JAW_EMPTY_MAX_RAD = 0.13


class StageFailure(Exception):
    """A stage could not reach its end condition; `stage` and `why` go back to the caller. `local_retry` is False when
    trying the same thing again cannot help -- the skill may only retry on the same target, so a grasp the arm cannot
    stand in front of from where the rover is belongs to the caller, not to another attempt."""

    def __init__(self, stage: str, why: str, local_retry: bool = True) -> None:
        super().__init__(f"{stage}: {why}")
        self.stage, self.why, self.local_retry = stage, why, local_retry


def quat_to_mat(x: float, y: float, z: float, w: float) -> np.ndarray:
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def rotvec_to_mat(v: np.ndarray) -> np.ndarray:
    ang = float(np.linalg.norm(v))
    if ang < 1e-9:
        return np.eye(3)
    k = v / ang
    Kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(ang) * Kx + (1 - math.cos(ang)) * Kx @ Kx


def mat_to_quat(R: np.ndarray) -> list[float]:
    t = np.trace(R)
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        return [(R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s, 0.25 * s]
    i = int(np.argmax(np.diag(R)))
    j, k = (i + 1) % 3, (i + 2) % 3
    s = math.sqrt(1.0 + R[i, i] - R[j, j] - R[k, k]) * 2
    q = [0.0, 0.0, 0.0, 0.0]
    q[i], q[j], q[k], q[3] = 0.25 * s, (R[j, i] + R[i, j]) / s, (R[k, i] + R[i, k]) / s, (R[k, j] - R[j, k]) / s
    return q


def mat_to_rotvec(R: np.ndarray) -> np.ndarray:
    """Axis-angle vector of a rotation matrix."""
    c = max(-1.0, min(1.0, (np.trace(R) - 1.0) / 2.0))
    ang = math.acos(c)
    if ang < 1e-9:
        return np.zeros(3)
    if ang > math.pi - 1e-6:
        w, v = np.linalg.eigh((R + np.eye(3)) / 2.0)
        return v[:, np.argmax(w)] * ang
    return np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) * ang / (2 * math.sin(ang))


@dataclass
class Frame:
    bgr: np.ndarray
    depth: np.ndarray
    K: np.ndarray
    stamp: float
    T_base_cam: np.ndarray  # 4x4, camera optical frame in chassis_base_link at grab time
    frame_id: str = ""
    arm_links: np.ndarray | None = None  # where the arm's own links were when the frame was taken (self-filtering)

    @property
    def up_cam(self) -> np.ndarray:
        return self.T_base_cam[:3, :3].T @ np.array([0.0, 0.0, 1.0])

    def to_base(self, p_cam) -> np.ndarray:
        return self.T_base_cam[:3, :3] @ np.asarray(p_cam) + self.T_base_cam[:3, 3]


class WristRGBD:
    """The wrist D435i's colour + registered depth, subscribed on the runner node."""

    def __init__(self, node: Any, camera: str = "wrist") -> None:
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import CameraInfo, Image

        self._lock = threading.Lock()
        self._rgb: dict[float, Any] = {}
        self._depth: dict[float, Any] = {}
        self._info = None
        stamp = lambda m: m.header.stamp.sec + m.header.stamp.nanosec * 1e-9  # noqa: E731
        self._subs = [
            node.create_subscription(Image, f"/openral/cameras/{camera}/image", lambda m: self._put(self._rgb, stamp(m), m), qos_profile_sensor_data),
            node.create_subscription(Image, f"/openral/cameras/{camera}_depth/image", lambda m: self._put(self._depth, stamp(m), m), qos_profile_sensor_data),
            node.create_subscription(CameraInfo, f"/openral/cameras/{camera}/camera_info", lambda m: setattr(self, "_info", m), qos_profile_sensor_data),
        ]
        self._node = node

    def _put(self, store: dict, t: float, m: Any) -> None:
        with self._lock:
            store[t] = m
            for k in sorted(store)[:-4]:
                del store[k]

    def close(self) -> None:
        for s in self._subs:
            self._node.destroy_subscription(s)

    def latest_pair(self, after: float) -> tuple[float, Any, Any] | None:
        """The newest depth image stamped after `after`, with the colour image nearest to it in time. The two streams
        reach a subscriber at a few hertz and rarely with equal stamps; geometry comes from the depth, the colour image
        is for the vision model and the evidence."""
        with self._lock:
            if not self._depth or not self._rgb or self._info is None:
                return None
            t = max(self._depth)
            if t <= after:
                return None
            t_rgb = min(self._rgb, key=lambda k: abs(k - t))
            return t, self._rgb[t_rgb], self._depth[t]


def image_to_bgr(m: Any) -> np.ndarray:
    img = np.frombuffer(m.data, np.uint8).reshape(m.height, m.width, -1)
    if m.encoding == "bgr8":
        return img.copy()
    return cv2.cvtColor(img, {"rgb8": cv2.COLOR_RGB2BGR, "rgba8": cv2.COLOR_RGBA2BGR, "bgra8": cv2.COLOR_BGRA2BGR}[m.encoding])


class EyeInHandSkill(rSkillBase):
    """Base of a staged manipulation skill: worker thread + command pump + evidence."""

    #: subclasses: the procedure; raise StageFailure to return to the caller
    def procedure(self) -> None:  # pragma: no cover - abstract
        raise NotImplementedError

    def __init__(self, *, manifest, robot_description, prompt, prompt_metadata_json, goal_params_json="", tf_lookup=None,
                 clock=None, ros_node=None) -> None:
        super().__init__(name=manifest.name, version=manifest.version, role=manifest.role,
                         embodiment_tags=list(manifest.embodiment_tags),
                         latency_budget_ms=manifest.latency_budget.per_chunk_ms if manifest.latency_budget is not None else None)
        self.manifest, self.description = manifest, robot_description
        self._goal_params_json, self._tf, self._node = goal_params_json, tf_lookup, ros_node
        self._clock = clock if clock is not None else time.monotonic
        self.goal: dict[str, Any] = {}
        self._evidence: dict[str, Any] = {}
        self._cmd_lock = threading.Lock()
        self._final_sent = False
        self._twist: tuple | None = None
        self._joints: tuple[float, ...] | None = None
        self._jaw_open = True
        self._q: dict[str, float] = {}
        self._done: BaseException | None | bool = None
        self._worker: threading.Thread | None = None
        self.camera: WristRGBD | None = None
        self._ik_client = None
        self._fk_client = None
        self._valid_client = None
        self._plan_client = None

    # ---- lifecycle ------------------------------------------------------------------------------------------------
    def _configure_impl(self) -> None:
        import json

        self.goal = {**json.loads(self.manifest.procedural.default_goal_json), **json.loads(self._goal_params_json or "{}")}

    def _activate_impl(self) -> None:
        self._evidence = {"stages": []}
        self._done, self._final_sent, self._q, self._twist = None, False, {}, None
        self.camera = WristRGBD(self._node)
        self.t0 = self._clock()
        self.evidence_dir = Path(self.goal["evidence_dir"]) / f"{self.manifest.name.rsplit('-', 1)[-1]}_{time.strftime('%Y%m%d-%H%M%S')}"
        self.evidence_dir.mkdir(parents=True, exist_ok=True)
        self._evidence["evidence_dir"] = str(self.evidence_dir)
        self._worker = threading.Thread(target=self._run, daemon=True)
        self._worker.start()

    def _deactivate_impl(self) -> None:
        pass

    def _shutdown_impl(self) -> None:
        if self.camera is not None:
            self.camera.close()

    def evidence(self) -> dict:
        return self._evidence

    def _run(self) -> None:
        try:
            while not self._q:  # the first joint reading arrives with the first control tick
                time.sleep(0.01)
            self.hold_here()
            self.procedure()
            self._done = True
        except StageFailure as exc:
            self._evidence.update(failed_stage=exc.stage, why=exc.why)
            self._done = exc
        except Exception as exc:  # reason: any error in the procedure must end the goal with its reason, not hang it
            self._evidence.update(failed_stage=self._evidence.get("stage", "?"), why=f"{type(exc).__name__}: {exc}")
            self._done = exc
        finally:
            if self._q:
                self.hold_here()
            self._evidence["sim_s"] = round(self._clock() - self.t0, 2)

    def _step_impl(self, world_state) -> list[Action]:
        self._q = joint_positions_by_name(world_state)
        if self._final_sent:
            if self._done is True:
                raise ROSRskillGoalSatisfied(f"{self.name}: {self._evidence.get('outcome', 'done')}")
            raise ROSRuntimeError(f"{self.name} failed at stage {self._evidence.get('failed_stage')}: {self._evidence.get('why')}")
        done = self._done is not None
        with self._cmd_lock:
            joints, twist, jaw_open = self._joints, self._twist, self._jaw_open
        out = [Action(control_mode=ControlMode.GRIPPER_BINARY, horizon=1, gripper=[1.0 if jaw_open else 0.0])]
        if joints is not None:  # planned paths and postures are commanded in joint space; the last target holds the arm
            out.insert(0, Action(control_mode=ControlMode.JOINT_POSITION, horizon=1, joint_names=list(ARM_JOINT_NAMES),
                                 joint_targets=[full_width_joint_row(self.description, dict(zip(ARM_JOINT_NAMES, joints)))]))
        elif twist is not None:  # short precise moves are commanded as a tool twist (the arm tracks it to millimetres)
            out.insert(0, Action(control_mode=ControlMode.CARTESIAN_TWIST, horizon=1, cartesian_twist=[twist], frame_id=BASE_FRAME_ID))
        self._final_sent = done
        return out

    # ---- commands (worker side) -----------------------------------------------------------------------------------
    def set_twist(self, twist) -> None:
        with self._cmd_lock:
            self._twist, self._joints = tuple(float(v) for v in twist), None

    def hold_here(self) -> None:
        with self._cmd_lock:
            self._joints, self._twist = tuple(self.arm_q()), None

    def stage(self, name: str, **info) -> dict:
        rec = {"stage": name, "t_sim": round(self._clock() - self.t0, 2), **info}
        self._evidence["stage"] = name
        self._evidence["stages"].append(rec)
        return rec

    def wait(self, sim_s: float) -> None:
        t = self._clock()
        while self._clock() - t < sim_s:
            time.sleep(0.01)

    def jaw(self) -> float:
        return float(self._q.get(GRIPPER_JOINT_NAME, float("nan")))

    def tcp(self) -> tuple[np.ndarray, np.ndarray]:
        tf = self._tf(target_frame=BASE_FRAME_ID, source_frame=TCP_FRAME_ID)
        return np.array(tf.position, float), quat_to_mat(*tf.quaternion_xyzw)

    def T(self, target: str, source: str) -> np.ndarray:
        tf = self._tf(target_frame=target, source_frame=source)
        M = np.eye(4)
        M[:3, :3], M[:3, 3] = quat_to_mat(*tf.quaternion_xyzw), tf.position
        return M

    def frame(self, after: float | None = None, timeout_s: float = 5.0) -> Frame:
        """The next wrist RGB-D pair newer than `after` (sim seconds), with the camera pose at grab time."""
        after = -1.0 if after is None else after
        t = self._clock()
        while True:
            got = self.camera.latest_pair(after)
            if got is not None:
                break
            if self._clock() - t > timeout_s:
                raise StageFailure(self._evidence.get("stage", "?"), "no wrist camera frames")
            time.sleep(0.01)
        stamp, rgb, dm = got
        info = self.camera._info
        depth = np.frombuffer(dm.data, np.float32).reshape(dm.height, dm.width).copy()
        T_now, links = self.fk(self.arm_q(), self.ARM_LINKS)
        # the tool's own volume too: its fingers reach beyond the last link and show up in its camera
        tool = (T_now[:3, :3] @ np.array([[0.0, 0.0, z] for z in (0.0, 0.05, 0.10)]).T).T + T_now[:3, 3]
        links = np.vstack([links, tool])
        return Frame(bgr=image_to_bgr(rgb), depth=depth, K=np.array(info.k, float).reshape(3, 3), stamp=stamp,
                     T_base_cam=self.T(BASE_FRAME_ID, rgb.header.frame_id), frame_id=rgb.header.frame_id, arm_links=links)

    def save(self, name: str, img: np.ndarray) -> str:
        path = self.evidence_dir / name
        cv2.imwrite(str(path), img)
        return str(path)

    def set_jaw(self, open_: bool, stage: str, timeout_s: float = 8.0) -> float:
        """Open or close the jaws and wait until they stop; returns the jaw angle."""
        with self._cmd_lock:
            self._jaw_open = open_
        t, hist = self._clock(), []
        while self._clock() - t < timeout_s:
            self.wait(0.1)
            hist.append(self.jaw())
            if open_ and hist[-1] >= JAW_OPEN_MIN_RAD:
                return hist[-1]
            if not open_ and len(hist) >= 5 and max(hist[-5:]) - min(hist[-5:]) < 0.004 and hist[0] - hist[-1] > 0.05:
                return hist[-1]
        if open_:
            raise StageFailure(stage, f"jaws did not open (jaw {hist[-1]:.3f} rad)")
        return hist[-1]

    def move_joints(self, q, stage: str, tol_rad: float = 0.04, rate_rad_s: float = 0.5, timeout_s: float = 40.0) -> None:
        """Joint-space move along the straight line in joint space at a limited rate (the fastest joint sets the pace;
        a jump in the joint targets throws a held payload, research repo F49)."""
        q = np.array(q, float)
        q0 = np.array(self.arm_q())
        span = float(np.max(np.abs(q - q0)))
        t0 = self._clock()
        while self._clock() - t0 < timeout_s:
            a = min(1.0, (self._clock() - t0) * rate_rad_s / max(span, 1e-6))
            with self._cmd_lock:
                self._joints = tuple(float(v) for v in q0 + (q - q0) * a)
            self.wait(0.05)
            if a >= 1.0 and np.max(np.abs(np.array(self.arm_q()) - q)) < tol_rad:
                return
        err = np.array(self.arm_q()) - q
        worst = int(np.argmax(np.abs(err)))
        tcp = self.tcp()[0]
        raise StageFailure(stage, f"arm stopped short of the posture: joint{worst + 1} {err[worst]:+.2f} rad off, tool at "
                                  f"{[round(float(v), 3) for v in tcp]} (blocked by contact or a joint limit)")

    def ik(self, p: np.ndarray, R: np.ndarray, seed: list[float]) -> list[float] | None:
        """Arm joints that put the TCP at (p, R) in chassis_base_link, nearest the seed (MoveIt IK on the robot's own
        model and planning scene); None when no collision-free configuration reaches it."""
        from moveit_msgs.srv import GetPositionIK

        if self._ik_client is None:
            self._ik_client = self._node.create_client(GetPositionIK, "/compute_ik")
            # a request sent before discovery completes is never answered
            if not self._ik_client.wait_for_service(timeout_sec=30.0):
                raise StageFailure(self._evidence.get("stage", "?"), "the arm's inverse kinematics service is not available")
        req = GetPositionIK.Request()
        r = req.ik_request
        r.group_name, r.ik_link_name, r.avoid_collisions = "rm_group", TCP_FRAME_ID, True
        r.robot_state.joint_state.name = list(ARM_JOINT_NAMES)
        r.robot_state.joint_state.position = [float(v) for v in seed]
        r.pose_stamped.header.frame_id = BASE_FRAME_ID
        r.pose_stamped.pose.position.x, r.pose_stamped.pose.position.y, r.pose_stamped.pose.position.z = (float(v) for v in p)
        q = mat_to_quat(R)
        o = r.pose_stamped.pose.orientation
        o.x, o.y, o.z, o.w = q
        r.timeout.nanosec = 100_000_000
        from moveit_msgs.msg import JointConstraint

        for joint, (centre, half) in POSTURE.items():
            r.constraints.joint_constraints.append(JointConstraint(joint_name=joint, position=centre, tolerance_above=half,
                                                                   tolerance_below=half, weight=1.0))
        fut = self._ik_client.call_async(req)
        t = time.monotonic()
        while not fut.done():
            if time.monotonic() - t > 5.0:
                raise StageFailure(self._evidence.get("stage", "?"), "the arm's inverse kinematics service did not answer")
            time.sleep(0.005)
        res = fut.result()
        if res.error_code.val != 1:
            return None
        js = res.solution.joint_state
        return [float(js.position[list(js.name).index(j)]) for j in ARM_JOINT_NAMES]

    #: arm links whose position is checked against what the cameras measured (the elbow and forearm sweep too)
    ARM_LINKS = ("Link4", "Link5", "Link6", "Link7")

    def fk(self, q, links: tuple[str, ...] = ()) -> np.ndarray:
        """The TCP pose (4x4, chassis_base_link) of an arm configuration (MoveIt FK on the robot's own model); with
        `links`, also their positions, appended as rows of a second return value."""
        from moveit_msgs.srv import GetPositionFK

        if self._fk_client is None:
            self._fk_client = self._node.create_client(GetPositionFK, "/compute_fk")
            if not self._fk_client.wait_for_service(timeout_sec=30.0):
                raise StageFailure(self._evidence.get("stage", "?"), "the arm's forward kinematics service is not available")
        req = GetPositionFK.Request()
        req.header.frame_id, req.fk_link_names = BASE_FRAME_ID, [TCP_FRAME_ID, *links]
        req.robot_state.joint_state.name = list(ARM_JOINT_NAMES)
        req.robot_state.joint_state.position = [float(v) for v in q]
        fut = self._fk_client.call_async(req)
        while not fut.done():
            time.sleep(0.005)
        res = fut.result()
        by_name = dict(zip(res.fk_link_names, res.pose_stamped))
        ps = by_name[TCP_FRAME_ID].pose
        T = np.eye(4)
        T[:3, :3] = quat_to_mat(ps.orientation.x, ps.orientation.y, ps.orientation.z, ps.orientation.w)
        T[:3, 3] = [ps.position.x, ps.position.y, ps.position.z]
        if not links:
            return T
        pts = np.array([[by_name[l].pose.position.x, by_name[l].pose.position.y, by_name[l].pose.position.z] for l in links])
        return T, pts

    def jacobian(self, q: np.ndarray, T0: np.ndarray, delta: float = 1e-3) -> np.ndarray:
        """Geometric Jacobian of the TCP at `q` (6x7, chassis_base_link) by finite differences of the robot's own
        forward kinematics."""
        J = np.zeros((6, 7))
        for i in range(7):
            qi = np.array(q, float)
            qi[i] += delta
            Ti = self.fk(qi)
            J[:3, i] = (Ti[:3, 3] - T0[:3, 3]) / delta
            J[3:, i] = mat_to_rotvec(Ti[:3, :3] @ T0[:3, :3].T) / delta
        return J

    def resolved_rate_step(self, q_cmd: np.ndarray, dx: np.ndarray, dw: np.ndarray, max_dq: float, damping: float = 0.05,
                           posture_gain: float = 0.4, null_grad: np.ndarray | None = None, null_gain: float = 1.0) -> np.ndarray:
        """One joint step that moves the TCP by (dx, dw) from the measured configuration: damped least squares on the
        Jacobian, limited per joint and kept inside the joint stops. Local by construction -- unlike an inverse
        kinematics call, which may answer with a different arm configuration for a nearby pose. The redundancy is used
        to hold the arm on its working branch (`POSTURE`): free-floating, the 7-DoF arm drifts into a mirrored
        configuration in which it cannot move on (research repo F57)."""
        q = np.array(self.arm_q())
        J = self.jacobian(q, self.fk(q))
        dq = J.T @ np.linalg.solve(J @ J.T + damping ** 2 * np.eye(6), np.concatenate([dx, dw]))
        # null-space pull towards the working branch, scaled like the task step
        ref = np.array(q)
        for joint, (centre, _half) in POSTURE.items():
            ref[ARM_JOINT_NAMES.index(joint)] = centre
        want = (ref - q) * posture_gain
        if null_grad is not None:  # e.g. lift the arm away from what the cameras measured under it
            want = want + null_grad * null_gain
        null = (np.eye(7) - np.linalg.pinv(J) @ J) @ want
        dq = dq + null * min(1.0, max_dq / max(float(np.max(np.abs(null))), 1e-9))
        scale = min(1.0, max_dq / max(float(np.max(np.abs(dq))), 1e-9))
        q_next = q_cmd + dq * scale
        return np.clip(q_next, [-(l - 0.05) for l in JOINT_LIMITS_RAD], [l - 0.05 for l in JOINT_LIMITS_RAD])

    def state_valid(self, q: np.ndarray) -> bool:
        """Is this arm configuration collision-free in the robot's own planning scene (its own links, the rover, the
        surveyed site structures)? The servo steps the joints directly, so it checks what the planners check."""
        from moveit_msgs.srv import GetStateValidity

        if self._valid_client is None:
            self._valid_client = self._node.create_client(GetStateValidity, "/check_state_validity")
            if not self._valid_client.wait_for_service(timeout_sec=30.0):
                raise StageFailure(self._evidence.get("stage", "?"), "the arm's state validity service is not available")
        req = GetStateValidity.Request()
        req.group_name = "rm_group"
        req.robot_state.joint_state.name = list(ARM_JOINT_NAMES)
        req.robot_state.joint_state.position = [float(v) for v in q]
        req.robot_state.is_diff = True
        fut = self._valid_client.call_async(req)
        while not fut.done():
            time.sleep(0.005)
        return bool(fut.result().valid)

    def arm_q(self) -> list[float]:
        return [float(self._q[j]) for j in ARM_JOINT_NAMES]

    def plan_to(self, q_goal, stage: str, tol_rad: float = 0.02, rate_rad_s: float = 0.5) -> dict:
        """Plan a collision-free path to a configuration with the robot's own planner (MoveIt, the same one the
        capability boundary's arm moves use) and follow it with the joint command pump. A planner keeps the arm off its
        stops and out of the structures in its scene, which a straight joint or Cartesian move does not."""
        from moveit_msgs.msg import Constraints, JointConstraint, MotionPlanRequest, WorkspaceParameters
        from moveit_msgs.srv import GetMotionPlan

        if self._plan_client is None:
            self._plan_client = self._node.create_client(GetMotionPlan, "/plan_kinematic_path")
            if not self._plan_client.wait_for_service(timeout_sec=30.0):
                raise StageFailure(stage, "the arm's motion planning service is not available")
        req = GetMotionPlan.Request()
        r: MotionPlanRequest = req.motion_plan_request
        r.group_name, r.num_planning_attempts, r.allowed_planning_time = "rm_group", 10, 10.0
        r.max_velocity_scaling_factor = r.max_acceleration_scaling_factor = 0.3
        r.workspace_parameters = WorkspaceParameters()
        r.workspace_parameters.header.frame_id = BASE_FRAME_ID
        r.workspace_parameters.min_corner.x = r.workspace_parameters.min_corner.y = r.workspace_parameters.min_corner.z = -2.0
        r.workspace_parameters.max_corner.x = r.workspace_parameters.max_corner.y = r.workspace_parameters.max_corner.z = 2.0
        r.start_state.joint_state.name = list(ARM_JOINT_NAMES)
        r.start_state.joint_state.position = [float(v) for v in self.arm_q()]
        goal = Constraints()
        goal.joint_constraints = [JointConstraint(joint_name=j, position=float(v), tolerance_above=tol_rad,
                                                  tolerance_below=tol_rad, weight=1.0) for j, v in zip(ARM_JOINT_NAMES, q_goal)]
        r.goal_constraints = [goal]
        fut = self._plan_client.call_async(req)
        t = time.monotonic()
        while not fut.done():
            if time.monotonic() - t > 60.0:
                raise StageFailure(stage, "the arm's motion planning service did not answer")
            time.sleep(0.01)
        res = fut.result().motion_plan_response
        if res.error_code.val != 1 or not res.trajectory.joint_trajectory.points:
            raise StageFailure(stage, f"no collision-free path to that arm posture (planner error {res.error_code.val})")
        pts = res.trajectory.joint_trajectory.points
        names = list(res.trajectory.joint_trajectory.joint_names)
        order = [names.index(j) for j in ARM_JOINT_NAMES]
        # follow the path on its own clock (waiting for each point in turn would take minutes on a long path), then let
        # the last point settle
        t0 = self._clock()
        for pt in pts:
            due = pt.time_from_start.sec + pt.time_from_start.nanosec * 1e-9
            while self._clock() - t0 < due:
                time.sleep(0.01)
            with self._cmd_lock:
                self._joints = tuple(float(pt.positions[i]) for i in order)
        # the planned path ends where it ends: a few degrees of tracking error are not a failure (the visual servo and the
        # measured tool pose take it from here)
        self.move_joints(np.array(q_goal), stage, tol_rad=0.08, rate_rad_s=rate_rad_s, timeout_s=20.0)
        return {"waypoints": len(pts), "sim_s": round(self._clock() - t0, 1)}

    def move_to(self, p: np.ndarray, R: np.ndarray, stage: str, seed=None, rate_rad_s: float = 0.4) -> None:
        """A larger reconfiguration: one IK solution for the pose (working branch, seeded at `seed` or the measured
        joints), then a rate-limited joint-space move."""
        q = self.ik(p, R, list(seed) if seed is not None else self.arm_q())
        if q is None:
            raise StageFailure(stage, "no arm configuration on the working branch reaches that pose")
        self.move_joints(q, stage, rate_rad_s=rate_rad_s)

    def servo_twist(self, goal: Callable[[], tuple[np.ndarray, np.ndarray] | None], stage: str, *, tol_m: float, tol_rad: float,
                    max_speed_m_s: float = 0.05, max_rate_rad_s: float = 0.25, timeout_s: float = 40.0, stall_s: float = 4.0,
                    settle_cycles: int = 3, gain: float = 1.2, guard: Callable[[np.ndarray, np.ndarray], str] | None = None) -> dict:
        """Short precise move: a proportional tool twist towards `goal()` (re-evaluated every cycle), the path the arm's
        controller tracks to millimetres. `guard(p, R)` may stop the move before a pose that would touch something."""
        t0 = t_prev = self._clock()
        best, t_best, inside, last_goal = math.inf, t0, 0, None
        trace: list[dict] = []
        self._evidence.setdefault("servo_traces", {})[f"{stage}_{len(self._evidence.get('servo_traces', {}))}"] = trace
        while True:
            g = goal()
            if g is not None:
                last_goal = g
            now = self._clock()
            if last_goal is None:
                if now - t0 > stall_s:
                    self.hold_here()
                    raise StageFailure(stage, "target not measured")
                self.wait(0.05)
                continue
            p, R = self.tcp()
            gp, gR = last_goal
            e, r = gp - p, mat_to_rotvec(gR @ R.T)
            en, rn = float(np.linalg.norm(e)), float(np.linalg.norm(r))
            if en < tol_m and rn < tol_rad:
                inside += 1
                if inside >= settle_cycles:
                    self.hold_here()
                    return {"pos_err_m": round(en, 4), "rot_err_rad": round(rn, 4), "sim_s": round(now - t0, 2)}
            else:
                inside = 0
            if en + 0.1 * rn < best - 0.002:
                best, t_best = en + 0.1 * rn, now
            if now - t_best > stall_s or now - t0 > timeout_s:
                self.hold_here()
                why = "stopped making progress" if now - t_best > stall_s else "timed out"
                raise StageFailure(stage, f"the tool {why} {en * 100:.1f} cm / {math.degrees(rn):.0f} deg from its goal "
                                          f"(at {[round(float(v), 3) for v in p]})")
            if guard is not None:
                blocked = guard(p + e * min(1.0, 0.03 / max(en, 1e-9)), R)
                if blocked:
                    self.hold_here()
                    raise StageFailure(stage, f"the tool cannot continue towards the goal without touching {blocked} "
                                              f"({en * 100:.1f} cm away)")
            v = e * gain
            if np.linalg.norm(v) > max_speed_m_s:
                v = v / np.linalg.norm(v) * max_speed_m_s
            w = r * gain
            if np.linalg.norm(w) > max_rate_rad_s:
                w = w / np.linalg.norm(w) * max_rate_rad_s
            self.set_twist((*v, *w))
            if len(trace) < 40 and now - (trace[-1]["t"] if trace else -1) > 0.5:
                trace.append({"t": round(now - t0, 1), "tcp": [round(float(x), 3) for x in p], "err_m": round(en, 3)})
            t_prev = now
            self.wait(0.05)

    def servo(self, goal: Callable[[], tuple[np.ndarray, np.ndarray] | None], stage: str, *, tol_m: float, tol_rad: float,
              max_joint_rate_rad_s: float = 0.3, timeout_s: float = 40.0, stall_s: float = 4.0, settle_cycles: int = 3,
              step_m: float = 0.04, step_rad: float = 0.2, guard: Callable[[np.ndarray], str] | None = None,
              null_objective: Callable[[np.ndarray], np.ndarray] | None = None) -> dict:
        """Move the TCP (chassis_base_link) to `goal()` = (position, rotation), re-evaluated every cycle (visual
        servoing); `goal()` returning None keeps the last goal. Each cycle the goal, corrected by the integral of the
        remaining position error (arm sag under a load), goes through the arm's inverse kinematics on its working branch
        seeded at the measured joints, and the joint targets move towards the solution at a limited rate."""
        t0 = t_prev = self._clock()
        best, t_best, inside, last_goal, integ, blocked = math.inf, t0, 0, None, np.zeros(3), 0
        trace: list[dict] = []
        self._evidence.setdefault("servo_traces", {})[f"{stage}_{len(self._evidence.get('servo_traces', {}))}"] = trace
        q_cmd = np.array(self.arm_q())
        while True:
            g = goal()
            if g is not None:
                last_goal = g
            now = self._clock()
            dt = max(now - t_prev, 1e-3)
            t_prev = now
            if last_goal is None:
                if now - t0 > stall_s:
                    raise StageFailure(stage, "target not measured")
                self.wait(0.05)
                continue
            p, R = self.tcp()
            gp, gR = last_goal
            e = gp - p
            r = mat_to_rotvec(gR @ R.T)
            en, rn = float(np.linalg.norm(e)), float(np.linalg.norm(r))
            if en < tol_m and rn < tol_rad:
                inside += 1
                if inside >= settle_cycles:
                    return {"pos_err_m": round(en, 4), "rot_err_rad": round(rn, 4), "sim_s": round(now - t0, 2)}
            else:
                inside = 0
            if en + 0.1 * rn < best - 0.002:
                best, t_best = en + 0.1 * rn, now
            if now - t_best > stall_s:
                lag = q_cmd - q_meas
                at_stop = [f"joint{i + 1} {q_meas[i]:+.2f} (limit +-{JOINT_LIMITS_RAD[i]})" for i in range(7) if abs(q_meas[i]) > JOINT_LIMITS_RAD[i] - 0.15]
                self._evidence.setdefault("stalls", []).append(
                    {"stage": stage, "tcp": [round(float(v), 3) for v in p], "goal": [round(float(v), 3) for v in gp],
                     "joints": np.round(q_meas, 2).tolist(), "command_minus_measured": np.round(lag, 2).tolist(),
                     "image": self.save(f"stall_{stage}_{len(self._evidence.get('stalls', []))}.jpg", self.frame(after=self._clock() - 0.05).bgr)})
                raise StageFailure(stage, f"arm stopped making progress {en * 100:.1f} cm / {math.degrees(rn):.0f} deg from its goal; "
                                          f"tool at {[round(float(v), 3) for v in p]}, joints {np.round(q_meas, 2).tolist()}, "
                                          f"commanded minus measured {np.round(lag, 2).tolist()}"
                                          + ("; at a joint stop: " + ", ".join(at_stop) if at_stop else ""))
            if now - t0 > timeout_s:
                raise StageFailure(stage, f"arm timed out {en * 100:.1f} cm / {math.degrees(rn):.0f} deg from its goal")
            if en < 0.03:
                integ = np.clip(integ + e * dt * 0.8, -0.05, 0.05)
            # move the tool a short way along the straight line to the goal, through the Jacobian: a step, not a new
            # arm configuration
            dx = (e + integ) * min(1.0, step_m / max(en, 1e-6))
            dw = r * min(1.0, step_rad / max(rn, 1e-6))
            q_meas = np.array(self.arm_q())
            q_next = self.resolved_rate_step(q_cmd, dx, dw, max_joint_rate_rad_s * dt,
                                             null_grad=None if null_objective is None else null_objective(q_meas))
            why = "" if self.state_valid(q_next) else "the robot's own planning scene"
            if not why and guard is not None:
                why = guard(q_next)
            if why:
                blocked += 1
                if blocked > 10:
                    raise StageFailure(stage, f"the arm cannot continue towards the goal without touching {why} "
                                              f"({en * 100:.1f} cm / {math.degrees(rn):.0f} deg away)")
            else:
                blocked, q_cmd = 0, q_next
                with self._cmd_lock:
                    self._joints = tuple(float(v) for v in q_cmd)
            if len(trace) < 60 and now - (trace[-1]["t"] if trace else -1) > 0.5:
                trace.append({"t": round(now - t0, 1), "tcp": [round(float(v), 3) for v in p], "goal": [round(float(v), 3) for v in gp],
                              "err_m": round(en, 3)})
            self.wait(0.05)

    def command_towards(self, p: np.ndarray, R: np.ndarray, q_cmd: np.ndarray, max_dq: float, max_jump: float | None = None):
        """One rate-limited joint command towards the IK solution of a TCP pose; None when IK has no solution, or when
        the solution is a different arm configuration (a joint swinging further than `max_jump`)."""
        q_meas = np.array(self.arm_q())
        q_goal = self.ik(p, R, list(q_meas))
        if q_goal is None or (max_jump is not None and float(np.max(np.abs(np.array(q_goal) - q_meas))) > max_jump):
            return None
        q_cmd = q_cmd + np.clip(np.array(q_goal) - q_cmd, -max_dq, max_dq)
        with self._cmd_lock:
            self._joints = tuple(float(v) for v in q_cmd)
        return q_cmd
