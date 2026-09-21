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
#: how far a seeded IK solution may sit from the seed before it is another arm configuration rather than a nearby one
BRANCH_JUMP_RAD = 1.2
#: How far the carried item pivots about the jaw closing axis while it hangs from its handle: the largest tilt from
#: vertical measured over every carry bag (g8s 14.1, g8t 4.7, g9m 12.5, g9o 15.4 deg), rounded up. The same bags say
#: its yaw does not change -- modulo the box's own symmetry it stays within 14 deg of where it started -- so a
#: carried item is not checked as a body that turns freely about the vertical (research F79).
HELD_PIVOT_RAD = math.radians(16.0)
#: The RM-75's working branch (joint: centre, half-range, rad): shoulder and forearm roll near zero, wrist roll unflipped.
#: Unconstrained IK on this 7-DoF arm lands on arbitrary null-space branches (shoulder turned 90 deg, wrist rolled to its
#: stop) from which the next small motion is impossible (research repo F57); the boundary's MoveIt posture preference is
#: the same idea.
#: The wrist roll (joint7) is not in here: the tool rotation asked for determines it, and holding it near zero excluded
#: every pose with the camera above the tool axis (half a turn, 3.14 rad) and made the servo's null space unroll the
#: wrist it had just turned (research repo F57).
POSTURE = {"joint1": (0.0, 1.5), "joint2": (-0.4, 0.9), "joint3": (0.0, 1.2), "joint4": (-1.5, 0.85),
           "joint5": (0.0, 1.6)}
#: The jaw bands, in the angle the EG2's own encoder reports: 0.10 closed on nothing, ~0.17 on a 12 mm neck, 0.33 on
#: the ORU's bail, 0.82 commanded open. "Open" is not that commanded value: the jaws come to rest a few hundredths
#: below it (0.776 after a snapshot restore, run g4p), so a threshold set just under the nominal reads an open jaw as
#: a held item -- which refused `arm_ready` in g4p and would have failed `place` at its own settle check. The edge of
#: the holding band belongs midway between the widest item this gripper can take and where the jaws come to rest open,
#: not at the nominal figure; this is the one definition of it (the boundary reads it from here).
JAW_OPEN_MIN_RAD = 0.6
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
    depth_semantics: dict | None = None  # sensor-sourced range/no-return meaning; absent means unknown

    @property
    def up_cam(self) -> np.ndarray:
        return self.T_base_cam[:3, :3].T @ np.array([0.0, 0.0, 1.0])

    def to_base(self, p_cam) -> np.ndarray:
        return self.T_base_cam[:3, :3] @ np.asarray(p_cam) + self.T_base_cam[:3, 3]


class WristRGBD:
    """The wrist D435i's colour + registered depth, subscribed on the runner node.

    The node owns these subscriptions for its life. A skill that created and destroyed them per call destroyed them
    from the worker thread while the executor was spinning on its own, which raised `InvalidHandle` inside
    `executor.spin()` and killed the runner -- every later goal on it, from any caller, then returned nothing
    (research repo F58)."""

    def __init__(self, node: Any, camera: str = "wrist") -> None:
        import json
        from std_msgs.msg import String
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import CameraInfo, Image

        self._lock = threading.Lock()
        self._rgb: dict[float, Any] = {}
        self._depth: dict[float, Any] = {}
        self._info = None
        self.depth_semantics = None
        stamp = lambda m: m.header.stamp.sec + m.header.stamp.nanosec * 1e-9  # noqa: E731
        self._subs = [
            node.create_subscription(Image, f"/openral/cameras/{camera}/image", lambda m: self._put(self._rgb, stamp(m), m), qos_profile_sensor_data),
            node.create_subscription(Image, f"/openral/cameras/{camera}_depth/image", lambda m: self._put(self._depth, stamp(m), m), qos_profile_sensor_data),
            node.create_subscription(CameraInfo, f"/openral/cameras/{camera}/camera_info", lambda m: setattr(self, "_info", m), qos_profile_sensor_data),
            node.create_subscription(String, f"/openral/cameras/{camera}/depth_semantics",
                                     lambda m: setattr(self, "depth_semantics", json.loads(m.data)), qos_profile_sensor_data),
        ]

    def _put(self, store: dict, t: float, m: Any) -> None:
        with self._lock:
            store[t] = m
            for k in sorted(store)[:-4]:
                del store[k]

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
        self._stop_requested = threading.Event()
        self._final_sent = False
        self._twist: tuple | None = None
        self._joints: tuple[float, ...] | None = None
        # A new arm operation must preserve a held object. Gripper commands begin
        # only when the procedure explicitly calls set_jaw (e.g. place's release).
        self._jaw_open: bool | None = None
        self._q: dict[str, float] = {}
        self._done: BaseException | None | bool = None
        self._worker: threading.Thread | None = None
        self.camera: WristRGBD | None = None
        self._ik_client = None
        self._fk_client = None
        self._valid_client = None
        self._held = None  # the carried item's catalogue envelope, checked with the robot while it hangs from the jaws
        self._plan_client = None

    # ---- lifecycle ------------------------------------------------------------------------------------------------
    def _configure_impl(self) -> None:
        import json

        self.goal = {**json.loads(self.manifest.procedural.default_goal_json), **json.loads(self._goal_params_json or "{}")}

    def _activate_impl(self) -> None:
        self._stop_requested.clear()
        self._evidence = {"stages": []}
        self._done, self._final_sent, self._q, self._twist = None, False, {}, None
        self._jaw_open = None
        # the wrist camera belongs to the robot, not to one call: one subscriber per node, kept for its life. Creating
        # it per skill and destroying it on shutdown raced the executor -- a subscription destroyed while the executor
        # held it in its wait set killed the runner with InvalidHandle, and with it every later skill (F57).
        self.camera = getattr(self._node, "_wrist_rgbd", None)
        if self.camera is None:
            self.camera = WristRGBD(self._node)
            self._node._wrist_rgbd = self.camera
        # the declaration channel is opened with the skill, not at the moment it is first needed: a publisher's first
        # message is lost before discovery completes, and that message is the one that says what the tool is about to
        # be inside (research repo F59)
        self.working_on(None)
        self.t0 = self._clock()
        self.evidence_dir = Path(self.goal["evidence_dir"]) / f"{self.manifest.name.rsplit('-', 1)[-1]}_{time.strftime('%Y%m%d-%H%M%S')}"
        self.evidence_dir.mkdir(parents=True, exist_ok=True)
        self._evidence["evidence_dir"] = str(self.evidence_dir)
        self._worker = threading.Thread(target=self._run, daemon=True)
        self._worker.start()

    def _deactivate_impl(self) -> None:
        self._stop_requested.set()
        self.finish_stop()

    def _shutdown_impl(self) -> None:
        self._stop_requested.set()
        self.finish_stop()

    def stop_actions(self, world_state) -> list[Action]:
        self._q = joint_positions_by_name(world_state)
        with self._cmd_lock:
            self._stop_requested.set()
            self._joints, self._twist = tuple(self.arm_q()), None
            actions = self._command_actions()
        self._evidence["stop"] = {"requested": True, "worker_stopped": False,
                                  "arm_hold_rad": list(self._joints),
                                  "jaw_command_open": self._jaw_open,
                                  "physical_stop_confirmed": False}
        return actions

    def finish_stop(self) -> None:
        if self._worker is not None:
            self._worker.join()
        self._evidence.setdefault("stop", {})["worker_stopped"] = True

    def _check_stop(self) -> None:
        if self._stop_requested.is_set():
            raise StageFailure("stopping", "operation stopped", local_retry=False)

    def _sleep(self, seconds: float) -> None:
        self._stop_requested.wait(seconds)
        self._check_stop()

    def evidence(self) -> dict:
        return self._evidence

    def _run(self) -> None:
        try:
            while not self._q:  # the first joint reading arrives with the first control tick
                self._sleep(0.01)
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
            out = self._command_actions()
        self._final_sent = done
        return out

    def _command_actions(self) -> list[Action]:
        joints, twist, jaw_open = self._joints, self._twist, self._jaw_open
        out = []
        if jaw_open is not None:
            out.append(Action(control_mode=ControlMode.GRIPPER_BINARY, horizon=1, gripper=[1.0 if jaw_open else 0.0]))
        if joints is not None:  # planned paths and postures are commanded in joint space; the last target holds the arm
            out.insert(0, Action(control_mode=ControlMode.JOINT_POSITION, horizon=1, joint_names=list(ARM_JOINT_NAMES),
                                 joint_targets=[full_width_joint_row(self.description, dict(zip(ARM_JOINT_NAMES, joints)))]))
        elif twist is not None:  # short precise moves are commanded as a tool twist (the arm tracks it to millimetres)
            out.insert(0, Action(control_mode=ControlMode.CARTESIAN_TWIST, horizon=1, cartesian_twist=[twist], frame_id=BASE_FRAME_ID))
        return out

    # ---- commands (worker side) -----------------------------------------------------------------------------------
    def set_twist(self, twist) -> None:
        with self._cmd_lock:
            self._check_stop()
            self._twist, self._joints = tuple(float(v) for v in twist), None

    def hold_here(self) -> None:
        with self._cmd_lock:
            self._joints, self._twist = tuple(self.arm_q()), None

    def wait_until_still(self, still_m: float = 0.0005, window_s: float = 0.3, timeout_s: float = 10.0) -> None:
        """Hold the arm where it is and wait until the tool stops moving, so that what a wrist camera sees moving
        is the world and not the camera. A servo ends when the tool is within tolerance, with the command still ahead
        of the arm by its lag, and the arm keeps going for a second or more."""
        self.hold_here()
        p_prev = self.tcp()[0]
        t0 = self._clock()
        while self._clock() - t0 < timeout_s:
            self.wait(window_s)
            p = self.tcp()[0]
            if float(np.linalg.norm(p - p_prev)) < still_m:
                return
            p_prev = p

    def contact_scene(self, prepare: bool) -> None:
        """Acquire/release an applied shared scene for one stationary contact operation."""
        import json
        from std_srvs.srv import Trigger

        name = "/space/prepare_contact_scene" if prepare else "/space/release_contact_scene"
        client = self._node.create_client(Trigger, name)
        try:
            if not client.wait_for_service(timeout_sec=30.0):
                raise StageFailure("contact_scene", f"scene service unavailable: {name}", local_retry=False)
            future = client.call_async(Trigger.Request())
            started = time.monotonic()
            while not future.done():
                if time.monotonic() - started > 30.0:
                    raise StageFailure("contact_scene", "scene acknowledgement not received", local_retry=False)
                time.sleep(0.01)  # also complete release when cancellation is already set
            result = future.result()
            if not result.success:
                raise StageFailure("contact_scene", result.message, local_retry=False)
            self._contact_scene_owned = prepare
            self._evidence["contact_scene" if prepare else "contact_scene_release"] = (
                json.loads(result.message) if prepare else result.message)
        finally:
            self._node.destroy_client(client)

    def contact_permission(self, phase: str, region=None, dimensions=None, evidence_ref="") -> None:
        """Apply a phase-scoped contact declaration and retain the consumer acknowledgement."""
        import json
        from openral_msgs.srv import ConfigureContact

        client = self._node.create_client(ConfigureContact, "/space/configure_contact")
        try:
            if not client.wait_for_service(timeout_sec=30.0):
                raise StageFailure("contact_scene", "contact permission service unavailable", local_retry=False)
            request = ConfigureContact.Request(phase=phase, evidence_ref=evidence_ref)
            if region is not None:
                request.region.header.frame_id = "odom"
                request.region.pose.position.x, request.region.pose.position.y, request.region.pose.position.z = map(float, region[:3, 3])
                q = mat_to_quat(region[:3, :3])
                orientation = request.region.pose.orientation
                orientation.x, orientation.y, orientation.z, orientation.w = q
                request.dimensions.x, request.dimensions.y, request.dimensions.z = map(float, dimensions)
            future = client.call_async(request)
            started = time.monotonic()
            while not future.done():
                if time.monotonic() - started > 30.0:
                    raise StageFailure("contact_scene", "contact permission acknowledgement missing", local_retry=False)
                time.sleep(0.01)  # revoke remains possible after cancellation
            result = future.result()
            self._evidence.setdefault("contact_permissions", []).append(json.loads(result.evidence_json))
            if not result.success:
                raise StageFailure("contact_scene", result.evidence_json, local_retry=False)
        finally:
            self._node.destroy_client(client)

    def working_on(self, p_base: np.ndarray | None) -> None:
        """Record the operation target; never delete its geometry from collision checks."""
        if p_base is None:
            if getattr(self, "_contact_scene_owned", False):
                self.contact_scene(False)
        else:
            self._evidence["operation_target_base_m"] = list(map(float, p_base))

    def stage(self, name: str, **info) -> dict:
        self._check_stop()
        rec = {"stage": name, "t_sim": round(self._clock() - self.t0, 2), **info}
        self._evidence["stage"] = name
        self._evidence["stages"].append(rec)
        return rec

    def wait(self, sim_s: float) -> None:
        t = self._clock()
        while self._clock() - t < sim_s:
            self._sleep(0.01)

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
            self._sleep(0.01)
        stamp, rgb, dm = got
        info = self.camera._info
        depth = np.frombuffer(dm.data, np.float32).reshape(dm.height, dm.width).copy()
        T_now, links = self.fk(self.arm_q(), self.ARM_LINKS)
        # the tool's own volume too: its fingers reach beyond the last link and show up in its camera
        tool = (T_now[:3, :3] @ np.array([[0.0, 0.0, z] for z in (0.0, 0.05, 0.10)]).T).T + T_now[:3, 3]
        links = np.vstack([links, tool])
        return Frame(bgr=image_to_bgr(rgb), depth=depth, K=np.array(info.k, float).reshape(3, 3), stamp=stamp,
                     T_base_cam=self.T(BASE_FRAME_ID, rgb.header.frame_id), frame_id=rgb.header.frame_id, arm_links=links,
                     depth_semantics=self.camera.depth_semantics)

    def save(self, name: str, img: np.ndarray) -> str:
        path = self.evidence_dir / name
        cv2.imwrite(str(path), img)
        return str(path)

    def set_jaw(self, open_: bool, stage: str, timeout_s: float = 8.0) -> float:
        """Open or close the jaws and wait until they stop; returns the jaw angle."""
        with self._cmd_lock:
            self._check_stop()
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
                self._check_stop()
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
        r.robot_state.joint_state.name = list(self._q)
        r.robot_state.joint_state.position = [float(self._q[j]) for j in self._q]
        for joint, value in zip(ARM_JOINT_NAMES, seed):
            r.robot_state.joint_state.position[r.robot_state.joint_state.name.index(joint)] = float(value)
        r.robot_state.is_diff = True
        r.pose_stamped.header.frame_id = BASE_FRAME_ID
        r.pose_stamped.pose.position.x, r.pose_stamped.pose.position.y, r.pose_stamped.pose.position.z = (float(v) for v in p)
        q = mat_to_quat(R)
        o = r.pose_stamped.pose.orientation
        o.x, o.y, o.z, o.w = q
        r.timeout.nanosec = 100_000_000
        fut = self._ik_client.call_async(req)
        t = time.monotonic()
        while not fut.done():
            if time.monotonic() - t > 5.0:
                raise StageFailure(self._evidence.get("stage", "?"), "the arm's inverse kinematics service did not answer")
            self._sleep(0.005)
        res = fut.result()
        self._evidence["last_ik"] = {"error_code": int(res.error_code.val), "goal_m": list(map(float, p))}
        if res.error_code.val != 1:
            return None
        js = res.solution.joint_state
        q = [float(js.position[list(js.name).index(j)]) for j in ARM_JOINT_NAMES]
        # What the redundancy must not do is jump to the mirrored arm configuration mid-operation (research F57).
        # Neither half of that says it alone. The joint window by itself refused poses the arm was already in and
        # made every bring_in height unreachable (g9l, F78); the distance from the seed by itself refused every
        # deliberate reconfiguration, because the postures seeded at READY turn the wrist to look down -- all four
        # `find` overviews solve on the working branch and still sit 1.6-2.6 rad from READY, in joint6/joint7, which
        # are not in POSTURE (g9n). A solution is this arm's own configuration when it is on the working branch, or
        # when it is next to the one the arm is in; a mirrored branch is neither, and reach, collision and the servo
        # decide the rest.
        jump = max(abs(a - b) for a, b in zip(q, seed))
        off_branch = [j for j, (centre, half) in POSTURE.items()
                      if abs(q[ARM_JOINT_NAMES.index(j)] - centre) > half]
        if off_branch and jump > BRANCH_JUMP_RAD:
            self._evidence["last_ik"].update(branch_jump_rad=round(jump, 3), off_branch=off_branch)
            return None
        return q

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
        t = time.monotonic()
        while not fut.done():
            if time.monotonic() - t > 5.0:  # every service call ends, or the stage does: a wedged planner is a failure
                raise StageFailure(self._evidence.get("stage", "?"), "the arm's forward kinematics service did not answer")
            self._sleep(0.005)
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

    def carry_item(self, held: dict | None) -> None:
        """Account for the item hanging from the jaws in every collision check, or stop doing so.

        `held` is the equipment catalogue's envelope (size_m, handle_above_base_m, neck_height_m). The item hangs
        gravity-vertical below the grasp, turning freely about the vertical, so it is checked as an upright square
        prism around its possible yaws, extended downwards by the neck's length it can slide in the jaws. It is
        checked against the robot's own body and against the surveyed site structures; only the live measurement is
        exempt, because it contains the item itself and the place it stood. Exempting the surveyed structures too let
        bring_in drag the hanging item into the stand it came from until the servo stalled, and the first metre of the
        drive tore it out of the jaws (g9i, research F78)."""
        self._held = held
        if held is None:
            return
        from moveit_msgs.msg import AllowedCollisionEntry, PlanningScene, PlanningSceneComponents
        from moveit_msgs.srv import ApplyPlanningScene, GetPlanningScene

        get = self._node.create_client(GetPlanningScene, "/get_planning_scene")
        apply = self._node.create_client(ApplyPlanningScene, "/apply_planning_scene")
        try:
            for client in (get, apply):
                if not client.wait_for_service(timeout_sec=30.0):
                    raise StageFailure(self._evidence.get("stage", "?"), "planning scene service unavailable", local_retry=False)
            req = GetPlanningScene.Request()
            req.components.components = (PlanningSceneComponents.ALLOWED_COLLISION_MATRIX
                                         | PlanningSceneComponents.WORLD_OBJECT_NAMES)
            fut = get.call_async(req)
            while not fut.done():
                self._sleep(0.005)
            scene = fut.result().scene
            acm = scene.allowed_collision_matrix
            world = [o.id for o in scene.world.collision_objects if o.id.startswith("measured")]
            for name in ["held_item", *world]:
                if name not in acm.entry_names:
                    acm.entry_names.append(name)
                    for entry in acm.entry_values:
                        entry.enabled.append(False)
                    acm.entry_values.append(AllowedCollisionEntry(enabled=[False] * len(acm.entry_names)))
            i = acm.entry_names.index("held_item")
            for name in world:
                j = acm.entry_names.index(name)
                acm.entry_values[i].enabled[j] = acm.entry_values[j].enabled[i] = True
            fut = apply.call_async(ApplyPlanningScene.Request(scene=PlanningScene(is_diff=True, allowed_collision_matrix=acm)))
            while not fut.done():
                self._sleep(0.005)
            self._evidence["held_envelope"] = {**held, "exempt_world": world, "applied": bool(fut.result().success)}
        finally:
            self._node.destroy_client(get)
            self._node.destroy_client(apply)

    def held_bottom_below_tcp(self) -> float:
        """How far below the tool the carried item's envelope reaches."""
        return float(self._held["handle_above_base_m"]) + float(self._held["neck_height_m"])

    def _held_body(self, drop_m: float = 0.0):
        from moveit_msgs.msg import AttachedCollisionObject, CollisionObject
        from geometry_msgs.msg import Pose
        from shape_msgs.msg import SolidPrimitive

        size = np.array(self._held["size_m"], float)
        height = float(size[2]) + float(self._held["neck_height_m"])
        # The item hangs from its handle and turns with the jaws, not about the vertical: checking it as a prism
        # swept over every yaw claimed 0.10 m per side of material that is not there, and that phantom cost bring_in
        # exactly the travel it needed to carry the ORU past the stand it came from (0.228 m when nothing was checked,
        # 0.105 m with the sweep; it needs its own half depth). What the bags do show is a pivot about the jaw closing
        # axis, which swings the item across that axis (research F79).
        along = float(size[0])  # the item's own x, the axis the jaws close along
        across = float(size[1]) * math.cos(HELD_PIVOT_RAD) + height * math.sin(HELD_PIVOT_RAD)
        _, R = self.tcp()  # the servo keeps the tool's orientation; the offset is vertical in the rover frame
        centre = R.T @ np.array([0.0, 0.0, -self.held_bottom_below_tcp() + height / 2 - drop_m])
        # upright in the rover frame, turned with the jaws: the item hangs gravity-vertical whatever the tool's pitch
        jaw_axis = R[:, 0] * np.array([1.0, 1.0, 0.0])
        jaw_axis = jaw_axis / np.linalg.norm(jaw_axis)
        upright = np.column_stack([jaw_axis, [-jaw_axis[1], jaw_axis[0], 0.0], [0.0, 0.0, 1.0]])
        body = AttachedCollisionObject(link_name=TCP_FRAME_ID, touch_links=[f"eg2_link{i}" for i in range(1, 7)] + ["eg2_base"])
        body.object = CollisionObject(id="held_item", operation=CollisionObject.ADD)
        body.object.header.frame_id = TCP_FRAME_ID
        body.object.primitives = [SolidPrimitive(type=SolidPrimitive.BOX, dimensions=[along, across, height])]
        pose = Pose()
        pose.position.x, pose.position.y, pose.position.z = map(float, centre)
        qx, qy, qz, qw = mat_to_quat(R.T @ upright)
        pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = qx, qy, qz, qw
        body.object.primitive_poses = [pose]
        return body

    def state_valid(self, q: np.ndarray, held_drop_m: float = 0.0) -> bool:
        """Is this arm configuration collision-free in the robot's own planning scene (its own links, the rover, the
        surveyed site structures)? The servo steps the joints directly, so it checks what the planners check."""
        from moveit_msgs.srv import GetStateValidity

        if self._valid_client is None:
            self._valid_client = self._node.create_client(GetStateValidity, "/check_state_validity")
            if not self._valid_client.wait_for_service(timeout_sec=30.0):
                raise StageFailure(self._evidence.get("stage", "?"), "the arm's state validity service is not available")
        req = GetStateValidity.Request()
        # Whole-robot validation includes the measured articulated gripper, not
        # just the arm's planning group or an implicit zero-width jaw state.
        req.robot_state.joint_state.name = list(self._q)
        req.robot_state.joint_state.position = [float(self._q[j]) for j in self._q]
        for joint, value in zip(ARM_JOINT_NAMES, q):
            req.robot_state.joint_state.position[req.robot_state.joint_state.name.index(joint)] = float(value)
        req.robot_state.is_diff = True
        if self._held is not None:
            req.robot_state.attached_collision_objects = [self._held_body(held_drop_m)]
        fut = self._valid_client.call_async(req)
        while not fut.done():
            self._sleep(0.005)
        result = fut.result()
        self._evidence["last_state_validity"] = {
            "valid": bool(result.valid),
            "contacts": [{"body1": c.contact_body_1, "body2": c.contact_body_2,
                          "depth_m": float(c.depth), "frame": c.header.frame_id,
                          "position_m": [float(getattr(c.position, axis)) for axis in "xyz"],
                          "normal": [float(getattr(c.normal, axis)) for axis in "xyz"]}
                         for c in result.contacts],
            "joint_positions": dict(zip(req.robot_state.joint_state.name,
                                         req.robot_state.joint_state.position)),
        }
        return bool(result.valid)

    def arm_q(self) -> list[float]:
        return [float(self._q[j]) for j in ARM_JOINT_NAMES]

    def gripper_surface_points(self, with_links: bool = False):
        """Current articulated collision-mesh vertices in TCP coordinates, from the robot model."""
        from ament_index_python.packages import get_package_share_directory
        from moveit_msgs.srv import GetPositionFK

        directory = Path(get_package_share_directory("rm_75_config")) / "meshes/eg2"
        paths = sorted(directory.glob("*.stl"))
        if not paths:
            raise StageFailure("contact_geometry", "robot gripper collision meshes unavailable", local_retry=False)
        request = GetPositionFK.Request()
        request.header.frame_id = BASE_FRAME_ID
        request.fk_link_names = [p.stem for p in paths] + [TCP_FRAME_ID]
        request.robot_state.joint_state.name = list(self._q)
        request.robot_state.joint_state.position = list(map(float, self._q.values()))
        request.robot_state.is_diff = True
        client = self._node.create_client(GetPositionFK, "/compute_fk")
        try:
            if not client.wait_for_service(timeout_sec=30.0):
                raise StageFailure("contact_geometry", "robot FK service unavailable", local_retry=False)
            future = client.call_async(request)
            while not future.done():
                self._sleep(0.005)
            result = future.result()
        finally:
            self._node.destroy_client(client)
        if result.error_code.val != 1:
            raise StageFailure("contact_geometry", "gripper FK failed", local_retry=False)
        points, point_links = [], []
        tcp = result.pose_stamped[-1].pose
        q = tcp.orientation
        R_tcp = quat_to_mat(q.x, q.y, q.z, q.w)
        p_tcp = np.array([tcp.position.x, tcp.position.y, tcp.position.z])
        dtype = np.dtype([("normal", "<f4", (3,)), ("vertices", "<f4", (3, 3)), ("attribute", "<u2")])
        for path, pose in zip(paths, result.pose_stamped[:-1]):
            triangles = np.frombuffer(path.read_bytes(), dtype=dtype, offset=84)
            vertices = np.unique(triangles["vertices"].reshape(-1, 3), axis=0)
            o, p = pose.pose.orientation, pose.pose.position
            in_base = vertices @ quat_to_mat(o.x, o.y, o.z, o.w).T + [p.x, p.y, p.z]
            points.append((in_base - p_tcp) @ R_tcp)
            point_links.extend([path.stem] * len(vertices))
        return (np.vstack(points), np.asarray(point_links)) if with_links else np.vstack(points)

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
            self._sleep(0.01)
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
                self._sleep(0.01)
            with self._cmd_lock:
                self._check_stop()
                self._joints = tuple(float(pt.positions[i]) for i in order)
        # the planned path ends where it ends: a few degrees of tracking error are not a failure (the visual servo and the
        # measured tool pose take it from here)
        self.move_joints(np.array(q_goal), stage, tol_rad=0.08, rate_rad_s=rate_rad_s, timeout_s=20.0)
        return {"waypoints": len(pts), "sim_s": round(self._clock() - t0, 1)}

    def move_to(self, p: np.ndarray, R: np.ndarray, stage: str, seed=None, rate_rad_s: float = 0.4) -> None:
        """A larger reconfiguration: one IK solution for the pose (this arm's own configuration, seeded at `seed` or
        the measured joints), then a rate-limited joint-space move."""
        q = self.ik(p, R, list(seed) if seed is not None else self.arm_q())
        if q is None:
            # Which of the two it was decides what the caller can do about it, so say it rather than "unreachable".
            why = self._evidence.get("last_ik", {})
            raise StageFailure(stage, f"the only configuration for that pose is a different one, {why['branch_jump_rad']} "
                                      f"rad away at {'/'.join(why['off_branch'])}" if why.get("off_branch") else
                                      "the arm has no configuration that reaches that pose")
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
              null_objective: Callable[[np.ndarray], np.ndarray] | None = None,
              max_speed_m_s: float | None = None, sag_integral: bool = True, posture_gain: float = 0.4,
              check_scene: bool = True, gain_per_s: float | None = None, advance_m_s: float | None = None,
              advance_ramp_s: float = 0.0, scene_from_measured: bool = False) -> dict:
        """Move the TCP (chassis_base_link) to `goal()` = (position, rotation), re-evaluated every cycle (visual
        servoing); `goal()` returning None keeps the last goal. Each cycle the goal, corrected by the integral of the
        remaining position error (arm sag under a load), goes through the arm's inverse kinematics on its working branch
        seeded at the measured joints, and the joint targets move towards the solution at a limited rate.
        A short empty-jaw insertion turns off the sag integral and the posture pull: with the joints lagging their
        targets by up to a second, both carried the tool past and beside a close-in goal (g8q replay, F78).

        `gain_per_s` is the loop's speed per metre (and per radian) of error. Left None the step is the whole remaining error every cycle
        (limited only by `max_speed_m_s`): a command that integrates the measured error at 20 per second, through an
        arm that answers a command 0.5-0.75 s late (gb7's bag: the command's peaks lead the measured joints' by that
        much), is a limit cycle for any delay over 0.3 s -- the tool swings +-1 cm about the goal and never sits inside
        a 3 mm band, and the stage ends 'arm stopped making progress 2.9 cm from its goal' (gb7r1; gb2's 2.6 cm
        stall was the same). A gain of 1 per second is stable for delays up to a second on the simulated loop
        (`experiments/servo_loop.py`) with no sag integral; that integral adds a second integrator and is only for
        stages that carry a load.

        `advance_m_s` makes the loop track a reference point that leaves the tool's starting position along the line
        to the goal at that speed, instead of the goal itself: every component of the error is then a few
        millimetres and takes the same gain. Aimed at the far goal, the speed cap is applied to the whole error
        vector, so a 12 cm axial error throttled the sideways correction by the same factor (0.42), while the arm's
        path drifts sideways by about 0.23 mm for every millimetre it advances: sideways error settled at 8-11 mm in
        gb7r4 (the model in `experiments/lateral.py` gives 11.1 mm) against a slot that leaves a finger 5 mm each
        side. At 1 cm/s that error is 2.2 mm. The tolerance, the progress test and the stall test still read the
        distance to the true goal. `advance_ramp_s` brings the reference up to speed smoothly: the arm's joints do not
        start together, and the breakaway threw the tool 9 mm high in gb7r5; the loop integrated that into its
        command, which fell 6.5 mm below the goal when the tool came back and put a finger on the collar. A model of
        that loop (`experiments/dip.py`): a ramp that halves the disturbance and a gain of 0.5 per second hold the
        command within 4 mm of the goal (a 12 mm dip at a gain of 1). Even so, gb7r7 was refused with the tool 0.3 mm
        from the goal and the command 6 mm below it: the arm settles about 6 mm above what it is told, so a loop that
        integrates the error holds a command below the goal, where the fingers are on the collar in the model and
        clear of it in the world. `scene_from_measured` asks the scene the physical question: from where the arm
        is, does the next step touch anything."""
        t0 = t_prev = self._clock()
        best, t_best, inside, last_goal, integ, blocked = math.inf, t0, 0, None, np.zeros(3), 0
        p_start, t_adv = None, t0
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
            ec = e  # what the loop controls on: the error to the goal, or to a reference point advancing towards it
            if advance_m_s is not None:
                if p_start is None:
                    p_start, t_adv = p.copy(), now
                line = gp - p_start
                reach = float(np.linalg.norm(line))
                te = now - t_adv
                travelled = advance_m_s * (te * te / (2 * advance_ramp_s) if te < advance_ramp_s else te - advance_ramp_s / 2) \
                    if advance_ramp_s else advance_m_s * te
                ec = p_start + line / max(reach, 1e-9) * min(reach, travelled) - p
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
            if sag_integral and en < 0.03:
                integ = np.clip(integ + e * dt * 0.8, -0.05, 0.05)
            # move the tool a short way along the straight line to the goal, through the Jacobian: a step, not a new
            # arm configuration
            dx = (ec + integ) * min(1.0, step_m / max(float(np.linalg.norm(ec)), 1e-6))
            if gain_per_s is not None:
                dx = dx * min(1.0, gain_per_s * dt)
            if max_speed_m_s is not None:
                dx *= min(1.0, max_speed_m_s * dt / max(float(np.linalg.norm(dx)), 1e-9))
            dw = r * min(1.0, step_rad / max(rn, 1e-6))
            if gain_per_s is not None:
                dw = dw * min(1.0, gain_per_s * dt)  # the orientation loop has the same plant: gb7r3 swung +-9 mm sideways at
                # 1.6 Hz with the position gain fixed and this one at 20/s -- a wrist swing about a 0.2 m lever
            q_meas = np.array(self.arm_q())
            q_next = self.resolved_rate_step(q_cmd, dx, dw, max_joint_rate_rad_s * dt, posture_gain=posture_gain,
                                             null_grad=None if null_objective is None else null_objective(q_meas))
            q_check = q_meas + (q_next - q_cmd) if scene_from_measured else q_next
            why = "" if not check_scene or self.state_valid(q_check) else "the robot's own planning scene"
            if not why and guard is not None:
                why = guard(q_next)
            if why:
                blocked += 1
                if blocked == 1:
                    self._evidence.setdefault("blocked_at", []).append(
                        {"stage": stage, "t": round(now - t0, 2), "measured_minus_goal_mm": [round(float(v) * 1000, 1) for v in p - gp],
                         "command_minus_goal_mm": [round(float(v) * 1000, 1) for v in self.fk(q_check)[:3, 3] - gp]})
                if blocked > 10:
                    raise StageFailure(stage, f"the arm cannot continue towards the goal without touching {why} "
                                              f"({en * 100:.1f} cm / {math.degrees(rn):.0f} deg away)")
            else:
                blocked, q_cmd = 0, q_next
                with self._cmd_lock:
                    self._check_stop()
                    self._joints = tuple(float(v) for v in q_cmd)
            if len(trace) < 200 and now - t0 - (trace[-1]["t"] if trace else -1) > 0.25:
                trace.append({"t": round(now - t0, 2), "tcp_minus_goal_mm": [round(float(v) * 1000, 1) for v in p - gp], "err_m": round(en, 3)})
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
            self._check_stop()
            self._joints = tuple(float(v) for v in q_cmd)
        return q_cmd
