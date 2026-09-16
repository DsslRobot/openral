"""MoveBaseToPoseRskill — deterministic BODY_TWIST servo of the mobile base to a planar pose.

``kind: "procedural"``. Nav2 gets the base to within its goal tolerance (0.25 m here) and, on a
base that brakes as slowly as LunarBot on regolith, overshoots along track (research repo F47);
a work pose in front of a shelf has 0.2 m of clearance. This skill closes the last metre: it
servos the base on live TF (``frame_id -> base frame``) to an exact (x, y, yaw) using the
vehicle's full planar command ``(vx, vy, wz)``: a 4WIS base crabs, so the goal is entered holding
the goal heading and translating in whatever direction the remaining error points. Two phases:
**enter** (reach the corridor entry ``approach_m`` in front of the goal on its own axis, turning
to the goal heading on the way) and **approach** (translate along the axis into the goal, nulling
the lateral offset with ``vy``, at a speed the vehicle can still stop from). Docking controller,
not a planner: it assumes the approach corridor is free, which is the caller's decision (the same
contract as ``move_ee_to_pose`` — this skill encodes no mission semantics).

Raises ``ROSRskillGoalSatisfied`` once position and heading are within tolerance and the base
has settled; ``ROSRuntimeError`` on ``timeout_s`` or ``stall_timeout_s`` of no progress.
"""

from __future__ import annotations

import json
import math
import time
from typing import TYPE_CHECKING, Any

from openral_core.exceptions import ROSConfigError, ROSRskillGoalSatisfied, ROSRuntimeError
from openral_core.schemas import Action, ControlMode, RobotDescription, RSkillManifest, WorldState

from openral_rskill.base import rSkillBase

if TYPE_CHECKING:
    from openral_state_adapter import TfLookup

__all__ = ["MoveBaseToPoseRskill"]

_K_P_LINEAR = 0.8  # 1/s
_K_P_ANGULAR = 1.5  # 1/s
_K_LATERAL = 1.5  # radians of heading bias per metre of offset from the goal's axis
_LATERAL_BIAS_MAX_RAD = 0.3  # cap on that bias, so the approach never turns into a manoeuvre
_CRAWL_M_S = 0.05  # speed floor on the approach: a nonholonomic base cannot null a lateral offset at rest
_TURN_FIRST_RAD = math.radians(60.0)  # beyond this bearing error the base turns in place before driving
_PROGRESS_EPS_M = 0.005
_PROGRESS_EPS_RAD = 0.01
_TF_WARMUP_GRACE_S = 1.0


def _wrap(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def _yaw_from_quat_xyzw(q: tuple[float, float, float, float]) -> float:
    x, y, z, w = q
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


class MoveBaseToPoseRskill(rSkillBase):
    """Closed-loop BODY_TWIST servo to a planar pose in a TF frame.

    Goal schema (``ProceduralIntegration.default_goal_json`` merged with per-dispatch
    ``goal_params_json``):
        frame_id: tf2 frame of the target, default ``"odom"``.
        base_frame: the vehicle frame, default ``"chassis_base_link"``.
        position: [x, y] target of ``base_frame``'s origin in ``frame_id`` (m). Required.
        yaw: target heading of ``base_frame`` +X in ``frame_id`` (rad). Required.
        xy_tolerance_m: default 0.03.  yaw_tolerance_rad: default 0.03.
        max_linear_speed_m_s: default 0.15.  max_angular_speed_rad_s: default 0.3.
        decel_m_s2: stopping deceleration the speed profile assumes, default 0.2.
        approach_m: how far in front of the goal (along its own +X axis) the corridor entry sits,
            default 1.5.
        corridor_m: half-width of that corridor, default 0.35.
        settle_s: seconds the base must stay inside tolerance before finishing, default 0.5.
        timeout_s: default 60.  stall_timeout_s: default 8.
    """

    def __init__(
        self,
        *,
        manifest: RSkillManifest,
        robot_description: RobotDescription | None,
        prompt: str,
        prompt_metadata_json: str,
        goal_params_json: str = "",
        tf_lookup: TfLookup | None = None,
        clock: Any = None,
    ) -> None:
        del robot_description
        if manifest.procedural is None:
            raise ROSConfigError(f"MoveBaseToPoseRskill requires manifest.procedural (kind={manifest.kind!r}).")
        if tf_lookup is None:
            raise ROSConfigError(f"MoveBaseToPoseRskill({manifest.name!r}) needs a live TF lookup, but none was wired.")
        super().__init__(
            name=manifest.name,
            version=manifest.version,
            role=manifest.role,
            embodiment_tags=list(manifest.embodiment_tags),
            latency_budget_ms=(manifest.latency_budget.per_chunk_ms if manifest.latency_budget is not None else None),
        )
        self.manifest = manifest
        self._clock = clock if clock is not None else time.monotonic
        self._prompt = prompt
        self._prompt_metadata_json = prompt_metadata_json
        self._goal_params_json = goal_params_json
        self._tf_lookup: TfLookup = tf_lookup
        self._goal: dict[str, Any] = {}
        self._start_s = 0.0
        self._last_improve_s = 0.0
        self._best_dist_m = math.inf
        self._best_yaw_err_rad = math.inf
        self._inside_since_s: float | None = None
        self._reverse: bool | None = None  # chosen once, at the first fix, so the base does not flip mid-approach
        self._last_fix: tuple[float, float, float] | None = None  # (x, y, t) of the previous TF fix
        self._speed = math.inf  # ground speed over the last >= 50 ms of clock
        self._phase: str | None = None  # stage -> align -> approach (see _step_impl)
        self._phase_best = math.inf

    # ── Lifecycle ────────────────────────────────────────────────────────────

    def _configure_impl(self) -> None:
        integration = self.manifest.procedural
        assert integration is not None
        goal = json.loads(integration.default_goal_json)
        if self._goal_params_json:
            overrides = json.loads(self._goal_params_json)
            if not isinstance(overrides, dict):
                raise ROSConfigError(f"{self.name}: goal_params_json must decode to a JSON object.")
            goal = {**goal, **overrides}
        if "position" not in goal or "yaw" not in goal:
            raise ROSConfigError(f"{self.name}: goal needs 'position' [x, y] and 'yaw'; got {sorted(goal)}.")
        if len(goal["position"]) != 2:
            raise ROSConfigError(f"{self.name}: 'position' must be [x, y]; got {goal['position']!r}.")
        self._goal = goal

    def _activate_impl(self) -> None:
        self._start_s = self._clock()
        self._last_improve_s = self._start_s
        self._best_dist_m = math.inf
        self._best_yaw_err_rad = math.inf
        self._inside_since_s = None
        self._reverse = None
        self._last_fix = None
        self._speed = math.inf
        self._phase = None
        self._phase_best = math.inf

    def _deactivate_impl(self) -> None:
        pass

    def _shutdown_impl(self) -> None:
        pass

    # ── Hot path ─────────────────────────────────────────────────────────────

    def _step_impl(self, world_state: WorldState) -> Action:
        del world_state
        g = self._goal
        now = self._clock()
        elapsed_s = now - self._start_s
        frame_id = str(g.get("frame_id", "odom"))
        base_frame = str(g.get("base_frame", "chassis_base_link"))
        xy_tol = float(g.get("xy_tolerance_m", 0.03))
        yaw_tol = float(g.get("yaw_tolerance_rad", 0.03))
        v_max = float(g.get("max_linear_speed_m_s", 0.15))
        w_max = float(g.get("max_angular_speed_rad_s", 0.3))
        decel = float(g.get("decel_m_s2", 0.2))
        allow_reverse = bool(g.get("allow_reverse", True))
        settle_s = float(g.get("settle_s", 0.5))
        timeout_s = float(g.get("timeout_s", 60.0))
        stall_timeout_s = float(g.get("stall_timeout_s", 8.0))

        try:
            tf = self._tf_lookup(target_frame=frame_id, source_frame=base_frame)
        except Exception as exc:  # reason: tf2 exception types need rclpy; see move_ee_to_pose
            if elapsed_s < _TF_WARMUP_GRACE_S:
                return self._twist(0.0, 0.0)
            raise ROSRuntimeError(f"{self.name}: TF lookup {base_frame!r} in {frame_id!r} failed: {exc}") from exc

        x, y = tf.position[0], tf.position[1]
        yaw = _yaw_from_quat_xyzw(tf.quaternion_xyzw)
        tx, ty = float(g["position"][0]), float(g["position"][1])
        ex, ey = tx - x, ty - y
        dist = math.hypot(ex, ey)
        yaw_err = _wrap(float(g["yaw"]) - yaw)

        if dist < self._best_dist_m - _PROGRESS_EPS_M:
            self._best_dist_m = dist
            self._last_improve_s = now
        if abs(yaw_err) < self._best_yaw_err_rad - _PROGRESS_EPS_RAD:
            self._best_yaw_err_rad = abs(yaw_err)
            self._last_improve_s = now

        # ground speed from successive fixes: a base that brakes with a ~1 s lag can be inside
        # tolerance and still moving, so "settled" needs the pose AND the speed
        # On the sim clock the runner steps faster than /clock advances, so consecutive fixes often share
        # a stamp: keep the last estimate until the clock moves (inf on every same-stamp step never let the
        # base count as settled, and the stall timer failed a dock sitting on its goal, research repo F51).
        if self._last_fix is None:
            self._last_fix = (x, y, now)
        else:
            lx, ly, lt = self._last_fix
            if now - lt >= 0.05:
                self._speed = math.hypot(x - lx, y - ly) / (now - lt)
                self._last_fix = (x, y, now)
        speed = self._speed

        inside = dist < xy_tol and abs(yaw_err) < yaw_tol and speed < 0.02
        if inside:
            if self._inside_since_s is None:
                self._inside_since_s = now
            if now - self._inside_since_s >= settle_s:
                raise ROSRskillGoalSatisfied(
                    f"{self.name}: reached ({x:.3f}, {y:.3f}, yaw {yaw:.3f}) — dist {dist:.3f} m, "
                    f"yaw_err {yaw_err:.3f} rad after {elapsed_s:.1f}s."
                )
            return self._twist(0.0, 0.0)
        self._inside_since_s = None
        if elapsed_s >= timeout_s:
            raise ROSRuntimeError(f"{self.name}: timed out after {elapsed_s:.1f}s; dist {dist:.3f} m, yaw_err {yaw_err:.3f} rad.")
        if now - self._last_improve_s >= stall_timeout_s:
            raise ROSRuntimeError(f"{self.name}: no progress for {stall_timeout_s}s; dist {dist:.3f} m, yaw_err {yaw_err:.3f} rad.")

        # ---- docking on an omnidirectional base -------------------------------------------------
        # This 4WIS vehicle crabs: `(vx, vy, wz)` are commanded together, so the goal is entered
        # holding the goal heading and translating in whatever direction the error points. That is
        # both the vehicle's design intent and the slip-robust choice -- lateral motion rolls all four
        # wheels at one steering angle, while an in-place turn scrubs them (42-54 % of command at 4x
        # torque, research repo F50). In-place rotation stays available and is used freely; it is
        # simply no longer the ONLY way to change where the vehicle is pointing while it moves.
        #
        # The approach corridor is kept: entering along the goal's own axis is what keeps the vehicle
        # clear of whatever the work pose serves, and it is the caller's declared free space.
        approach_m = float(g.get("approach_m", 1.5))
        corridor_m = float(g.get("corridor_m", 0.35))
        d = (math.cos(float(g["yaw"])), math.sin(float(g["yaw"])))
        rel_x, rel_y = x - tx, y - ty
        along = rel_x * d[0] + rel_y * d[1]  # > 0 while still on the approach side of the goal
        lat = d[0] * rel_y - d[1] * rel_x  # signed offset from the goal's axis
        entry = (tx + d[0] * approach_m, ty + d[1] * approach_m)
        if self._phase is None:
            self._set_phase("approach" if (abs(lat) < corridor_m and -0.1 < along < approach_m + 1.0) else "enter", now)

        if self._phase == "enter":  # reach the corridor entry, already turning to the goal heading
            de = math.hypot(entry[0] - x, entry[1] - y)
            self._progress(de + abs(yaw_err), now)
            if de < 0.15 and abs(yaw_err) < 0.15:
                self._set_phase("approach", now)
            else:
                return self._body_twist(entry[0] - x, entry[1] - y, yaw, yaw_err, v_max, w_max, decel)

        # approach: hold the goal heading, translate along the axis, null the lateral offset with vy
        self._progress(dist + abs(yaw_err), now)
        if abs(lat) > 2.0 * corridor_m:  # pushed out of the corridor: come back round to the entry
            self._set_phase("enter", now)
            return self._twist(0.0, 0.0)
        return self._body_twist(tx - x, ty - y, yaw, yaw_err, v_max, w_max, decel)

    def _body_twist(self, ex: float, ey: float, yaw: float, yaw_err: float,
                    v_max: float, w_max: float, decel: float) -> Action:
        """Translate toward a world-frame error while holding a heading — the omni command this
        vehicle actually accepts. Speed is capped so it can still stop inside the remaining distance
        at its measured deceleration (~0.25 m/s² on regolith, F50)."""
        dist = math.hypot(ex, ey)
        speed = min(v_max, _K_P_LINEAR * dist, math.sqrt(2.0 * decel * dist) + _CRAWL_M_S)
        if dist > 1e-6:
            ux, uy = ex / dist * speed, ey / dist * speed
        else:
            ux = uy = 0.0
        c, sn = math.cos(yaw), math.sin(yaw)
        vx, vy = c * ux + sn * uy, -sn * ux + c * uy  # world -> body
        return self._twist(vx, _clip(_K_P_ANGULAR * yaw_err, w_max), vy)

    def _set_phase(self, phase: str, now: float) -> None:
        self._phase = phase
        self._phase_best = math.inf
        self._last_improve_s = now

    def _progress(self, err: float, now: float) -> None:
        """Stall detection against the CURRENT phase's error: driving to the staging point legitimately
        increases the distance to the goal, and aligning does not change it at all."""
        if err < self._phase_best - _PROGRESS_EPS_M:
            self._phase_best = err
            self._last_improve_s = now

    @staticmethod
    def _twist(v: float, w: float, vy: float = 0.0) -> Action:
        return Action(control_mode=ControlMode.BODY_TWIST, horizon=1, body_twist=[(v, vy, 0.0, 0.0, 0.0, w)])


def _clip(x: float, bound: float) -> float:
    return max(-bound, min(bound, x))
