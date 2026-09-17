"""MoveEEToPoseRskill — deterministic, non-learned CARTESIAN_TWIST skill.

``kind: "procedural"`` (execution_plan.md §8.3 item 3). Servos LunarBot's
gripper TCP (``tcp_frame`` — ``Link7`` + the arm's TCP offset, published by
``openral_hal_lunar_bot.sensor_bridge_node``) toward a target pose via a
proportional velocity controller on live TF, capped at the caller's declared
max speed. The *which* pose is entirely the caller's decision (from
``locate_object`` / world state) — this skill only knows how to servo to
whatever pose it is given (``docs/lunar_bot_capability_set_plan.md`` §3.3).

Raises ``ROSRskillGoalSatisfied`` once both the position and orientation
error are within tolerance; ``ROSRuntimeError`` on ``timeout_s`` or on
``stall_timeout_s`` of no measurable progress (kernel rejection has no
synchronous channel back to this skill and surfaces as one of those two,
same rationale as ``procedural_move_joints.MoveJointsRskill``).
"""

from __future__ import annotations

import json
import math
import time
from typing import TYPE_CHECKING, Any

from openral_core.exceptions import ROSConfigError, ROSRskillGoalSatisfied, ROSRuntimeError
from openral_core.schemas import Action, ControlMode, RobotDescription, RSkillManifest, WorldState

from openral_rskill._lunar_bot_arm import BASE_FRAME_ID, TCP_FRAME_ID
from openral_rskill.base import rSkillBase

if TYPE_CHECKING:
    from openral_state_adapter import TfLookup

__all__ = ["MoveEEToPoseRskill"]

#: Proportional gains (1/s) for the linear/angular velocity commanded from
#: the current pos/rot error — simple P-servo, clamped at the goal's
#: max_*_speed before it ever reaches the kernel envelope.
_K_P_LINEAR = 1.0
_K_P_ANGULAR = 1.0
#: Minimum pos_err improvement (m) that counts as "still making progress"
#: for the stall-timeout check.
_PROGRESS_EPS_M = 0.002
#: Minimum rot_err improvement (rad) that counts as progress, mirroring
#: _PROGRESS_EPS_M for the orientation axis.
_PROGRESS_EPS_RAD = 0.01
#: Grace period after activation during which a TF lookup failure (buffer
#: not warmed up yet — `tcp_frame` needs at least one sensor-bridge tick)
#: is treated as "not ready yet" rather than a hard failure.
_TF_WARMUP_GRACE_S = 1.0


def _merge_goal(default: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    return {**default, **overrides}


def _quat_mul(
    a: tuple[float, float, float, float], b: tuple[float, float, float, float]
) -> tuple[float, float, float, float]:
    """Hamilton product of two ``[x, y, z, w]`` quaternions, ``a * b``."""
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    )


def _quat_conjugate(
    q: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    x, y, z, w = q
    return (-x, -y, -z, w)


class MoveEEToPoseRskill(rSkillBase):
    """Deterministic CARTESIAN_TWIST servo rSkill for LunarBot's gripper TCP.

    Goal schema (JSON object, ``ProceduralIntegration.default_goal_json``
    merged with per-dispatch ``goal_params_json``):
        frame_id: tf2 frame the target pose is expressed in, default
            ``"chassis_base_link"``.
        position: [x, y, z] target TCP position (m) in ``frame_id``. Required.
        quaternion_xyzw: [x, y, z, w] target TCP orientation in ``frame_id``.
            Required.
        pos_tolerance_m: default 0.02.
        rot_tolerance_rad: default 0.05.
        max_linear_speed_m_s: default 0.15 (under the kernel's
            ``max_ee_speed_m_s: 0.2`` envelope, ``robots/lunar_bot/robot.yaml``).
        max_angular_speed_rad_s: default 0.2 (under ``max_ee_angular_speed_rad_s: 0.3``).
        timeout_s: default 20.0.
        stall_timeout_s: default 4.0.
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
        del robot_description  # unused — this skill closes its loop on TF, not joint_state.
        if manifest.procedural is None:
            raise ROSConfigError(
                f"MoveEEToPoseRskill requires manifest.procedural (kind={manifest.kind!r}); "
                "this manifest declares no procedural block."
            )
        if tf_lookup is None:
            raise ROSConfigError(
                f"MoveEEToPoseRskill({manifest.name!r}) needs a live TF lookup to servo "
                "against tcp_frame, but none was wired. The runner only supplies one from "
                "on_configure()'s _init_tf_lookup() — see make_default_skill_resolver's "
                "procedural branch / tf_lookup_getter."
            )
        super().__init__(
            name=manifest.name,
            version=manifest.version,
            role=manifest.role,
            embodiment_tags=list(manifest.embodiment_tags),
            latency_budget_ms=(
                manifest.latency_budget.per_chunk_ms if manifest.latency_budget is not None else None
            ),
        )
        self.manifest = manifest
        self._clock = clock if clock is not None else time.monotonic
        self._prompt = prompt
        self._prompt_metadata_json = prompt_metadata_json
        self._goal_params_json = goal_params_json
        self._tf_lookup: TfLookup = tf_lookup
        self._goal: dict[str, Any] = {}
        self._target_frame: str = BASE_FRAME_ID
        self._target_pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
        self._target_quat: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
        self._start_s: float = 0.0
        self._best_pos_err_m: float = math.inf
        self._best_rot_err_rad: float = math.inf
        self._last_improve_s: float = 0.0

    # ── Lifecycle ────────────────────────────────────────────────────────────

    def _configure_impl(self) -> None:
        integration = self.manifest.procedural
        assert integration is not None  # enforced by __init__
        try:
            default_goal = json.loads(integration.default_goal_json)
        except json.JSONDecodeError as exc:
            raise ROSConfigError(
                f"MoveEEToPoseRskill({self.name!r}): default_goal_json is not valid JSON: {exc}"
            ) from exc
        goal = default_goal
        if self._goal_params_json:
            try:
                overrides = json.loads(self._goal_params_json)
            except json.JSONDecodeError as exc:
                raise ROSConfigError(
                    f"MoveEEToPoseRskill({self.name!r}): goal_params_json from the "
                    f"reasoner / action goal is not valid JSON: {exc}"
                ) from exc
            if not isinstance(overrides, dict):
                raise ROSConfigError(
                    f"MoveEEToPoseRskill({self.name!r}): goal_params_json must decode "
                    f"to a JSON object; got {type(overrides).__name__}."
                )
            goal = _merge_goal(default_goal, overrides)
        position = goal.get("position")
        quat = goal.get("quaternion_xyzw")
        if not isinstance(position, list) or len(position) != 3:
            raise ROSConfigError(
                f"MoveEEToPoseRskill({self.name!r}): 'position' must be a 3-element "
                f"list; got {position!r}."
            )
        if not isinstance(quat, list) or len(quat) != 4:
            raise ROSConfigError(
                f"MoveEEToPoseRskill({self.name!r}): 'quaternion_xyzw' must be a "
                f"4-element list; got {quat!r}."
            )
        self._goal = goal
        self._target_frame = str(goal.get("frame_id", BASE_FRAME_ID))
        self._target_pos = tuple(float(v) for v in position)  # type: ignore[assignment]
        self._target_quat = tuple(float(v) for v in quat)  # type: ignore[assignment]

    def _activate_impl(self) -> None:
        self._start_s = self._clock()
        self._best_pos_err_m = math.inf
        self._best_rot_err_rad = math.inf
        self._last_improve_s = self._start_s
        self._stopped = False

    def _deactivate_impl(self) -> None:
        pass

    def _shutdown_impl(self) -> None:
        pass

    # ── Hot path ─────────────────────────────────────────────────────────────

    def _step_impl(self, world_state: WorldState) -> Action:
        del world_state  # this skill closes its loop on live TF, not joint_state.
        now = self._clock()
        elapsed_s = now - self._start_s
        timeout_s = float(self._goal.get("timeout_s", 20.0))
        stall_timeout_s = float(self._goal.get("stall_timeout_s", 4.0))
        pos_tol_m = float(self._goal.get("pos_tolerance_m", 0.02))
        rot_tol_rad = float(self._goal.get("rot_tolerance_rad", 0.05))
        max_linear = float(self._goal.get("max_linear_speed_m_s", 0.15))
        max_angular = float(self._goal.get("max_angular_speed_rad_s", 0.2))

        try:
            tf = self._tf_lookup(target_frame=self._target_frame, source_frame=TCP_FRAME_ID)
        except Exception as exc:  # reason: tf2 lookup exception types require rclpy/tf2_ros,
            # which this module deliberately does not import (unit-testable without ROS,
            # same rationale as procedural_body_twist.py). Only tolerated during the
            # startup grace window — the sensor bridge needs one tick to publish
            # `tcp_frame` before the buffer has anything to interpolate against.
            if elapsed_s < _TF_WARMUP_GRACE_S:
                return Action(control_mode=ControlMode.CARTESIAN_TWIST, horizon=1, cartesian_twist=[(0.0,) * 6])
            raise ROSRuntimeError(
                f"{self.name}: TF lookup of {TCP_FRAME_ID!r} in {self._target_frame!r} "
                f"failed after the {_TF_WARMUP_GRACE_S}s warm-up grace period: {exc}"
            ) from exc

        cx, cy, cz = tf.position
        cqx, cqy, cqz, cqw = tf.quaternion_xyzw

        tx, ty, tz = self._target_pos
        ex, ey, ez = tx - cx, ty - cy, tz - cz
        pos_err_m = math.sqrt(ex * ex + ey * ey + ez * ez)

        qx, qy, qz, qw = _quat_mul(self._target_quat, _quat_conjugate((cqx, cqy, cqz, cqw)))
        if qw < 0.0:  # shortest-path convention (q and -q are the same rotation)
            qx, qy, qz, qw = -qx, -qy, -qz, -qw
        rot_err_rad = 2.0 * math.atan2(math.hypot(qx, qy, qz), qw)

        # Progress on EITHER axis resets the stall clock -- position can
        # legitimately converge well before orientation does (or vice
        # versa) for a redundant 7-DoF arm's IK solve, so gating the stall
        # timeout on position alone would time out a run that is still
        # honestly converging its orientation (or the reverse).
        made_progress = False
        if pos_err_m < self._best_pos_err_m - _PROGRESS_EPS_M:
            self._best_pos_err_m = pos_err_m
            made_progress = True
        if rot_err_rad < self._best_rot_err_rad - _PROGRESS_EPS_RAD:
            self._best_rot_err_rad = rot_err_rad
            made_progress = True
        if made_progress:
            self._last_improve_s = now

        if pos_err_m < pos_tol_m and rot_err_rad < rot_tol_rad:
            # Stop before completing: the simulator keeps applying the last twist it received until the next command,
            # and its twist integrator holds the pose under a zero twist. Completing on a nonzero correction twist
            # left the arm creeping at K_p * pos_err (1.5 cm/s at 0.015 m) while the caller thought (research ob5).
            if not self._stopped:
                self._stopped = True
                return Action(control_mode=ControlMode.CARTESIAN_TWIST, horizon=1, cartesian_twist=[(0.0,) * 6],
                              frame_id=self._target_frame)
            raise ROSRskillGoalSatisfied(
                f"{self.name}: reached target (pos_err={pos_err_m:.4f} m, "
                f"rot_err={rot_err_rad:.4f} rad) after {elapsed_s:.2f}s."
            )
        if elapsed_s >= timeout_s:
            raise ROSRuntimeError(
                f"{self.name}: timed out after {elapsed_s:.2f}s (timeout_s={timeout_s}); "
                f"pos_err={pos_err_m:.4f} m, rot_err={rot_err_rad:.4f} rad remaining."
            )
        if now - self._last_improve_s >= stall_timeout_s:
            raise ROSRuntimeError(
                f"{self.name}: no progress for {stall_timeout_s}s (best pos_err="
                f"{self._best_pos_err_m:.4f} m / current {pos_err_m:.4f} m; "
                f"best rot_err={self._best_rot_err_rad:.4f} rad / current {rot_err_rad:.4f} rad)."
            )

        # Deliberately NOT gated to zero once "close enough": SRB's arm_ik
        # action term re-syncs its target to the CURRENT joint state every
        # step and provides no passive holding stiffness against drift
        # (openral_hal.lunar_bot_srb's module docstring, confirmed live
        # here too -- commanding zero twist on a converged axis let it
        # drift right back out of tolerance a few ticks later, an on/off
        # limit cycle around the tolerance boundary rather than a stable
        # hold). Continuous small proportional correction, even this close
        # to the target, is the honest way to counteract that.
        if pos_err_m > 1e-6:
            linear_speed = min(max_linear, _K_P_LINEAR * pos_err_m)
            vx, vy, vz = (
                ex / pos_err_m * linear_speed,
                ey / pos_err_m * linear_speed,
                ez / pos_err_m * linear_speed,
            )
        else:
            vx, vy, vz = (0.0, 0.0, 0.0)

        axis_norm = math.hypot(qx, qy, qz)
        if axis_norm > 1e-6:
            angular_speed = min(max_angular, _K_P_ANGULAR * rot_err_rad)
            wx, wy, wz = (
                qx / axis_norm * angular_speed,
                qy / axis_norm * angular_speed,
                qz / axis_norm * angular_speed,
            )
        else:
            wx, wy, wz = (0.0, 0.0, 0.0)

        return Action(
            control_mode=ControlMode.CARTESIAN_TWIST,
            horizon=1,
            cartesian_twist=[(vx, vy, vz, wx, wy, wz)],
            frame_id=self._target_frame,
        )
