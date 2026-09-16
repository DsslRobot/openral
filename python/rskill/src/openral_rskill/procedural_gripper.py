"""GripperRskill — deterministic, non-learned GRIPPER_BINARY skill.

``kind: "procedural"`` (execution_plan.md §8.3 item 3). Serves both
``rskill-procedural-grasp`` (``mode: "close"``) and
``rskill-procedural-release`` (``mode: "open"``) — one class, since both
issue the same command every step and differ only in which joint-state
reading counts as done vs. failed (grasp: something between the jaws stops
the closing motion short of the fully-closed stroke; release: the jaws
simply reach the open stroke).

Commands a fixed GRIPPER_BINARY target every ``step()`` (SRB zero-order-holds
the last published command — same rationale as
``openral_hal.lunar_bot_srb``'s CARTESIAN_TWIST doc comment) and reads
``eg2_joint1``'s raw position/velocity from ``world_state.joint_state`` to
detect the physical outcome:

* **close (grasp):** done when the jaw has stopped moving (its position
  changed by less than ``_STOPPED_WINDOW_RAD`` over ``stable_steps``
  consecutive calls — a position window, because the mimic-coupled pads
  chatter at the closed stop and a velocity test never settled there,
  research repo F50) short of the closed stroke — something is between the
  jaws. Fails (``ROSRuntimeError``) if the jaw stops at the closed stroke
  (nothing grasped: ``GRIPPER_CLOSED_RAD`` is the pads' contact position)
  or if ``timeout_s`` elapses first.
* **open (release):** done when the jaw reaches the open stroke. Fails on
  ``timeout_s``.
"""

from __future__ import annotations

import json
import time
from typing import Any, Literal

from openral_core.exceptions import ROSConfigError, ROSRskillGoalSatisfied, ROSRuntimeError
from openral_core.schemas import Action, ControlMode, RobotDescription, RSkillManifest, WorldState

from openral_rskill._lunar_bot_arm import (
    GRIPPER_CLOSED_RAD,
    GRIPPER_JOINT_NAME,
    GRIPPER_STROKE_RAD,
    joint_positions_by_name,
    joint_velocities_by_name,
)
from openral_rskill.base import rSkillBase

__all__ = ["GripperRskill"]

#: Fraction of the full stroke a jaw must clear the closed end before a
#: stopped-motion reading counts as "something grasped" rather than
#: measurement noise at the fully-closed limit.
_CLOSED_MARGIN_RAD = 0.03
#: Fraction of the full stroke the jaw must reach before "open" counts as
#: satisfied — allows for actuator settling short of the exact limit.
_OPEN_MARGIN_RAD = 0.03
#: Minimum travel from the jaw's position when "close" was dispatched before
#: a stopped reading is trusted as "closed on an object" rather than the
#: pre-motion transient (the command has not reached the actuator yet, so
#: velocity still reads ~0 at the very first step()) being misread as
#: "already stopped, must be grasping something".
_CLOSE_MOTION_MARGIN_RAD = 0.05
#: Largest position change (rad) over ``stable_steps`` consecutive readings that still counts
#: as "stopped". Normal jaw travel is ~0.02 rad per 30 Hz step; pad chatter at the closed stop
#: is far below this, so the window settles where a velocity threshold kept resetting.
_STOPPED_WINDOW_RAD = 0.005


def _merge_goal(default: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    return {**default, **overrides}


class GripperRskill(rSkillBase):
    """Deterministic GRIPPER_BINARY rSkill for LunarBot's EG2-4C2 gripper.

    Goal schema (JSON object, ``ProceduralIntegration.default_goal_json``
    merged with per-dispatch ``goal_params_json``):
        mode: ``"close"`` (grasp) or ``"open"`` (release). Required — set by
            the manifest's default; the reasoner is not expected to override it.
        timeout_s: seconds before giving up, default 5.0.
        stable_steps: consecutive low-velocity ``step()`` calls required
            before a "close" motion counts as stopped, default 3.
        velocity_eps_rad_s: velocity magnitude below which the jaw counts as
            stationary, default 0.01.
    """

    def __init__(
        self,
        *,
        manifest: RSkillManifest,
        robot_description: RobotDescription | None,
        prompt: str,
        prompt_metadata_json: str,
        goal_params_json: str = "",
        tf_lookup: Any = None,
        clock: Any = None,
    ) -> None:
        del tf_lookup, robot_description  # unused — reads one named joint, not TF.
        if manifest.procedural is None:
            raise ROSConfigError(
                f"GripperRskill requires manifest.procedural (kind={manifest.kind!r}); "
                "this manifest declares no procedural block."
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
        self._goal: dict[str, Any] = {}
        self._mode: Literal["close", "open"] = "close"
        self._start_s: float = 0.0
        self._stable_count: int = 0
        self._close_start_position: float | None = None
        self._positions: list[float] = []  # recent jaw readings for the stopped-position window
        self._last_sample_s: float | None = None

    # ── Lifecycle ────────────────────────────────────────────────────────────

    def _configure_impl(self) -> None:
        integration = self.manifest.procedural
        assert integration is not None  # enforced by __init__
        try:
            default_goal = json.loads(integration.default_goal_json)
        except json.JSONDecodeError as exc:
            raise ROSConfigError(
                f"GripperRskill({self.name!r}): default_goal_json is not valid JSON: {exc}"
            ) from exc
        goal = default_goal
        if self._goal_params_json:
            try:
                overrides = json.loads(self._goal_params_json)
            except json.JSONDecodeError as exc:
                raise ROSConfigError(
                    f"GripperRskill({self.name!r}): goal_params_json from the "
                    f"reasoner / action goal is not valid JSON: {exc}"
                ) from exc
            if not isinstance(overrides, dict):
                raise ROSConfigError(
                    f"GripperRskill({self.name!r}): goal_params_json must decode "
                    f"to a JSON object; got {type(overrides).__name__}."
                )
            goal = _merge_goal(default_goal, overrides)
        mode = goal.get("mode")
        if mode not in ("close", "open"):
            raise ROSConfigError(
                f"GripperRskill({self.name!r}): 'mode' must be 'close' or 'open'; got {mode!r}."
            )
        self._goal = goal
        self._mode = mode

    def _activate_impl(self) -> None:
        self._start_s = self._clock()
        self._stable_count = 0
        self._close_start_position = None
        self._positions = []
        self._last_sample_s = None

    def _deactivate_impl(self) -> None:
        pass

    def _shutdown_impl(self) -> None:
        pass

    # ── Hot path ─────────────────────────────────────────────────────────────

    def _step_impl(self, world_state: WorldState) -> Action:
        timeout_s = float(self._goal.get("timeout_s", 5.0))
        stable_steps = int(self._goal.get("stable_steps", 3))
        velocity_eps = float(self._goal.get("velocity_eps_rad_s", 0.01))
        now = self._clock()
        elapsed_s = now - self._start_s

        del velocity_eps  # accepted for goal compatibility; "stopped" is a position window now
        positions = joint_positions_by_name(world_state)
        position = positions.get(GRIPPER_JOINT_NAME)

        # stopped: the last `stable_steps` readings (plus the one before them) span < window. Readings are
        # sampled >= 50 ms apart on the clock: the runner steps faster than a slow sim advances, and
        # identical same-frame readings mid-stroke would pass for a stopped jaw (research repo F51).
        if position is not None and (self._last_sample_s is None or now - self._last_sample_s >= 0.05):
            self._last_sample_s = now
            self._positions.append(position)
            recent = self._positions[-(stable_steps + 1):]
            if len(recent) == stable_steps + 1 and max(recent) - min(recent) < _STOPPED_WINDOW_RAD:
                self._stable_count += 1
            else:
                self._stable_count = 0

        if self._mode == "close":
            if position is not None and self._close_start_position is None:
                self._close_start_position = position
            if position is not None and self._close_start_position is not None:
                fully_closed = position <= GRIPPER_CLOSED_RAD + _CLOSED_MARGIN_RAD
                stopped = self._stable_count >= 1
                moved_enough = (
                    self._close_start_position - position
                ) >= _CLOSE_MOTION_MARGIN_RAD
                if fully_closed and stopped:
                    raise ROSRuntimeError(
                        f"{self.name}: jaw reached the fully-closed stroke "
                        f"(position={position:.4f} rad) — nothing grasped."
                    )
                if moved_enough and stopped and not fully_closed:
                    raise ROSRskillGoalSatisfied(
                        f"{self.name}: jaw stopped at position={position:.4f} rad, short of "
                        f"the fully-closed stroke after {elapsed_s:.2f}s — something grasped."
                    )
            if elapsed_s >= timeout_s:
                raise ROSRuntimeError(
                    f"{self.name}: timed out after {elapsed_s:.2f}s (timeout_s={timeout_s}) "
                    "without the jaw ever settling."
                )
            commanded_open = False
        else:  # "open"
            if position is not None and position >= GRIPPER_STROKE_RAD - _OPEN_MARGIN_RAD:
                raise ROSRskillGoalSatisfied(
                    f"{self.name}: jaw reached the open stroke (position={position:.4f} rad) "
                    f"after {elapsed_s:.2f}s."
                )
            if elapsed_s >= timeout_s:
                raise ROSRuntimeError(
                    f"{self.name}: timed out after {elapsed_s:.2f}s (timeout_s={timeout_s}) "
                    "before reaching the open stroke."
                )
            commanded_open = True

        return Action(control_mode=ControlMode.GRIPPER_BINARY, horizon=1, gripper=[1.0 if commanded_open else 0.0])
