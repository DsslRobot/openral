"""MoveJointsRskill — a deterministic, non-learned JOINT_POSITION skill.

``kind: "procedural"`` — closed loop against live ``world_state.joint_state``,
no learned weights, no external ROS action/service server (execution_plan.md
§8.3 item 3). Also the entrypoint for ``rskill-procedural-stow``: "stow" is
this same class dispatched with no ``goal_params_json`` override, so the
manifest's ``default_goal_json`` (the stow pose) is used verbatim — there is
no separate stow-specific behaviour to implement.

Emits one ``Action(control_mode=JOINT_POSITION, horizon=1)`` per ``step()``
holding the 7 arm-joint targets (zero-padded to the full 21-wide row F20
requires), until every arm joint is within ``tolerance_rad`` of its target,
then raises ``ROSRskillGoalSatisfied``. Raises ``ROSRuntimeError`` on
``timeout_s`` — the only other failure mode this skill can observe directly
(a target beyond the RM-75's declared range is rejected by the kernel, which
this skill has no synchronous channel to see; it simply times out, an
honest failure per ``docs/lunar_bot_capability_set_plan.md`` §3.3's table).
"""

from __future__ import annotations

import json
import time
from typing import Any

from openral_core.exceptions import ROSConfigError, ROSRskillGoalSatisfied, ROSRuntimeError
from openral_core.schemas import Action, ControlMode, RobotDescription, RSkillManifest, WorldState

from openral_rskill._lunar_bot_arm import ARM_JOINT_NAMES, full_width_joint_row, joint_positions_by_name
from openral_rskill.base import rSkillBase

__all__ = ["MoveJointsRskill"]


def _merge_goal(default: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Shallow-merge ``overrides`` over ``default`` — same rationale as
    ``procedural_body_twist._merge_goal``: this skill's goal has no nested
    dicts (``joint_targets`` is a flat list; the rest are leaves)."""
    return {**default, **overrides}


class MoveJointsRskill(rSkillBase):
    """Deterministic JOINT_POSITION rSkill for LunarBot's 7-DoF RM-75 arm.

    Goal schema (JSON object, ``ProceduralIntegration.default_goal_json``
    merged with per-dispatch ``goal_params_json``):
        joint_targets: [j1..j7] target angles (rad), in ``ARM_JOINT_NAMES``
            order. Required (no physically meaningful default beyond the
            manifest's own — see the manifest for ``move_joints`` vs ``stow``).
        tolerance_rad: per-joint convergence tolerance, default 0.02.
        timeout_s: seconds before giving up, default 15.0.
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
    ) -> None:
        del tf_lookup  # unused — this skill closes its loop on joint_state, not TF.
        if manifest.procedural is None:
            raise ROSConfigError(
                f"MoveJointsRskill requires manifest.procedural (kind={manifest.kind!r}); "
                "this manifest declares no procedural block."
            )
        if robot_description is None:
            raise ROSConfigError(
                "MoveJointsRskill requires a RobotDescription (needs the full joint "
                "order to build a kernel-valid JOINT_POSITION row, F20)."
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
        self._description = robot_description
        self._prompt = prompt
        self._prompt_metadata_json = prompt_metadata_json
        self._goal_params_json = goal_params_json
        self._goal: dict[str, Any] = {}
        self._targets_by_name: dict[str, float] = {}
        self._start_s: float = 0.0

    # ── Lifecycle ────────────────────────────────────────────────────────────

    def _configure_impl(self) -> None:
        integration = self.manifest.procedural
        assert integration is not None  # enforced by __init__
        try:
            default_goal = json.loads(integration.default_goal_json)
        except json.JSONDecodeError as exc:
            raise ROSConfigError(
                f"MoveJointsRskill({self.name!r}): default_goal_json is not valid JSON: {exc}"
            ) from exc
        goal = default_goal
        if self._goal_params_json:
            try:
                overrides = json.loads(self._goal_params_json)
            except json.JSONDecodeError as exc:
                raise ROSConfigError(
                    f"MoveJointsRskill({self.name!r}): goal_params_json from the "
                    f"reasoner / action goal is not valid JSON: {exc}"
                ) from exc
            if not isinstance(overrides, dict):
                raise ROSConfigError(
                    f"MoveJointsRskill({self.name!r}): goal_params_json must decode "
                    f"to a JSON object; got {type(overrides).__name__}."
                )
            goal = _merge_goal(default_goal, overrides)
        targets = goal.get("joint_targets")
        if not isinstance(targets, list) or len(targets) != len(ARM_JOINT_NAMES):
            raise ROSConfigError(
                f"MoveJointsRskill({self.name!r}): 'joint_targets' must be a "
                f"{len(ARM_JOINT_NAMES)}-element list; got {targets!r}."
            )
        self._goal = goal
        self._targets_by_name = dict(zip(ARM_JOINT_NAMES, (float(t) for t in targets), strict=True))

    def _activate_impl(self) -> None:
        self._start_s = time.monotonic()

    def _deactivate_impl(self) -> None:
        pass

    def _shutdown_impl(self) -> None:
        pass

    # ── Hot path ─────────────────────────────────────────────────────────────

    def _step_impl(self, world_state: WorldState) -> Action:
        timeout_s = float(self._goal.get("timeout_s", 15.0))
        tolerance_rad = float(self._goal.get("tolerance_rad", 0.02))
        elapsed_s = time.monotonic() - self._start_s

        current = joint_positions_by_name(world_state)
        errors = [
            abs(self._targets_by_name[jn] - current[jn]) for jn in ARM_JOINT_NAMES if jn in current
        ]
        if len(errors) == len(ARM_JOINT_NAMES) and max(errors) < tolerance_rad:
            raise ROSRskillGoalSatisfied(
                f"{self.name}: all {len(ARM_JOINT_NAMES)} arm joints within "
                f"{tolerance_rad} rad of target after {elapsed_s:.2f}s."
            )
        if elapsed_s >= timeout_s:
            raise ROSRuntimeError(
                f"{self.name}: timed out after {elapsed_s:.2f}s (timeout_s={timeout_s}); "
                f"max remaining joint error {max(errors) if errors else float('nan'):.4f} rad."
            )
        row = full_width_joint_row(self._description, self._targets_by_name)
        return Action(
            control_mode=ControlMode.JOINT_POSITION,
            horizon=1,
            joint_targets=[row],
            joint_names=list(ARM_JOINT_NAMES),
        )
