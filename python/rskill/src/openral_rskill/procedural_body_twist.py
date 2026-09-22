"""ProceduralBodyTwistRskill — a deterministic, non-learned BODY_TWIST skill.

Resolved when ``RSkillManifest.kind == "procedural"`` and
``procedural.entrypoint`` names this class (see
``openral_core.schemas.ProceduralIntegration``). Exists for procedure-driven /
non-learned mission execution (execution_plan.md §5.2's Baseline A) — no
existing rSkill kind fits "a scripted motion with no weights and no external
ROS action/service server to wrap" (``vla`` requires ``model_family`` +
``weights_uri`` and routes through the policy-adapter factory;
``ros_action``/``ros_service`` require a running server this project does not
own; ``playbook`` is a reasoner-side decision procedure that never emits an
``Action``).

Emits a fixed ``(linear, angular)`` body-twist velocity, one
``Action(control_mode=BODY_TWIST, horizon=1)`` per ``step()`` call, for
``duration_s`` seconds, then raises ``ROSRskillGoalSatisfied`` — the same
termination contract ``ROSActionRskill``'s result-only mode uses.
"""

from __future__ import annotations

import json
import time
from typing import Any

from openral_core.exceptions import ROSConfigError, ROSRskillGoalSatisfied
from openral_core.schemas import Action, ControlMode, RobotDescription, RSkillManifest, WorldState

from openral_rskill.base import rSkillBase

__all__ = ["ProceduralBodyTwistRskill"]


def _merge_goal(default: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Shallow-merge ``overrides`` over ``default`` — this skill's goal has no
    nested dicts (``linear``/``angular``/``duration_s`` are all leaves or flat
    lists), so a shallow merge is the honest implementation, not a
    simplification of ``RosIntegration``'s deep-merge contract."""
    return {**default, **overrides}


class ProceduralBodyTwistRskill(rSkillBase):
    """Deterministic BODY_TWIST rSkill — no learned weights, no ROS server.

    Args mirror ``ROSActionRskill``'s constructor shape so the runner's call
    site stays uniform across ``kind`` values.

    Goal schema (JSON object, ``ProceduralIntegration.default_goal_json``
    merged with per-dispatch ``goal_params_json``):
        linear: [vx, vy, vz] (m/s), default [0, 0, 0].
        angular: [wx, wy, wz] (rad/s), default [0, 0, 0].
        duration_s: seconds to hold the twist before finishing, default 1.0.
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
        ros_node: Any = None,  # the runner node; unused -- this skill subscribes to nothing
    ) -> None:
        del tf_lookup  # unused — a fixed body-twist hold needs no TF feedback;
        # accepted so every `kind: procedural` skill shares one constructor
        # shape (`make_default_skill_resolver`'s procedural branch forwards
        # it uniformly — see `procedural_move_ee_to_pose.py` for the skill
        # that actually consumes it).
        if manifest.procedural is None:
            raise ROSConfigError(
                f"ProceduralBodyTwistRskill requires manifest.procedural (kind={manifest.kind!r}); "
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
        self._description = robot_description
        self._prompt = prompt
        self._prompt_metadata_json = prompt_metadata_json
        self._goal_params_json = goal_params_json
        self._goal: dict[str, Any] = {}
        self._start_s: float = 0.0

    # ── Lifecycle ────────────────────────────────────────────────────────────

    def _configure_impl(self) -> None:
        integration = self.manifest.procedural
        assert integration is not None  # enforced by __init__ + manifest validator
        try:
            default_goal = json.loads(integration.default_goal_json)
        except json.JSONDecodeError as exc:
            raise ROSConfigError(
                f"ProceduralBodyTwistRskill({self.name!r}): default_goal_json is not valid JSON: {exc}"
            ) from exc
        goal = default_goal
        if self._goal_params_json:
            try:
                overrides = json.loads(self._goal_params_json)
            except json.JSONDecodeError as exc:
                raise ROSConfigError(
                    f"ProceduralBodyTwistRskill({self.name!r}): goal_params_json from the "
                    f"reasoner / action goal is not valid JSON: {exc}"
                ) from exc
            if not isinstance(overrides, dict):
                raise ROSConfigError(
                    f"ProceduralBodyTwistRskill({self.name!r}): goal_params_json must "
                    f"decode to a JSON object; got {type(overrides).__name__}."
                )
            goal = _merge_goal(default_goal, overrides)
        linear = goal.get("linear", [0.0, 0.0, 0.0])
        angular = goal.get("angular", [0.0, 0.0, 0.0])
        if len(linear) != 3 or len(angular) != 3:
            raise ROSConfigError(
                f"ProceduralBodyTwistRskill({self.name!r}): 'linear'/'angular' must each "
                f"have 3 components; got linear={linear!r} angular={angular!r}."
            )
        self._goal = goal

    def _activate_impl(self) -> None:
        self._start_s = self._clock()

    def _deactivate_impl(self) -> None:
        pass

    def _shutdown_impl(self) -> None:
        pass

    # ── Hot path ─────────────────────────────────────────────────────────────

    def _step_impl(self, world_state: WorldState) -> Action:
        duration_s = float(self._goal.get("duration_s", 1.0))
        elapsed_s = self._clock() - self._start_s
        if elapsed_s >= duration_s:
            raise ROSRskillGoalSatisfied(
                f"{self.name}: body_twist held for {elapsed_s:.2f}s (duration_s={duration_s})."
            )
        linear = self._goal.get("linear", [0.0, 0.0, 0.0])
        angular = self._goal.get("angular", [0.0, 0.0, 0.0])
        vx, vy, vz = (float(v) for v in linear)
        wx, wy, wz = (float(v) for v in angular)
        return Action(
            control_mode=ControlMode.BODY_TWIST,
            horizon=1,
            body_twist=[(vx, vy, vz, wx, wy, wz)],
        )
