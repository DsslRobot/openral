"""GripperRskill outcome criteria on synthetic jaw readings (real manifests + real lunar_bot description).

The physical facts these encode (research repo F49/F50): the EG2's mimic-coupled pads meet at
``GRIPPER_CLOSED_RAD`` (0.10 rad) and chatter there by a few mrad, so an empty close must read
"nothing grasped" from a position window, not a velocity threshold; a T-handle neck stops the jaw
at ~0.30 rad ("grasped"); the open stroke is 0.82 rad.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from openral_core.exceptions import ROSRskillGoalSatisfied, ROSRuntimeError
from openral_core.schemas import JointState, RobotDescription, RSkillManifest, RSkillState, WorldState

from openral_rskill._lunar_bot_arm import GRIPPER_CLOSED_RAD, GRIPPER_STROKE_RAD
from openral_rskill.procedural_gripper import GripperRskill

ROOT = Path(__file__).resolve().parents[2]


def _manifest(name: str) -> RSkillManifest:
    return RSkillManifest.model_validate(yaml.safe_load((ROOT / f"rskills/{name}/rskill.yaml").read_text()))


def _skill(name: str) -> GripperRskill:
    skill = GripperRskill(
        manifest=_manifest(name),
        robot_description=RobotDescription.from_yaml(str(ROOT / "robots/lunar_bot/robot.yaml")),
        prompt="test",
        prompt_metadata_json="",
    )
    skill.configure()
    skill.activate()
    assert skill.info.state is RSkillState.ACTIVE
    return skill


def _ws(jaw: float, t_ns: int) -> WorldState:
    return WorldState(
        joint_state=JointState(name=["eg2_joint1"], position=[jaw], velocity=[0.0], stamp_ns=t_ns),
        stamp_ns=t_ns,
    )


def _drive(skill: GripperRskill, readings: list[float]) -> None:
    for i, jaw in enumerate(readings):
        skill.step(_ws(jaw, i * 33_000_000))


def test_empty_close_settling_at_the_pad_contact_reads_nothing_grasped() -> None:
    skill = _skill("rskill-procedural-grasp")
    travel = [GRIPPER_STROKE_RAD - 0.02 * i for i in range(36)]  # 0.82 -> 0.12 at ~0.6 rad/s
    chatter = [GRIPPER_CLOSED_RAD + d for d in (0.004, 0.001, 0.003, 0.0, 0.002, 0.001, 0.003)]
    with pytest.raises(ROSRuntimeError, match="nothing grasped"):
        _drive(skill, travel + chatter)


def test_close_stopping_on_a_handle_reads_grasped() -> None:
    skill = _skill("rskill-procedural-grasp")
    travel = [GRIPPER_STROKE_RAD - 0.02 * i for i in range(27)]  # 0.82 -> 0.30
    with pytest.raises(ROSRskillGoalSatisfied, match="something grasped"):
        _drive(skill, travel + [0.296, 0.297, 0.296, 0.296, 0.297])


def test_pre_motion_stillness_is_not_a_grasp() -> None:
    """The jaw sits still at the open stroke before the command bites — not "grasped"."""
    skill = _skill("rskill-procedural-grasp")
    _drive(skill, [GRIPPER_STROKE_RAD] * 8)  # must not raise
    with pytest.raises(ROSRskillGoalSatisfied):
        _drive(skill, [GRIPPER_STROKE_RAD - 0.02 * i for i in range(1, 27)] + [0.3] * 5)


def test_open_reaches_the_stroke() -> None:
    skill = _skill("rskill-procedural-release")
    with pytest.raises(ROSRskillGoalSatisfied, match="open stroke"):
        _drive(skill, [0.1 + 0.02 * i for i in range(40)])
