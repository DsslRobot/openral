"""A new arm operation must not release an existing grasp on its first tick.

Real LunarBot/skill manifests and a recorded post-pick joint reading; no ROS or
physics substitutes. These tests check emitted command semantics, not grasp physics.
"""
import json
from pathlib import Path

import yaml
from openral_core.schemas import ControlMode, JointState, RobotDescription, RSkillManifest, WorldState
from openral_rskill._lunar_bot_arm import GRIPPER_JOINT_NAME
from openral_rskill.procedural_place import PlaceRskill

ROOT = Path(__file__).resolve().parents[2]


def recorded_place():
    manifest = RSkillManifest.model_validate(
        yaml.safe_load((ROOT / "rskills/rskill-procedural-place/rskill.yaml").read_text())
    )
    robot = RobotDescription.from_yaml(str(ROOT / "robots/lunar_bot/robot.yaml"))
    arm = json.loads((Path(__file__).parent / "fixtures/lunar_bot_held_joints.json").read_text())["arm"]
    names = [f"joint{i}" for i in range(1, 8)] + [GRIPPER_JOINT_NAME]
    state = WorldState(stamp_ns=0, joint_state=JointState(
        name=names, position=arm["joints_rad"] + [arm["jaw_rad"]], stamp_ns=0,
    ))
    return PlaceRskill(manifest=manifest, robot_description=robot, prompt="", prompt_metadata_json=""), state


def test_place_preserves_grasp_before_release():
    skill, state = recorded_place()
    first = skill._step_impl(state)
    assert not any(a.control_mode == ControlMode.GRIPPER_BINARY for a in first)
    skill.hold_here()
    actions = skill._step_impl(state)
    assert [a.control_mode for a in actions] == [ControlMode.JOINT_POSITION]


def test_explicit_release_still_emits_open_command():
    skill, state = recorded_place()
    skill._step_impl(state)
    # The recorded jaw remains closed: this call times out honestly, but its
    # command must still reach the runner rather than being suppressed.
    from openral_rskill._eye_in_hand import StageFailure
    import pytest

    with pytest.raises(StageFailure, match="jaws did not open"):
        skill.set_jaw(True, "release", timeout_s=0.1)
    actions = skill._step_impl(state)
    assert len(actions) == 1
    assert actions[0].control_mode == ControlMode.GRIPPER_BINARY
    assert actions[0].gripper == [1.0]


def test_stop_interrupts_worker_without_releasing_recorded_grasp():
    import threading

    from openral_rskill._eye_in_hand import StageFailure

    skill, state = recorded_place()
    skill._step_impl(state)
    skill._jaw_open = False  # previous explicit grasp command
    entered = threading.Event()
    stopped = []

    def wait_for_motion():
        entered.set()
        try:
            skill.wait(60.0)
        except StageFailure as exc:
            stopped.append(exc.stage)

    skill._worker = threading.Thread(target=wait_for_motion)
    skill._worker.start()
    assert entered.wait(1.0)
    actions = skill.stop_actions(state)
    skill.finish_stop()
    assert stopped == ["stopping"]
    assert not skill._worker.is_alive()
    assert actions[0].control_mode == ControlMode.JOINT_POSITION
    assert actions[1].gripper == [0.0]
    assert skill.evidence()["stop"]["worker_stopped"]
    assert not skill.evidence()["stop"]["physical_stop_confirmed"]


def test_stopped_worker_cannot_issue_a_late_release_or_twist():
    import pytest

    from openral_rskill._eye_in_hand import StageFailure

    skill, state = recorded_place()
    skill.stop_actions(state)
    with pytest.raises(StageFailure, match="operation stopped"):
        skill.set_jaw(True, "release")
    with pytest.raises(StageFailure, match="operation stopped"):
        skill.set_twist([1.0, 0, 0, 0, 0, 0])
    assert skill._jaw_open is None
    assert skill._twist is None
