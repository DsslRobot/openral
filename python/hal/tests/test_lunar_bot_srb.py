"""LunarBotSRBHAL: unifies BODY_TWIST (base) + CARTESIAN_TWIST/JOINT_POSITION
(arm) + GRIPPER_BINARY (gripper) into one mobile-manipulator HAL.

Uses the real ``robots/lunar_bot/robot.yaml`` manifest (CLAUDE.md §1.11 —
real fixtures, not placeholders) with an injected ``publish_fn`` capturing
what would have gone out over ROS — the sanctioned test seam at the
process boundary (the constructor's own docstring calls this the
``SimTransport``-shaped injection point), not a mock of the HAL itself.
"""

from __future__ import annotations

from typing import Any

import pytest


def _build_hal() -> tuple[Any, list[tuple[str, dict[str, object]]]]:
    from openral_core import RobotDescription
    from openral_hal.lunar_bot_srb import LunarBotSRBHAL

    published: list[tuple[str, dict[str, object]]] = []

    def _publish_fn(topic: str, msg: dict[str, object]) -> None:
        published.append((topic, msg))

    desc = RobotDescription.from_yaml("robots/lunar_bot/robot.yaml")
    hal = LunarBotSRBHAL(desc, publish_fn=_publish_fn, state_fn=lambda: {})
    hal.connect()
    return hal, published


def test_body_twist_forwards_physical_units_unconverted() -> None:
    """BODY_TWIST is a direct m/s / rad/s passthrough to cmd_vel (unchanged by this PR)."""
    from openral_core.schemas import Action, ControlMode

    hal, published = _build_hal()
    action = Action(
        control_mode=ControlMode.BODY_TWIST,
        horizon=1,
        body_twist=[(0.2, 0.0, 0.0, 0.0, 0.0, 0.1)],
    )
    hal.send_action(action)

    assert len(published) == 1
    topic, msg = published[0]
    assert topic == hal.cmd_vel_topic
    assert msg["linear"] == {"x": 0.2, "y": 0.0, "z": 0.0}  # type: ignore[comparison-overlap]
    assert msg["angular"] == {"x": 0.0, "y": 0.0, "z": 0.1}  # type: ignore[comparison-overlap]


def test_cartesian_twist_converts_physical_units_to_srb_raw() -> None:
    """CARTESIAN_TWIST divides by the calibrated raw-per-m/s (rad/s) constants."""
    from openral_core.schemas import Action, ControlMode
    from openral_hal.lunar_bot_srb import (
        _ARM_ANGULAR_RADPS_PER_RAW_UNIT,
        _ARM_LINEAR_MPS_PER_RAW_UNIT,
    )

    hal, published = _build_hal()
    action = Action(
        control_mode=ControlMode.CARTESIAN_TWIST,
        horizon=1,
        cartesian_twist=[(0.1, 0.0, 0.0, 0.0, 0.0, 0.2)],
    )
    hal.send_action(action)

    assert len(published) == 1
    topic, msg = published[0]
    assert topic == hal.arm_topic
    assert msg["mode"] == 0.0
    lin = msg["ik_linear"]  # type: ignore[index]
    ang = msg["ik_angular"]  # type: ignore[index]
    assert lin[0] == pytest.approx(0.1 / _ARM_LINEAR_MPS_PER_RAW_UNIT)  # type: ignore[index]
    assert lin[1] == pytest.approx(0.0)  # type: ignore[index]
    assert ang[2] == pytest.approx(0.2 / _ARM_ANGULAR_RADPS_PER_RAW_UNIT)  # type: ignore[index]
    assert msg["joint_targets"] == (0.0,) * 7  # type: ignore[comparison-overlap]


def _full_width_row_with_arm_targets(arm_targets: tuple[float, ...]) -> list[float]:
    """Build the full 21-wide RobotDescription.joints-order row a real
    JOINT_POSITION dispatch must send -- the safety kernel's structural
    check forces every JOINT_* chunk's n_dof to equal the envelope's full
    joint count, so a 7-wide arm-only row would be rejected before this
    HAL ever saw it (confirmed live -- see docs/... for the E2E test this
    mirrors). Non-arm slots zero-padded.
    """
    from openral_core import RobotDescription

    desc = RobotDescription.from_yaml("robots/lunar_bot/robot.yaml")
    joint_names = [j.name for j in desc.joints]
    row = [0.0] * len(joint_names)
    for jn, val in zip(
        ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"), arm_targets
    ):
        row[joint_names.index(jn)] = val
    return row


def test_joint_position_extracts_arm_slice_with_explicit_joint_names() -> None:
    """JOINT_POSITION with action.joint_names set (ADR-0102) as an extra
    check that the emitter meant the 7 arm joints -- the row itself is
    still the full 21-wide vector regardless."""
    from openral_core.schemas import Action, ControlMode
    from openral_hal.lunar_bot_srb import _ARM_JOINT_NAMES

    hal, published = _build_hal()
    arm_targets = (0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7)
    action = Action(
        control_mode=ControlMode.JOINT_POSITION,
        horizon=1,
        joint_targets=[_full_width_row_with_arm_targets(arm_targets)],
        joint_names=list(_ARM_JOINT_NAMES),
    )
    hal.send_action(action)

    assert len(published) == 1
    topic, msg = published[0]
    assert topic == hal.arm_topic
    assert msg["mode"] == 1.0
    assert msg["joint_targets"] == arm_targets  # type: ignore[comparison-overlap]
    assert msg["ik_linear"] == (0.0, 0.0, 0.0)  # type: ignore[comparison-overlap]


def test_joint_position_extracts_arm_slice_without_joint_names() -> None:
    """JOINT_POSITION with no joint_names: the whole-vector convention still
    resolves correctly since the row's width already matches the manifest."""
    from openral_core.schemas import Action, ControlMode

    hal, published = _build_hal()
    arm_targets = (1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7)
    action = Action(
        control_mode=ControlMode.JOINT_POSITION,
        horizon=1,
        joint_targets=[_full_width_row_with_arm_targets(arm_targets)],
    )
    hal.send_action(action)

    assert len(published) == 1
    topic, msg = published[0]
    assert topic == hal.arm_topic
    assert msg["mode"] == 1.0
    assert msg["joint_targets"] == arm_targets  # type: ignore[comparison-overlap]


def test_joint_position_rejects_wrong_joint_names() -> None:
    from openral_core.exceptions import ROSConfigError
    from openral_core.schemas import Action, ControlMode

    hal, _ = _build_hal()
    action = Action(
        control_mode=ControlMode.JOINT_POSITION,
        horizon=1,
        joint_targets=[_full_width_row_with_arm_targets((0.0,) * 7)],
        joint_names=["joint1", "joint2"],  # wrong -- must be all 7, in order
    )
    with pytest.raises(ROSConfigError, match="joint_names must be"):
        hal.send_action(action)


def test_joint_position_rejects_wrong_row_width() -> None:
    """A 7-wide row (the naive-looking shape) is wrong -- the kernel's
    structural check requires the full 21-wide vector for any JOINT_* mode."""
    from openral_core.exceptions import ROSConfigError
    from openral_core.schemas import Action, ControlMode

    hal, _ = _build_hal()
    action = Action(
        control_mode=ControlMode.JOINT_POSITION,
        horizon=1,
        joint_targets=[[0.0] * 7],
    )
    with pytest.raises(ROSConfigError, match="row has 7 values"):
        hal.send_action(action)


@pytest.mark.parametrize(
    ("commanded", "expect_srb_bool"),
    [
        (1.0, False),  # fully open -> SRB's inverted wire: False = open
        (0.5, False),  # boundary counts as open (>= 0.5)
        (0.4, True),  # closed -> SRB's inverted wire: True = close
        (0.0, True),
    ],
)
def test_gripper_binary_applies_inverted_srb_convention(
    commanded: float, expect_srb_bool: bool
) -> None:
    """GRIPPER_BINARY: >=0.5 is OPEN in OpenRAL terms, published as SRB's inverted bool."""
    from openral_core.schemas import Action, ControlMode

    hal, published = _build_hal()
    action = Action(
        control_mode=ControlMode.GRIPPER_BINARY,
        horizon=1,
        gripper=[commanded],
    )
    hal.send_action(action)

    assert len(published) == 1
    topic, msg = published[0]
    assert topic == hal.hand_topic
    assert msg["data"] is expect_srb_bool


def test_cartesian_twist_rejects_multi_step_horizon() -> None:
    """A multi-step CARTESIAN_TWIST chunk has no meaning over SRB's plain-Twist transport."""
    from openral_core.exceptions import ROSConfigError
    from openral_core.schemas import Action, ControlMode

    hal, _ = _build_hal()
    action = Action(
        control_mode=ControlMode.CARTESIAN_TWIST,
        horizon=2,
        cartesian_twist=[(0.1, 0, 0, 0, 0, 0), (0.1, 0, 0, 0, 0, 0)],
    )
    with pytest.raises(ROSConfigError, match="chunking semantics"):
        hal.send_action(action)


def test_gripper_binary_rejects_empty_payload() -> None:
    from openral_core.exceptions import ROSConfigError
    from openral_core.schemas import Action, ControlMode

    hal, _ = _build_hal()
    action = Action(control_mode=ControlMode.GRIPPER_BINARY, horizon=1, gripper=[])
    with pytest.raises(ROSConfigError, match="empty gripper payload"):
        hal.send_action(action)


def test_unsupported_control_mode_rejected() -> None:
    from openral_core.exceptions import ROSConfigError
    from openral_core.schemas import Action, ControlMode

    hal, _ = _build_hal()
    action = Action(
        control_mode=ControlMode.JOINT_VELOCITY,
        horizon=1,
        joint_velocities=[[0.0] * 21],
    )
    with pytest.raises(
        ROSConfigError, match="body_twist / cartesian_twist / joint_position / gripper_binary"
    ):
        hal.send_action(action)
