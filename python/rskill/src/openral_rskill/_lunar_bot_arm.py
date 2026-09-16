"""Shared LunarBot arm/gripper constants for ``kind: procedural`` rSkills.

Duplicated from ``openral_hal.lunar_bot_srb`` / SRB's ``lunarbot.py`` rather
than imported across a layer boundary (rSkill is layer 3, HAL is layer 0) or
a separate repository (SRB is a sibling submodule this package does not
depend on) — same rationale as the HAL's own calibration constants and the
sensor bridge's ``_TCP_OFFSET``. A change to the robot's joint naming or
gripper stroke must be applied in all these places; there is no single
source of truth today because none of these layers may depend on each other.
"""

from __future__ import annotations

from openral_core.schemas import RobotDescription, WorldState

#: The 7-DoF RM-75 arm's joint order — must match ``LunarBotSRBHAL``'s
#: ``_ARM_JOINT_NAMES`` (``openral_hal.lunar_bot_srb``) and
#: ``SwitchableArmActionCfg.joint_names`` in SRB's ``lunarbot.py``.
ARM_JOINT_NAMES: tuple[str, ...] = (
    "joint1",
    "joint2",
    "joint3",
    "joint4",
    "joint5",
    "joint6",
    "joint7",
)

#: Representative EG2-4C2 gripper joint (of the 6 mimic-linked joints) used
#: to read the jaw stroke fraction from ``world_state.joint_state`` —
#: ``eg2_joint1``'s sign is +1.0 in ``lunarbot.py``'s ``_GRIPPER_SIGN``, so
#: its raw radians run the same direction as the physical stroke: 0.0 rad =
#: fully closed, ``GRIPPER_STROKE_RAD`` = fully open.
GRIPPER_JOINT_NAME = "eg2_joint1"
GRIPPER_STROKE_RAD = 0.82
#: the empty jaws' pad-to-pad contact, which is also SRB's closed command target
#: (``lunarbot.py`` ``_GRIPPER_CLOSED_RAD``): commanded to 0 the pads were driven into each
#: other and never settled (research repo F50). A grasp stops well above this (0.296 rad on
#: the spares' 12 mm T-handle neck).
GRIPPER_CLOSED_RAD = 0.10

#: Published by ``openral_hal_lunar_bot.sensor_bridge_node`` — ``Link7`` +
#: the arm's TCP offset, one definition shared by this skill and the
#: mission verifier (execution_plan.md §8.3 item 3 / item 7).
TCP_FRAME_ID = "tcp_frame"
BASE_FRAME_ID = "chassis_base_link"


def joint_positions_by_name(world_state: WorldState) -> dict[str, float]:
    """``{joint name: current position (rad)}`` from ``world_state.joint_state``.

    A name-keyed dict rather than an index into ``JointState.position`` —
    cheap per step, and does not assume the wire order without checking.
    """
    joint_state = world_state.joint_state
    return dict(zip(joint_state.name, joint_state.position, strict=True))


def joint_velocities_by_name(world_state: WorldState) -> dict[str, float]:
    """``{joint name: current velocity (rad/s)}`` from ``world_state.joint_state``."""
    joint_state = world_state.joint_state
    if not joint_state.velocity:
        return {}
    return dict(zip(joint_state.name, joint_state.velocity, strict=True))


def full_width_joint_row(
    robot_description: RobotDescription, targets_by_name: dict[str, float]
) -> list[float]:
    """Zero-pad ``targets_by_name`` into the full ``RobotDescription.joints`` order.

    The C++ safety kernel's structural check requires a JOINT_* chunk's
    ``n_dof`` to equal the envelope's full width (LunarBot's 21 joints), not
    just the arm joints being commanded (F20, ``lunar_bot_joint_position_
    control.md``) — ``LunarBotSRBHAL`` only forwards the slots named in
    ``action.joint_names`` to SRB, so zero-padding the rest is a structural
    formality, never an actual commanded position for those joints.
    """
    return [targets_by_name.get(j.name, 0.0) for j in robot_description.joints]
