"""LunarBotSRBHAL — bridges OpenRAL's safety-checked chunk path to Space
Robotics Bench (SRB), an externally-owned Isaac Sim simulator reached over
ROS 2, not OpenRAL's own in-process MuJoCo physics.

Structurally mirrors ``openral_hal.ros_control.RosControlHAL``: the adapter
imports no ``rclpy`` itself so it stays unit-testable without a live ROS 2
install, and is wired to a real transport after construction via
``attach_transport`` — ``build_hal`` constructs the HAL from the manifest
alone, before any ROS node exists (see ``ros_control.py``'s own
``attach_transport`` docstring for why).

Drives all four of LunarBot's actuator surfaces, unifying base + arm +
gripper into one HAL so this is a genuine mobile manipulator rather than
independent single-actuator slices:

* BODY_TWIST → SRB's ``.../action/cmd_vel`` (``geometry_msgs/Twist``,
  the base's ``FourWheelSteerActionCfg``). Real m/s and rad/s directly —
  SRB's own action-term config already applies the (empirically verified)
  sign, so this HAL forwards the commanded values unconverted.
* CARTESIAN_TWIST and JOINT_POSITION both → SRB's
  ``.../switchable_arm`` (``std_msgs/Float32MultiArray``, 14 floats: a
  mode flag, the 6-component IK twist, and 7 joint-position targets — see
  ``SwitchableArmAction`` in the SRB fork). The arm's own controller picks
  which sub-command actually drives the joints each step, exactly like a
  real manipulator's controller supports switching control modes on
  demand (per the user's explicit direction on this) — not two
  independent SRB action terms that would silently fight over the same 7
  joints every physics step. CARTESIAN_TWIST is **not a direct unit
  passthrough** — see ``_ARM_LINEAR_MPS_PER_RAW_UNIT``/
  ``_ARM_ANGULAR_RADPS_PER_RAW_UNIT`` below for why and how this HAL
  converts; JOINT_POSITION **is** a direct radians passthrough (SRB's
  sub-term uses ``joint_pos_scale=1.0``, no HAL-side conversion needed).
* GRIPPER_BINARY → SRB's ``.../binary_joint_position``
  (``std_msgs/Bool``, the EG2-4C2 gripper). **Inverted from the naive
  reading** — see ``_gripper_open_to_srb_bool`` below.

None of these chunk over time on SRB's side (a plain ``Twist``/``Bool``/
``Float32MultiArray`` command is instantaneous, not a trajectory), so
``send_action`` requires ``horizon == 1`` for all of them — a multi-step
chunk would have no well-defined meaning over this transport.

JOINT_POSITION has one real wrinkle: the C++ safety kernel's structural
check requires a JOINT_* chunk's ``n_dof`` to equal the envelope's
``n_dof`` (LunarBot's full 21 joints), not just the 7 arm joints being
commanded (``cpp/openral_safety_kernel/src/validator.cpp``'s
``is_joint_mode`` branch — a pre-existing, unrelated-to-this-HAL kernel
requirement, not something introduced here). So a JOINT_POSITION dispatch
to this HAL must carry a full 21-wide ``action.joint_targets`` row (steering/
wheel/gripper slots zero-padded — safe, since none of those declare a
position limit that excludes 0.0) with ``action.joint_names`` set to the 7
arm joint names (ADR-0102) so this HAL knows which of the 21 slots to
actually forward to SRB's joint-position sub-command.
"""

from __future__ import annotations

import time
from collections.abc import Callable

import structlog
from openral_core.exceptions import (
    ROSConfigError,
    ROSEStopRequested,
    ROSPerceptionStale,
    ROSRuntimeError,
)
from openral_core.schemas import Action, ControlMode, JointState, RobotDescription

from openral_hal._base import HALBase, _raw_floats

__all__ = ["LunarBotSRBHAL"]

log = structlog.get_logger(__name__)

# (topic: str, msg: dict[str, object]) -> None — same shape as RosControlHAL's
# injectable transport, so the same test-time SimTransport pattern applies.
_PublishFn = Callable[[str, dict[str, object]], None]


def _default_publish(topic: str, msg: dict[str, object]) -> None:  # pragma: no cover
    """No-op publish used before a real transport is attached.

    In production, ``attach_transport`` replaces this at lifecycle-node
    configure time. Reached only in isolated unit tests that inject nothing.
    """
    log.debug("hal.publish", topic=topic, fields=list(msg.keys()))


# ── Arm (CARTESIAN_TWIST) raw-unit calibration ──────────────────────────────
#
# SRB's switchable-arm action term's IK sub-mode (the same differential-IK
# math Isaac Lab's DifferentialInverseKinematicsAction uses,
# use_relative_mode=True, ik_scale=0.1) does NOT speak physical m/s or rad/s
# on the wire: `process_actions()` scales the raw twist by ik_scale and
# treats the result as a per-control-step POSITION delta added to the
# end-effector's
# CURRENT measured pose (confirmed by reading Isaac Lab's own
# task_space_actions.py / differential_ik.py, not guessed from the topic's
# message type). Since the ROS bridge holds the last published Twist as a
# zero-order-hold buffer and feeds it into process_actions() every step,
# sustaining a constant raw value DOES produce continuous EE motion whose
# rate is proportional to the raw magnitude -- functionally a velocity from
# this HAL's point of view, just realised through repeated small position
# deltas rather than a native rate controller (unlike the base's cmd_vel,
# which SRB's own FourWheelSteerActionCfg speaks directly in m/s and rad/s).
#
# Measured live 2026-09-13 (SRB `_ground_manipulation env.robot=lunar_bot`,
# `agent ros`, fresh env at its init pose), holding one raw axis at 1.0 for
# 2.0s and measuring Link7's TF displacement:
#
#   linear.x=1.0  -> 0.431 m/s   | linear.y=1.0  -> 0.293 m/s
#   angular.z=1.0 -> 0.388 rad/s | angular.x=1.0 -> 0.461 rad/s
#
# This is a genuinely POSE-DEPENDENT ratio (a 7-DoF arm's Jacobian
# conditioning varies across the workspace; ik_method="dls" bounds but does
# not eliminate this), not a single physical constant like the base's -- so
# these are deliberately conservative, ROUNDED-UP-FROM-THE-OBSERVED-MAXIMUM
# provisional bring-up constants (same epistemic status as
# robot.yaml's own BODY_TWIST bounds: "conservative... not a validated
# operating envelope"), not a claimed precise calibration. Rounding up
# (using MORE raw-per-desired-m/s than the observed maximum needed) means a
# commanded speed at or under the safety envelope's max_ee_speed_m_s /
# max_ee_angular_speed_rad_s is unlikely to be exceeded at the poses
# measured; it is not a guarantee against every possible configuration.
_ARM_LINEAR_MPS_PER_RAW_UNIT = 0.5
_ARM_ANGULAR_RADPS_PER_RAW_UNIT = 0.6

# Arm joint order for the JOINT_POSITION sub-command -- must match
# SwitchableArmActionCfg's own joint_names list in lunarbot.py exactly
# (both are "joint1".."joint7", declared explicitly rather than a single
# regex, precisely so this ordering is guaranteed rather than incidental).
_ARM_JOINT_NAMES = ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7")


# ── Gripper (GRIPPER_BINARY) direction convention ───────────────────────────
#
# OpenRAL's Action.gripper is a normalised jaw fraction in [0, 1]
# (JointSpec docstring); this HAL treats >= 0.5 as OPEN, < 0.5 as CLOSE --
# the common parallel-gripper convention, and the first real GRIPPER_BINARY
# implementation in this project (no prior robot HAL to mirror).
#
# SRB's binary_joint_position wire is INVERTED from the naive reading of
# "True means open": Isaac Lab's BinaryJointAction maps a positive raw
# value to `open_command_expr` and negative to `close_command_expr`, and
# SRB's ROS bridge extractor is
# `lambda msg: [-1.0 if msg.data else 1.0]` (srb/interfaces/interface/ros.py)
# -- so `Bool(data=True)` -> raw=-1.0 -> CLOSE, `Bool(data=False)` ->
# raw=+1.0 -> OPEN. Confirmed live 2026-09-13, not just read from source:
# publishing True moved eg2_joint1 from ~0.82 rad (open) to ~0.0 rad
# (closed); publishing False reopened it.
def _gripper_open_to_srb_bool(*, open_: bool) -> bool:
    """OpenRAL 'is this gripper command OPEN?' -> SRB's inverted wire bool."""
    return not open_


class LunarBotSRBHAL(HALBase):
    """SRB-bridge HAL adapter for the LunarBot mobile manipulator.

    Args:
        description: The ``lunar_bot`` ``RobotDescription``
            (``robots/lunar_bot/robot.yaml``).
        cmd_vel_topic: SRB topic ``send_action`` publishes BODY_TWIST
            commands on. Defaults to the per-env0 topic ``agent ros`` mode
            exposes for ``lunar_bot``'s ``FourWheelSteerActionCfg`` action
            term (verified live in the research repo's
            ``docs/srb_integration_notes.md`` §3.2).
        arm_topic: SRB topic ``send_action`` publishes CARTESIAN_TWIST and
            JOINT_POSITION commands on (the 7-DoF arm's single switchable
            action term -- see ``SwitchableArmAction`` in the SRB fork).
        hand_topic: SRB topic ``send_action`` publishes GRIPPER_BINARY
            commands on (the EG2-4C2 gripper's binary joint-position
            action term).
        joint_state_topic: SRB topic ``read_state`` reads from. Defaults to
            the topic verified live in the same integration notes.
        publish_fn / state_fn: Inject a test transport (or a
            ``SimTransport``-shaped callable pair) at construction time;
            production wiring uses ``attach_transport`` instead, matching
            ``RosControlHAL``.
        staleness_limit_s: Maximum age of a ``read_state()`` reading before
            ``ROSPerceptionStale`` is raised.

    Raises:
        ROSConfigError: If ``description.joints`` is empty.
    """

    def __init__(
        self,
        description: RobotDescription,
        *,
        cmd_vel_topic: str = "/srb/env0/action/cmd_vel",
        arm_topic: str = "/srb/env0/robot/robot/switchable_arm",
        hand_topic: str = "/srb/env0/robot/robot/binary_joint_position",
        joint_state_topic: str = "/srb/env0/robot/joint_states",
        publish_fn: _PublishFn | None = None,
        state_fn: Callable[[], dict[str, object]] | None = None,
        staleness_limit_s: float = 0.5,
    ) -> None:
        """Initialise the adapter; does not open any connection yet."""
        if not description.joints:
            raise ROSConfigError(
                f"RobotDescription '{description.name}' has no joints; "
                "cannot initialise LunarBotSRBHAL."
            )
        self.description = description
        self._cmd_vel_topic = cmd_vel_topic
        self._arm_topic = arm_topic
        self._hand_topic = hand_topic
        self._joint_state_topic = joint_state_topic
        self._publish_fn: _PublishFn = publish_fn or _default_publish
        self._state_fn = state_fn
        self._staleness_limit_s = staleness_limit_s

        self._connected: bool = False
        self._last_state_time: float = 0.0
        # Set by `attach_transport` to the transport's real per-message
        # arrival clock; see `read_state` for why `None` means "no freshness
        # check beyond connect() liveness".
        self._stamp_fn: Callable[[], float] | None = None
        self._joint_names: list[str] = [j.name for j in description.joints]
        # Where each of the 7 arm joints sits within the whole-robot 21-wide
        # joint order -- resolved from the manifest, not hardcoded, so a
        # future robot.yaml joint reordering can't silently desync this.
        missing = [jn for jn in _ARM_JOINT_NAMES if jn not in self._joint_names]
        if missing:
            raise ROSConfigError(
                f"RobotDescription '{description.name}' is missing arm joints {missing}; "
                "cannot resolve JOINT_POSITION indices for LunarBotSRBHAL."
            )
        self._arm_joint_indices: list[int] = [
            self._joint_names.index(jn) for jn in _ARM_JOINT_NAMES
        ]

    # ── Transport wiring ─────────────────────────────────────────────────────

    def attach_transport(
        self,
        publish_fn: _PublishFn,
        state_fn: Callable[[], dict[str, object]],
        stamp_fn: Callable[[], float] | None = None,
    ) -> None:
        """Bind this HAL to a live transport after construction.

        Same seam as ``RosControlHAL.attach_transport`` — ``build_hal`` runs
        before any ROS node exists, so a real deployment cannot pass
        ``publish_fn``/``state_fn`` to ``__init__``. The
        ``openral_hal_lunar_bot`` lifecycle node calls this once it has a
        node to create the SRB-facing publisher/subscription on.

        Args:
            publish_fn: Sends one command dict to an SRB action topic.
            state_fn: Returns the newest SRB joint state as a raw dict.
            stamp_fn: Returns the ``time.monotonic()`` timestamp of the
                newest joint state. Without it, freshness can only be
                measured from ``connect()``.
        """
        if stamp_fn is not None and not callable(stamp_fn):
            raise ROSConfigError(
                f"attach_transport(stamp_fn=...) needs a callable returning the newest "
                f"joint state's time.monotonic() timestamp, got {type(stamp_fn).__name__}."
            )
        self._publish_fn = publish_fn
        self._state_fn = state_fn
        self._stamp_fn = stamp_fn
        self._last_state_time = time.monotonic()

    @property
    def joint_state_topic(self) -> str:
        """The SRB ``sensor_msgs/JointState`` topic this HAL reads."""
        return self._joint_state_topic

    @property
    def cmd_vel_topic(self) -> str:
        """The SRB ``geometry_msgs/Twist`` (base) topic this HAL publishes to."""
        return self._cmd_vel_topic

    @property
    def arm_topic(self) -> str:
        """The SRB ``Float32MultiArray`` (arm) topic this HAL publishes to."""
        return self._arm_topic

    @property
    def hand_topic(self) -> str:
        """The SRB ``std_msgs/Bool`` (gripper) topic this HAL publishes to."""
        return self._hand_topic

    # ── Lifecycle ────────────────────────────────────────────────────────────

    def connect(self) -> None:
        """Mark the adapter connected.

        Opens no resource itself (unlike ``ROSPublishingHAL``, this adapter
        holds no ``rclpy`` node of its own — the real publisher/subscription
        are created by the lifecycle node and handed in via
        ``attach_transport``).

        Raises:
            ROSRuntimeError: If already connected.
        """
        if self._connected:
            raise ROSRuntimeError(f"LunarBotSRBHAL('{self.description.name}') is already connected.")
        log.info(
            "hal.connect",
            robot=self.description.name,
            cmd_vel_topic=self._cmd_vel_topic,
            joint_state_topic=self._joint_state_topic,
        )
        self._connected = True
        self._last_state_time = time.monotonic()

    # ── Hot path ─────────────────────────────────────────────────────────────

    def read_state(self) -> JointState:
        """Return the latest joint state snapshot read from SRB.

        Raises:
            ROSRuntimeError: If not connected.
            ROSPerceptionStale: If the last reading is older than
                ``staleness_limit_s``.
        """
        self._require_connected("read_state")
        if self._stamp_fn is not None:
            last = self._stamp_fn()
            age = float("inf") if last <= 0.0 else time.monotonic() - last
        else:
            age = time.monotonic() - self._last_state_time
        if age > self._staleness_limit_s:
            raise ROSPerceptionStale(
                f"SRB joint state is {age:.3f} s old (limit {self._staleness_limit_s} s)."
            )

        n = len(self._joint_names)
        raw: dict[str, object] = self._state_fn() if self._state_fn is not None else {}
        return JointState(
            name=self._joint_names,
            position=_raw_floats(raw, "position", n),
            velocity=_raw_floats(raw, "velocity", n),
            effort=_raw_floats(raw, "effort", n),
            stamp_ns=int(time.time_ns()),
        )

    def send_action(self, action: Action) -> None:
        """Forward one action to the SRB topic for its control mode.

        Args:
            action: The ``Action`` produced by a Skill. Must have
                ``control_mode`` in ``{BODY_TWIST, CARTESIAN_TWIST,
                JOINT_POSITION, GRIPPER_BINARY}`` and ``horizon == 1``
                (none of the four chunk over time on SRB's side).

        Raises:
            ROSRuntimeError: If not connected.
            ROSConfigError: If ``action.control_mode`` is unsupported,
                ``horizon != 1``, or the mode's payload field is empty.
        """
        self._require_connected("send_action")
        if action.control_mode is ControlMode.BODY_TWIST:
            self._send_body_twist(action)
        elif action.control_mode is ControlMode.CARTESIAN_TWIST:
            self._send_cartesian_twist(action)
        elif action.control_mode is ControlMode.JOINT_POSITION:
            self._send_joint_position(action)
        elif action.control_mode is ControlMode.GRIPPER_BINARY:
            self._send_gripper_binary(action)
        else:
            raise ROSConfigError(
                f"LunarBotSRBHAL supports body_twist / cartesian_twist / "
                f"joint_position / gripper_binary; got {action.control_mode!r}."
            )

    def _require_single_step(self, action: Action, *, payload_name: str) -> None:
        if action.horizon != 1:
            raise ROSConfigError(
                f"LunarBotSRBHAL: {action.control_mode.value} has no chunking "
                f"semantics on SRB's plain-topic transport; got "
                f"horizon={action.horizon} (expected 1)."
            )
        if not getattr(action, payload_name):
            raise ROSConfigError(
                f"LunarBotSRBHAL: {action.control_mode.value} Action has empty "
                f"{payload_name} payload."
            )

    def _send_body_twist(self, action: Action) -> None:
        """Publish a BODY_TWIST action to SRB's base ``cmd_vel`` topic.

        Real m/s / rad/s directly — no HAL-side scale conversion (see the
        module docstring's cmd_vel bullet).
        """
        self._require_control_mode(action, ControlMode.BODY_TWIST)
        self._require_single_step(action, payload_name="body_twist")
        vx, vy, vz, wx, wy, wz = action.body_twist[0]  # type: ignore[index]
        msg: dict[str, object] = {
            "linear": {"x": vx, "y": vy, "z": vz},
            "angular": {"x": wx, "y": wy, "z": wz},
        }
        self._publish_fn(self._cmd_vel_topic, msg)
        log.debug(
            "hal.send_action",
            robot=self.description.name,
            control_mode=action.control_mode,
            linear=(vx, vy, vz),
            angular=(wx, wy, wz),
        )

    def _send_cartesian_twist(self, action: Action) -> None:
        """Publish a CARTESIAN_TWIST action to SRB's switchable-arm topic
        (mode flag 0.0 -- IK/twist sub-mode).

        Converts from the physical m/s / rad/s the safety kernel validated
        (``max_ee_speed_m_s`` / ``max_ee_angular_speed_rad_s``) into SRB's
        raw per-step Twist units via the calibration constants above.
        """
        self._require_control_mode(action, ControlMode.CARTESIAN_TWIST)
        self._require_single_step(action, payload_name="cartesian_twist")
        vx, vy, vz, wx, wy, wz = action.cartesian_twist[0]  # type: ignore[index]
        raw_linear = tuple(v / _ARM_LINEAR_MPS_PER_RAW_UNIT for v in (vx, vy, vz))
        raw_angular = tuple(w / _ARM_ANGULAR_RADPS_PER_RAW_UNIT for w in (wx, wy, wz))
        msg: dict[str, object] = {
            "mode": 0.0,
            "ik_linear": raw_linear,
            "ik_angular": raw_angular,
            "joint_targets": (0.0,) * len(_ARM_JOINT_NAMES),
        }
        self._publish_fn(self._arm_topic, msg)
        log.debug(
            "hal.send_action",
            robot=self.description.name,
            control_mode=action.control_mode,
            physical_linear_m_s=(vx, vy, vz),
            physical_angular_rad_s=(wx, wy, wz),
            raw_linear=raw_linear,
            raw_angular=raw_angular,
        )

    def _send_joint_position(self, action: Action) -> None:
        """Publish a JOINT_POSITION action to SRB's switchable-arm topic
        (mode flag 1.0 -- joint-position sub-mode).

        The safety kernel's structural check requires a JOINT_* chunk's
        ``n_dof`` to equal the envelope's full 21 (LunarBot's whole joint
        count), not just the 7 arm joints being commanded here (see the
        module docstring) -- so ``action.joint_targets[0]`` is a full
        21-wide row (steering/wheel/gripper slots zero-padded by the
        caller) and ``action.joint_names`` names the 7 arm joints this HAL
        should extract from it. Direct radians passthrough: SRB's
        joint-position sub-term uses ``joint_pos_scale=1.0``.
        """
        self._require_control_mode(action, ControlMode.JOINT_POSITION)
        self._require_single_step(action, payload_name="joint_targets")
        row = action.joint_targets[0]  # type: ignore[index]
        # The wire-level row is ALWAYS the full 21-wide RobotDescription.joints
        # vector, never just the 7 arm values -- the C++ kernel's structural
        # check forces chunk.n_dof == envelope.n_dof for every JOINT_* mode
        # (validator.cpp's is_joint_mode branch), so a 7-wide chunk would be
        # rejected as kNdofMismatch before this HAL ever saw it. This holds
        # whether or not action.joint_names is set -- that field only NAMES
        # which of the 21 slots are the ones actually being commanded
        # (ADR-0102); it never changes the row's width.
        if len(row) != len(self._joint_names):
            raise ROSConfigError(
                f"LunarBotSRBHAL: JOINT_POSITION row has {len(row)} values but "
                f"robot '{self.description.name}' has {len(self._joint_names)} joints "
                "(the safety kernel's structural check requires the full-width vector)."
            )
        if action.joint_names and list(action.joint_names) != list(_ARM_JOINT_NAMES):
            raise ROSConfigError(
                f"LunarBotSRBHAL: JOINT_POSITION action.joint_names must be "
                f"{list(_ARM_JOINT_NAMES)!r} (the 7 arm joints, ADR-0102) when set; "
                f"got {list(action.joint_names)!r}."
            )
        arm_targets = tuple(row[i] for i in self._arm_joint_indices)
        msg: dict[str, object] = {
            "mode": 1.0,
            "ik_linear": (0.0, 0.0, 0.0),
            "ik_angular": (0.0, 0.0, 0.0),
            "joint_targets": arm_targets,
        }
        self._publish_fn(self._arm_topic, msg)
        log.debug(
            "hal.send_action",
            robot=self.description.name,
            control_mode=action.control_mode,
            arm_joint_targets=arm_targets,
        )

    def _send_gripper_binary(self, action: Action) -> None:
        """Publish a GRIPPER_BINARY action to SRB's gripper topic.

        Thresholds the ``[0, 1]`` jaw-fraction command at 0.5 (>= 0.5 =
        OPEN) and applies SRB's inverted wire convention (see
        ``_gripper_open_to_srb_bool`` above).
        """
        self._require_control_mode(action, ControlMode.GRIPPER_BINARY)
        self._require_single_step(action, payload_name="gripper")
        commanded_open = action.gripper[0] >= 0.5  # type: ignore[index]
        srb_bool = _gripper_open_to_srb_bool(open_=commanded_open)
        self._publish_fn(self._hand_topic, {"data": srb_bool})
        log.debug(
            "hal.send_action",
            robot=self.description.name,
            control_mode=action.control_mode,
            commanded_open=commanded_open,
            srb_bool=srb_bool,
        )

    # ── Safety ───────────────────────────────────────────────────────────────

    def estop(self) -> None:
        """Trigger an emergency stop.

        Raises:
            ROSEStopRequested: Always.
        """
        log.critical("hal.estop", robot=self.description.name)
        self._connected = False
        raise ROSEStopRequested(f"Emergency stop triggered on robot '{self.description.name}'.")
