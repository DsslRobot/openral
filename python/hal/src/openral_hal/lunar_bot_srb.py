"""LunarBotSRBHAL — bridges OpenRAL's safety-checked chunk path to Space
Robotics Bench (SRB), an externally-owned Isaac Sim simulator reached over
ROS 2, not OpenRAL's own in-process MuJoCo physics.

Structurally mirrors ``openral_hal.ros_control.RosControlHAL``: the adapter
imports no ``rclpy`` itself so it stays unit-testable without a live ROS 2
install, and is wired to a real transport after construction via
``attach_transport`` — ``build_hal`` constructs the HAL from the manifest
alone, before any ROS node exists (see ``ros_control.py``'s own
``attach_transport`` docstring for why).

First integration slice: BODY_TWIST only (drives SRB's per-action-term
``.../action/cmd_vel`` topic, a ``geometry_msgs/Twist``). CARTESIAN_TWIST
(SRB's ``differential_inverse_kinematics`` action, for the 7-DoF arm) and
GRIPPER_BINARY (SRB's ``binary_joint_position`` action) are follow-up work
once this path is proven end to end — see ``robots/lunar_bot/robot.yaml``.

Body twist has no chunking semantics on SRB's side (a plain ``Twist`` is an
instantaneous velocity command, not a trajectory), so ``send_action`` only
accepts ``horizon == 1`` — a multi-step BODY_TWIST chunk would have no
well-defined meaning over this transport.
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
        """The SRB ``geometry_msgs/Twist`` topic this HAL publishes to."""
        return self._cmd_vel_topic

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
        """Forward a BODY_TWIST action to SRB's ``cmd_vel`` action topic.

        Args:
            action: The ``Action`` produced by a Skill. Must have
                ``control_mode == BODY_TWIST`` and ``horizon == 1``.

        Raises:
            ROSRuntimeError: If not connected.
            ROSConfigError: If ``action.control_mode`` is not BODY_TWIST,
                ``horizon != 1``, or ``action.body_twist`` is empty.
        """
        self._require_connected("send_action")
        self._require_control_mode(action, ControlMode.BODY_TWIST)
        if action.horizon != 1:
            raise ROSConfigError(
                f"LunarBotSRBHAL: BODY_TWIST has no chunking semantics on SRB's "
                f"plain-Twist transport; got horizon={action.horizon} (expected 1)."
            )
        if not action.body_twist:
            raise ROSConfigError(
                "LunarBotSRBHAL: BODY_TWIST Action has empty body_twist payload."
            )
        vx, vy, vz, wx, wy, wz = action.body_twist[0]
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

    # ── Safety ───────────────────────────────────────────────────────────────

    def estop(self) -> None:
        """Trigger an emergency stop.

        Raises:
            ROSEStopRequested: Always.
        """
        log.critical("hal.estop", robot=self.description.name)
        self._connected = False
        raise ROSEStopRequested(f"Emergency stop triggered on robot '{self.description.name}'.")
