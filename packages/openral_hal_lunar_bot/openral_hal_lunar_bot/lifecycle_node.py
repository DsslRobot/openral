#!/usr/bin/env python3
r"""lunar_bot HAL lifecycle node entry point.

Subclasses ``openral_hal.lifecycle.ManifestHALLifecycleNode`` (not the plain
manifest-driven wrapper every other robot package uses) because
``LunarBotSRBHAL`` needs a real transport wired to it after ``build_hal``
constructs it and before it is used — the same problem
``ros_control.py``'s ``RosControlHAL`` has, solved there via
``_attach_ros_control_transport``'s ``RosControlDrivable`` structural check.
That check is specific to ros2_control's ``ControllerKind``/joint-trajectory
message shape, which SRB's plain-topic interface does not use, so this
package supplies its own transport-attach hook instead of extending the
generic one. ``_create_hal`` (manifest-driven: reads ``robot_yaml`` +
``hal_mode``, routes through ``build_hal``) is inherited unchanged.

Usage::

    ros2 run openral_hal_lunar_bot lifecycle_node \
        --ros-args -p robot_yaml:=robots/lunar_bot/robot.yaml -p hal_mode:=real
"""

from __future__ import annotations

import structlog

from openral_hal.lifecycle import ManifestHALLifecycleNode, _ROS2_AVAILABLE, log

__all__ = ["main"]

_log = structlog.get_logger(__name__)


def _make_main() -> None:
    """Build and run the lunar_bot lifecycle node.

    Mirrors ``make_lifecycle_main_from_manifest``'s body (it hardcodes
    ``ManifestHALLifecycleNode``, so it cannot be reused for a subclass —
    see the module docstring) minus the parts specific to that function's
    closure.
    """
    if not _ROS2_AVAILABLE:
        log.error("rclpy not found — cannot start lifecycle node without ROS 2.")
        raise SystemExit(1)

    import rclpy
    from rclpy.executors import ExternalShutdownException

    from openral_observability import configure_observability

    configure_observability(service_name="openral.hal.lunar_bot")

    rclpy.init()
    node = LunarBotHALLifecycleNode("openral_hal_lunar_bot")
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if _ROS2_AVAILABLE:
    from rclpy.lifecycle import TransitionCallbackReturn

    from openral_hal.lifecycle import ManifestHALLifecycleNode as _ManifestHALLifecycleNode

    class LunarBotHALLifecycleNode(_ManifestHALLifecycleNode):  # type: ignore[misc]
        """``ManifestHALLifecycleNode`` + SRB transport wiring for ``LunarBotSRBHAL``.

        ``on_configure_post_hal`` runs after the base class has built and
        connected ``self._hal`` (a ``LunarBotSRBHAL``, per
        ``robots/lunar_bot/robot.yaml``'s ``hal.real`` entry) — the right
        point to create the real ``geometry_msgs/Twist`` publisher and
        ``sensor_msgs/JointState`` subscriber on SRB's topics and hand them
        to the HAL via ``attach_transport``, the same seam
        ``RosControlHAL`` uses for real ros2_control hardware.
        """

        def on_configure_post_hal(self) -> TransitionCallbackReturn:
            from openral_hal.lunar_bot_srb import LunarBotSRBHAL

            result = super().on_configure_post_hal()
            if result != TransitionCallbackReturn.SUCCESS:
                return result

            hal = self._hal
            if not isinstance(hal, LunarBotSRBHAL):
                # hal_mode=sim (or any non-LunarBotSRBHAL) — nothing to wire.
                return TransitionCallbackReturn.SUCCESS

            self._lunar_bot_transport = _LunarBotSRBTransport(
                self,
                cmd_vel_topic=hal.cmd_vel_topic,
                arm_topic=hal.arm_topic,
                hand_topic=hal.hand_topic,
                joint_state_topic=hal.joint_state_topic,
            )
            hal.attach_transport(
                self._lunar_bot_transport.publish,
                self._lunar_bot_transport.state,
                self._lunar_bot_transport.last_arrival,
            )
            _log.info(
                "lunar_bot_hal.transport_attached",
                cmd_vel_topic=hal.cmd_vel_topic,
                arm_topic=hal.arm_topic,
                hand_topic=hal.hand_topic,
                joint_state_topic=hal.joint_state_topic,
            )
            return TransitionCallbackReturn.SUCCESS

        def on_deactivate_pre_teardown(self) -> None:
            super().on_deactivate_pre_teardown()
            transport = getattr(self, "_lunar_bot_transport", None)
            if transport is not None:
                transport.destroy()
                self._lunar_bot_transport = None

    class _LunarBotSRBTransport:
        """Real ``rclpy`` publishers/subscription for ``LunarBotSRBHAL``.

        Owned by ``LunarBotHALLifecycleNode`` so the publishers/subscription
        live on the same node as the rest of the HAL lifecycle plumbing
        (QoS / executor / shutdown all in one place, matching
        ``ROSPublishingHAL``'s own rationale for taking a host node instead
        of opening its own). One ``Twist`` publisher for the base
        (``cmd_vel``), one ``Float32MultiArray`` publisher for the arm
        (``arm`` -- SRB's ``SwitchableArmAction``, 14 floats: mode flag +
        6-component IK twist + 7 joint-position targets), and one ``Bool``
        publisher for the gripper (``hand``) — ``publish()`` dispatches on
        the ``topic`` argument to pick the right one, so ``LunarBotSRBHAL``
        stays transport-agnostic (it only ever calls ``publish(topic,
        msg)``).
        """

        def __init__(
            self,
            node: "LunarBotHALLifecycleNode",
            *,
            cmd_vel_topic: str,
            arm_topic: str,
            hand_topic: str,
            joint_state_topic: str,
        ) -> None:
            import time

            from geometry_msgs.msg import Twist
            from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy
            from sensor_msgs.msg import JointState as RosJointState
            from std_msgs.msg import Bool, Float32MultiArray

            qos = QoSProfile(
                reliability=QoSReliabilityPolicy.RELIABLE,
                durability=QoSDurabilityPolicy.VOLATILE,
                depth=10,
            )
            self._node = node
            self._time = time
            self._cmd_vel_topic = cmd_vel_topic
            self._arm_topic = arm_topic
            self._hand_topic = hand_topic
            self._cmd_vel_pub = node.create_publisher(Twist, cmd_vel_topic, qos)
            self._arm_pub = node.create_publisher(Float32MultiArray, arm_topic, qos)
            self._hand_pub = node.create_publisher(Bool, hand_topic, qos)
            self._latest_state: dict[str, object] = {}
            self._last_arrival_s: float = 0.0
            self._state_sub = node.create_subscription(
                RosJointState, joint_state_topic, self._on_joint_state, qos
            )

        def _on_joint_state(self, msg: object) -> None:
            self._latest_state = {
                "position": list(msg.position),  # type: ignore[attr-defined]
                "velocity": list(msg.velocity),  # type: ignore[attr-defined]
                "effort": list(msg.effort),  # type: ignore[attr-defined]
            }
            self._last_arrival_s = self._time.monotonic()

        def publish(self, topic: str, msg: dict[str, object]) -> None:
            from geometry_msgs.msg import Twist, Vector3
            from std_msgs.msg import Bool, Float32MultiArray

            if topic == self._hand_topic:
                self._hand_pub.publish(Bool(data=bool(msg["data"])))
                return
            if topic == self._arm_topic:
                mode = float(msg["mode"])  # type: ignore[arg-type]
                ik_linear = msg["ik_linear"]  # type: ignore[index]
                ik_angular = msg["ik_angular"]  # type: ignore[index]
                joint_targets = msg["joint_targets"]  # type: ignore[index]
                data = [mode, *ik_linear, *ik_angular, *joint_targets]  # type: ignore[misc]
                self._arm_pub.publish(Float32MultiArray(data=[float(v) for v in data]))
                return
            lin = msg["linear"]  # type: ignore[index]
            ang = msg["angular"]  # type: ignore[index]
            self._cmd_vel_pub.publish(
                Twist(
                    linear=Vector3(x=lin["x"], y=lin["y"], z=lin["z"]),  # type: ignore[index]
                    angular=Vector3(x=ang["x"], y=ang["y"], z=ang["z"]),  # type: ignore[index]
                )
            )

        def state(self) -> dict[str, object]:
            return self._latest_state

        def last_arrival(self) -> float:
            return self._last_arrival_s

        def destroy(self) -> None:
            self._node.destroy_publisher(self._cmd_vel_pub)
            self._node.destroy_publisher(self._arm_pub)
            self._node.destroy_publisher(self._hand_pub)
            self._node.destroy_subscription(self._state_sub)


main = _make_main

if __name__ == "__main__":
    main()
