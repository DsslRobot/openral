#!/usr/bin/env python3
"""Bridge Nav2's ``/cmd_vel`` onto ``/openral/candidate_action`` under kernel supervision.

``rskill-nav2-mobile_base-navigate_to_pose``'s own manifest documents the gap this closes: Nav2's
behaviour tree publishes ``geometry_msgs/Twist`` on ``/cmd_vel`` straight to the base controller,
bypassing the safety kernel entirely and relying only on Nav2's own costmap + velocity_smoother.
``openral_hal.mobile_base_bridge.MobileBaseBridge._on_cmd_vel`` documents the same gap from the HAL
side and names the fix: "run an external ``twist_to_action`` relay onto
``/openral/candidate_action`` for the supervised path." This node is that relay — §8.3 item 5
(execution_plan.md).

Mechanism: wrap each ``/cmd_vel`` message as a ``BODY_TWIST`` ``openral_core.Action`` (identical
field mapping to ``MobileBaseBridge._on_cmd_vel``: ``[linear.x, linear.y, 0, 0, 0, angular.z]``,
frame = the robot's base frame) and publish it through ``openral_runner.ROSPublishingHAL`` — the
same serialiser every other candidate_action producer uses — so it lands on
``/openral/candidate_action`` and is checked by ``openral_safety_kernel`` exactly like any rSkill's
action, then re-published as ``/openral/safe_action`` for the per-robot HAL to apply.

Mutually exclusive with ``MobileBaseBridge``'s own ``/cmd_vel`` handling. That bridge attaches
automatically (``ManifestHALLifecycleNode.on_activate_post_subs``) whenever the manifest declares
``base_joints``, and its ``/cmd_vel`` subscription is the intentional bypass path (direct
``hal.send_action``, no kernel). Running both for the same robot would double-actuate every Nav2
velocity command. This node therefore refuses to subscribe — logs why and stays otherwise idle,
rather than guessing a safe combination — when the loaded ``robot_yaml`` declares ``base_joints``
(``MobileBaseBridge`` already owns that robot) or omits ``body_twist`` from
``capabilities.supported_control_modes`` (nothing to relay to). LunarBot is the first robot this
applies to: a real 4WIS/4WID rover with no virtual planar ``base_joints``, so ``MobileBaseBridge``
never attaches for it and its ``/cmd_vel`` path was previously unbridged in either direction.

Not gated on e-stop client-side, unlike ``MobileBaseBridge._on_cmd_vel``: that bridge bypasses the
kernel so it must police the estop latch itself; this relay's whole point is routing through the
kernel, which is the sole safety authority for a candidate_action publisher and will reject/drop
during a latched stop like any other producer. Duplicating that check here would be a second,
divergeable copy of state the kernel already owns.
"""

from __future__ import annotations

from typing import Any

__all__ = ["main"]


def main(args: Any = None) -> None:
    """Entry point for ``ros2 run openral_nav2_bringup twist_to_action_relay_node.py``."""
    import rclpy
    from openral_core.schemas import Action, ControlMode, RobotDescription
    from openral_runner.ros_publishing_hal import ROSPublishingHAL
    from rclpy.executors import ExternalShutdownException
    from rclpy.node import Node
    from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy

    class TwistToActionRelayNode(Node):  # type: ignore[misc]  # reason: rclpy.node.Node is untyped
        """Subscribes ``/cmd_vel``, republishes as a kernel-checked ``BODY_TWIST`` chunk."""

        def __init__(self) -> None:
            super().__init__("openral_nav2_twist_to_action_relay")
            self.declare_parameter("robot_yaml", "")
            self.declare_parameter("cmd_vel_topic", "/cmd_vel")
            self.declare_parameter("candidate_action_topic", "/openral/candidate_action")

            gp = self.get_parameter
            robot_yaml = gp("robot_yaml").get_parameter_value().string_value
            cmd_vel_topic = gp("cmd_vel_topic").get_parameter_value().string_value
            candidate_action_topic = (
                gp("candidate_action_topic").get_parameter_value().string_value
            )

            self._hal: ROSPublishingHAL | None = None
            self._cmd_vel_sub: Any = None
            self._base_frame = ""

            if not robot_yaml:
                self.get_logger().error(
                    "twist_to_action_relay: no `robot_yaml` — refusing to subscribe "
                    "`/cmd_vel` (nothing to check the robot's base_joints / "
                    "supported_control_modes against)."
                )
                return

            description = RobotDescription.from_yaml(robot_yaml)
            if description.base_joints:
                self.get_logger().error(
                    f"twist_to_action_relay: {description.name!r} declares `base_joints` — "
                    "MobileBaseBridge already owns its `/cmd_vel` (bypass path); refusing to "
                    "also subscribe here to avoid double-actuating every Nav2 velocity command."
                )
                return
            if ControlMode.BODY_TWIST not in description.capabilities.supported_control_modes:
                self.get_logger().error(
                    f"twist_to_action_relay: {description.name!r} does not declare "
                    "`body_twist` in `capabilities.supported_control_modes` — nothing to "
                    "relay `/cmd_vel` to; refusing to subscribe."
                )
                return

            self._base_frame = description.base_frame
            self._hal = ROSPublishingHAL(
                node=self,
                description=description,
                candidate_action_topic=candidate_action_topic,
            )
            self._hal.connect()

            cmd_vel_qos = QoSProfile(
                reliability=QoSReliabilityPolicy.RELIABLE,
                durability=QoSDurabilityPolicy.VOLATILE,
                depth=1,
            )
            from geometry_msgs.msg import Twist

            self._cmd_vel_sub = self.create_subscription(
                Twist, cmd_vel_topic, self._on_cmd_vel, cmd_vel_qos
            )
            self.get_logger().info(
                f"twist_to_action_relay: {description.name!r} — {cmd_vel_topic} -> "
                f"{candidate_action_topic} (frame={self._base_frame!r})"
            )

        def _on_cmd_vel(self, msg: Any) -> None:
            """Wrap one ``geometry_msgs/Twist`` as a ``BODY_TWIST`` Action and publish it.

            Same field mapping as ``MobileBaseBridge._on_cmd_vel``: only ``linear.x`` /
            ``linear.y`` / ``angular.z`` carry signal for a planar base. A send failure
            (kernel rejection, transport error) is logged by ``ROSPublishingHAL``/
            ``send_action``'s own error path — nothing to add here; this callback must stay
            non-blocking (single ungrouped chunk, ``tick_group_size=1``, so
            ``send_action`` never waits on ``/openral/action_applied``).
            """
            if self._hal is None:
                return
            linear, angular = msg.linear, msg.angular
            action = Action(
                control_mode=ControlMode.BODY_TWIST,
                horizon=1,
                body_twist=[
                    (
                        float(linear.x),
                        float(linear.y),
                        0.0,
                        0.0,
                        0.0,
                        float(angular.z),
                    )
                ],
                frame_id=self._base_frame,
                tick_group_size=1,
            )
            self._hal.send_action(action)

    rclpy.init(args=args)
    node = TwistToActionRelayNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
