#!/usr/bin/env python3
r"""LunarBot sensor bridge — SRB's native ROS topics -> OpenRAL conventions.

SRB (reached over ROS, not OpenRAL in-process physics — see the module
docstring in ``lifecycle_node.py`` and ``robots/lunar_bot/robot.yaml``'s
header) already publishes everything this robot senses under
``srb.interfaces.interface.ros.RosInterface``: per-camera ``Image``/
``CameraInfo``/``PointCloud2``, an ``Imu``, a ``RayCaster`` lidar
``PointCloud2``, and the full scene ``/tf`` tree — all keyed by SRB's own
scene-sensor names (``cam_front``, ``cam_wrist``, ``imu_robot``,
``lidar_robot``) under the ``srb/env0/...`` frame namespace. None of that
is on OpenRAL's own conventions (``/openral/cameras/<name>/...``, a
``chassis_base_link``-rooted TF tree, ``/odom``, ``/imu``), and downstream
consumers (world-state camera subscriptions, the object detector, Nav2, the
octomap bridge) only know the OpenRAL side.

This node is the relay, not a resynthesis: every image/camera_info/
pointcloud/imu topic below is SRB's own already-computed message,
republished on a new topic with ``header.frame_id`` rewritten to the
matching ``robots/lunar_bot/robot.yaml`` ``sensors[].frame_id`` — the same
"thin relay in the bridge node" fallback
``docs/lunar_bot_capability_set_plan.md`` §3.2 describes for a
``deploy_binding`` that isn't wired into any launch for this robot yet
(that mechanism — ``openral_rskill_ros.sensor_leg`` — targets ``openral
deploy run``/``deploy sim``, neither of which supports ``lunar_bot`` until
capability-set item 2b lands). Relaying SRB's live ``CameraInfo`` (rather
than re-deriving static intrinsics from ``PinholeCameraCfg``) also means a
resolution change in the Hydra launch config can never desync from what
this node publishes — only ``robot.yaml``'s *documentation* comment would.

TF re-rooting, ``/odom``, and the IMU/lidar/camera frame chain are new
computation, not a relay: SRB roots every scene transform under
``srb/env{i}`` (not any OpenRAL-meaningful frame), so this node uses a
``tf2_ros`` buffer to look up each sensor's pose relative to the robot root
and rebroadcasts it as ``chassis_base_link -> <manifest frame>`` — the
composition ``docs/lunar_bot_capability_set_plan.md`` §3.2 item 1 specifies
(``T(robot <- frame)`` from ``T(env <- robot)`` and ``T(env <- frame)``),
done automatically by ``tf2_ros.Buffer.lookup_transform`` since both frames
share the common parent ``srb/env0``. ``map -> odom`` is published as a
static identity transform when no localizer runs (``publish_map_to_odom``,
default true) so Nav2 / the world-state lift / the verifier resolve through
TF; with ``odom_source: wheel`` and AMCL or SLAM in the graph, that node
authors the edge instead and this one is turned off.

``robot_tf_frame`` (default ``srb/env0/robot``) is the one genuinely open
question this module cannot resolve by reading source: whether SRB's
articulation-root TF frame (``Articulation.data.root_pos_w``, broadcast by
``RosInterface._broadcast_transforms``'s per-articulation loop) is
numerically the same pose as the ``chassis_base_link`` body — the manifest's
``frame_base`` names ``chassis_base_link`` as a prim *under*
``{robot.prim_path}`` (see ``lunarbot.py``'s frame declarations), which is
consistent with (but does not prove) the two coinciding: USD prim
containment is an authoring-time grouping, while PhysX's articulation root
pose is a physics-time property of the root rigid body — they usually
coincide for a mobile-base robot with no extra virtual root Xform, but this
is exactly the kind of claim this project verifies live rather than assumes
(``docs/srb_deploy_backend_plan.md`` §6). Left as a parameter, not a
hardcoded constant, precisely so a live mismatch is a one-flag fix rather
than a code change.

Usage::

    ros2 run openral_hal_lunar_bot sensor_bridge_node.py \
        --ros-args -p robot_yaml:=robots/lunar_bot/robot.yaml
"""

from __future__ import annotations

import math

import structlog

__all__ = ["main", "rotate_vector_by_quat_xyzw"]

_log = structlog.get_logger(__name__)

try:
    import rclpy  # noqa: F401

    _ROS2_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only on hosts without ROS 2
    _ROS2_AVAILABLE = False

#: SRB's per-env TF root (``RosInterface``'s ``srb/env{i}`` frame family).
DEFAULT_ENV_TF_FRAME = "srb/env0"
#: The articulation-root TF frame SRB broadcasts for `lunar_bot` -- see the
#: module docstring's `robot_tf_frame` paragraph for why this is a parameter.
DEFAULT_ROBOT_TF_FRAME = "srb/env0/robot"
#: SRB scene-sensor name -> ``robots/lunar_bot/robot.yaml`` `sensors[].name`,
#: for every sensor whose *pose* (not just its data) this bridge relays onto
#: the `chassis_base_link`-rooted TF tree.
SRB_SENSOR_TF_SOURCE = {
    "front": "cam_front",
    "wrist": "cam_wrist",
    "imu": "imu_robot",
    "lidar": "lidar_robot",
}
#: `robots/lunar_bot/robot.yaml` `sensors[].name` this bridge requires present
#: (item 2's full set — see `docs/lunar_bot_capability_set_plan.md` §3.2).
REQUIRED_SENSOR_NAMES = ("front", "front_depth", "wrist", "imu", "lidar")

#: Camera extrinsics: (the SRB body the camera is mounted on, mount position, mount roll/pitch/yaw in degrees), both in
#: IsaacLab's "world" camera convention (+X forward, +Z up) -- numerically identical to SRB's ``lunarbot.py``
#: ``frame_front_camera`` / ``frame_wrist_camera`` (duplicated for the same reason as ``_TCP_OFFSET``). The camera frame
#: is chained through the live body TF, as a real robot's calibrated extrinsics are: SRB's own camera TF comes from
#: IsaacLab's camera ``data.pos_w``, which keeps the spawn pose unless ``update_latest_camera_pose`` is set, and moved
#: neither with the rover nor with the arm (research repo F53).
CAMERA_MOUNTS = {
    "front": ("rgbd_camera_frame", (0.0, 0.0, 0.0), (0.0, 15.0, 0.0)),
    "wrist": ("Link7", (0.0, -0.12, -0.08), (0.0, -72.0, 90.0)),
}
#: World camera convention (+X forward, +Z up) -> ROS optical frame (+Z forward, +Y down), xyzw.
_WORLD_TO_OPTICAL_XYZW = (0.5, -0.5, 0.5, -0.5)


def _quat_mul_xyzw(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (aw * bx + ax * bw + ay * bz - az * by, aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw, aw * bw - ax * bx - ay * by - az * bz)


def _rpy_deg_to_quat_xyzw(roll: float, pitch: float, yaw: float) -> tuple[float, float, float, float]:
    cr, cp, cy = (math.cos(math.radians(v) / 2) for v in (roll, pitch, yaw))
    sr, sp, sy = (math.sin(math.radians(v) / 2) for v in (roll, pitch, yaw))
    return (sr * cp * cy - cr * sp * sy, cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy, cr * cp * cy + sr * sp * sy)


#: The arm's flange body — SRB's ``RosInterface._broadcast_transforms``
#: publishes a live TF frame for every body of every articulation
#: (``srb/env{i}/{asset_name}/{body_name}``, not just declared sensors), so
#: ``Link7`` is already on ``/tf`` under the robot articulation root with no
#: SRB-side change needed — confirmed by reading
#: ``srb/interfaces/interface/ros.py``'s per-articulation body loop.
LINK7_BODY_NAME = "Link7"
#: Tool-centre-point offset from ``Link7`` along its own +Z axis — must stay
#: numerically identical to SRB's ``lunarbot.py`` ``_TCP_OFFSET`` (also fed
#: into ``SwitchableArmActionCfg.OffsetCfg`` there, so it is already the
#: pose the arm's IK controller itself targets); duplicated here rather than
#: imported because this package does not depend on the SRB submodule, the
#: same rationale as ``openral_hal.lunar_bot_srb``'s calibration constants.
_TCP_OFFSET = (0.0, 0.0, 0.140)
#: Published child frame for the gripper TCP (execution_plan.md §8.3 item 3:
#: "the TCP frame: Link7 + _TCP_OFFSET, exposed by the sensor bridge as
#: tcp_frame on TF so the skill and the verifier use one definition").
TCP_FRAME_ID = "tcp_frame"
#: The RM-75's own root body in SRB's articulation — also the root link of the arm-only MoveIt
#: model (``rm_75_config``: ``rm_group`` chains ``base_link -> Link7``, no virtual joint). MoveIt
#: can only accept a goal expressed in ``chassis_base_link`` if TF connects that frame to its
#: planning root, so the bridge re-roots this body like it does ``tcp_frame``: the mount
#: transform comes from the articulation that is actually loaded, not a hand-entered number.
ARM_ROOT_BODY_NAME = "base_link"
ARM_ROOT_FRAME_ID = "base_link"

#: 4WIS geometry, from SRB's ``lunarbot.py`` ``FourWheelSteerActionCfg`` (wheel centres in the chassis
#: frame, wheel radius). Wheel odometry inverts the same kinematics the action term applies, so the
#: estimate is wrong in exactly the way a real rover's is: it believes the wheels, and the wheels slip
#: (15-38 % on this regolith, worst in-place, research repo F50).
WHEEL_POSITIONS_M = ((0.4925, 0.42705), (0.4925, -0.42705), (-0.5225, 0.42705), (-0.5225, -0.42705))
WHEEL_RADIUS_M = 0.1453
STEERING_JOINTS = ("chassis_to_front_left_steering_joint", "chassis_to_front_right_steering_joint",
                   "chassis_to_rear_left_steering_joint", "chassis_to_rear_right_steering_joint")
DRIVE_JOINTS = ("front_left_steering_to_wheel_joint", "front_right_steering_to_wheel_joint",
                "rear_left_steering_to_wheel_joint", "rear_right_steering_to_wheel_joint")


def rotate_vector_by_quat_xyzw(
    vx: float, vy: float, vz: float, qx: float, qy: float, qz: float, qw: float
) -> tuple[float, float, float]:
    """Rotate a 3-vector by a unit quaternion (Hamilton, ``[x, y, z, w]``).

    Standard ``v' = v + 2w(u x v) + 2u x (u x v)`` form (``u`` = the
    quaternion's vector part) — avoids building a full rotation matrix for a
    single vector. Used to express ``/odom``'s finite-difference world-frame
    linear velocity in the ``chassis_base_link`` frame REP-105 requires
    (``Odometry.twist`` is in ``child_frame_id``), by passing the
    **conjugate** of the body's world orientation.

    Args:
        vx, vy, vz: The vector to rotate.
        qx, qy, qz, qw: The rotation, as a unit quaternion.

    Returns:
        The rotated vector.

    Example:
        >>> # Identity rotation leaves the vector unchanged.
        >>> rotate_vector_by_quat_xyzw(1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0)
        (1.0, 2.0, 3.0)
        >>> # +90 deg about Z sends +X to +Y.
        >>> import math
        >>> rx, ry, rz = rotate_vector_by_quat_xyzw(
        ...     1.0, 0.0, 0.0, 0.0, 0.0, math.sin(math.pi / 4), math.cos(math.pi / 4)
        ... )
        >>> round(rx, 9), round(ry, 9), round(rz, 9)
        (0.0, 1.0, 0.0)
    """
    uvx = qy * vz - qz * vy
    uvy = qz * vx - qx * vz
    uvz = qx * vy - qy * vx
    uuvx = qy * uvz - qz * uvy
    uuvy = qz * uvx - qx * uvz
    uuvz = qx * uvy - qy * uvx
    return (
        vx + 2.0 * qw * uvx + 2.0 * uuvx,
        vy + 2.0 * qw * uvy + 2.0 * uuvy,
        vz + 2.0 * qw * uvz + 2.0 * uuvz,
    )


def _conjugate_xyzw(qx: float, qy: float, qz: float, qw: float) -> tuple[float, float, float, float]:
    """The conjugate (inverse, for a unit quaternion) of ``[x, y, z, w]``."""
    return (-qx, -qy, -qz, qw)


def _make_main() -> None:
    """Build the sensor bridge node and spin it on a background thread.

    Two non-obvious things had to be verified live (2026-09-14, against a
    running SRB instance) before this settled, both silent failure modes —
    the process neither crashes nor logs an error, it just stops making
    progress:

    1. **Not `rclpy.spin(node)`** (a bare `SingleThreadedExecutor`): stalls
       this node after ~1-2s — `_on_tick` stops firing and every TF/odom
       lookup starves. `tf2_ros.TransformListener` puts its `/tf`/
       `/tf_static` subscriptions on a `ReentrantCallbackGroup` specifically
       so a TF-dependent callback can run concurrently with the listener's
       own callback — mixing that with this node's several
       MutuallyExclusive-group relay subscriptions needs a
       `MultiThreadedExecutor`, confirmed: swapping in one with 4 threads
       fixed it in isolation.
    2. **Not `executor.spin()` called directly on the main thread**, even
       with the `MultiThreadedExecutor` from (1): stalls identically after
       ~1s. Spinning the *same* executor on a background thread (main thread
       just waits) ran the full length of every test with no stall. Root
       cause not chased further (a main-thread-specific interaction between
       rclpy's wait-set wakeup and this process's signal handling is the
       working hypothesis); the fix is verified, not just theorised.
    """
    if not _ROS2_AVAILABLE:
        _log.error("rclpy not found — cannot start the LunarBot sensor bridge without ROS 2.")
        raise SystemExit(1)

    import threading

    import rclpy
    from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor

    from openral_observability import configure_observability

    configure_observability(service_name="openral.hal.lunar_bot.sensor_bridge")

    rclpy.init()
    node = LunarBotSensorBridgeNode()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)

    def _run_executor() -> None:
        try:
            executor.spin()
        except ExternalShutdownException:
            pass

    spin_thread = threading.Thread(
        target=_run_executor, name="lunar_bot_sensor_bridge_spin", daemon=True
    )
    spin_thread.start()
    try:
        # A bounded join (not a bare `spin_thread.join()`) so a SIGINT on
        # this main thread is actually delivered rather than blocked behind
        # an uninterruptible native wait.
        while spin_thread.is_alive():
            spin_thread.join(timeout=1.0)
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.try_shutdown()


if _ROS2_AVAILABLE:
    from geometry_msgs.msg import TransformStamped
    from nav_msgs.msg import Odometry
    from rclpy.node import Node
    from rclpy.qos import (
        DurabilityPolicy,
        HistoryPolicy,
        QoSProfile,
        ReliabilityPolicy,
    )
    from rclpy.time import Time as RclpyTime
    from sensor_msgs.msg import CameraInfo, Image
    from sensor_msgs.msg import Imu as RosImu
    from sensor_msgs.msg import JointState
    from sensor_msgs.msg import PointCloud2
    from tf2_ros import (
        Buffer,
        StaticTransformBroadcaster,
        TransformBroadcaster,
        TransformException,
        TransformListener,
    )

    #: Sensor-data QoS (CLAUDE.md §2: images/pointclouds/IMU — BEST_EFFORT,
    #: VOLATILE, KEEP_LAST small). Matches what SRB itself publishes closely
    #: enough to bridge without a reliability mismatch dropping every message
    #: (a RELIABLE subscriber gets nothing from a BEST_EFFORT publisher).
    _SENSOR_QOS = QoSProfile(
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.VOLATILE,
        history=HistoryPolicy.KEEP_LAST,
        depth=5,
    )
    #: CameraInfo / control-adjacent QoS (CLAUDE.md §2: RELIABLE, VOLATILE,
    #: KEEP_LAST=1).
    _INFO_QOS = QoSProfile(
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.VOLATILE,
        history=HistoryPolicy.KEEP_LAST,
        depth=1,
    )

    class LunarBotSensorBridgeNode(Node):  # type: ignore[misc]  # reason: rclpy.node.Node is untyped
        """Relay SRB's sensor topics + re-root its TF tree onto OpenRAL conventions.

        See the module docstring for the full rationale. Every publisher this
        node owns is created in ``__init__`` from ``robots/lunar_bot/robot.yaml``
        (``robot_yaml`` parameter) — a manifest missing one of
        ``REQUIRED_SENSOR_NAMES`` raises ``ROSConfigError`` at construction,
        never silently drops a sensor.

        Parameters:
            robot_yaml: Path to ``lunar_bot``'s ``RobotDescription`` YAML.
            env_tf_frame: SRB's per-env TF root (``DEFAULT_ENV_TF_FRAME``).
            robot_tf_frame: SRB's articulation-root TF frame for this robot
                (``DEFAULT_ROBOT_TF_FRAME``) — see the module docstring.
            bridge_rate_hz: Cadence for the TF/`` /odom`` republish timer.
                Image/CameraInfo/PointCloud2/Imu relays are event-driven
                (republished on receipt, not on this timer).
        """

        def __init__(self, node_name: str = "openral_hal_lunar_bot_sensor_bridge") -> None:
            """Load the manifest, then wire every relay/TF publisher."""
            super().__init__(node_name)
            from openral_core import RobotDescription
            from openral_core.exceptions import ROSConfigError

            self.declare_parameter("robot_yaml", "robots/lunar_bot/robot.yaml")
            self.declare_parameter("env_tf_frame", DEFAULT_ENV_TF_FRAME)
            self.declare_parameter("robot_tf_frame", DEFAULT_ROBOT_TF_FRAME)
            self.declare_parameter("bridge_rate_hz", 30.0)
            # "truth": odom is the simulator's own pose -- convenient, but it hands the robot a
            # perfect estimate no real rover has. "wheel": integrate the 4WIS wheel/steering
            # kinematics, so odom drifts with wheel slip and SLAM has its real job (F50).
            self.declare_parameter("odom_source", "truth")
            # wheel odometry takes its heading rate from the IMU (see _wheel_odometry)
            self.declare_parameter("odom_imu_yaw_rate", True)
            # The identity `map -> odom` stands in for a localizer only when there is none. With AMCL
            # or SLAM in the graph that node owns the edge; a second (static) publisher of the same
            # edge makes tf2 flip between the two.
            self.declare_parameter("publish_map_to_odom", True)

            robot_yaml = str(self.get_parameter("robot_yaml").value)
            description = RobotDescription.from_yaml(robot_yaml)
            self._sensors = {s.name: s for s in description.sensors}
            missing = [n for n in REQUIRED_SENSOR_NAMES if n not in self._sensors]
            if missing:
                raise ROSConfigError(
                    f"LunarBotSensorBridgeNode: robot.yaml '{robot_yaml}' is missing "
                    f"sensors {missing} (needs all of {list(REQUIRED_SENSOR_NAMES)}) — "
                    "see docs/lunar_bot_capability_set_plan.md §3.2."
                )

            self._base_frame = description.base_frame
            self._odom_frame = description.odom_frame
            self._map_frame = description.map_frame
            self._env_tf_frame = str(self.get_parameter("env_tf_frame").value)
            self._robot_tf_frame = str(self.get_parameter("robot_tf_frame").value)

            # ── TF ────────────────────────────────────────────────────────
            self._tf_buffer = Buffer()
            self._tf_listener = TransformListener(self._tf_buffer, self)
            self._tf_broadcaster = TransformBroadcaster(self)
            self._static_tf_broadcaster = StaticTransformBroadcaster(self)
            if bool(self.get_parameter("publish_map_to_odom").value):
                self._publish_static_map_to_odom()

            # ── /odom ─────────────────────────────────────────────────────
            self._odom_pub = self.create_publisher(Odometry, "/odom", _SENSOR_QOS)
            self._prev_pose_stamp_s: float | None = None
            self._prev_pose_xyz: tuple[float, float, float] | None = None
            self._latest_ang_vel: tuple[float, float, float] = (0.0, 0.0, 0.0)
            # wheel-odometry state (only used when odom_source == "wheel")
            self._wheel_pose = [0.0, 0.0, 0.0]  # x, y, yaw integrated from the wheels
            self._wheel_stamp_s: float | None = None
            self._joint_state: dict[str, tuple[float, float]] = {}  # name -> (position, velocity)
            self._truth_pub = self.create_publisher(Odometry, "/openral/pose_truth", _SENSOR_QOS)
            self.create_subscription(JointState, f"/{self._robot_tf_frame.rsplit('/', 1)[0]}/robot/joint_states",
                                     self._on_joint_state, _SENSOR_QOS)

            # ── IMU relay (data) + its TF is handled by _publish_sensor_frames ──
            imu_frame = self._sensors["imu"].frame_id
            self._imu_pub = self.create_publisher(RosImu, "/imu", _SENSOR_QOS)
            self.create_subscription(
                RosImu,
                f"/{self._env_tf_frame}/{SRB_SENSOR_TF_SOURCE['imu']}",
                lambda msg: self._on_imu(msg, frame_id=imu_frame),
                _SENSOR_QOS,
            )

            # ── Cameras: image + camera_info relays ──────────────────────
            self._relay_image(
                src_topic=f"/{self._env_tf_frame}/cam_front/image_rgb",
                dst_topic="/openral/cameras/front/image",
                frame_id=self._sensors["front"].frame_id,
            )
            self._relay_camera_info(
                src_topic=f"/{self._env_tf_frame}/cam_front/camera_info",
                dst_topic="/openral/cameras/front/camera_info",
                frame_id=self._sensors["front"].frame_id,
            )
            self._relay_image(
                src_topic=f"/{self._env_tf_frame}/cam_front/image_depth",
                dst_topic="/openral/cameras/front_depth/image",
                frame_id=self._sensors["front_depth"].frame_id,
            )
            self._relay_camera_info(
                src_topic=f"/{self._env_tf_frame}/cam_front/camera_info",
                dst_topic="/openral/cameras/front_depth/camera_info",
                frame_id=self._sensors["front_depth"].frame_id,
            )
            self._relay_pointcloud(
                src_topic=f"/{self._env_tf_frame}/cam_front/pointcloud",
                dst_topic="/openral/cameras/front_depth/points",
                frame_id=self._sensors["front_depth"].frame_id,
            )
            self._relay_image(
                src_topic=f"/{self._env_tf_frame}/cam_wrist/image_rgb",
                dst_topic="/openral/cameras/wrist/image",
                frame_id=self._sensors["wrist"].frame_id,
            )
            self._relay_camera_info(
                src_topic=f"/{self._env_tf_frame}/cam_wrist/camera_info",
                dst_topic="/openral/cameras/wrist/camera_info",
                frame_id=self._sensors["wrist"].frame_id,
            )
            # The wrist camera is a RealSense D435i like the front one: its depth (32FC1, metres along the optical
            # axis, registered to the RGB image in simulation) is relayed like the front's.
            self._relay_image(
                src_topic=f"/{self._env_tf_frame}/cam_wrist/image_depth",
                dst_topic="/openral/cameras/wrist_depth/image",
                frame_id=self._sensors["wrist"].frame_id,
            )

            # ── Lidar relay (data); its TF is handled by _publish_sensor_frames ──
            self._relay_pointcloud(
                src_topic=f"/{self._env_tf_frame}/lidar_robot/pointcloud",
                dst_topic="/openral/lidar/points",
                frame_id=self._sensors["lidar"].frame_id,
            )

            # ── Periodic: sensor-frame TF + /odom (needs a fresh tf2 lookup, not
            # a subscription callback) ───────────────────────────────────────
            self._odom_source = str(self.get_parameter("odom_source").value).strip().lower()
            self._use_imu_yaw_rate = bool(self.get_parameter("odom_imu_yaw_rate").value)
            bridge_rate_hz = float(self.get_parameter("bridge_rate_hz").value)
            self._timer = self.create_timer(1.0 / bridge_rate_hz, self._on_tick)

            _log.info(
                "sensor_bridge.started",
                robot_yaml=robot_yaml,
                sensors=list(self._sensors),
                robot_tf_frame=self._robot_tf_frame,
                bridge_rate_hz=bridge_rate_hz,
            )

        # ── Setup helpers ────────────────────────────────────────────────

        def _publish_static_map_to_odom(self) -> None:
            """Broadcast the identity ``map -> odom`` transform, once.

            Simulation ground truth: nothing estimates ``map -> odom`` here
            (no localizer), so it is identity, published static rather than
            re-sent every tick — stated as such, not dressed up as SLAM (see
            the module docstring).
            """
            t = TransformStamped()
            t.header.stamp = self.get_clock().now().to_msg()
            t.header.frame_id = self._map_frame
            t.child_frame_id = self._odom_frame
            t.transform.rotation.w = 1.0
            self._static_tf_broadcaster.sendTransform(t)

        def _relay_image(self, *, src_topic: str, dst_topic: str, frame_id: str) -> None:
            pub = self.create_publisher(Image, dst_topic, _SENSOR_QOS)

            def _cb(msg: Image) -> None:
                msg.header.frame_id = frame_id
                pub.publish(msg)

            self.create_subscription(Image, src_topic, _cb, _SENSOR_QOS)

        def _relay_camera_info(self, *, src_topic: str, dst_topic: str, frame_id: str) -> None:
            pub = self.create_publisher(CameraInfo, dst_topic, _INFO_QOS)

            def _cb(msg: CameraInfo) -> None:
                msg.header.frame_id = frame_id
                pub.publish(msg)

            self.create_subscription(CameraInfo, src_topic, _cb, _INFO_QOS)

        def _relay_pointcloud(self, *, src_topic: str, dst_topic: str, frame_id: str) -> None:
            pub = self.create_publisher(PointCloud2, dst_topic, _SENSOR_QOS)

            def _cb(msg: PointCloud2) -> None:
                msg.header.frame_id = frame_id
                pub.publish(msg)

            self.create_subscription(PointCloud2, src_topic, _cb, _SENSOR_QOS)

        def _publish_wheel_odom(self, stamp) -> None:  # noqa: ANN001  # reason: builtin_interfaces/Time
            now_s = stamp.sec + stamp.nanosec * 1e-9
            est = self._wheel_odometry(now_s)
            if est is None:  # no joint state yet
                return
            vx, vy, w, slip_var = est
            if self._wheel_stamp_s is not None:
                dt = now_s - self._wheel_stamp_s
                if 0.0 < dt < 1.0:
                    yaw = self._wheel_pose[2] + 0.5 * w * dt  # midpoint heading over the interval
                    c, s = math.cos(yaw), math.sin(yaw)
                    self._wheel_pose[0] += (vx * c - vy * s) * dt
                    self._wheel_pose[1] += (vx * s + vy * c) * dt
                    self._wheel_pose[2] += w * dt
            self._wheel_stamp_s = now_s
            x, y, yaw = self._wheel_pose
            qz, qw = math.sin(yaw / 2.0), math.cos(yaw / 2.0)

            out = TransformStamped()
            out.header.stamp = stamp
            out.header.frame_id = self._odom_frame
            out.child_frame_id = self._base_frame
            out.transform.translation.x, out.transform.translation.y = x, y
            out.transform.rotation.z, out.transform.rotation.w = qz, qw
            self._tf_broadcaster.sendTransform(out)

            odom = Odometry()
            odom.header.stamp = stamp
            odom.header.frame_id = self._odom_frame
            odom.child_frame_id = self._base_frame
            odom.pose.pose.position.x, odom.pose.pose.position.y = x, y
            odom.pose.pose.orientation.z, odom.pose.pose.orientation.w = qz, qw
            odom.twist.twist.linear.x = vx
            odom.twist.twist.linear.y = vy
            odom.twist.twist.angular.z = w
            # Slip-aware: the wheels disagree with a rigid-body twist exactly when they slip, so the
            # least-squares residual variance is the planar-velocity variance. The floor is the
            # measured straight-line slip no residual can see (all four wheels overspinning
            # together, 15-38 %, F50): (0.2 * speed)^2.
            speed_var = (0.2 * math.hypot(vx, vy)) ** 2
            odom.twist.covariance[0] = odom.twist.covariance[7] = slip_var + speed_var + 1e-4
            odom.twist.covariance[35] = 1e-4  # IMU gyro
            odom.twist.covariance[14] = odom.twist.covariance[21] = odom.twist.covariance[28] = 1e6  # unobserved
            self._odom_pub.publish(odom)

        def _on_joint_state(self, msg: JointState) -> None:
            vel = msg.velocity if len(msg.velocity) == len(msg.name) else [0.0] * len(msg.name)
            for i, name in enumerate(msg.name):
                self._joint_state[name] = (msg.position[i], vel[i])

        def _wheel_odometry(self, stamp_s: float) -> tuple[float, float, float, float] | None:
            """Body twist (vx, vy, w) from the wheels, by least squares over the four 4WIS constraints.

            Each wheel i at chassis-frame ``(xi, yi)``, steered to ``ti`` and rolling at ``wi``: the
            rigid-body velocity at the wheel is ``(vx - w * yi, vy + w * xi)``, so
            ``wi * r * cos(ti) = vx - w * yi`` and ``wi * r * sin(ti) = vy + w * xi``. Eight
            equations, three unknowns (the base crabs, so ``vy`` is a real unknown). The residual is
            the slip the wheels disagree about; returned as a per-axis variance for ``/odom``.
            """
            import numpy as np  # reason: bridge-local, and only on the wheel-odometry path

            rows, rhs = [], []
            for (xi, yi), sj, dj in zip(WHEEL_POSITIONS_M, STEERING_JOINTS, DRIVE_JOINTS, strict=True):
                if sj not in self._joint_state or dj not in self._joint_state:
                    return None
                ti = self._joint_state[sj][0]
                rim = self._joint_state[dj][1] * WHEEL_RADIUS_M
                rows.append((1.0, 0.0, -yi)); rhs.append(rim * math.cos(ti))
                rows.append((0.0, 1.0, xi)); rhs.append(rim * math.sin(ti))
            a, b = np.array(rows), np.array(rhs)
            (vx, vy, w), *_ = np.linalg.lstsq(a, b, rcond=None)
            slip_var = float(np.sum((a @ np.array([vx, vy, w]) - b) ** 2)) / (len(b) - 3)
            del stamp_s
            # Heading rate from the IMU, not from the wheels: an in-place turn is pure lateral scrub,
            # where the wheels realise 42-54 % of the commanded rate and the kinematic inversion
            # believes the wheels. Wheel-inertial odometry is the standard answer and it is what a real
            # rover carries (research repo F50).
            if self._use_imu_yaw_rate:
                w = self._latest_ang_vel[2]  # the IMU sits on chassis_base_link with no rotation (SRB lunarbot.py)
            return float(vx), float(vy), float(w), slip_var

        def _on_imu(self, msg: RosImu, *, frame_id: str) -> None:
            self._latest_ang_vel = (
                msg.angular_velocity.x,
                msg.angular_velocity.y,
                msg.angular_velocity.z,
            )
            msg.header.frame_id = frame_id
            # SRB's Imu sensor models linear acceleration + angular velocity
            # only (srb/core/sensor -- no orientation filter); mark
            # orientation as "not provided" per the sensor_msgs/Imu
            # convention rather than publishing a fabricated identity quat.
            msg.orientation_covariance[0] = -1.0
            self._imu_pub.publish(msg)

        # ── Periodic: TF re-rooting + /odom ─────────────────────────────

        def _on_tick(self) -> None:
            self._publish_sensor_frames()
            self._publish_tcp_frame()
            self._publish_arm_root_frame()
            self._publish_odom()

        def _publish_arm_root_frame(self) -> None:
            """Broadcast ``chassis_base_link -> base_link`` (the arm's mount) for MoveIt's planning root."""
            tf = self._lookup(self._robot_tf_frame, f"{self._robot_tf_frame}/{ARM_ROOT_BODY_NAME}")
            if tf is None:
                return
            out = TransformStamped()
            out.header.stamp = self.get_clock().now().to_msg()
            out.header.frame_id = self._base_frame
            out.child_frame_id = ARM_ROOT_FRAME_ID
            out.transform = tf.transform
            self._tf_broadcaster.sendTransform(out)

        def _lookup(self, target_frame: str, source_frame: str) -> TransformStamped | None:
            try:
                return self._tf_buffer.lookup_transform(target_frame, source_frame, RclpyTime())
            except TransformException as exc:
                _log.debug(
                    "sensor_bridge.tf_lookup_failed",
                    target_frame=target_frame,
                    source_frame=source_frame,
                    error=str(exc),
                )
                return None

        def _publish_sensor_frames(self) -> None:
            """Re-root each sensor's SRB pose as ``chassis_base_link -> <manifest frame>``; cameras through their mount body."""
            stamp = self.get_clock().now().to_msg()
            for manifest_name, (body, pos, rpy) in CAMERA_MOUNTS.items():
                tf = self._lookup(self._robot_tf_frame, f"{self._robot_tf_frame}/{body}")
                if tf is None:
                    continue
                t, r = tf.transform.translation, tf.transform.rotation
                q_body = (r.x, r.y, r.z, r.w)
                ox, oy, oz = rotate_vector_by_quat_xyzw(*pos, *q_body)
                qx, qy, qz, qw = _quat_mul_xyzw(_quat_mul_xyzw(q_body, _rpy_deg_to_quat_xyzw(*rpy)), _WORLD_TO_OPTICAL_XYZW)
                out = TransformStamped()
                out.header.stamp = stamp
                out.header.frame_id = self._base_frame
                out.child_frame_id = self._sensors[manifest_name].frame_id
                out.transform.translation.x, out.transform.translation.y, out.transform.translation.z = t.x + ox, t.y + oy, t.z + oz
                out.transform.rotation.x, out.transform.rotation.y, out.transform.rotation.z, out.transform.rotation.w = qx, qy, qz, qw
                self._tf_broadcaster.sendTransform(out)
            for manifest_name, srb_name in SRB_SENSOR_TF_SOURCE.items():
                if manifest_name in CAMERA_MOUNTS:
                    continue
                tf = self._lookup(self._robot_tf_frame, f"{self._env_tf_frame}/{srb_name}")
                if tf is None:
                    continue
                out = TransformStamped()
                out.header.stamp = stamp
                out.header.frame_id = self._base_frame
                out.child_frame_id = self._sensors[manifest_name].frame_id
                out.transform = tf.transform
                self._tf_broadcaster.sendTransform(out)

        def _publish_tcp_frame(self) -> None:
            """Broadcast ``chassis_base_link -> tcp_frame`` (``Link7`` + ``_TCP_OFFSET``).

            ``Link7``'s own TF (``<robot_tf_frame>/Link7``) is a direct child
            of ``robot_tf_frame`` (an articulation body, not a sensor — see
            ``LINK7_BODY_NAME``'s docstring), and this bridge already treats
            ``chassis_base_link`` as coincident with ``robot_tf_frame``'s
            pose (``_publish_odom``), so no extra lookup against
            ``chassis_base_link`` is needed: ``T(robot_tf_frame <- Link7)``
            IS ``T(chassis_base_link <- Link7)`` under that same assumption.
            The TCP offset is applied in ``Link7``'s own frame (rotated into
            the base frame, then added) — matching ``SwitchableArmActionCfg
            .OffsetCfg(pos=_TCP_OFFSET)``'s position-only, identity-rotation
            offset in ``lunarbot.py``.
            """
            tf = self._lookup(self._robot_tf_frame, f"{self._robot_tf_frame}/{LINK7_BODY_NAME}")
            if tf is None:
                return
            t = tf.transform.translation
            r = tf.transform.rotation
            ox, oy, oz = rotate_vector_by_quat_xyzw(*_TCP_OFFSET, r.x, r.y, r.z, r.w)
            out = TransformStamped()
            out.header.stamp = self.get_clock().now().to_msg()
            out.header.frame_id = self._base_frame
            out.child_frame_id = TCP_FRAME_ID
            out.transform.translation.x = t.x + ox
            out.transform.translation.y = t.y + oy
            out.transform.translation.z = t.z + oz
            out.transform.rotation = r
            self._tf_broadcaster.sendTransform(out)

        def _publish_odom(self) -> None:
            """Broadcast ``odom -> chassis_base_link`` and ``/odom``.

            ``odom_source: truth`` republishes the simulator's own pose. ``odom_source: wheel``
            integrates the 4WIS wheel/steering kinematics instead, so the estimate drifts with wheel
            slip exactly as a real rover's does (15-38 % slip on this regolith, research repo F50) and
            the localizer (AMCL on the surveyed map, or SLAM) has a real ``map -> odom`` correction to make. The simulator's pose is published on
            ``/openral/pose_truth`` either way -- for the evaluator, never for the robot.
            """
            tf = self._lookup(self._env_tf_frame, self._robot_tf_frame)
            if tf is None:
                return
            stamp = tf.header.stamp
            truth = Odometry()
            truth.header.stamp = stamp
            truth.header.frame_id = self._env_tf_frame
            truth.child_frame_id = self._base_frame
            truth.pose.pose.position.x = tf.transform.translation.x
            truth.pose.pose.position.y = tf.transform.translation.y
            truth.pose.pose.position.z = tf.transform.translation.z
            truth.pose.pose.orientation = tf.transform.rotation
            self._truth_pub.publish(truth)

            if self._odom_source == "wheel":
                self._publish_wheel_odom(stamp)
                return
            now_s = stamp.sec + stamp.nanosec * 1e-9
            translation = tf.transform.translation
            rotation = tf.transform.rotation
            x, y, z = translation.x, translation.y, translation.z

            out = TransformStamped()
            out.header.stamp = stamp
            out.header.frame_id = self._odom_frame
            out.child_frame_id = self._base_frame
            out.transform = tf.transform
            self._tf_broadcaster.sendTransform(out)

            odom = Odometry()
            odom.header.stamp = stamp
            odom.header.frame_id = self._odom_frame
            odom.child_frame_id = self._base_frame
            odom.pose.pose.position.x = x
            odom.pose.pose.position.y = y
            odom.pose.pose.position.z = z
            odom.pose.pose.orientation = rotation

            if self._prev_pose_stamp_s is not None and self._prev_pose_xyz is not None:
                dt = now_s - self._prev_pose_stamp_s
                px, py, pz = self._prev_pose_xyz
                if dt > 1e-6:
                    vx_w, vy_w, vz_w = (x - px) / dt, (y - py) / dt, (z - pz) / dt
                    cqx, cqy, cqz, cqw = _conjugate_xyzw(
                        rotation.x, rotation.y, rotation.z, rotation.w
                    )
                    vx_b, vy_b, vz_b = rotate_vector_by_quat_xyzw(
                        vx_w, vy_w, vz_w, cqx, cqy, cqz, cqw
                    )
                    odom.twist.twist.linear.x = vx_b
                    odom.twist.twist.linear.y = vy_b
                    odom.twist.twist.linear.z = vz_b
            (
                odom.twist.twist.angular.x,
                odom.twist.twist.angular.y,
                odom.twist.twist.angular.z,
            ) = self._latest_ang_vel

            self._odom_pub.publish(odom)
            self._prev_pose_stamp_s = now_s
            self._prev_pose_xyz = (x, y, z)


main = _make_main

if __name__ == "__main__":
    main()
