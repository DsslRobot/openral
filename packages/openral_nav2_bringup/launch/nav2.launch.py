#!/usr/bin/env python3
"""Stand-alone launch for the Nav2 stack.

Includes upstream ``nav2_bringup/launch/navigation_launch.py`` — brings up ``bt_navigator``,
``planner_server``, ``controller_server``, ``smoother_server``, ``behavior_server``,
``velocity_smoother`` and ``lifecycle_manager_navigation`` (drives them to ``ACTIVE``). Params come
from ``config/nav2_panda_mobile.yaml`` (a copy of upstream ``nav2_params.yaml``). With a
``robot_yaml`` arg, ``RewrittenYaml`` substitutes ``robot_radius`` (from ``footprint_radius``),
costmap ``inflation_radius`` (footprint + clearance) and MPPI ``motion_model`` (from
``base_kinematics``) via ``RobotDescription.nav2_param_overrides()``, so one base file serves any
mobile base. Base ships panda_mobile's values (``robot_radius: 0.35``, ``inflation_radius: 0.40``,
``motion_model: Omni``, ``vy_min: -0.5``) — a no-op rewrite for panda. Velocity bounds stay Nav2
tuning in the base file, not robot identity.

Unlike slam_toolbox (idles until the Reasoner activates it), Nav2 is always-on: each sub-node is a
``LifecycleNode`` driven by ``lifecycle_manager_navigation`` (``autostart=true``). The Reasoner
triggers navigation by dispatching the ``OpenRAL/rskill-nav2-mobile_base-navigate_to_pose-none``
wrapped-action rSkill (``NavigateToPose`` goal to ``/navigate_to_pose``), not by
lifecycle-transitioning the planner.

Composed into ``packages/openral_rskill_ros/launch/deploy_e2e.launch.py`` when ``enable_nav2=true``.
"""

from __future__ import annotations

import os
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    GroupAction,
    IncludeLaunchDescription,
    OpaqueFunction,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration

#: How long ``lifecycle_manager_navigation`` waits for a managed server's bond
#: heartbeat before declaring it dead and tearing down the whole Nav2 stack.
#:
#: Nav2's default is 4.0 s, and on a loaded host that is short enough to be
#: missed by a scheduling hiccup rather than by a real fault. The cascade is
#: silent and total: the manager deactivates every server, the graph stays up
#: doing nothing, and the run burns its deadline with no error printed. In the
#: 2026-09-06 ceiling battery this killed 31 of 89 valid runs, every one of
#: which the harness then scored as a policy failure (`deadline-no-grasp`) —
#: see ``docs/reference/collision-validation-evidence.md`` and issue #256.
#: Twenty-five of the 31 named ``controller_server``.
#:
#: This is a liveness timeout on the navigation stack, NOT a safety check: the
#: E-stop path is ``openral_safety_kernel``, which is unaffected. Raising it
#: trades a slower reaction to a genuinely hung server for not mistaking a
#: descheduled one for a dead one. It is not disabled (``0.0``), so a server
#: that really dies is still caught.
BOND_TIMEOUT_S = 30.0


def _params_path_for_backend(backend: str) -> str:
    """Pick the Nav2 base params for the SLAM backend.

    ``visual`` (cuVSLAM + nvblox) uses ``nav2_visual.yaml`` — global+local
    costmaps consume the backend-agnostic ``/map`` via ``static_layer``.
    Anything else (``lidar``/``none``) uses the ``/scan``-based base config.
    """
    share = get_package_share_directory("openral_nav2_bringup")
    fname = "nav2_visual.yaml" if backend.strip().lower() == "visual" else "nav2_panda_mobile.yaml"
    return os.path.join(share, "config", fname)


def _default_params_path() -> str:
    return _params_path_for_backend("lidar")


def _upstream_navigation_launch() -> str:
    share = get_package_share_directory("nav2_bringup")
    return os.path.join(share, "launch", "navigation_launch.py")


def generate_launch_description() -> LaunchDescription:
    """Stand-alone bring-up for the upstream Nav2 navigation stack."""
    args = [
        DeclareLaunchArgument(
            "params_file",
            default_value="",
            description=(
                "YAML parameter file for Nav2. Empty (default) selects the "
                "base config by `slam_backend`: nav2_panda_mobile.yaml (lidar) "
                "or nav2_visual.yaml (visual). Set explicitly to override."
            ),
        ),
        DeclareLaunchArgument(
            "slam_backend",
            default_value="lidar",
            description=(
                "Which SLAM backend feeds the costmap: `visual` "
                "(cuVSLAM+nvblox; costmaps consume `/map` via static_layer) or "
                "`lidar`/`none` (costmaps ray-cast `/scan`). Selects the base "
                "params file when `params_file` is empty."
            ),
        ),
        DeclareLaunchArgument(
            "use_sim_time",
            default_value="true",
            description="Pass-through to Nav2's `use_sim_time`.",
        ),
        DeclareLaunchArgument(
            "autostart",
            default_value="true",
            description=(
                "Drive Nav2's lifecycle_manager_navigation to ACTIVE "
                "automatically. Nav2 sits idle until a NavigateToPose "
                "goal arrives, so always-on is the right default — "
                "the Reasoner triggers navigation by dispatching the "
                "wrapped-action rSkill, not by lifecycle-transition."
            ),
        ),
        DeclareLaunchArgument(
            "use_composition",
            default_value="False",
            description=(
                "When True, run all Nav2 components in a single "
                "process. Off by default — composition makes "
                "per-component lifecycle introspection harder."
            ),
        ),
        DeclareLaunchArgument(
            "robot_yaml",
            default_value="",
            description=(
                "Path to the robot's robot.yaml. When set, the base "
                "params_file is rewritten with the robot's "
                "`RobotDescription.nav2_param_overrides()` (robot_radius + "
                "inflation_radius from footprint_radius, motion_model from "
                "base_kinematics) so one shared base file serves any mobile "
                "base. Empty string uses params_file verbatim."
            ),
        ),
        DeclareLaunchArgument(
            "payload_scan_filter",
            default_value="true",
            description=(
                "Run the payload scan filter alongside Nav2: it removes the "
                "robot's OWN returns, and any carried object's, from the scan "
                "the costmaps and the collision monitor read. `robot_yaml` "
                "enables its self half (the manifest's bare chassis outline); "
                "without one only the payload half runs. Lidar backend only — "
                "the base params point every observation source at its output, "
                "so turning this off means re-pointing them at `/scan`. "
                "There is no footprint publisher: Nav2 is base-only and takes "
                "its polygon statically from the manifest (see the README)."
            ),
        ),
        DeclareLaunchArgument(
            "pointcloud_to_laserscan",
            default_value="true",
            description=(
                "Run `pointcloud_to_laserscan_node` alongside Nav2: converts "
                "`/openral/lidar/points` into `/scan` for robots whose only "
                "lidar is a 3-D point cloud (e.g. LunarBot's Mid-360) rather "
                "than a native 2-D scan (§8.3 item 5). Lidar backend only "
                "(mirrors `payload_scan_filter`). Requires "
                "`ros-humble-pointcloud-to-laserscan`."
            ),
        ),
        DeclareLaunchArgument(
            "cmd_vel_relay",
            default_value="true",
            description=(
                "Run `twist_to_action_relay_node.py` alongside Nav2: bridges its "
                "`/cmd_vel` output onto `/openral/candidate_action` so the safety "
                "kernel checks every Nav2 velocity command instead of it reaching "
                "the base controller directly (§8.3 item 5). Requires `robot_yaml`; "
                "self-disables (logs why, stays idle) for a robot whose manifest "
                "declares `base_joints` — `MobileBaseBridge` already bridges "
                "`/cmd_vel` for that case, via its own bypass path — or that omits "
                "`body_twist` from `capabilities.supported_control_modes`."
            ),
        ),
    ]

    # ``OpaqueFunction`` (upstream ``launch.actions``) defers the callback
    # to launch-execution time, where the ``robot_yaml`` / ``params_file``
    # LaunchConfiguration values are finally resolved — we need them to
    # build the per-robot RewrittenYaml, which can't happen at parse time.
    return LaunchDescription([*args, OpaqueFunction(function=_nav2_include_with_robot_overrides)])


def _payload_scan_filter_nodes(
    *, robot_yaml: str, slam_backend: str, use_sim_time: object
) -> list[object]:
    """The scan filter that rides with Nav2.

    ``openral_nav2_payload_scan_filter`` removes the robot's own returns and any carried object's
    from the scan the costmaps and collision monitor consume. Reads the attachment set off
    ``/openral/world_state_fast`` (the safety kernel's own source) and takes the chassis outline
    from ``robot_yaml``.

    No footprint publisher: Nav2 is base-only, and the costmaps' ``footprint`` comes statically
    from the manifest via ``RobotDescription.nav2_param_overrides()`` (see README, "Nav2 is
    base-only").

    ``robot_yaml`` is optional: without it only the payload half runs (the node warns). Skipped on
    the ``visual`` backend, which has no ``/scan``.
    """
    from launch_ros.actions import Node  # reason: launch-time only

    nodes: list[object] = []
    if slam_backend.strip().lower() == "visual":
        return nodes
    params: dict[str, object] = {"use_sim_time": use_sim_time}
    if robot_yaml:
        params["robot_yaml"] = robot_yaml
    nodes.append(
        Node(
            package="openral_nav2_bringup",
            executable="payload_scan_filter_node.py",
            name="openral_nav2_payload_scan_filter",
            output="screen",
            parameters=[params],
        )
    )
    return nodes


def _pointcloud_to_laserscan_nodes(*, slam_backend: str, use_sim_time: object) -> list[object]:
    """The `/openral/lidar/points` -> `/scan` conversion for lidar-only robots (e.g. LunarBot).

    §8.3 item 5 / `docs/srb_deploy_backend_plan.md` §5: LunarBot's Mid-360 reaches OpenRAL as a
    `sensor_msgs/PointCloud2` on `/openral/lidar/points` (`openral_hal_lunar_bot.sensor_bridge_node`,
    F25); Nav2's lidar profile (`config/nav2_panda_mobile.yaml`) consumes only a 2-D `/scan` (through
    `payload_scan_filter_node.py`'s `/scan` -> `/openral/nav2/scan`, already wired) — no separate
    full-cloud costmap feed exists to use instead. Requires `ros-humble-pointcloud-to-laserscan`
    (not installed by this launch file — `apt install ros-humble-pointcloud-to-laserscan`; the node
    simply fails to start until it is, same as any other missing exec dependency). Visual backend
    has no `/scan` consumer, so this is skipped there (mirrors `_payload_scan_filter_nodes`).

    Height band / range bounds below are a first-pass choice (base-frame-relative slice through the
    Mid-360's real -7..+52 deg vertical FOV), not yet live-tuned against real terrain returns —
    same "provisional, revisit after a live run" status as `robot.yaml`'s new `footprint_radius`.
    """
    from launch_ros.actions import Node  # reason: launch-time only

    if slam_backend.strip().lower() == "visual":
        return []
    return [
        Node(
            package="pointcloud_to_laserscan",
            executable="pointcloud_to_laserscan_node",
            name="openral_nav2_pointcloud_to_laserscan",
            output="screen",
            remappings=[("cloud_in", "/openral/lidar/points"), ("scan", "/scan")],
            parameters=[
                {
                    "use_sim_time": use_sim_time,
                    "target_frame": "chassis_base_link",
                    "transform_tolerance": 0.1,
                    "min_height": 0.05,
                    "max_height": 0.6,
                    "angle_min": -3.14159,
                    "angle_max": 3.14159,
                    "angle_increment": 0.0087,  # ~0.5 deg
                    "scan_time": 0.1,
                    "range_min": 0.1,
                    "range_max": 40.0,
                    "use_inf": True,
                    "concurrency_level": 1,
                }
            ],
        )
    ]


def _twist_to_action_relay_nodes(*, robot_yaml: str) -> list[object]:
    """The `/cmd_vel` -> `/openral/candidate_action` relay that rides with Nav2.

    See `twist_to_action_relay_node.py`'s module docstring for the full rationale (§8.3 item 5)
    and the mutual-exclusion check against `MobileBaseBridge`'s bypass path — that check happens
    inside the node itself (it needs to load and inspect `robot_yaml`), not here.
    """
    from launch_ros.actions import Node  # reason: launch-time only

    if not robot_yaml:
        return []
    return [
        Node(
            package="openral_nav2_bringup",
            executable="twist_to_action_relay_node.py",
            name="openral_nav2_twist_to_action_relay",
            output="screen",
            parameters=[{"robot_yaml": robot_yaml}],
        )
    ]


def _with_use_sim_time(params_file: str, use_sim_time: bool) -> str:
    """Copy of the params file with ``use_sim_time`` set in every ``ros__parameters`` block.

    Upstream ``navigation_launch.py`` hands each server only the params file and rewrites
    ``use_sim_time`` through ``RewrittenYaml``, which replaces keys that already exist and
    adds none. The shared base files carry no ``use_sim_time``, so on a simulation clock
    (SRB publishes ``/clock``) every Nav2 server still ran on wall time: paths were stamped
    ~1.8e9 s against a TF tree at a few seconds of sim time, the controller logged "Transform
    data too old when converting from map to odom" and reported the goal reached without
    moving (research repo F47).
    """
    import tempfile

    import yaml

    data = yaml.safe_load(Path(params_file).read_text())

    def mark(node: object) -> None:
        if isinstance(node, dict):
            for k, v in node.items():
                if k == "ros__parameters" and isinstance(v, dict):
                    v["use_sim_time"] = use_sim_time
                mark(v)

    mark(data)
    out = tempfile.NamedTemporaryFile("w", prefix="openral_nav2_time_", suffix=".yaml", delete=False)  # noqa: SIM115
    with out:
        yaml.safe_dump(data, out, sort_keys=False)
    return out.name


def _apply_path_overrides(params_file: str, rewrites: dict[str, str]) -> tuple[str, dict[str, str]]:
    """Substitute overrides whose key is a dotted path from the file's top level; return (file, other rewrites).

    ``RewrittenYaml`` rewrites every key of that name anywhere in the file, which is right for
    ``robot_radius`` and wrong for ``width``: the global and local costmaps both have one, and a
    worksite-sized global window must not become the local window. A key containing ``.`` is
    resolved as ``node.node.ros__parameters.key`` into a temporary copy of the file; scalars are
    parsed with YAML so numbers stay numbers.
    """
    import tempfile

    import yaml

    paths = {k: v for k, v in rewrites.items() if "." in k}
    if not paths:
        return params_file, rewrites
    data = yaml.safe_load(Path(params_file).read_text())
    for key, value in paths.items():
        node = data
        *parents, leaf = key.split(".")
        for p in parents:
            node = node[p]
        if leaf not in node:
            raise KeyError(f"Nav2 override {key!r}: {leaf!r} is not a parameter in {params_file}")
        parsed = yaml.safe_load(value)
        # keep the base file's declared type: rclcpp refuses a double for an integer parameter
        node[leaf] = int(parsed) if isinstance(node[leaf], int) and not isinstance(node[leaf], bool) else parsed
    out = tempfile.NamedTemporaryFile("w", prefix="openral_nav2_path_", suffix=".yaml", delete=False)  # noqa: SIM115
    with out:
        yaml.safe_dump(data, out, sort_keys=False)
    return out.name, {k: v for k, v in rewrites.items() if k not in paths}


def _apply_list_overrides(params_file: str, rewrites: dict[str, str]) -> tuple[str, dict[str, str]]:
    """Substitute overrides for keys that are lists in the base file; return (file, other rewrites).

    ``RewrittenYaml(convert_types=True)`` only converts scalars: a ``"[0.57, 0.0, 0.47]"``
    rewrite of ``velocity_smoother.max_velocity`` reaches the node as a *string* and its
    ``configure`` throws ("parameter 'max_velocity' ... double_array ... string"), which
    aborts ``lifecycle_manager_navigation`` before ``bt_navigator`` activates. Whether a
    ``"[...]"`` value is a list or a string follows the base file (the costmap ``footprint``
    is a string parameter and must stay one), so only keys whose base value is a YAML list
    are parsed here and written into a temporary copy of the file.
    """
    import tempfile

    import yaml

    data = yaml.safe_load(Path(params_file).read_text())
    list_keys: set[str] = set()

    def collect(node: object) -> None:
        if isinstance(node, dict):
            for k, v in node.items():
                if isinstance(v, list):
                    list_keys.add(k)
                collect(v)

    collect(data)
    parsed = {k: yaml.safe_load(v) for k, v in rewrites.items() if k in list_keys}
    if not parsed:
        return params_file, rewrites
    for k, v in parsed.items():
        if not isinstance(v, list):
            raise ValueError(f"Nav2 override {k}={rewrites[k]!r} is not a list, but {k} is a list in {params_file}")

    def substitute(node: object) -> None:
        if isinstance(node, dict):
            for k in list(node):
                if k in parsed and isinstance(node[k], list):
                    node[k] = parsed[k]
                else:
                    substitute(node[k])

    substitute(data)
    out = tempfile.NamedTemporaryFile("w", prefix="openral_nav2_", suffix=".yaml", delete=False)  # noqa: SIM115
    with out:
        yaml.safe_dump(data, out, sort_keys=False)
    return out.name, {k: v for k, v in rewrites.items() if k not in parsed}


def _nav2_include_with_robot_overrides(context: object) -> list[object]:
    """Rewrite the base Nav2 params with per-robot overrides, then include.

    Runs at launch time (via ``OpaqueFunction``) so it can read the
    resolved ``robot_yaml`` / ``params_file`` launch args off the
    ``context``.

    Keeps the bringup generic: ``robot.yaml`` is the single
    source for the per-robot Nav2 geometry/kinematics. ``RewrittenYaml``
    substitutes the matching keys in the shared base param file; an empty
    ``robot_yaml`` (or a fixed-base arm) yields no rewrites and the base
    file is used verbatim.
    """
    from nav2_common.launch import (
        RewrittenYaml,  # reason: nav2 dep, launch-time only
    )

    params_file = LaunchConfiguration("params_file").perform(context)  # type: ignore[attr-defined]
    robot_yaml = LaunchConfiguration("robot_yaml").perform(context)  # type: ignore[attr-defined]
    slam_backend = LaunchConfiguration("slam_backend").perform(context)  # type: ignore[attr-defined]
    description = None
    if robot_yaml:
        from openral_core import (
            RobotDescription,  # reason: defer schema import to launch time
        )

        description = RobotDescription.from_yaml(robot_yaml)
    # An empty params_file selects the robot's own base file when it declares one
    # (`nav2_params_file`), else the shared config by SLAM backend (visual → nav2_visual.yaml
    # consuming `/map`; lidar → the /scan base).
    if not params_file and description is not None and description.nav2_params_file:
        params_file = os.path.join(get_package_share_directory("openral_nav2_bringup"), "config", description.nav2_params_file)
    if not params_file:
        params_file = _params_path_for_backend(slam_backend)

    rewrites: dict[str, str] = {}
    if description is not None:
        rewrites = description.nav2_param_overrides()
        params_file, rewrites = _apply_path_overrides(params_file, rewrites)
        params_file, rewrites = _apply_list_overrides(params_file, rewrites)
    use_sim_time_value = LaunchConfiguration("use_sim_time").perform(context)  # type: ignore[attr-defined]
    params_file = _with_use_sim_time(params_file, use_sim_time_value.strip().lower() in ("true", "1", "yes"))

    resolved_params = RewrittenYaml(
        source_file=params_file,
        param_rewrites=rewrites,
        convert_types=True,
    )
    from launch_ros.actions import SetParameter  # reason: launch-time only

    actions: list[object] = [
        # `bond_timeout` cannot go in the params file: upstream
        # `navigation_launch.py` hands `lifecycle_manager_navigation` only
        # `{autostart, node_names}` and never the params file, so a
        # `lifecycle_manager_navigation:` block there is silently ignored. Set
        # as a scoped override instead; the servers that do not declare it
        # ignore the override.
        GroupAction(
            scoped=True,
            actions=[
                SetParameter(name="bond_timeout", value=BOND_TIMEOUT_S),
                IncludeLaunchDescription(
                    PythonLaunchDescriptionSource(_upstream_navigation_launch()),
                    launch_arguments={
                        "params_file": resolved_params,
                        "use_sim_time": LaunchConfiguration("use_sim_time"),
                        "autostart": LaunchConfiguration("autostart"),
                        "use_composition": LaunchConfiguration("use_composition"),
                    }.items(),
                ),
            ],
        )
    ]
    payload_scan_filter = LaunchConfiguration("payload_scan_filter").perform(context)  # type: ignore[attr-defined]
    if payload_scan_filter.strip().lower() in ("true", "1", "yes"):
        use_sim_time = LaunchConfiguration("use_sim_time").perform(context)  # type: ignore[attr-defined]
        actions.extend(
            _payload_scan_filter_nodes(
                robot_yaml=robot_yaml,
                slam_backend=slam_backend,
                use_sim_time=use_sim_time.strip().lower() in ("true", "1", "yes"),
            )
        )
    pointcloud_to_laserscan = LaunchConfiguration("pointcloud_to_laserscan").perform(context)  # type: ignore[attr-defined]
    if pointcloud_to_laserscan.strip().lower() in ("true", "1", "yes"):
        use_sim_time = LaunchConfiguration("use_sim_time").perform(context)  # type: ignore[attr-defined]
        actions.extend(
            _pointcloud_to_laserscan_nodes(
                slam_backend=slam_backend,
                use_sim_time=use_sim_time.strip().lower() in ("true", "1", "yes"),
            )
        )
    cmd_vel_relay = LaunchConfiguration("cmd_vel_relay").perform(context)  # type: ignore[attr-defined]
    if cmd_vel_relay.strip().lower() in ("true", "1", "yes"):
        actions.extend(_twist_to_action_relay_nodes(robot_yaml=robot_yaml))
    return actions


# Used by ``test/test_nav2_launch.py`` for hermetic argument validation
# without spawning a real ROS 2 graph.
DEFAULT_PARAMS_PATH = Path(__file__).resolve().parent.parent / "config" / "nav2_panda_mobile.yaml"
