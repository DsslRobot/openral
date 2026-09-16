"""Simulator bridge nodes in deploy_e2e.launch.py (ExternalSimulatorSpec.bridges).

Nav2's local_costmap cannot activate before odom -> base TF exists; for a ROS-attached
simulator that TF comes from a bridge node, not the HAL (research repo F37). These pin
that the launch builds the declared bridges with the graph clock and the manifest path,
and collects the topics Nav2 is gated on.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from openral_core import ExternalSimulatorSpec, SimulatorBridgeSpec

REPO_ROOT = Path(__file__).resolve().parents[2]
LAUNCH = REPO_ROOT / "packages/openral_rskill_ros/launch/deploy_e2e.launch.py"


@pytest.fixture(scope="module")
def launch_module() -> object:
    for dep in ("launch", "launch_ros", "lifecycle_msgs", "openral_foxglove_bringup"):
        pytest.importorskip(dep, reason=f"{dep} is a module-level import of the launch file")
    spec = importlib.util.spec_from_file_location("deploy_e2e_launch_bridges", LAUNCH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["deploy_e2e_launch_bridges"] = module
    spec.loader.exec_module(module)
    return module


def _sim(*bridges: SimulatorBridgeSpec) -> ExternalSimulatorSpec:
    return ExternalSimulatorSpec(argv=["srb", "agent", "ros"], bridges=list(bridges))


def test_no_bridges_means_no_nodes_and_no_gate(launch_module: object) -> None:
    sim = _sim()
    assert launch_module._simulator_bridge_nodes(sim, "robots/x/robot.yaml", True) == []
    assert launch_module._simulator_bridge_topics(sim) == []


def test_bridge_nodes_get_sim_clock_and_manifest(launch_module: object) -> None:
    a = SimulatorBridgeSpec(package="pkg_a", executable="a.py", name="bridge_a", parameters={"rate": 30.0},
                            pass_robot_yaml=True, publishes=["/odom", "/imu"])
    b = SimulatorBridgeSpec(package="pkg_b", executable="b.py", name="bridge_b", publishes=["/imu", "/scan_in"])
    nodes = launch_module._simulator_bridge_nodes(_sim(a, b), "/abs/robot.yaml", True)
    assert [n.node_package for n in nodes] == ["pkg_a", "pkg_b"]
    assert launch_module._simulator_bridge_params(a, "/abs/robot.yaml", True) == {
        "rate": 30.0, "robot_yaml": "/abs/robot.yaml", "use_sim_time": True}
    assert launch_module._simulator_bridge_params(b, "/abs/robot.yaml", False) == {"use_sim_time": False}
    assert launch_module._simulator_bridge_topics(_sim(a, b)) == ["/odom", "/imu", "/scan_in"]


def test_bridge_spec_rejects_unknown_fields() -> None:
    with pytest.raises(ValueError):
        SimulatorBridgeSpec(package="p", executable="e", name="n", remap={"a": "b"})


def test_motion_planning_include_is_scoped_to_the_graph_clock(launch_module: object) -> None:
    """move_group comes from the robot manifest and runs on the graph's clock (F49)."""
    from openral_core import LaunchInclude, RobotDescription

    spec = RobotDescription.from_yaml(str(REPO_ROOT / "robots/lunar_bot/robot.yaml")).motion_planning
    assert spec == LaunchInclude(package="rm_75_config", launch_file="move_group.launch.py")
    group = launch_module._build_motion_planning_include(spec, use_sim_time=True)
    kinds = [type(a).__name__ for a in group.get_sub_entities()]
    assert "SetParameter" in kinds and "IncludeLaunchDescription" in kinds
    include = next(a for a in group.get_sub_entities() if type(a).__name__ == "IncludeLaunchDescription")
    assert include is not None  # path comes from get_package_share_directory + the declared file name


def test_motion_planning_include_refuses_an_unbuilt_package(launch_module: object) -> None:
    """A manifest naming a MoveIt config that is not on the ament path fails at launch parse."""
    from openral_core import LaunchInclude
    from openral_core.exceptions import ROSConfigError

    with pytest.raises(ROSConfigError, match="not on the ament path"):
        launch_module._build_motion_planning_include(
            LaunchInclude(package="no_such_moveit_config", launch_file="move_group.launch.py"), use_sim_time=True
        )
