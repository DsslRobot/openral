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
