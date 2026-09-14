"""Unit tests for `ManifestHALLifecycleNode`'s E-stop recovery paths.

Regression coverage for F17/F33 (docs/research_findings.md, research repo):
the HAL used to clear its own `_estopped` latch ONLY on a separate
`/openral/estop_cleared` broadcast, so a caller that only invoked the real
kernel's `/openral/estop_reset` service (which already publishes an
unlatched `/openral/safety_status`) left the HAL silently dropping every
`safe_action` even though the kernel had genuinely recovered. `_on_safety_status`
now delegates into the existing `_on_estop_cleared` recovery policy (real
per-HAL recovery, not bypassed) whenever it observes `latched: False`.

Skipped where ROS 2 (`rclpy`) is unavailable (e.g. a GPU-less CI runner).
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("rclpy")

from openral_hal.lifecycle import _ManifestHALLifecycleNode
from rclpy.parameter import Parameter

REPO_ROOT = Path(__file__).resolve().parents[2]


def _build_configured(node_name: str, robot_id: str) -> object:
    """A node with a real sim HAL attached (`_create_hal` + `_hal` set), not activated."""
    node = _ManifestHALLifecycleNode(node_name)
    node.set_parameters(
        [
            Parameter("robot_yaml", value=str(REPO_ROOT / "robots" / robot_id / "robot.yaml")),
            Parameter("hal_mode", value="sim"),
        ]
    )
    node._hal = node._create_hal()
    return node


def _safety_status(*, latched: bool) -> object:
    """A real `openral_msgs/SafetyStatus` message, not a stub/mock."""
    from openral_msgs.msg import SafetyStatus

    msg = SafetyStatus()
    msg.latched = latched
    return msg


@pytest.mark.usefixtures("_rclpy_ctx")
class TestSafetyStatusAutoClear:
    """`/openral/safety_status` reporting `latched: false` must clear `_estopped`."""

    def test_latched_false_clears_the_estop_latch(self) -> None:
        node = _build_configured("t_franka_status_clear", "franka_panda")
        try:
            node._estopped = True
            node._on_safety_status(_safety_status(latched=False))
            assert node._estopped is False
        finally:
            node.destroy_node()

    def test_latched_true_does_not_clear_the_estop_latch(self) -> None:
        """A latched status must never be mistaken for a clear."""
        node = _build_configured("t_franka_status_no_clear", "franka_panda")
        try:
            node._estopped = True
            node._on_safety_status(_safety_status(latched=True))
            assert node._estopped is True
        finally:
            node.destroy_node()

    def test_not_estopped_is_a_no_op(self) -> None:
        """No latch to clear -- must not raise or flip anything."""
        node = _build_configured("t_franka_status_noop", "franka_panda")
        try:
            node._estopped = False
            node._on_safety_status(_safety_status(latched=False))
            assert node._estopped is False
        finally:
            node.destroy_node()
