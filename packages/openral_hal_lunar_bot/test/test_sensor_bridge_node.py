"""``rotate_vector_by_quat_xyzw`` — the one piece of ``/odom``'s twist math
that has nothing to do with ROS and can be pinned without rclpy: rotating
SRB's world-frame finite-difference velocity into ``chassis_base_link``
(REP-105 requires ``Odometry.twist`` in ``child_frame_id``).

Import-only smoke coverage for the module's ``_ROS2_AVAILABLE`` guard lives
implicitly here too: this file imports the module on a host with no ROS 2
sourced (ament pytest runs standalone), so a failure to import at all would
mean the guard is broken, not just untested rotation math.
"""

from __future__ import annotations

import math

import pytest
from openral_hal_lunar_bot.sensor_bridge_node import _conjugate_xyzw, rotate_vector_by_quat_xyzw


def test_identity_rotation_is_a_no_op() -> None:
    assert rotate_vector_by_quat_xyzw(1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0) == (1.0, 2.0, 3.0)


def test_quarter_turn_about_z_sends_x_to_y() -> None:
    qz, qw = math.sin(math.pi / 4), math.cos(math.pi / 4)
    rx, ry, rz = rotate_vector_by_quat_xyzw(1.0, 0.0, 0.0, 0.0, 0.0, qz, qw)
    assert (rx, ry, rz) == pytest.approx((0.0, 1.0, 0.0), abs=1e-9)


def test_half_turn_about_x_negates_y_and_z() -> None:
    # 180 deg about X: qx=1, qw=0 (unnormalised representation of a pure
    # 180 deg rotation is still (sin(pi/2), 0, 0, cos(pi/2)) = (1, 0, 0, 0)).
    rx, ry, rz = rotate_vector_by_quat_xyzw(1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0)
    assert (rx, ry, rz) == pytest.approx((1.0, -2.0, -3.0), abs=1e-9)


def test_conjugate_round_trip_recovers_the_original_vector() -> None:
    """Exactly ``_publish_odom``'s use: rotate world->body, then back, is a no-op.

    An arbitrary (non-axis-aligned) unit quaternion, so the test cannot pass
    by accident the way an axis-aligned one could.
    """
    # Unit quaternion for a ~73.9 deg rotation about (1, 1, 1)/sqrt(3).
    axis = (1.0 / math.sqrt(3),) * 3
    half_angle = 0.645
    s = math.sin(half_angle)
    qx, qy, qz, qw = axis[0] * s, axis[1] * s, axis[2] * s, math.cos(half_angle)

    v_world = (0.37, -1.21, 5.5)
    v_body = rotate_vector_by_quat_xyzw(*v_world, qx, qy, qz, qw)
    cqx, cqy, cqz, cqw = _conjugate_xyzw(qx, qy, qz, qw)
    v_roundtrip = rotate_vector_by_quat_xyzw(*v_body, cqx, cqy, cqz, cqw)

    assert v_roundtrip == pytest.approx(v_world, abs=1e-9)
    # A rotation must preserve magnitude — catches a sign/formula error that
    # a component-wise comparison alone could miss.
    mag_before = math.dist((0.0, 0.0, 0.0), v_world)
    mag_after = math.dist((0.0, 0.0, 0.0), v_body)
    assert mag_after == pytest.approx(mag_before, abs=1e-9)


def test_conjugate_flips_only_the_vector_part() -> None:
    assert _conjugate_xyzw(0.1, -0.2, 0.3, 0.9) == (-0.1, 0.2, -0.3, 0.9)
