#!/usr/bin/env python3
"""Block until every named ROS 2 topic has at least one publisher.

Used by ``packages/openral_rskill_ros/launch/deploy_e2e.launch.py`` to hold a
dependent stack (Nav2) until an external simulator's bridge nodes are
publishing what it consumes (``DeployScene.simulator.bridges[*].publishes``).
Exits 0 when all topics have a publisher, 1 when ``--timeout-s`` elapses first
(the launch still proceeds, but the missing topics are named on stderr).
"""

from __future__ import annotations

import argparse
import sys
import time

import rclpy


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topic", action="append", required=True, dest="topics", help="Topic to wait for. Repeatable.")
    parser.add_argument("--timeout-s", type=float, default=120.0, help="Total budget across all topics.")
    parser.add_argument("--label", default="", help="Who is waiting (for the log line).")
    args = parser.parse_args()

    rclpy.init()
    node = rclpy.create_node("openral_wait_for_topics")
    try:
        deadline = time.monotonic() + args.timeout_s
        remaining = set(args.topics)
        while remaining and time.monotonic() < deadline:
            remaining = {t for t in remaining if node.count_publishers(t) < 1}
            if remaining:
                rclpy.spin_once(node, timeout_sec=0.5)
        if remaining:
            print(f"wait-for-topics[{args.label}]: no publisher on {sorted(remaining)} after {args.timeout_s:.0f}s", file=sys.stderr)
            return 1
        print(f"wait-for-topics[{args.label}]: {sorted(args.topics)} all have publishers")
        return 0
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    sys.exit(main())
