#!/usr/bin/env python3
"""Drive a ROS 2 LifecycleNode through CONFIGURE → ACTIVATE with retries.

Used by ``packages/openral_rskill_ros/launch/deploy_e2e.launch.py`` to
auto-activate ``/openral_slam_toolbox``, not via ``ros2 lifecycle set``:

1. Discovery race: a robocasa-kitchen first boot can spend ~30s importing
   robosuite/robocasa before the node is visible, so a fixed-delay
   ``TimerAction`` either finds "Node not found" or overwaits fast boots.
2. launch_ros's ``lifecycle_event_manager`` logs a false transition failure
   on Jazzy whenever ``response.success=false`` — slam_toolbox 2.8.4's
   ``on_configure`` (``src/slam_toolbox_common.cpp:139``) actually returns
   SUCCESS and the FSM does transition; not patchable from this tree.

Waits ``--service-timeout-s`` for ``<node>/change_state``, then drives
CONFIGURE then ACTIVATE, each bounded by ``--transition-timeout-s`` (must
cover a robocasa-kitchen ``on_configure``, which can exceed a minute: MuJoCo
+ robosuite import, ``env.reset``, a possible ``uv`` rebuild). Exits 0 on
success, non-zero only if the service never appears or the FSM state never
advances.

Also drives ``deploy_e2e.launch.py``'s readiness gate for a ROS-attached
external simulator (``DeployScene.simulator``, e.g. SRB): with one or more
``--wait-for-topic``, this script blocks BEFORE the change_state wait until
every named topic has at least one publisher (or
``--wait-for-topics-timeout-s`` elapses, which is a hard failure — unlike an
absent ``change_state`` service, a scene that declared these topics and
never got them is a real problem, not "not spawned yet"). Distinct from the
transition-drive's own retry logic: the simulator's process, not a lifecycle
FSM, is what's being waited on here.
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import Any

import rclpy
from lifecycle_msgs.msg import Transition
from lifecycle_msgs.srv import ChangeState, GetState

_STATE_TO_TRANSITION = {
    "inactive": [Transition.TRANSITION_CONFIGURE],
    "active": [Transition.TRANSITION_CONFIGURE, Transition.TRANSITION_ACTIVATE],
}


def _service_path(node: str, suffix: str) -> str:
    return f"{node.rstrip('/')}/{suffix}"


def _wait_for_service(
    node: Any,
    service_name: str,
    timeout_s: float,
    srv_type: type,
) -> Any:
    deadline = time.monotonic() + timeout_s
    client = node.create_client(srv_type, service_name)
    while time.monotonic() < deadline:
        if client.wait_for_service(timeout_sec=1.0):
            return client
        rclpy.spin_once(node, timeout_sec=0.0)
    msg = f"service {service_name!r} never appeared within {timeout_s:.1f}s"
    raise TimeoutError(msg)


def _read_state(node: Any, target_node: str, get_state_client: Any) -> str:
    del target_node  # used by callers for log context; not needed here
    req = GetState.Request()
    future = get_state_client.call_async(req)
    rclpy.spin_until_future_complete(node, future, timeout_sec=5.0)
    resp = future.result()
    if resp is None:
        return ""
    label: str = resp.current_state.label
    return label


def _drive_transition(
    node: Any,
    target_node: str,
    change_state_client: Any,
    get_state_client: Any,
    transition_id: int,
    transition_label: str,
    transition_timeout_s: float,
) -> None:
    req = ChangeState.Request()
    req.transition.id = transition_id
    future = change_state_client.call_async(req)
    # ``change_state`` runs on_<transition> synchronously on the single-
    # threaded executor, so the future resolves only when it returns.
    # robocasa-kitchen ``configure`` can block >1 min (MuJoCo + robosuite
    # import, ``env.reset``, a cold ``uv`` build measured at ~27s) — a
    # fixed 30s timeout previously returned ``future.result()=None`` and
    # false-failed a transition that was about to succeed. Wait the
    # caller-supplied budget instead.
    rclpy.spin_until_future_complete(node, future, timeout_sec=transition_timeout_s)
    resp = future.result()
    # Post-call state is the source of truth, not ``resp.success``: Jazzy's
    # first CONFIGURE returns ``success=false`` even though the FSM
    # transitions, and a spin timing out at the deadline yields
    # ``resp=None`` even if ``on_configure`` finished microseconds later.
    # Grace-poll the state so an in-flight settle isn't misread as failure.
    grace_deadline = time.monotonic() + 5.0
    while True:
        post_state = _read_state(node, target_node, get_state_client)
        if post_state in {"inactive", "active"}:
            return
        if resp is not None and resp.success:
            return
        if time.monotonic() >= grace_deadline:
            break
        rclpy.spin_once(node, timeout_sec=0.2)
    msg = (
        f"transition {transition_label!r} on {target_node!r} did not advance the "
        f"FSM within {transition_timeout_s:.1f}s (post-call state={post_state!r}, "
        f"response.success={getattr(resp, 'success', None)!r})"
    )
    raise RuntimeError(msg)


def _wait_for_topic_publishers(node: Any, topics: list[str], timeout_s: float) -> list[str]:
    """Block until every ``topics`` entry has >=1 publisher, or ``timeout_s`` elapses.

    ``node.count_publishers(topic)`` reflects local graph-cache knowledge, which
    updates via discovery independent of any subscription — no subscriber to
    ``topics`` is created here.

    Args:
        node: A live ``rclpy`` node.
        topics: Topic names to wait for (e.g. ``["/clock",
            "/srb/env0/robot/joint_states"]``).
        timeout_s: Total budget across all topics, not per-topic.

    Returns:
        Topics still missing a publisher when the budget ran out (empty = all found).
    """
    deadline = time.monotonic() + timeout_s
    remaining = set(topics)
    while remaining and time.monotonic() < deadline:
        remaining = {t for t in remaining if node.count_publishers(t) < 1}
        if remaining:
            rclpy.spin_once(node, timeout_sec=0.5)
    return sorted(remaining)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--node", required=True, help="Target lifecycle node name (e.g. /openral_slam_toolbox)."
    )
    parser.add_argument(
        "--target",
        choices=("inactive", "active"),
        default="active",
        help="Goal state: drive CONFIGURE → INACTIVE, or +ACTIVATE → ACTIVE.",
    )
    parser.add_argument(
        "--service-timeout-s",
        type=float,
        default=30.0,
        help="Seconds to wait for the change_state service to appear.",
    )
    parser.add_argument(
        "--transition-timeout-s",
        type=float,
        default=300.0,
        help=(
            "Seconds to wait for each CONFIGURE / ACTIVATE transition to "
            "complete. Must cover the node's slowest on_configure — a "
            "robocasa-kitchen HAL first-boot (MuJoCo + robosuite import + "
            "env.reset, plus a possible uv rebuild) can exceed a minute."
        ),
    )
    parser.add_argument(
        "--wait-for-topic",
        action="append",
        default=[],
        dest="wait_for_topics",
        metavar="TOPIC",
        help=(
            "Block until this topic has >=1 publisher, before even trying "
            "change_state. Repeatable. For a scene's ROS-attached external "
            "simulator (DeployScene.simulator.ready_topics) — the HAL "
            "should not be asked to configure before its transport exists."
        ),
    )
    parser.add_argument(
        "--wait-for-topics-timeout-s",
        type=float,
        default=600.0,
        help="Total budget for every --wait-for-topic to gain a publisher.",
    )
    args = parser.parse_args()

    rclpy.init()
    node = rclpy.create_node("openral_lifecycle_autostart")
    try:
        if args.wait_for_topics:
            missing = _wait_for_topic_publishers(
                node, args.wait_for_topics, args.wait_for_topics_timeout_s
            )
            if missing:
                print(
                    "lifecycle-autostart: external simulator never published "
                    f"{missing} within {args.wait_for_topics_timeout_s:.1f}s — "
                    f"not driving {args.node!r} to {args.target!r} "
                    "(the simulator process likely failed to boot; check its "
                    "own log).",
                    file=sys.stderr,
                )
                return 1
        change_state_name = _service_path(args.node, "change_state")
        get_state_name = _service_path(args.node, "get_state")
        try:
            change_state_client = _wait_for_service(
                node, change_state_name, args.service_timeout_s, ChangeState
            )
            get_state_client = _wait_for_service(
                node, get_state_name, args.service_timeout_s, GetState
            )
        except TimeoutError as exc:
            print(f"lifecycle-autostart: {exc}", file=sys.stderr)
            return 0  # don't log an [ERROR] from the process; absent server is informational

        current = _read_state(node, args.node, get_state_client)
        transitions = _STATE_TO_TRANSITION[args.target]
        labels = {
            Transition.TRANSITION_CONFIGURE: "configure",
            Transition.TRANSITION_ACTIVATE: "activate",
        }
        for tid in transitions:
            label = labels[tid]
            if current == "active":
                # Already at goal.
                break
            if current == "inactive" and label == "configure":
                continue  # already configured; only need activate
            _drive_transition(
                node,
                args.node,
                change_state_client,
                get_state_client,
                tid,
                label,
                args.transition_timeout_s,
            )
            current = _read_state(node, args.node, get_state_client)
        print(
            f"lifecycle-autostart: {args.node} reached state={current!r} (target={args.target!r})"
        )
        return 0
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    sys.exit(main())
