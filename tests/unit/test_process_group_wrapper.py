"""``tools/process_group_wrapper.py`` — kills a WHOLE process tree on
SIGTERM, even when the wrapped command forks a child without ``exec``
(exactly what NVIDIA's Isaac Sim ``python.sh`` launcher does — see the
wrapper's own module docstring).

Reproduces the failure mode directly (a tiny bash script that forks a
``sleep`` child as a background job and ``wait``s on it, never ``exec``s)
rather than depending on a real Isaac Sim install — a real process/OS
boundary, so spawning real subprocesses here is the CLAUDE.md §1.11
sanctioned kind of test double, not a mock. A control test confirms
sending SIGTERM directly to the SAME script (no wrapper) — exactly what
``ros2 launch``'s ``ExecuteProcess`` does today — leaves the grandchild
running, proving this is a real bug the wrapper fixes, not a test that
would pass either way.
"""

from __future__ import annotations

import signal
import subprocess
import sys
import time
from pathlib import Path

import psutil
import pytest

_TOOL_PATH = Path(__file__).resolve().parents[2] / "tools" / "process_group_wrapper.py"

# A minimal stand-in for Isaac Sim's `python.sh`: forks a child WITHOUT
# `exec` (`$python_exe "$@" || error_exit`, no `exec` — read verbatim from
# the installed script), so the parent (this script) and the child are two
# distinct PIDs; the parent supervises via `wait`, same shape as python.sh.
_FORK_WITHOUT_EXEC_SCRIPT = "sleep 60 &\nwait $!\n"

_SPAWN_SETTLE_S = 2.0
_TERM_WAIT_S = 15.0


@pytest.fixture
def fork_without_exec_script(tmp_path: Path) -> Path:
    script = tmp_path / "fork_without_exec.sh"
    script.write_text(_FORK_WITHOUT_EXEC_SCRIPT)
    script.chmod(0o755)
    return script


def _wait_for_one_child(pid: int, timeout_s: float = _SPAWN_SETTLE_S) -> psutil.Process:
    """The forked ``sleep`` grandchild, once bash has actually forked it."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        children = psutil.Process(pid).children(recursive=True)
        if children:
            return children[0]
        time.sleep(0.05)
    raise TimeoutError(f"pid {pid} never forked a child within {timeout_s}s")


def _alive(proc: psutil.Process) -> bool:
    return proc.is_running() and proc.status() != psutil.STATUS_ZOMBIE


def test_wrapper_kills_the_whole_tree_on_sigterm(fork_without_exec_script: Path) -> None:
    """The wrapper's group-kill reaches a grandchild the direct child never `exec`'d into."""
    proc = subprocess.Popen(
        [sys.executable, str(_TOOL_PATH), "--", "bash", str(fork_without_exec_script)]
    )
    grandchild = None
    try:
        grandchild = _wait_for_one_child(proc.pid)
        assert _alive(grandchild), "test setup: grandchild should be alive before SIGTERM"

        proc.send_signal(signal.SIGTERM)
        proc.wait(timeout=_TERM_WAIT_S)

        assert not _alive(grandchild), (
            "process_group_wrapper left the forked-without-exec grandchild running "
            "after SIGTERM — the whole point of this wrapper"
        )
    finally:
        if proc.poll() is None:
            proc.kill()
        if grandchild is not None and _alive(grandchild):
            grandchild.kill()  # don't leak a live `sleep` if the assertion above failed


def test_without_the_wrapper_the_grandchild_survives_sigterm(
    fork_without_exec_script: Path,
) -> None:
    """Control: proves this is a real bug, not a test that would pass either way.

    SIGTERM straight to the bash script (no wrapper) is exactly what
    `ros2 launch`'s `ExecuteProcess` does today — confirms the forked
    grandchild is left running, orphaned. Always reaped at the end.
    """
    proc = subprocess.Popen(["bash", str(fork_without_exec_script)])
    grandchild = _wait_for_one_child(proc.pid)
    try:
        assert _alive(grandchild)
        proc.send_signal(signal.SIGTERM)
        proc.wait(timeout=5.0)
        # This IS the bug: the direct child (bash) is gone, but a plain
        # single-PID SIGTERM never touched its already-forked grandchild.
        assert _alive(grandchild), (
            "control assumption broke: the grandchild died without the wrapper too — "
            "if this starts failing, the wrapper test above may no longer be proving "
            "anything real"
        )
    finally:
        if _alive(grandchild):
            grandchild.kill()
