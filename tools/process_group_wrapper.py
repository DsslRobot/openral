#!/usr/bin/env python3
r"""Run a command as a new process-group leader; kill the WHOLE group on
SIGTERM/SIGINT, not just the direct child.

Why this exists — confirmed live 2026-09-14, not theoretical: NVIDIA Isaac
Sim's ``python.sh`` launcher (SRB's own interpreter) runs the real Kit
process as a plain foreground command, not an ``exec``:

.. code-block:: bash

    $python_exe "${filtered_args[@]}" $args || error_exit

No ``exec`` means bash **forks** a child for ``$python_exe`` and stays
alive supervising it — so the heavy Kit/PhysX/RTX process is a distinct
PID, not the same process as the one whatever spawned ``python.sh`` is
tracking. ``ros2 launch``'s own ``ExecuteProcess`` (``launch.actions.
execute_local.ExecuteLocal``) has no process-group handling anywhere in
its source (confirmed by reading it — no ``setsid``/``killpg``/
``preexec_fn``) — it only ever signals the one PID it directly spawned.
Killing that PID (the ``python.sh`` bash wrapper) kills bash; the
already-forked Kit process is unaffected and is orphaned, reparented to
init, and keeps running — observed surviving over an hour, still burning
~100% CPU and several GB of GPU memory with zero ROS topics publishing,
because nothing was left to signal it.

This wrapper is the standard fix for exactly this class of problem:
become a new session/process-group leader (``os.setsid``, so this
process's PID becomes its own PGID), spawn the real command as a child
(every further fork it makes — ``python.sh``'s bash child, that child's
Kit process, Kit's own internal helper processes like "Omniverse Hub" —
inherits the SAME group unless it explicitly calls ``setsid`` itself,
which none of these do), and on SIGTERM/SIGINT send the signal to the
WHOLE group (``os.killpg``) rather than just the one child. Escalates to
SIGKILL after a grace period if the group does not exit on SIGTERM.

Usage::

    process_group_wrapper.py -- <command> [args...]

Used by ``deploy_e2e.launch.py`` to spawn ``DeployScene.simulator``
(``ExternalSimulatorSpec``) instead of running its ``argv`` directly.
"""

from __future__ import annotations

import contextlib
import os
import signal
import subprocess
import sys

#: Seconds to wait for the group to exit after SIGTERM before escalating to
#: SIGKILL. Mirrors launch's own default ``sigterm_timeout``-class budgets
#: (short — this is a last-resort group-kill, not the primary shutdown path,
#: which is `ros2 launch` politely signalling the wrapper itself first).
_SIGKILL_GRACE_S = 10.0


def main() -> int:
    """Spawn ``argv`` (after a literal ``--``) as a new process-group leader."""
    argv = sys.argv[1:]
    if argv and argv[0] == "--":
        argv = argv[1:]
    if not argv:
        print(
            "process_group_wrapper: usage: process_group_wrapper.py -- <command> [args...]",
            file=sys.stderr,
        )
        return 2

    os.setsid()
    child = subprocess.Popen(argv)  # noqa: S603  # reason: argv is this project's own scene config, not untrusted input
    own_pgid = os.getpgrp()

    def _forward(sig: int, _frame: object) -> None:
        # Ignore further delivery of this same signal to THIS process — we
        # are about to send it to the whole group, ourselves included, and
        # must not have our own escalation logic interrupted by that.
        signal.signal(sig, signal.SIG_IGN)
        with contextlib.suppress(ProcessLookupError):
            os.killpg(own_pgid, signal.SIGTERM)
        try:
            child.wait(timeout=_SIGKILL_GRACE_S)
        except subprocess.TimeoutExpired:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(own_pgid, signal.SIGKILL)
        sys.exit(child.returncode if child.returncode is not None else 143)

    signal.signal(signal.SIGTERM, _forward)
    signal.signal(signal.SIGINT, _forward)

    return child.wait()


if __name__ == "__main__":
    sys.exit(main())
