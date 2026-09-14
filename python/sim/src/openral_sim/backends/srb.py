r"""SRB scene adapter — a ROS-attached external simulator, not a stepped one.

Space Robotics Bench (SRB, a Space Robot Harness research-repo submodule) is
Isaac Lab under a different CLI (``srb agent ros``). Structurally it is
**not** like this package's other Isaac backend (``isaac_sim.py``, a ZMQ
sidecar this process pings/steps): SRB owns no wire protocol OpenRAL speaks.
It is reached only over plain ROS 2 topics
(``srb.interfaces.interface.ros.RosInterface``) — the same way a real
robot's vendor driver is reached. A robot deployed against SRB (``lunar_bot``
today) drives it through its own topic-bridge HAL
(``openral_hal.lunar_bot_srb:LunarBotSRBHAL``), which ``robot.yaml`` declares
as **both** ``hal.sim`` and ``hal.real`` — the adapter class is identical;
only which process is on the other end differs.

So this module registers the ``srb`` scene id purely for two things
``openral_sim.SCENES`` already does generically:

1. **Preflight provisioning** (``provision_srb``) — ``openral deploy sim``
   calls a scene's ``provision=`` hook in front of ``ros2 launch``
   (``openral_cli.deploy_sim._preflight_scene_assets``), independent of
   whether anything ever calls the factory below. This is where "is `srb`
   actually installed on this host" gets checked, with an actionable error,
   instead of surfacing as an opaque `ExecuteProcess` failure deep in the
   launch graph.
2. **Registry membership** — so ``scene: {id: "srb", ...}`` in a
   ``DeployScene`` YAML resolves to *something* registered, for any tooling
   that walks ``SCENES`` (dry-run rendering, docs generation).

The factory's returned rollout's ``reset``/``step`` deliberately raise: this
scene is never scene-attached (``SimAttachedHAL``) for ``lunar_bot`` — its
``_ROBOT_HAL_REGISTRY`` entry is ``bare_twin_sim=True`` precisely so
``deploy_sim.py`` never injects ``sim_env_yaml``/scene-attaches it, and the
manifest-driven HAL node builds ``hal.sim`` (``LunarBotSRBHAL``) directly
instead. The actual SRB *process* is spawned and gated by the launch itself
from ``DeployScene.simulator`` (an ``ExternalSimulatorSpec``,
``deploy_e2e.launch.py``'s ``_maybe_add_external_simulator``), never through
this rollout. A future scene that genuinely wants to step SRB in-process
through this Protocol would need a real bridge here; none exists today, and
faking one (returning zeroed observations) would be a silent lie about what
happened — CLAUDE.md §1.11 forbids exactly that, hence the typed raise
instead of a stub.

See the research repo's ``docs/srb_deploy_backend_plan.md`` for the full
design and ``docs/srb_integration_notes.md`` / the ``srb-native-install``
memory for the (host-specific, not-pip-installable) install this
provisioner checks for.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from openral_core.exceptions import ROSConfigError

from openral_sim.registry import SCENES

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray
    from openral_core import SceneSpec, SimEnvironment, TaskSpec

    from openral_sim.rollout import Observation, StepResult

_SRB_SCENE_ID = "srb"

_NOT_STEPPED_MESSAGE = (
    "srb is a ROS-attached external simulator, not stepped in-process. "
    "`openral deploy sim` spawns and gates it via DeployScene.simulator and "
    "drives the robot through its own topic-bridge HAL (robot.yaml "
    "hal.sim), never through SimRollout.reset/step -- reaching this means "
    "something tried to scene-attach the `srb` scene id, which no "
    "registered robot does today (see srb.py's module docstring)."
)

#: Conventional Isaac Sim install root on this project's hosts
#: (`srb-native-install` memory) — the fallback when `ISAAC_SIM_PATH` is
#: unset in *this* process's env. The spawned `srb` process's own
#: `ISAAC_SIM_PATH` always comes from `ExternalSimulatorSpec.env_set`
#: (the scene YAML), not from here; this constant only backstops the
#: preflight "does an install exist somewhere" check.
_DEFAULT_ISAAC_SIM_PATH = Path.home() / "Programs" / "isaac-sim"


def _isaac_sim_path_candidate() -> Path | None:
    """An Isaac Sim install directory this host actually has, or ``None``."""
    override = os.environ.get("ISAAC_SIM_PATH")
    if override:
        p = Path(override).expanduser()
        return p if p.is_dir() else None
    return _DEFAULT_ISAAC_SIM_PATH if _DEFAULT_ISAAC_SIM_PATH.is_dir() else None


def provision_srb() -> None:
    """Verify the ``srb`` CLI + its Isaac Sim install are reachable.

    Cheap (two filesystem/PATH checks, no process spawned) — unlike
    ``isaac_sim.provision_isaac_sim``, there is no auto-provision path:
    Isaac Sim + SRB is a native, host-specific install this project does
    not know how to reproduce unattended (``srb-native-install`` memory —
    four undocumented-upstream fixes were needed to get it working).

    Raises:
        ROSConfigError: ``srb`` is not on ``PATH``, or no Isaac Sim install
            is reachable (neither ``ISAAC_SIM_PATH`` nor the conventional
            ``~/Programs/isaac-sim``).
    """
    if shutil.which("srb") is None:
        raise ROSConfigError(
            "srb (Space Robotics Bench) is not on PATH. It is a native, "
            "host-specific install on top of NVIDIA Isaac Sim, not a "
            "pip-installable dependency this project auto-provisions — see "
            "the research repo's docs/srb_integration_notes.md for the "
            "working install recipe (and the four fixes the documented "
            "upstream install misses). Install it, then re-run."
        )
    if _isaac_sim_path_candidate() is None:
        raise ROSConfigError(
            "srb is on PATH but no Isaac Sim install was found. This "
            "preflight check looks for ISAAC_SIM_PATH or the conventional "
            f"{_DEFAULT_ISAAC_SIM_PATH} — set one, or install Isaac Sim "
            "there. (The srb PROCESS this scene spawns gets its own "
            "ISAAC_SIM_PATH from DeployScene.simulator.env_set regardless; "
            "this check only confirms an install exists somewhere on the "
            "host before committing to `ros2 launch`.)"
        )


@dataclass
class _SRBExternalRollout:
    """``SimRollout``-shaped placeholder for the ``srb`` scene id.

    Exists only so ``SCENES.register("srb", ...)`` has a factory to decorate
    (registry membership + the ``provision`` hook are what this scene id is
    actually for — see the module docstring). ``reset``/``step`` always
    raise; nothing in the ``lunar_bot`` deploy path calls them.
    """

    scene: SceneSpec
    task: TaskSpec

    def reset(self, seed: int | None = None) -> Observation:
        """Never called in the current deploy path.

        Raises:
            ROSConfigError: Always.
        """
        raise ROSConfigError(_NOT_STEPPED_MESSAGE)

    def step(self, action: NDArray[np.float32]) -> StepResult:
        """Never called in the current deploy path.

        Raises:
            ROSConfigError: Always.
        """
        raise ROSConfigError(_NOT_STEPPED_MESSAGE)

    def render(self) -> NDArray[np.uint8] | None:
        """No frame — this rollout renders nothing; the launch owns SRB's own cameras."""
        return None

    def close(self) -> None:
        """No-op — this rollout owns no process; the launch spawns/terminates SRB."""


@SCENES.register(_SRB_SCENE_ID, fixed_robot=None, provision=provision_srb)
def _build_srb_scene(env_cfg: SimEnvironment) -> _SRBExternalRollout:
    """Build the ``srb`` scene id's registry entry.

    Re-runs ``provision_srb`` (idempotent, cheap) so the build path can
    never drift from the preflight check per CLAUDE.md's "same callable"
    rule for ``provision=`` hooks, then returns a rollout whose ``reset``/
    ``step`` raise — see the module docstring for why a stub would be
    dishonest here rather than merely lazy.
    """
    provision_srb()
    return _SRBExternalRollout(scene=env_cfg.scene, task=env_cfg.task)
