"""deploy sim forwards the scene YAML to manifest-driven arms in sim mode.

Uses ``scenes/deploy/libero_pnp.yaml`` — a DeployScene (env-only, no task)
that resolves to ``franka_panda`` via ``SCENES.fixed_robot("libero_spatial")``.
``openral deploy sim --config`` is strict on DeployScene, so the
fixture must be DeployScene-shaped.

Parents depth: tests/unit/test_*.py → parents[0]=tests/unit, parents[1]=tests,
parents[2]=repo root (matches every other test in this directory, e.g.
test_cli_deploy_sim.py line 38).
"""

from __future__ import annotations

from pathlib import Path

from openral_cli.deploy_sim import resolve_launch_invocation

_REPO = Path(__file__).resolve().parents[2]
_SCENE = _REPO / "scenes/deploy/libero_pnp.yaml"


def test_sim_mode_injects_sim_env_yaml() -> None:
    """Manifest-driven HAL (franka_panda) gets sim_env_yaml in sim mode."""
    assert _SCENE.is_file(), f"missing fixture: {_SCENE}"
    inv = resolve_launch_invocation(
        config=_SCENE,
        robot_override="franka_panda",
        dashboard_port=4318,
        reset_to_pose_service=None,
        hal_mode="sim",
    )
    assert inv.hal_params["sim_env_yaml"] == str(_SCENE.resolve())


def test_real_mode_does_not_inject_scene() -> None:
    """Manifest-driven HAL (franka_panda) does not get sim_env_yaml in real mode."""
    inv = resolve_launch_invocation(
        config=None,
        robot_override="franka_panda",
        dashboard_port=4318,
        reset_to_pose_service=None,
        hal_mode="real",
    )
    assert "sim_env_yaml" not in inv.hal_params


def test_sim_mode_without_simulator_does_not_forward_deploy_config() -> None:
    """A scene with no `simulator:` (every scene except SRB-backed ones today)
    still does not get `deploy_config:=` in sim mode — this is the pre-existing
    behaviour ``ExternalSimulatorSpec`` must not change for every other robot.
    """
    inv = resolve_launch_invocation(
        config=_SCENE,
        robot_override="franka_panda",
        dashboard_port=4318,
        reset_to_pose_service=None,
        hal_mode="sim",
    )
    assert not any(arg.startswith("deploy_config:=") for arg in inv.argv_template)


_SRB_SCENE = _REPO / "scenes/deploy/srb_panel_remount.yaml"


def test_lunar_bot_srb_scene_is_bare_twin_not_scene_attached() -> None:
    """lunar_bot's `bare_twin_sim=True` registry entry never injects
    `sim_env_yaml` — it is never scene-attached through `openral_sim.SCENES`
    (SRB is ROS-attached, not stepped; see `openral_sim.backends.srb`).
    """
    assert _SRB_SCENE.is_file(), f"missing fixture: {_SRB_SCENE}"
    inv = resolve_launch_invocation(
        config=_SRB_SCENE,
        robot_override=None,
        dashboard_port=4318,
        reset_to_pose_service=None,
        hal_mode="sim",
    )
    assert "sim_env_yaml" not in inv.hal_params
    assert inv.hal.package == "openral_hal_lunar_bot"


def test_lunar_bot_srb_scene_forwards_deploy_config_in_sim_mode() -> None:
    """A `DeployScene.simulator` (SRB) is the one case sim mode DOES need
    `deploy_config:=` — `compose_runtime_graph` reads it to spawn + gate the
    external simulator process (the launch's own concern, not the HAL's).
    """
    inv = resolve_launch_invocation(
        config=_SRB_SCENE,
        robot_override=None,
        dashboard_port=4318,
        reset_to_pose_service=None,
        hal_mode="sim",
    )
    assert f"deploy_config:={_SRB_SCENE.resolve()}" in inv.argv_template


def test_srb_scenes_declare_the_sensor_bridge_for_nav2() -> None:
    """F37: Nav2 wedges unless the SRB topic/TF bridge runs; the scene must own it."""
    from openral_core import DeployScene

    for scene in ("srb_panel_remount.yaml", "srb_panel_remount_gui.yaml", "srb_lunar_base.yaml"):
        sim = DeployScene.from_yaml(_REPO / "scenes/deploy" / scene).simulator
        assert sim is not None, scene
        bridge = next(b for b in sim.bridges if b.executable == "sensor_bridge_node.py")
        assert bridge.pass_robot_yaml
        assert "/odom" in bridge.publishes
