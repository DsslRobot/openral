# Provenance — vendored, not authored here

This package is a filtered subset of `rm_moveit2_config/rm_75_config` from
[RealManRobot/ros2_rm_robot](https://github.com/RealManRobot/ros2_rm_robot),
branch `humble`, commit `c941b565e4f9174afa36561f143ef5fbbb744750`.

Kept: `config/` (SRDF, kinematics, joint limits, controllers, the
config-level URDF wrapper + ros2_control xacro, `.setup_assistant`) and only
the `demo.launch.py` + its transitive pieces (`move_group.launch.py`,
`moveit_rviz.launch.py`, `rsp.launch.py`, `spawn_controllers.launch.py`,
`static_virtual_joint_tfs.launch.py`, `warehouse_db.launch.py`). Dropped:
the `_6f`/`_6fb` end-effector variant configs (`lunar_bot` uses the default
`Link7` tip), Gazebo/real-hardware launch variants (out of scope — this
project drives the RM-75 through SRB/OpenRAL, not this package's own
controllers), and `rviz` config.

**License:** upstream declares bare `<license>BSD</license>` with no
accompanying license text file — ambiguous, not a clean grant. Vendored
under the same recorded research-use exception as the sibling
`rm_description` package — see `docs/research_findings.md` F31 in the
research harness repo. Do not upstream, redistribute, or publish this
package standalone without resolving the license question first.

**Known gap, not fixed here:** `rm_group`'s chain root is `base_link`
(the arm's own base, per upstream's SRDF/URDF), not `lunar_bot`'s actual
`chassis_base_link` (the mount point differs — see
`robots/lunar_bot/robot.yaml`'s comment on inferred parent/child links).
This config plans the arm in isolation (matching the pattern the existing
Franka `moveit_resources_panda_moveit_config` demo already uses — a
standalone `move_group` against its own URDF root, not the full vehicle).
It is sufficient for `rskill-moveit-*` dispatch (the resulting
`joint1`..`joint7` trajectory targets are robot-agnostic once resolved by
name), but self-collision/world-collision checking during planning does
not account for the mobile base or wheels. Full-body collision-aware
planning is future work, not required for item 6's exit condition
(a `moveit-eef-pose` dispatch reaching a pose above the panel through the
kernel).
