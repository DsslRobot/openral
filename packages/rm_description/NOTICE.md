# Provenance — vendored, not authored here

This package is a filtered subset of `rm_description` from
[RealManRobot/ros2_rm_robot](https://github.com/RealManRobot/ros2_rm_robot),
branch `humble`, commit `c941b565e4f9174afa36561f143ef5fbbb744750`.

Kept: `urdf/rm_75.urdf`, `urdf/rm_75.urdf.xacro`, and the eight RM-75
meshes (`base_link`, `link1`..`link7`) referenced by that URDF's default
(`Link7`) variant. Everything else in the upstream package (other arm
models' URDF/meshes, `launch/`, `rviz/`, `scripts/`) was dropped as
out of scope for this project (only the RM-75 is used).

**Modified from upstream:** the seven `joint1`..`joint7` position limits in
both URDF files were tightened to the exact RM-75 datasheet values already
declared in `robots/lunar_bot/robot.yaml` (`docs/lunar_bot_rm75_spec.md`) —
upstream ships rounded approximations (e.g. `joint1` ±3.1 rad vs the exact
±3.1067 rad = ±178°). Joint names/links/kinematics are otherwise verbatim.

**License:** upstream declares `<license>TODO: License declaration</license>`
in `package.xml` — i.e. no license is actually declared. Vendored here under
an explicit, recorded research-use exception to this repo's CLAUDE.md §1.9
(uniform Apache-2.0 posture) — see `docs/research_findings.md` F31 in the
research harness repo (`Space_Robot_Harness/docs/research_findings.md`) for
the decision and its rationale/scope. Do not upstream, redistribute, or
publish this package standalone without resolving the license question with
RealMan first.
