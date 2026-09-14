# openral_hal_lunar_bot

HAL lifecycle node for `lunar_bot` (4WIS/4WID rover + 7-DoF RealMan RM-75 arm +
EG2-4C2 gripper), bridging OpenRAL's safety-checked chunk path to Space
Robotics Bench (SRB) over ROS 2.

First integration slice: **BODY_TWIST only** (mobile base). Publishes
`geometry_msgs/Twist` on SRB's `.../action/cmd_vel`, reads
`sensor_msgs/JointState` from SRB's `.../robot/joint_states` and republishes
it on the standard `/joint_states` topic. CARTESIAN_TWIST (arm) and
GRIPPER_BINARY (gripper) are follow-up work — see
`robots/lunar_bot/robot.yaml` and `python/hal/src/openral_hal/lunar_bot_srb.py`.

```bash
ros2 run openral_hal_lunar_bot lifecycle_node \
    --ros-args -p robot_yaml:=robots/lunar_bot/robot.yaml -p hal_mode:=real
```

Requires an SRB instance already running with `--interface ros` (see the
research repo's `docs/srb_integration_notes.md` for the exact launch
invocation and the environment fixes this host needs).
