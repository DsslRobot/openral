"""The arm's joint-space rules: which turn of the wrist roll an IK answer is taken at, and how fast the posture pull moves.

Real recorded numbers (mc4's stand-off solution, gc3's carry pose); no ROS or physics.
"""
import math
from pathlib import Path

import numpy as np
import yaml
from openral_core.schemas import RobotDescription, RSkillManifest
from openral_rskill._eye_in_hand import JOINT_LIMITS_RAD, POSTURE, nearest_equivalent
from openral_rskill.procedural_place import PlaceRskill

ROOT = Path(__file__).resolve().parents[2]


def a_skill():
    manifest = RSkillManifest.model_validate(yaml.safe_load((ROOT / "rskills/rskill-procedural-place/rskill.yaml").read_text()))
    robot = RobotDescription.from_yaml(str(ROOT / "robots/lunar_bot/robot.yaml"))
    return PlaceRskill(manifest=manifest, robot_description=robot, prompt="", prompt_metadata_json="")


def test_wrist_roll_is_taken_at_the_turn_nearest_the_arm():
    # mc4: the solver answered joint7 = -5.78 for a pose the arm reached at +0.50; the arm stood at joint7 = 1.88
    answer = [0.61, -1.58, -0.42, -1.84, 0.55, 1.58, -5.78]
    stood = [0.30, -1.27, -0.25, -2.33, 0.17, 1.80, 1.88]
    q = nearest_equivalent(answer, stood)
    assert math.isclose(q[6], -5.78 + 2 * math.pi)
    assert q[:6] == answer[:6]


def test_a_turn_beyond_the_stops_margin_is_not_taken():
    stood = [0.0, -1.0, 0.0, -1.4, 0.0, 1.2, 6.2]
    q = nearest_equivalent([0.0, -1.0, 0.0, -1.4, 0.0, 1.2, -0.5], stood)
    assert math.isclose(q[6], -0.5 + 2 * math.pi)  # 5.78: the turn the arm is at, inside the stop
    q = nearest_equivalent([0.0, -1.0, 0.0, -1.4, 0.0, 1.2, -0.02], stood)
    assert math.isclose(q[6], -0.02)  # 6.263 would be past the stop's margin (6.233)


def test_posture_pull_is_a_rate_not_a_step_at_the_joint_rate_limit():
    skill = a_skill()
    q = np.array([-0.11, -0.67, 0.02, -0.56, 0.06, -0.45, 1.61])  # gc3's carry pose, elbow nearly straight
    skill.arm_q = lambda: list(q)
    skill.fk = lambda _q: np.eye(4)
    J = np.zeros((6, 7))
    for row, joint in enumerate([0, 1, 2, 4, 5, 6]):
        J[row, joint] = 1.0  # the elbow (joint 4) moves nothing the task cares about: it is the null space
    skill.jacobian = lambda _q, _T: J
    dt, max_dq = 0.05, 0.3 * 0.05
    step = skill.resolved_rate_step(q, np.zeros(3), np.zeros(3), max_dq, dt) - q
    want = 0.05 * dt * (POSTURE["joint4"][0] - q[3])  # 0.05 per second of its distance from the window's centre
    assert math.isclose(step[3], want, rel_tol=1e-6)
    assert abs(step[3]) < max_dq / 4  # it was the whole joint-rate limit, from a standstill, every tick
    assert np.allclose(np.delete(step, 3), 0.0)
