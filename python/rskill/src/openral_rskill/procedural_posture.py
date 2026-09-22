"""PostureRskill -- take the arm to a named configuration the way a planner does (one physical operation).

``kind: procedural``, LunarBot. The same end state as ``procedural_move_joints``, reached differently: the robot's own
planner (MoveIt) finds a path through the scene it has -- the surveyed structures *and* whatever the robot's cameras
have measured standing in the arm's work zone (``openral_ext/planning_scene``) -- and the skill follows that path.

A straight interpolation in joint space does not look where it is going. Raising the arm out of its travel pose while
the rover stood at the depot swept it through the staged ORU and pushed the item off its slot (research repo F57); the
item is in no map, but it is in the depth image, and a planned path goes round it.

Failure returns the stage and why: no path through the scene, or the arm stopped short of the posture.
"""

from __future__ import annotations

import numpy as np

from openral_rskill._eye_in_hand import EyeInHandSkill

__all__ = ["PostureRskill"]


class PostureRskill(EyeInHandSkill):
    def procedure(self) -> None:
        q = np.array([float(v) for v in self.goal["joint_targets"]], float)
        self._evidence["joint_targets"] = [round(float(v), 4) for v in q]
        info = self.plan_to(q, "move", tol_rad=float(self.goal.get("tolerance_rad", 0.08)))
        self._evidence.update(info, reached=[round(float(v), 4) for v in self.arm_q()],
                              tcp=[round(float(v), 4) for v in self.tcp()[0]])
