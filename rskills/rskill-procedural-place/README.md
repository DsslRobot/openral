# Place

`OpenRAL/rskill-procedural-lunar_bot-place` lowers a held item onto a named support, opens the jaws, retreats and records settling evidence. It executes through the common OpenRAL safety path.

The `settle` evidence includes the before/after images, depth image, gripper mask, jaw angle and `fx` from that frame’s camera calibration. An independent verifier uses `fx` to convert physical settling tolerance into pixels. The skill’s success flag alone does not establish placement.

The manifest declares joint-position control as used by its resolved-rate servo, alongside Cartesian twist and the gripper.

The calibration field was missing from the original producer although the independent consumer required it. Syntax and real manifest schema checked; payload placement validation remains pending in the Mission B development run.

The shared eye-in-hand runner preserves the gripper command until an operation explicitly requests opening or closing. Starting place must not release the item. The regression test uses a recorded held-joint state and real manifests to check the emitted commands (2 tests passed); physical placement remains to be validated.
