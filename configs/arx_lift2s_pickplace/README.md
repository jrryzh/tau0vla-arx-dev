# ARX LIFT2s pick-place fallback route

This route accepts only the 30 FPS, 14D dataset contract with
`action_semantics=state_t_plus_1` and `action_offset_frames=1`. It is an
explicit fallback until the SDK exposes real joint-position commands. Do not
mix it with `joint_position_command` data or deploy its smoke checkpoints to
hardware.

The native order is:

`left_j0..j5, left_gripper, right_j0..j5, right_gripper`

The DataLoader explicitly uses PyAV. The model sees unified 40D state/action;
arm targets are relative joint deltas and grippers remain absolute.
