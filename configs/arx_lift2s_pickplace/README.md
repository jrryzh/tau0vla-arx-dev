# ARX LIFT2s pick-place state-as-action route

This route accepts only the 30 FPS, 14D dataset contract with
`action_semantics=state_t_plus_1` and `action_offset_frames=1`. Under this
contract, the next observed joint state is intentionally used as the action
target. Keep that semantic declaration with the checkpoint so it is not
confused with `joint_position_command` data.

The native order is:

`left_j0..j5, left_gripper, right_j0..j5, right_gripper`

The DataLoader explicitly uses PyAV. The model sees unified 40D state/action;
arm targets are relative joint deltas and grippers remain absolute.

The task text is required when converting HDF5 and is stored in LeRobot's
`task` field. Training reads that field directly; the data config does not
replace it with a fixed prompt.

## Data flow and configuration

1. `tools/convert_official_hdf5_to_lerobot.py` converts the official 60 FPS
   HDF5 episodes to the 30 FPS LeRobot v3 dataset. `--task` is mandatory and
   `--action-mode state_t_plus_1` writes each output action from the following
   output state.
2. `data.py` registers `arx_lift2s_pickplace_ft`, points to the converted repo,
   declares the three cameras and canonical prompt, maps native 14D values to
   unified 40D, and loads `norm_stats.json`.
3. `train.yaml` selects that route through
   `data_args.config_name: arx_lift2s_pickplace_ft`, selects the base model and
   defines optimization/runtime settings. `scripts/deepspeed/zero1.json`
   supplies the referenced DeepSpeed configuration.
4. `src/tau0_vla/adapters/arx_lift2s/layout.py` owns the native joint order,
   dataset-contract checks, and ARX-to-unified modality layout.
5. `src/tau0_vla/data/robots/unified.py` performs the 14D-to-40D scatter and
   converts arm targets to state-relative values; gripper targets stay
   absolute. `src/tau0_vla/adapters/arx_lift2s/deploy_io.py` restores inferred
   unified actions to the native 14D order.

The converted repo defaults to
`data/handel_pickplace/lerobot_v3_30fps_state_t_plus_1`. Override it without
editing the config by setting `ARX_LEROBOT_ROOT`.

The dataset used by this route was produced with:

```bash
python tools/convert_official_hdf5_to_lerobot.py \
  --input data/handel_pickplace --start 0 --end 50 \
  --source-fps 60 --fps 30 \
  --task "Pick up the handle and place it into the tray." \
  --action-mode state_t_plus_1 \
  --output data/handel_pickplace/lerobot_v3_30fps_state_t_plus_1 \
  --repo-id arx_lift2s/handel_pickplace_30fps_state_t_plus_1
```

After recomputing `norm_stats.json`, launch training with
`bash scripts/train.sh configs/arx_lift2s_pickplace/train.yaml`. A relocated
virtual environment can select its interpreter explicitly with
`PYTHON_BIN=/path/to/python`.

`train.yaml` remains the local single-GPU regression configuration.
`train_h200.yaml` is the full-parameter production configuration: 16 H200s,
per-rank batch 8, global batch 128, 10,000 steps, and 20 retained checkpoints
at 500-step intervals. Submit its guarded smoke/formal workflow with
`scripts/qzcli_arx_h200.sh`; the smoke run uses a distinct output root,
20 steps, no checkpoint saving, and `AUTO_RESUME=0`.
