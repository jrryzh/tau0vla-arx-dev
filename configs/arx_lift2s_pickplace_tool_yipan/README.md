# ARX LIFT2s tool-to-tray finetuning

This configuration is isolated from `arx_lift2s_pickplace_ft`: it uses only
the 49 tool-to-tray episodes under `data/0904_pickplace_tool_yipan`, its own
40D unified normalization statistics, and the instruction:

`Pick up the tool and place it into the tray.`

The converted dataset is 30 FPS LeRobot v3 with three camera streams, native
14D state/action, and `action(t) = qpos(t+1)` at the output frame rate. Source
episodes 10 and 28 are intentionally absent; `meta/arx.json` records the
requested 1--51 range, selected source episode numbers, and missing numbers.

Conversion command:

```bash
python tools/convert_official_hdf5_to_lerobot.py \
  --input data/0904_pickplace_tool_yipan --start 1 --end 51 \
  --allow-missing-episodes --source-fps 60 --fps 30 \
  --task "Pick up the tool and place it into the tray." \
  --action-mode state_t_plus_1 \
  --output data/0904_pickplace_tool_yipan/lerobot_v3_30fps_state_t_plus_1 \
  --repo-id arx_lift2s/0904_pickplace_tool_yipan_30fps_state_t_plus_1
```

Validate every source image and every converted video frame, plus exact
state/action alignment, with:

```bash
python scripts/validate_arx_lerobot_conversion.py \
  data/0904_pickplace_tool_yipan/lerobot_v3_30fps_state_t_plus_1 \
  --source data/0904_pickplace_tool_yipan \
  --expected-episodes 49 --expected-frames 16717 \
  --expected-task "Pick up the tool and place it into the tray." \
  --expected-missing 10 28
```

Fit the training-faithful unified 40D statistics over the empty frame filters
and merge the partial into this configuration's `norm_stats.json`:

```bash
PYTHONPATH=src:. python scripts/norm_stats/compute_unified_ft_stats.py \
  --body arx_lift2s_unified \
  --repos data/0904_pickplace_tool_yipan/lerobot_v3_30fps_state_t_plus_1 \
  --action-horizon 30 --positive-labels --negative-labels \
  --partials-dir /tmp/arx_tool_yipan_norm_partials
PYTHONPATH=src:. python scripts/norm_stats/merge_stats.py \
  --partials /tmp/arx_tool_yipan_norm_partials \
  --out configs/arx_lift2s_pickplace_tool_yipan/norm_stats.json
```

`train.yaml` is the single-GPU one-step regression route. `train_h200.yaml` is
the production full-parameter route: 10,000 steps, ZeRO-1, bf16/SDPA, cosine
learning rate 2e-5 with 100 warmup steps, and checkpointing every 500 steps.

Launch the guarded H200 workflow with:

```bash
scripts/qzcli_arx_h200.sh auto --profile tool-yipan --credentials /secure/qzcli.txt
```

The tool-yipan profile first tests one 8-H200 node at per-GPU batch 16. Only an
explicit CUDA OOM permits fallback to two 8-H200 nodes at per-GPU batch 8; the
effective global batch remains 128.
