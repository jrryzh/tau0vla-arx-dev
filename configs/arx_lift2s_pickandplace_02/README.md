# ARX LIFT2s pick-and-place 02

Independent 6,000-step full-parameter finetuning from `tau-0-vla-base`.
Instruction: `Pick up the tool and place it into the tray.`

Source episodes 26–53 (28 episodes) produce 9,517 frames at 30 FPS.
Three 640×480 RGB cameras and native 14D state/action are retained;
`action(t) = state(t+1)` at the output FPS. `meta/arx.json` records source
numbers; output episodes are numbered from zero. The horizon of 30 removes
29 anchors per episode, leaving 8,705 training samples. With 16 H200s,
batch 8 per GPU, accumulation 1 and incomplete batches dropped, 6k steps
are approximately 88.24 data epochs; verify actual `vla_epoch` in logs.

This configuration has its own dataset registration, environment override
(`ARX_PICKANDPLACE_02_LEROBOT_ROOT`), unified 40D statistics and outputs.
The empty positive/negative frame filters and image augmentation match the
existing ARX routes. RTC is explicitly disabled. LR 2e-5, cosine over 6k,
warmup 100, weight decay 0.01, bf16, TF32, SDPA, ZeRO-1 and seed 42 are used.
Checkpoints are saved every 500 steps, with retention 20.

## Prepare and validate

Run with the project environment (`source .venv/bin/activate`):

```bash
python tools/convert_official_hdf5_to_lerobot.py \
  --input data/0904_pickandplace_02 --start 26 --end 53 \
  --source-fps 60 --fps 30 --action-mode state_t_plus_1 \
  --task "Pick up the tool and place it into the tray." \
  --output data/0904_pickandplace_02/lerobot_v3_30fps_state_t_plus_1 \
  --repo-id arx_lift2s/0904_pickandplace_02_30fps_state_t_plus_1
python scripts/validate_arx_lerobot_conversion.py data/0904_pickandplace_02/lerobot_v3_30fps_state_t_plus_1 \
  --source data/0904_pickandplace_02 --expected-episodes 28 --expected-frames 9517 \
  --expected-task "Pick up the tool and place it into the tray."
PYTHONPATH=src:. python scripts/norm_stats/compute_unified_ft_stats.py \
  --body arx_lift2s_unified --repos data/0904_pickandplace_02/lerobot_v3_30fps_state_t_plus_1 \
  --action-horizon 30 --positive-labels --negative-labels \
  --partials-dir outputs/0904_pickandplace_preparation/stats_02
PYTHONPATH=src:. python scripts/norm_stats/merge_stats.py \
  --partials outputs/0904_pickandplace_preparation/stats_02 --out configs/arx_lift2s_pickandplace_02/norm_stats.json
```

## Launch and monitor

```bash
scripts/qzcli_arx_h200.sh auto --profile pickandplace-02 --credentials /secure/qzcli.txt
```

The profile is pinned to “罗剑岚老师项目专用” / “vla分区”. It refreshes the
resource query and queues for two eight-H200 nodes, runs a separate
20-step smoke with no weight saving,
then starts the 6k run from base. Both profiles can run concurrently when
four nodes are free. The controller never stops existing jobs; any resource
release is handled separately under explicit user authorization.

The controller checks smoke loss, gradients, memory and all parameter
groups, validates checkpoint-500 plus continued optimization, and tracks
the formal job through scheduler success and checkpoint-6000 validation.
A per-profile lock prevents duplicate controllers. After an interruption,
rerun `formal` with the same profile: a running job is monitored; a failed
job resumes only from this run's latest complete, compatible checkpoint.
An incomplete newest save is skipped. A directory holding only incomplete
checkpoints fails explicitly. Smoke checkpoints are never resumed.

Model: `outputs/arx_lift2s_pickandplace_02_h200_formal/arx_lift2s_pickandplace_02_h200_formal/checkpoint-6000`.
Controller evidence: `outputs/qzcli_arx_h200_pickandplace_02/`.
The latter includes `formal_job_id`, checkpoint validation logs and
`dashboard/{metrics.csv,training_curves.png,summary.json,index.html}`.
`summary.json` records the latest loss and actual `vla_epoch`.

Final validation checks weights, all 16 optimizer/data-state shards,
scheduler, processor, statistics and deployment metadata. The checkpoint's
`finch_data_spec` is a relative link to the run-level data contract; follow
symlinks when copying a checkpoint for deployment (`cp -rL` or `rsync -L`).
Robot success rate requires subsequent deployment evaluation.
