# Blue + T 混训 v1

新增 `0907-bluet-joint-vr`、`0907-bluet-joint-feedback`、`0907-bluet-eef-vr` 三个独立实验。每组包含 Blue 全部 52 条、T（原 L）全部 53 条，共 105 条、61,801 个输出帧和 58,756 个有效窗口。Blue、T 分别保留原任务指令；按有效窗口均匀采样，约 59.54% Blue、40.46% T，不额外做任务重采样。

复用已验证的 `arx-open-baseline-v1` 校准及三种标签契约，逐分量重新与原 HDF5 校验。原始文件 SHA256 和视频 SHA256 重新核验，视频硬链接到已全量解码、逐像素验证的无损 GOP2 文件。原始数据和已有六组派生数据不修改；混训重新编号 episode、全局帧和 task_index，保留来源映射。新数据位于 `data/0907_blue_t_v1/BlueT/<mode>`。

每组重新拟合实际有效窗口上的 40D 统计，并核对其充分统计量等于两来源之和。三组各自有配置和统计：`configs/arx_lift2s_0907_bluet_<mode下划线>/`。真实 DataLoader 每 rank 为 459 batches，10k 对应约 21.7865 个实际 vla_epoch。检查覆盖全部 105 条的首末有效窗口、两种任务指令、mask、统计来源和训练／推理往返。

训练参数沿用原六组：原 `tau-0-vla-base` 初始化，全参数，16 张 H200、每卡 batch 8、累积 1、全局 batch 128；lr 2e-5、cosine 10k、warmup 100、weight decay 0.01、bf16、TF32、SDPA、ZeRO-1、seed 42、RTC 关闭。先独立 20-step smoke，不保存权重；通过后从 base 重新启动正式训练，每 500 steps 保存、最多 20 份。阶段交付要求完整 checkpoint-500 和两次后续增长检查；自动监控和自身完整 checkpoint 恢复继续至 10k。

```bash
PYTHONPATH=src:. .venv/bin/python scripts/prepare_bluet_mixed_training.py
PYTHONPATH=src:. .venv/bin/python scripts/run_blue_t_campaign.py --mixed --manage-resources --credentials /absolute/path/qzcli.txt
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/validate_bluet_mixed_inference.py --base
# checkpoint-500 完整后，回放三个模型的 Blue 和 T 请求并落盘：
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/validate_bluet_mixed_inference.py
```

服务端继续使用 `arx-calibrated-v3`，请求显式携带原始 joint／EEF、左右全开基线和任务指令；实机 EEF 发布与 VR→夹爪驱动映射仍待驱动契约。离线检查不代表实机验证。

资源管理通过 `scripts/manage_blue_t_resources.py --mixed` 使用独立台账，仅为本组三个 profile 的实际缺口释放已授权占卡任务，三个正式任务提交后逐个补交相同资源。与原六组资源操作共用互斥锁；不会把原六组训练当成混训资源候选。

[混训实时报告](../outputs/0907_bluet_mixed_preparation/report.md) · [源文件核验](../outputs/0907_bluet_mixed_preparation/source_integrity.json) · [资源台账](../outputs/0907_bluet_mixed_preparation/resources/ledger.json)
