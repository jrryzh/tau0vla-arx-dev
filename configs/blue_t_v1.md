# Blue / T 夹爪校准训练 v1

本次六组实验使用 Blue 全部 52 条与原 L 全部 53 条记录。原始 HDF5 只读；派生数据和模型统一使用 T 名称。六组均从原 `tau-0-vla-base` 初始化，目标 10,000 steps，每 500 steps 保存，保留 20 份。阶段验收要求六组各有完整 checkpoint-500，随后至少两次检查 step 增长且无新增训练异常。此阶段验收不代表完成 10k 或完成实机验证。

| Profile | 原始目录 | 派生数据 | 机械臂标签 | 夹爪标签 |
|---|---|---|---|---|
| 0907-blue-joint-vr | data/Blue | data/0907_blue_t_v1/Blue/joint-vr | qpos[2k+2] | 1−poscmd[2k]/5 |
| 0907-blue-joint-feedback | data/Blue | data/0907_blue_t_v1/Blue/joint-feedback | qpos[2k+2] | qpos[2k+2]−o |
| 0907-blue-eef-vr | data/Blue | data/0907_blue_t_v1/Blue/eef-vr | poscmd[2k] 的 12D pose | 1−poscmd[2k]/5 |
| 0907-t-joint-vr | data/L | data/0907_blue_t_v1/T/joint-vr | qpos[2k+2] | 1−poscmd[2k]/5 |
| 0907-t-joint-feedback | data/L | data/0907_blue_t_v1/T/joint-feedback | qpos[2k+2] | qpos[2k+2]−o |
| 0907-t-eef-vr | data/L | data/0907_blue_t_v1/T/eef-vr | poscmd[2k] 的 12D pose | 1−poscmd[2k]/5 |

配置目录为 `configs/arx_lift2s_<profile中连字符替换为下划线>/`，每组单独保存 `data.py`、`train_h200.yaml` 和 `norm_stats.json`。状态均取源帧 2k；共同去掉最后一个降采样帧。三种实验共享相同图像和完整 horizon 30 的有效起点，60→30 FPS。

每侧基线 `o` 独立估计：筛选连续 VR≥4.99 的至少 30 帧区间，仅使用该区间最后 10 帧反馈；其 P90−P10≤0.003，前 5 帧与后 5 帧中位数差≤0.0015。基线取合格平台中位数的 P10。缺少平台或同条合格平台变化超过 0.02 时，准备过程失败，不产生 ready 标记。`q−o` 保留行程单位与负噪声；不拟合闭合端点，不按 min/max 拉伸。

joint 使用原生 14D，40D 活跃槽位为 18、19、24:30、32:38。EEF 原生存储为 `[校准 joint/gripper 14D, 双臂 pose 12D]`；后 12D 由原 14D EEF 去掉索引 6、13 得到，模型只启用 0:20，关节槽位 state/action mask 全部关闭。前 14D 的关节数值在 EEF 模型中不参与输入或监督。EEF state 来自 `observations/eef`，命令来自 `action_poscmd`；`action_eef` 不作为命令源。

姿态单位为 xyz 米、RPY 弧度，XYZ 外旋 `R=Rz(yaw)@Ry(pitch)@Rx(roll)`。内部使用 xyz+rot6d，动作相对当前 EEF state；服务端恢复为绝对 xyz+四元数 xyzw。数据契约和 checkpoint 均保存分量时间偏移、夹爪语义、校准版本及姿态约定。无源时间戳，不推测控制延迟。

准备、检查与启动：

```bash
.venv/bin/python scripts/prepare_blue_t_training.py --dataset Blue --workers 4
.venv/bin/python scripts/prepare_blue_t_training.py --dataset T --workers 4
PYTHONPATH=src:. .venv/bin/python scripts/finish_blue_t_preparation.py Blue
PYTHONPATH=src:. .venv/bin/python scripts/finish_blue_t_preparation.py T
PYTHONPATH=src:. .venv/bin/python scripts/run_blue_t_campaign.py --credentials /absolute/path/qzcli.txt --manage-resources
```

准备过程全量解码源图像，无损生成 RGB H264 视频，并将每个输出像素的 SHA256 与选中源帧对比。GOP2 阶段再次逐帧验证，保证随机读取效率。三种实验的视频通过硬链接共享；向量、统计和契约独立。所有源 SHA256、episode 映射、平台区间及验证记录位于 `outputs/0907_blue_t_preparation` 和各数据集 `meta/arx.json`。

每组先运行 20-step smoke，不保存权重；通过后重新从 base 启动正式训练。两节点共 16 H200、每卡 batch 8、累积 1、全局 batch 128；全参数、lr 2e-5、cosine、warmup 100、weight decay 0.01、bf16、TF32、SDPA、ZeRO-1、seed 42，RTC 关闭。控制器仅接受该组路径及配置匹配的完整 checkpoint 恢复。资源操作由独立台账记录，仅按实际缺口释放已授权 owner/group/name/command/16GPU 匹配的占卡任务；六组正式任务提交后逐项补回。资源不足时排队等待。

初始释放额度用尽后，若六组正式任务已提交、smoke 全部结束、前次释放和正式排队均已超过 120 秒，仍有实际节点缺口，可人工运行资源脚本的 `release-one --reassess-deficit`。它再次检查当前分配、空闲、队列顺序及占卡任务身份，每次只释放一项，并把实际缺口写入同一补交台账；自动控制器不会自行突破初始释放额度。

实时状态：[report.md](../outputs/0907_blue_t_preparation/report.md)、[campaign_report.json](../outputs/0907_blue_t_preparation/campaign_report.json)、[对比曲线](../outputs/0907_blue_t_preparation/training_comparison.png)。每组 DataLoader 检查记录真实长度、各 rank batch 数和实际 vla_epoch 换算；正式曲线直接读取训练日志的 vla_epoch。不同标签空间的训练 loss 不代表可直接比较的实机成功率。
