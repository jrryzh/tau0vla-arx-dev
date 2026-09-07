# Deployment

The `deploy` package loads post-trained τ₀-VLA checkpoints for local inference,
HTTP serving, and open-loop evaluation. Public v1 HTTP serving supports
joint-control checkpoints only.

| entry point | purpose |
|---|---|
| `policy.py` | `Tau0VLAPolicy.from_checkpoint(...).infer(payload)` |
| `server.py` | HTTP policy server |
| `openloop.py` | local checkpoint evaluation |
| `openloop_with_server.py` | evaluation through a running server |
| `check_parity.py` | compare local and server inference |

## Checkpoint contract

A post-trained serving checkpoint carries a Data Spec for each route. It records
the robot name, registry key, config modules, cameras, prompt, dimensions, and
normalization contract. Installed adapter/registry code resolves those IDs. The
matching route artifacts live under `finch_data_spec/<route>/`.

Deployment uses that contract to reconstruct the training path:

```text
SDK or canonical payload
    -> adapter instruction, camera, and state mapping
    -> checkpoint Data Spec transforms and 40D assembly
    -> model
    -> unnormalize and undo relative action
    -> semantic action slices
    -> adapter action keys or flat SDK order
```

`--adapter <dotted.package>` is an explicit override. Normally the adapter is
resolved from the checkpoint and must remain registered under the same
`robot_name`.

## Policy server

```bash
python3 -m deploy.server --model /path/to/checkpoint
```

For a multi-route checkpoint:

```bash
python3 -m deploy.server --model /path/to/checkpoint --route YOUR_ROUTE
```

The server exposes:

- `POST /act` — canonical `{prompt, images, state, meta}` input and a
  name-keyed semantic action dictionary;
- `POST /act_lerobot_bytes` — embodiment SDK payload and a flat positional
  action chunk;
- `GET /health` — health check.

For an ARX LIFT2s checkpoint the same process also exposes a versioned robot
control contract. It creates one active session at a time and returns native
30-step, 14D joint-position chunks:

- `GET /api/v1/arx-lift2s/policy-contract`
- `POST /api/v1/arx-lift2s/sessions`
- `POST /api/v1/arx-lift2s/sessions/{session_id}/action-chunks`

Training-time RTC checkpoints additionally expose the compatible v2 family:

- `GET /api/v2/arx-lift2s/policy-contract` reports `rtc_enabled`,
  `rtc_max_delay`, and `rtc_delay_unit=action_steps`.
- v2 action requests add `rtc_delay` and a compact native-ARX
  `action_prefix` shaped `[rtc_delay, 14]`; delay zero uses an empty prefix.
- The response echoes `rtc_delay` and always remains `[30, 14]`. The returned
  conditioned prefix is overwritten exactly after native action permutation.

The v1 routes remain available and do not require an RTC-trained checkpoint.

The action-chunk endpoint accepts JSON metadata plus `head`, `left_wrist`, and
`right_wrist` JPEG multipart fields. It validates monotonically increasing
request IDs and echoes the request/session identity. This robot-facing API is
additive; `/act` and `/act_lerobot_bytes` keep their existing contracts.

For the ARX1 deployment, bind the server to the dedicated point-to-point
Ethernet address `192.168.50.2:8000` and allow the robot source
`192.168.50.1`. Keep Wi-Fi for management only; the bundled
`scripts/start_arx_lift2s_server.sh` uses these defaults.

Both POST request bodies are `.npz` bundles produced by
`deploy.wire.pack_payload`: boolean, integer, and floating-point numpy arrays
travel as named entries and the nested dictionary/list structure travels in a
tagged JSON envelope. The server validates the archive and always decodes it
with `allow_pickle=False`. FastAPI serializes
responses as JSON: `/act` returns a semantic dictionary and the legacy-named
`/act_lerobot_bytes` returns a nested numeric list, not raw bytes.

The wire format never unpickles request bodies — pickled payloads are remote
code execution for anyone who can reach the port. The server caps the request
while streaming it and separately caps the archive's total uncompressed size,
so compressed payloads cannot bypass `MAX_BODY_BYTES`. Existing clients must
replace `pickle.dumps(payload)` with `pack_payload(payload)`; there is no unsafe
pickle fallback. The server binds to `127.0.0.1` by default. The safe wire
format does not provide authentication or encryption, so expose the server
beyond localhost only inside a trusted, isolated network or behind an
authenticated TLS proxy.

For `/act`, send raw task text in `payload["prompt"]`; `encode_payload` applies
the saved prompt template. Sending already-templated text wraps it twice.
Dictionary insertion order in the response is irrelevant. The flat endpoint is
positional and must match the SDK exactly.

## Adapter input mapping

`adapters/<robot>/deploy_io.py` owns live payload conversion:

1. parse the instruction, images, and SDK state into the adapter's observation;
2. load state fields from the checkpoint's `field_descriptions.json`;
3. scatter each SDK state channel into those declared indices;
4. map SDK camera keys to the canonical names stored in the Data Spec;
5. produce `{prompt, images, state, meta}` for the common policy.

`_STATE_CHANNELS` keys must equal checkpoint field-description names. Every
active checkpoint state field needs exactly one accessor, and the constructed
state must be checked for uncovered indices. The template scatter assumes a
field's indices are contiguous; use indexed assignment instead when they are
not.

For a unified route, keep the three mapping layers separate:

1. `repack.raw` selects the dataset vector;
2. registry groups select native indices and scatter them into 40D;
3. checkpoint field descriptions plus `_STATE_CHANNELS` rebuild that same flat
   vector from a live SDK payload.

Every state index referenced by an active registry group must be filled exactly
once.

Every camera name and left/right view must match training. Missing images fail
loudly; also check that the instruction is non-empty.

EEF columns and providers remain training/data-pipeline features. The public v1
server rejects routes with EEF action slices and does not accept or return an
EEF control contract. The public G1 adapter also never derives EEF state from
joints.

## Action restoration and SDK order

For unified routes, model output is first restored to absolute semantic values.
The compact flat order is:

```text
left_eef, right_eef, left_gripper, right_gripper,
waist, chassis_velocity, left_arm, right_arm
```

This is the data-level restoration order. Public v1 serving uses only its
joint-control branches; EEF slices are not a public server output.

Only active slices are included. For `g1_a2d_joint_unified` this becomes:

```text
[left_gripper, right_gripper, left_arm x7, right_arm x7]
```

The A2D SDK instead consumes:

```text
[left_arm x7, right_arm x7, left_gripper, right_gripper]
```

`build_sdk_action_perm` maps compact semantic offsets back to native positions
using registry `action_groups`. It must reject:

- an active semantic slice with no SDK-native group;
- duplicate or uncovered native positions;
- EEF output sent to a joint-only SDK.

If the SDK vector contains uncontrolled or pass-through columns, a permutation
alone is insufficient; implement an explicit fill/preserve policy in that
adapter. Returning `None` is valid only when restored component order already
equals SDK order.

The bundled `g1_agibot_36` and `g1_daas_36` registry layouts contain native
action gaps. For example, `g1_agibot_36` controls native columns `0`, `1`, and
`16:30`, while a 36D SDK vector also contains columns `2:16` and `30:36`. The
policy does not define values for those gaps, so a permutation cannot construct
the complete 36D command safely.

In public v1:

- use canonical `/act` for `g1_agibot_36` and `g1_daas_36`;
- use `/act_lerobot_bytes` only for a layout with a complete, contiguous action
  mapping, such as `g1_a2d_joint_unified`.

Before hardware use, log and inspect semantic slices, permutation, final width,
and a known action vector. Compare one identical observation through the
dataset encoder and SDK payload encoder.

## Open-loop evaluation

```bash
python3 deploy/openloop.py --ckpt /path/to/checkpoint --no-plot
```

Select another route/config or save plots when needed:

```bash
python3 deploy/openloop.py \
    --ckpt /path/to/checkpoint \
    --route YOUR_ROUTE \
    --config YOUR_CONFIG_NAME \
    --out-dir /path/to/output
```

Evaluate through HTTP:

```bash
python3 deploy/openloop_with_server.py \
    --server-url http://127.0.0.1:10088 \
    --config-module your_package.your_config_module \
    --no-plot
```

For an external config module, pass `--config-module` so the same
`@register_config` runs on the evaluation side.
# ARX 校准协议 v3

Blue/T 校准模型只开放 `/arx/v3/policy-contract` 和 `/arx/v3/action-chunks`；旧 `/act` 及 ARX v1/v2 路由不可用于这些模型。旧模型继续使用原协议。

启动仍使用 `python -m deploy.server --model <checkpoint> --device cuda`。POST 请求采用 multipart：三个 JPEG 字段 `head`、`left_wrist`、`right_wrist`，以及 JSON 字符串字段 `metadata`。metadata 必须包含：

```json
{
  "protocol_version": "arx-calibrated-v3",
  "calibration_version": "arx-open-baseline-v1",
  "experiment": "eef-vr",
  "request_id": 1,
  "sample_monotonic_ns": 123456789,
  "task_instruction": "Pick up the T-shaped part and place it in its designated position on the board.",
  "raw_joint_feedback": [0,0,0,0,0,0,-2,0,0,0,0,0,0,-3],
  "raw_eef_feedback": [0,0,0,0,0,0,0,0,0,0,0,0,0,0],
  "open_baselines": {"left": -2, "right": -3}
}
```

示例数值仅用于说明结构，实际请求须使用真实反馈与当前校准值。两种反馈都保持采集 14D 顺序 `[left6, gripper, right6, gripper]`。服务端扣除一次全开基线；请求不得传入已经校准的 state。`experiment` 必须匹配 checkpoint 的 joint-vr、joint-feedback 或 eef-vr。

返回 `actions` 为具名分量。joint 返回左右绝对关节角；EEF 返回左右绝对 xyz 米和四元数 xyzw。左右夹爪以单列单独返回，`gripper_semantics` 明确它是 VR 闭合意图比例或基线校准后的反馈位置。`pose_convention` 保存源 RPY 约定。服务端结果始终包含 `robot_client_adapted: false`：ROS EEF 发布和 VR→底层夹爪驱动映射仍待驱动契约明确后实现。

每次推理在 `outputs/arx_calibrated_inference/<model_id>/` 保存 NPZ，包含原始 joint/EEF、基线、校准后输入、三相机、动作、请求时间和模型标识。支持无需机器人在线的回放：

```bash
PYTHONPATH=src:. .venv/bin/python scripts/replay_blue_t_request.py \
  --model <checkpoint> --request <recorded-request.npz> --out outputs/offline_replay
```
