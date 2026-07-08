# 数据准备指南

本文说明如何把一个 LeRobot v3 数据集接入 RhinoVLA native72 微调入口。核心流程是编写一份
native72 mapping,把数据集原生 `observation.state` 和 `action` 显式映射到 RhinoVLA 固定的
72D state/action 槽位,再生成训练配置中引用的 `meta/norm.json`。

训练启动、checkpoint 和日志说明见 [FINETUNING_GUIDE.md](FINETUNING_GUIDE.md)。

---

## 0. 需要准备的文件

1. 一个 LeRobot v3 磁盘数据集,包含 `data/`、`videos/` 和 `meta/`。
2. 一份 native72 mapping:`configs/data_mappings/<robot>.yaml`。
3. 一份归一化统计:`<dataset_root>/meta/norm.json`,由 `datasets/compute_norm_json.py` 生成。
4. 一份训练配置,在 `datasets.vla_data` 中引用 mapping 和 norm。

仓库自带的最小示例是:

```text
configs/data_mappings/demo_lerobot_native72.yaml
configs/training/demo_ae_finetune.yaml
configs/training/demo_full_finetune.yaml
```

---

## 1. LeRobot v3 数据布局

推荐目录结构:

```text
datasets/<your_lerobot_v3>/
├── data/chunk-000/file-000.parquet
├── videos/observation.images.<cam>/chunk-000/file-000.mp4
└── meta/
    ├── info.json
    ├── tasks.parquet            # 或 tasks.jsonl
    ├── norm.json                # 由 compute_norm_json.py 生成
    └── episode_valid_intervals.parquet   # 仅当 valid_chunk_filter: valid_intervals
```

每帧 parquet 至少需要:

| 列 | 要求 |
|---|---|
| `observation.state` | 必需,任意长度的状态向量 |
| `action` | 必需,任意长度的动作/目标向量,长度可与 state 不同 |
| `episode_index` / `frame_index` / `timestamp` / `task_index` | 必需 |
| 图像列 | 必需,列名由 mapping 的 `views[].key` 决定 |
| 指令列 | 推荐 `subtask_prompt`,可用 `instruction.fallback_keys` 回退到 `task` 或 `prompt` |

采样按行索引切 action chunk:`[i : i + action_horizon]`。当前 loader 支持三种
`sampling.valid_chunk_filter`:

| 模式 | 数据要求 | 适用场景 |
|---|---|---|
| `episode_only` | 无额外 sidecar;只要求 chunk 不跨 episode | 普通数据集、快速验证 |
| `valid_chunk_start` | parquet 中有 `valid_chunk_start` 布尔列 | 数据转换阶段已标注有效起点 |
| `valid_intervals` | `meta/episode_valid_intervals.parquet` | 需要按人工/规则区间过滤有效片段 |

---

## 2. 72D 槽位契约

RhinoVLA 固定使用 72D state/action。未映射的槽位填 0,mask 填 0,不参与 action loss。

| 槽位 | 语义 |
|---|---|
| `0..6` | 左臂 7D 关节 |
| `7..13` | 右臂 7D 关节 |
| `14` / `15` | 左 / 右夹爪 |
| `16..31` / `32..47` | 左 / 右灵巧手 |
| `48..50` | 头部 roll/pitch/yaw |
| `51..52` | 躯干 pitch/lift |
| `53..54` | 折叠升降腿关节 |
| `55..57` | 腰部 roll/pitch/yaw |
| `58..60` | 移动底盘速度指令 vx/vy/yaw_rate |
| `61..71` | 保留 |

mapping 只负责“源向量下标到 72D 槽位”的机械映射,不推断机器人语义。

---

## 3. 编写 native72 mapping

示例骨架:

```yaml
format: lerobot
target_dim: 72
fps: 30
action_horizon: 30
image_size: [256, 256]

instruction:
  source_key: subtask_prompt
  fallback_keys: [task, prompt]

datasets:
  - dataset_id: <id>
    root: datasets/<your_lerobot_v3>
    repo_id: local/<id>
    robot_instance_id: <robot>

    views:
      - {key: observation.images.head_color, role: top_head, modality: rgb, required: true}
      - {key: observation.images.hand_left, role: hand_left, modality: rgb, required: false}
      - {key: observation.images.hand_right, role: hand_right, modality: rgb, required: false}

    state_source: observation.state
    action_source: action
    active_state_slots:  [0, 1, 2]
    active_action_slots: [0, 1, 2]

    sampling:
      dataset_weight: 1.0
      valid_chunk_filter: episode_only

    native_joint_groups:
      - slot_group: arm0_joint
        state_source_indices:  [0, 1, 2]
        action_source_indices: [0, 1, 2]
        target_slots:          [0, 1, 2]
```

核心规则:

```text
rhino72_state[target_slots[i]]  = observation.state[state_source_indices[i]]
rhino72_action[target_slots[i]] = action[action_source_indices[i]]
```

注意事项:

- `state_source_indices[i]`、`action_source_indices[i]` 和 `target_slots[i]` 必须一一对应。
- state 与 action 源维度顺序一致时,两组 index 通常相同;不一致时必须分别写清楚。
- action-only 维度写 `state_source: none` 和空 `state_source_indices: []`。
- `views[].role` 必须在训练配置的 `view_role_vocab` 内。
- 夹爪默认不做方向变换、不单独缩放;如果数据协议方向与目标语义相反,应在 mapping 中显式使用
  `value_transform` 并重新生成 norm。

---

## 4. 生成归一化统计

```bash
python datasets/compute_norm_json.py datasets/<your_lerobot_v3> \
  --mapping-path configs/data_mappings/<robot>.yaml \
  --mapping-dataset-id <id> \
  --overwrite
```

默认统计为 quantile p01/p99 no-clip:

```text
mean = (p01 + p99) / 2
std  = (p99 - p01) / 2
std < 0.01 的维度使用 1.0
```

脚本会在 `_meta.norm_diagnostics` 中记录每个源维度的 min/max、非有限值计数和归一化后范围。希望在
CI 或交付检查中直接拦截风险时,加:

```bash
python datasets/compute_norm_json.py datasets/<your_lerobot_v3> \
  --mapping-path configs/data_mappings/<robot>.yaml \
  --mapping-dataset-id <id> \
  --fail-on-norm-warning \
  --overwrite
```

当某些 action-only 命令维度有明确物理范围,并且数据分布可能偏置时,可在 mapping group 内声明
norm 覆盖项:

```yaml
      - slot_group: base_velocity_cmd
        state_source: none
        state_source_indices: []
        action_source_indices: [16, 17, 18]
        target_slots: [58, 59, 60]
        norm:
          action_mean: [0.0, 0.0, 0.0]
          action_std: [0.13, 0.13, 0.26]
          method: physical_symmetric_range
```

mapping 改动后必须用同一份 mapping 重新生成 `meta/norm.json`。

---

## 5. 数据检查

`compute_norm_json.py` 生成 `norm.json` 时会同步记录源 state/action 每一维的 min/max、非有限值计数和
归一化后范围。需要直接在终端查看范围检查表时,加 `--range-check-format table`:

```bash
python datasets/compute_norm_json.py datasets/<your_lerobot_v3> \
  --mapping-path configs/data_mappings/<robot>.yaml \
  --mapping-dataset-id <id> \
  --overwrite \
  --range-check-format table
```

需要单独保存范围检查 JSON 时:

```bash
python datasets/compute_norm_json.py datasets/<your_lerobot_v3> \
  --mapping-path configs/data_mappings/<robot>.yaml \
  --mapping-dataset-id <id> \
  --output /tmp/<robot>_norm_preview.json \
  --overwrite \
  --range-check-format json > outputs/<robot>_range_check.json
```

---

## 6. 上线前检查

1. `datasets[].root` 指向 LeRobot 根目录,里面直接有 `data/ videos/ meta/`。
2. parquet 的 state/action 维度顺序与 mapping 完全一致。
3. action-only 维度是否需要 `state_source: none` 已确认。
4. 夹爪、底盘、腰部等有方向或单位约定的维度已人工确认。
5. `meta/norm.json` 是用当前 mapping 生成的,训练配置的 `norm_stats_path` 指向它。
6. `views[].key` 与 parquet 图像列名一致,`views[].role` 与训练配置的角色词表一致。
