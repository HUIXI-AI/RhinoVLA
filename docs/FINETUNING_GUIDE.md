# Finetuning Guide

目标：用当前仓库，在任意机器人、任意 X/Y 维 `observation.state` / `action` 的 LeRobot v3 数据集上，从 `checkpoints/rhinovla_pretrain.ckpt` 初始化并微调 RhinoVLA。核心做法是写一份 mapping，把数据集原始维度显式映射到 RhinoVLA 固定 72D state/action slots，并用 mask 决定哪些 slot 参与训练。

当前保留的训练配置示例：

- `configs/training/demo_ae_finetune.yaml`：使用仓库自带的 `datasets/example_lerobot_v3`，batch size 1，用于验证完整链路能启动。

对应保留的 mapping 示例：

- `configs/data_mappings/demo_lerobot_native72.yaml`：本地 example 数据集的 native72 mapping 示例。

## 快速启动 Demo

先从 Hugging Face 下载 `checkpoints/rhinovla_pretrain.ckpt`，然后运行：

```bash
./scripts/train/demo_train.sh
```

脚本只做三件事：进入仓库根目录、检查 `checkpoints/rhinovla_pretrain.ckpt`、启动 `configs/training/demo_ae_finetune.yaml`。默认不启用 SwanLab，只写本地 `outputs/demo_train/**/metrics.jsonl`。

如果要上报 SwanLab：

```bash
SWANLAB_API_KEY=... TRACKERS='[jsonl,swanlab]' ./scripts/train/demo_train.sh
```

本地部署 SwanLab 需要额外指定地址：

```bash
SWANLAB_API_KEY=... \
SWANLAB_WEB_HOST=http://your-swanlab-host:8000 \
TRACKERS='[jsonl,swanlab]' \
./scripts/train/demo_train.sh
```

`SWANLAB_API_HOST` 也可替代 `SWANLAB_WEB_HOST`。其他训练参数可以用命令行覆盖，例如：

```bash
./scripts/train/demo_train.sh trainer.max_train_steps=10 run_id=debug_10steps
```

## LeRobot v3 数据格式

数据集目录由 training config 的 `datasets.vla_data.norm_stats_path` 和 mapping 的 `root` 指向，推荐使用仓库相对路径。最小结构：

```text
datasets/your_lerobot_v3/
  data/chunk-000/file-000.parquet
  videos/...
  meta/info.json
  meta/tasks.parquet
  meta/norm.json
```

parquet 里至少需要：

- `observation.state`：每帧机器人 state 向量，长度可以是任意 X。
- `action`：每帧 action/target 向量，长度可以是任意 Y。
- `episode_index`、`frame_index`、`timestamp`、`task_index`。
- 图像列，例如 `observation.images.head_color` 或 `observation.images.chest`。列名由 mapping 的 `views` 决定。
- 指令列推荐 `subtask_prompt`，也可以用 mapping 的 `instruction.fallback_keys` 回退到 `task` 或 `prompt`。

chunk 采样默认按 30Hz 行索引切 `[i : i + H]`，`H` 来自 `action_horizon`。`sampling.valid_chunk_filter` 只保留三种行为名：`episode_only` 是默认模式，只要求 chunk 不跨 episode；`valid_chunk_start` 是帧/起点级过滤，需要数据侧提供 `valid_chunk_start` 列；`valid_intervals` 是区间级过滤，需要 `meta/episode_valid_intervals.parquet`。

## 72D Slots 语义

RhinoVLA 固定使用 72D state/action 和对应 mask。没有映射的 slot 填 0，mask 填 0，不参与 loss。

| slot 范围 | 语义 |
|---|---|
| `0..6` | 左臂 7D 关节 |
| `7..13` | 右臂 7D 关节 |
| `14` | 左夹爪 |
| `15` | 右夹爪 |
| `16..31` | 左灵巧手 |
| `32..47` | 右灵巧手 |
| `48..50` | 头部 |
| `51..52` | 躯干 pitch/lift |
| `53..54` | 折叠升降腿关节 |
| `55..57` | 腰部 roll/pitch/yaw |
| `58..60` | 移动底盘速度指令 |
| `61..71` | 保留 |

训练 loss 是 action mask 有效 slot 上的 flow-matching MSE。state mask 只描述当前输入 state 的有效 slot。

## X 维数据映射到 72D

推荐对新机器人使用 `native_joint_groups`，每个 group 写清楚源向量 index 到 72D slot 的映射：

```yaml
native_joint_groups:
  - slot_group: arm0_joint
    state_source_indices: [0, 1, 2, 3, 4, 5, 6]
    action_source_indices: [0, 1, 2, 3, 4, 5, 6]
    target_slots: [0, 1, 2, 3, 4, 5, 6]

  - slot_group: gripper
    state_source_indices: [7, 15]
    action_source_indices: [7, 15]
    target_slots: [14, 15]
```

核心字段只保留这些：

- `format: lerobot`
- `target_dim: 72`
- `fps`、`action_horizon`、`image_size`
- `datasets[].dataset_id`、`root`、`repo_id`、`robot_instance_id`
- `views`
- `state_source`、`action_source`
- `active_state_slots`、`active_action_slots`
- `native_joint_groups`
- `instruction`
- `sampling`

一个 `native_joint_groups` 条目同时描述两件事：当前时刻的 state 输入，以及未来 H 步的 action 监督。因此有两套源 index：

- `state_source_indices`：从 `state_source` 指向的列读取，默认是 `observation.state`，用于构造模型输入的 72D `state`。
- `action_source_indices`：从 `action_source` 指向的列读取，默认是 `action`，用于构造训练监督的 H 步 72D `actions`。

如果你的 LeRobot 数据里 `observation.state` 和 `action` 的维度顺序完全一致，这两项通常写成一样；如果 action 列比 state 多底盘速度、末端目标、delta action，或顺序不同，就必须分别配置。某些 action-only 维度，例如底盘速度，state 侧不存在，可以写：

```yaml
- slot_group: base_velocity_cmd
  state_source: none
  state_source_indices: []
  action_source_indices: [19, 20, 21]
  target_slots: [58, 59, 60]
```

`state_source_indices` 的 index 顺序由 LeRobot parquet 里 `observation.state` 向量的维度顺序决定，`action_source_indices` 由 `action` 向量的维度顺序决定。代码不会自动识别语义，只会执行：

```text
state_selected = state_vector[state_source_indices]
action_selected = action_vector[action_source_indices]
rhino72_state[target_slots] = state_selected
rhino72_action[target_slots] = action_selected
```

因此 `state_source_indices[i]` / `action_source_indices[i]` 必须和 `target_slots[i]` 一一对应。比如 `state_source_indices: [0, 1]`、`action_source_indices: [0, 1]`、`target_slots: [51, 52]` 表示 state/action 源第 0 维写入 RhinoVLA slot 51，源第 1 维写入 slot 52。源向量维度顺序应来自数据转换脚本、`meta/info.json` 的 features 描述或机器人数据导出协议；mapping 只是把这个协议显式写出来。

`demo_lerobot_native72.yaml` 使用 `datasets + native_joint_groups` 结构；后续新增机器人也按这套结构扩展，不再新增第二种 mapping 形态。

夹爪维度和其他维度一样处理：不做额外开合方向变换，不单独缩放，直接按 `norm.json` 里的 quantile p01/p99 no-clip 统计归一化。

## Norm 文件

训练必须显式提供 `norm.json`，不要依赖 fallback。生成方式：

```bash
python datasets/compute_norm_json.py datasets/example_lerobot_v3 \
  --mapping-path configs/data_mappings/demo_lerobot_native72.yaml \
  --mapping-dataset-id example_lerobot_v3 \
  --overwrite
```

统计方式是 quantile p01/p99 no-clip：

```text
mean = (p01 + p99) / 2
std  = (p99 - p01) / 2
std < 0.01 的维度用 1.0
```

dataloader 会把源维度 norm stats 按 mapping 扩展到 72D；未激活 slot 的 mean=0、std=1。

## 训练配置

关键字段在 `configs/training/*.yaml`：

```yaml
framework:
  name: RhinoVLA
  qwenvl:
    base_vlm: rhinovla/assets/qwen3_vl_processor
    freeze_qwen: true
    freeze_vision: false

datasets:
  vla_data:
    dataset_py: lerobot_native72
    mapping_path: configs/data_mappings/demo_lerobot_native72.yaml
    mapping_dataset_id: example_lerobot_v3
    norm_stats_path: datasets/example_lerobot_v3/meta/norm.json
    image_resize_mode: letterbox
    use_view_registry_prompt: true

trainer:
  pretrained_checkpoint: checkpoints/rhinovla_pretrain.ckpt
```

`base_vlm` 指向仓库内 Qwen3-VL config/tokenizer/processor asset，用来构建模型结构和图文预处理；它不包含、也不会加载官方 Qwen3-VL 权重。微调权重来自 `trainer.pretrained_checkpoint: checkpoints/rhinovla_pretrain.ckpt`。

`use_view_registry_prompt` 控制图文组合方式：

- `true`：使用 RhinoVLA 预训练一致的 View Registry prompt，需要配置 `view_roles` 和 `view_modalities`。
- `false`：使用裸 `<图><图><图> + instruction` prompt。

## 解冻模式

支持两种训练模式：

- `ae_only`：默认微调模式。`framework.qwenvl.freeze_qwen: true`，只训练 ActionExpert/action IO/projector。
- `full`：`framework.qwenvl.freeze_qwen: false` 且 `framework.qwenvl.freeze_vision: true`，VLM 里除了视觉 encoder/VIT 外都参与训练，同时训练 ActionExpert。

`trainer.pretrained_checkpoint: checkpoints/rhinovla_pretrain.ckpt` 是 RhinoVLA 50K 预训练 ckpt 转成当前结构后的 strict-load 权重。

## 使用流程清单

1. 准备 LeRobot v3 数据集目录，确认 parquet 里有 state/action、相机列、episode/frame/timestamp/task 字段。
2. 按数据集真实向量顺序编写 `configs/data_mappings/<robot>.yaml`，把源 index 映射到 72D slots。
3. 用 `datasets/compute_norm_json.py` 生成 `meta/norm.json`。
4. 复制或修改 `configs/training/demo_ae_finetune.yaml`，配置 `mapping_path`、`mapping_dataset_id`、`norm_stats_path`、batch size、解冻模式和日志。
5. 确认 `checkpoints/rhinovla_pretrain.ckpt` 存在。
6. 启动训练：`python -m rhinovla.training.train --config_yaml configs/training/<your_config>.yaml`，或参考 `scripts/train/demo_train.sh` 写自己的启动脚本。
