# RPU Backend Inference

本文说明如何用即将开源的 `rpu_backend` 相关库，在 RPU/R1 侧运行本仓库训练出的 RhinoVLA checkpoint。

## 依赖边界

本仓库只提供 RhinoVLA 侧的轻量 wrapper，入口在：

```python
from rhinovla.inference.rpu_backend import RhinoVLARPUBackendExecutor
```

它对 `rpu_backend` 的依赖边界很窄，只使用 public API：

```python
from rpu_backend.api import RhinoVLAPolicy
```

当前 public API 仍以 RPU prepared artifact 为入口：

```python
RhinoVLAPolicy.from_prepare_artifact(...).to("rpu")
```

因此运行时需要以下依赖和资产：

| 依赖/资产 | 作用 |
|---|---|
| `rpu_backend` Python 包 | 即将开源的 RPU backend 库，必须暴露 `rpu_backend.api.RhinoVLAPolicy`。实际包名、版本和安装方式以 `rpu_backend` 发布说明为准。 |
| RPU runtime/system libraries | 板端 RPU 运行时依赖，例如驱动、runtime shared libraries、必要的 `LD_PRELOAD`。具体以目标板端环境和 `rpu_backend` 发布说明为准。 |
| `prepare_artifact.pt` | RPU backend 可直接消费的 prepared artifact。它通常锁定模型图、prompt tokenization、view layout、图像数量、部分 runtime metadata。 |
| RhinoVLA training `config.yaml` | 训练时使用的模型和数据配置，用于让 runtime 找到正确的 RhinoVLA 结构和预处理约定。 |
| RhinoVLA `checkpoint` | 本仓库训练代码保存的 PyTorch 模型权重，例如 `steps_<n>_pytorch_model.pt`。 |
| `norm.json` / norm sidecar | state/action 的归一化和反归一化统计。 |
| native72 mapping YAML | 当 norm stats 仍是原始数据维度而不是 72D 时，用它展开到 RhinoVLA 72D slots。 |
| 本仓库 `rhinovla` 包 | 提供 wrapper、norm/mapping 处理和本仓库模型代码路径。 |

本 wrapper 不导入 `rpu_backend.adapters.*`、`rpu_backend.runtime.*`、`rpu_backend.tests.*` 等 private module，也不包含旧 board-kit 代码。

## `prepare_artifact` 与 `checkpoint`

二者不是同一个东西：

| 参数 | 含义 |
|---|---|
| `checkpoint` | 训练产物。它是 PyTorch 权重，需要配套 training config 才能还原 RhinoVLA 模型。 |
| `prepare_artifact` | RPU 部署产物。它是 `rpu_backend` 当前 public API 的加载入口，由 `RhinoVLAPolicy.from_prepare_artifact(...)` 消费。 |

在本 wrapper 里，`prepare_artifact` 直接传给 `RhinoVLAPolicy.from_prepare_artifact(...)`；`checkpoint` 和 `train_config` 会写入 runtime env：

```text
RHINOVLA_CONFIG=/path/to/config.yaml
RHINOVLA_CKPT=/path/to/steps_<n>_pytorch_model.pt
```

如果未来 `rpu_backend` 开源版本提供 direct checkpoint API 或 checkpoint-to-artifact API，可以在本 wrapper 里另加入口；当前实现不假设这类私有能力。

## Python API

```python
from PIL import Image
import numpy as np

from rhinovla.inference.rpu_backend import RhinoVLARPUBackendExecutor

executor = RhinoVLARPUBackendExecutor(
    prepare_artifact="/path/to/rhinovla_prepare_artifact.pt",
    train_config="/path/to/config.yaml",
    checkpoint="/path/to/steps_1000_pytorch_model.pt",
    norm_stats_path="/path/to/norm.json",
    mapping_path="/path/to/configs/data_mappings/robot_native72.yaml",
    mapping_dataset_id="robot_dataset",
    instruction="pick up the object",
    num_steps=5,
    action_hz=30.0,
    active_slots=None,
    view_roles=["top_head", "hand_left", "hand_right"],
    view_modalities=["rgb", "rgb", "rgb"],
    rhino_repo="/path/to/rhinovla",
    runtime_env={"RPU_RHINOVLA_PREFIX_ALIAS": "1"},
    artifact_strict=False,
    noise_seed=0,
)

result = executor.infer(
    [
        Image.open("/path/to/head.png"),
        Image.open("/path/to/left.png"),
        Image.open("/path/to/right.png"),
    ],
    np.asarray([...], dtype=np.float32),
)
print(result.actions_raw.shape)
executor.close()
```

### 构造参数

| 参数 | 必填 | 含义 |
|---|---:|---|
| `prepare_artifact` | 是 | RPU prepared artifact 路径，传给 `RhinoVLAPolicy.from_prepare_artifact()`。 |
| `norm_stats_path` | 是 | matching norm stats JSON。可指向训练数据 `meta/norm.json`，也可指向 checkpoint sidecar，例如 `steps_<n>_norm_stats.json`。 |
| `instruction` | 是 | 本次 RPU policy 的自然语言指令。当前 RPU artifact prompt 是 locked 的，推理时不能临时换 prompt；需要换 prompt 时应重建 executor/匹配 artifact。 |
| `train_config` | 否 | 训练 YAML 路径。若提供，会作为 `RHINOVLA_CONFIG` 写入 `runtime_env`，除非用户已显式覆盖同名 key。 |
| `checkpoint` | 否 | 训练 checkpoint 路径。若提供，会作为 `RHINOVLA_CKPT` 写入 `runtime_env`，除非用户已显式覆盖同名 key。 |
| `num_steps` | 否 | flow-matching denoise/inference steps，默认 `5`。应和 artifact、评测或部署 profile 对齐。 |
| `action_hz` | 否 | 输出 action chunk 的动作频率 metadata，默认 `30.0`。 |
| `active_slots` | 否 | 需要输出/执行的 72D action slots。未提供时优先使用 norm/mapping 里的 active action slots；72D stats 默认回退到 `0-15`。 |
| `mapping_path` | 否 | native72 mapping YAML。native-dim norm stats 必须提供；72D stats 可不提供。 |
| `mapping_dataset_id` | 否 | mapping YAML 中 `datasets` 列表的选择项。mapping 只有一个 dataset 时可省略。 |
| `view_roles` | 否 | 图像视角角色，顺序必须和输入图片一致，默认 `["top_head", "hand_left", "hand_right"]`。 |
| `view_modalities` | 否 | 图像模态，顺序必须和输入图片一致，默认 `["rgb", "rgb", "rgb"]`。 |
| `rhino_repo` | 否 | 传给 `rpu_backend` 的 RhinoVLA repo override。部署路径和 artifact metadata 不一致时使用。 |
| `runtime_env` | 否 | 传给 `RhinoVLAPolicy.from_prepare_artifact(..., runtime_env=...)` 的环境变量字典，用于设置 RPU runtime/profile 开关或覆盖路径。 |
| `artifact_strict` | 否 | 是否让 `rpu_backend` 严格校验 artifact metadata，默认 `False`。 |
| `noise_seed` | 否 | 初始噪声 seed。用于数值复现/回归对齐，默认 `0`。 |

### `infer()` 参数

| 参数 | 含义 |
|---|---|
| `images` | 图像序列，顺序必须和 `view_roles` / `view_modalities` 一致。元素可以是 PIL Image，也可以是可转为 uint8 image array 的对象。 |
| `raw_state` | 未归一化的机器人 state。可以是完整 72D，也可以是长度等于 active state slots 的压缩 state。压缩 state 会先写入 72D active state slots，再做归一化。 |
| `instruction` | 可选校验参数。若传入值和构造 executor 时的 `instruction` 不一致，会报错，因为当前 RPU prompt 是 locked 的。 |

### 输出字段

| 字段 | 含义 |
|---|---|
| `result.actions_norm` | `rpu_backend` 返回的归一化 action chunk，形状通常为 `[H, D]`。 |
| `result.actions_raw` | 用 `norm_stats_path` 反归一化后的物理 action chunk。 |
| `result.action_hz` | 输出动作频率 metadata。 |
| `result.latency_ms` | backend 返回的单次推理延迟。 |
| `result.extra` | 包含 `active_slots`、`active_state_slots`、`prepare_artifact`、`train_config`、`checkpoint`、`timing_ms` 和 `rpu_backend` backend metadata。 |

## CLI Usage

```bash
PREPARE_ARTIFACT=/path/to/rhinovla_prepare_artifact.pt \
CONFIG=/path/to/config.yaml \
CHECKPOINT=/path/to/steps_1000_pytorch_model.pt \
NORM_STATS=/path/to/norm.json \
MAPPING=/path/to/configs/data_mappings/robot_native72.yaml \
MAPPING_DATASET_ID=robot_dataset \
INSTRUCTION='pick up the object' \
STATE='[0.0, 0.1, 0.2]' \
IMAGES='/path/head.png /path/left.png /path/right.png' \
OUT=./rpu_backend_infer_output.json \
  bash scripts/infer/run_rpu_backend_infer.sh
```

`STATE` 支持 JSON list、逗号分隔 list，或指向 JSON array 的文件路径。

### CLI / shell 参数

| Shell 变量 | CLI 参数 | 必填 | 含义 |
|---|---|---:|---|
| `PREPARE_ARTIFACT` | `--prepare-artifact` | 是 | RPU prepared artifact 路径。 |
| `CONFIG` | `--config` / `--train-config` | 是 | 训练 YAML 路径，同时默认写入 `RHINOVLA_CONFIG`。 |
| `CHECKPOINT` | `--checkpoint` | 是 | 训练 checkpoint 路径，同时默认写入 `RHINOVLA_CKPT`。 |
| `NORM_STATS` | `--norm-stats` | 是 | matching norm stats JSON。 |
| `INSTRUCTION` | `--instruction` | 是 | 本次推理指令。 |
| `STATE` | `--state` | 是 | raw state，支持 JSON list、逗号分隔 list 或 JSON 文件路径。 |
| `IMAGES` | `--image` | 是 | shell 脚本中为空格分隔的图片路径；CLI 中可重复传 `--image`，顺序就是视角顺序。 |
| `OUT` | `--output` | 否 | 输出 JSON 路径；未设置时脚本默认 `./rpu_backend_infer_output.json`。 |
| `NUM_STEPS` | `--num-steps` | 否 | denoise/inference steps，默认 `5`。 |
| `ACTION_HZ` | `--action-hz` | 否 | action frequency metadata，默认 `30`。 |
| `NOISE_SEED` | `--noise-seed` | 否 | 初始噪声 seed，默认 `0`。 |
| `ACTIVE_SLOTS` | `--active-slots` | 否 | action active slots，如 `0-15` 或 `0-6,14,15`。 |
| `VIEW_ROLES` | `--view-roles` | 否 | 逗号分隔 view roles，默认 `top_head,hand_left,hand_right`。 |
| `VIEW_MODALITIES` | `--view-modalities` | 否 | 逗号分隔 view modalities，默认 `rgb,rgb,rgb`。 |
| `MAPPING` | `--mapping` | 否 | native72 mapping YAML。native-dim norm stats 必填。 |
| `MAPPING_DATASET_ID` | `--mapping-dataset-id` | 否 | 选择 mapping YAML 中的 dataset entry。 |
| `RPU_RHINO_REPO` | `--rpu-rhino-repo` | 否 | 传给 `rpu_backend` 的 RhinoVLA repo path override。 |
| `RPU_ENV_FILE` | `--rpu-env-file` | 否 | TOML 文件，读取其中 `[env]` table 作为 runtime env。 |
| `RPU_ENV` | `--rpu-env` | 否 | runtime env override，shell 中为空格分隔 `KEY=VALUE`；CLI 中可重复传。 |
| `RPU_ARTIFACT_STRICT` | `--rpu-artifact-strict` | 否 | shell 中设为 `1` 时开启 strict artifact metadata 校验。 |

`RPU_ENV_FILE` 和 `RPU_ENV` 的合并顺序是：

1. 先读取 TOML `[env]`；
2. 再应用 `--rpu-env KEY=VALUE` 覆盖；
3. 最后如果还没有 `RHINOVLA_CONFIG` / `RHINOVLA_CKPT`，由 `--config` / `--checkpoint` 自动补上。

## Norm 和 Mapping 约定

`norm_stats_path` 可以指向：

- 训练数据目录中的 `meta/norm.json`，此时 stats 可能仍是 native source dimensions；
- checkpoint sidecar，例如 `steps_<n>_norm_stats.json`；
- 扁平 JSON，包含 `state_mean`、`state_std`、`action_mean` 和 `action_std`。

如果 stats 是 native source dimensions，需要同时传 matching native72 mapping 和 dataset id。wrapper 会把 source stats 展开到 RhinoVLA 72D slots，inactive mean 填 `0.0`，inactive std 填 `1.0`，并使用 mapping 中的 `active_state_slots` / `active_action_slots` 处理短 state 输入和 action metadata。

如果 stats 已经是 72D，mapping 可以不传。
