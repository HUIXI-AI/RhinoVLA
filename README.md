<p align="center">
  <img src="assets/huixi_logo_cropped.png" alt="辉羲智能" height="72" />
</p>

<h1 align="center">
  <img src="assets/rhinovla_logo.png" alt="RhinoVLA logo" height="56" style="vertical-align:middle;" />
  RhinoVLA
</h1>

<p align="center">
  <img src="https://img.shields.io/badge/License-Apache--2.0-green" alt="License" />
  <a href="https://arxiv.org/abs/2606.07383"><img src="https://img.shields.io/badge/Paper-arXiv%202606.07383-red" alt="Paper" /></a>
</p>

<p align="center">
  <b>中文</b> |
  <a href="./README_EN.md">English</a> |
  <a href="https://arxiv.org/abs/2606.07383">Paper</a>
</p>

<p align="center">
  <b>面向机器人端侧实时控制的跨本体 Vision-Language-Action 模型</b>
</p>

RhinoVLA 是辉羲智能自研的跨本体 VLA 系统，面向机器人本体端实时控制设计。技术报告展示了辉羲自研 VLA、辉羲 R1 芯片，以及智元、银河通用机器人组成的模型-芯片-机器人部署链路。

技术报告记录了 RhinoVLA 在辉羲 R1 芯片上 **11.69Hz** 的端到端推理频率及跨本体部署结果。这些既有结果不代表本分支 V3 的验证结果。

## 最新动态

- [2026-10-15] 发布了 RhinoVLA-V3 版本，动作/状态空间扩展至 96D，新增双臂末端执行器（EEF）控制及人形机器人腿足关节槽位，并使用更多机器人数据进行预训练。
- [2026-08-27] 支持 Training RTC（[Training-Time Action Conditioning for Efficient Real-Time Chunking](https://www.alphaxiv.org/pdf/2512.05964)）；训练配置通过 `framework.action_expert.training_time_rtc.max_delay` 指定最大 prefix 长度（训练时在 0 到该值之间采样）。
- [2026-07-30] 扩充训练数据和训练步数，支持动态图像宽高比，并优化图像 patch 的 RoPE 位置计算。
- [2026-07-08] 首次发布 RhinoVLA 预训练权重和微调代码。

V2 已冻结；本分支仅支持 V3，不兼容 V2 权重。

## 主要特性

- **端侧部署接口**：提供 RPU backend wrapper；V3 部署需匹配的 backend 与 artifact，见 [部署说明](docs/RPU_BACKEND_INFERENCE.md)。
- **算法-系统联合优化**：采用 Qwen3-VL Backbone + Action Expert 结构，并结合芯片、算子和运行时优化降低端侧开销。
- **跨本体适配**：通过 View Registry 和统一的 V3 96D 状态-动作接口处理视觉输入与动作空间差异。

<details>
<summary>技术报告原始架构（非本分支 V3 微调结构）</summary>

<p align="center">
  <img src="assets/rhinovla_architecture.png" alt="RhinoVLA technical report architecture" width="95%" />
</p>
</details>

## 🚀 快速开始 Quickstart

训练环境、数据 mapping、checkpoint 路径和启动脚本配置请参考 [Finetuning Guide](docs/FINETUNING_GUIDE.md)。
数据准备见 [数据准备指南](docs/DATA_PREPARATION.md)。

### 预训练权重

V3 发布仓库：[ModelScope](https://www.modelscope.cn/models/huixiAI/RhinoVLA-V3) ·
[Hugging Face](https://huggingface.co/HuixiAI/RhinoVLA-V3)。

下载 V3 权重后，将 `trainer.pretrained_checkpoint`
设为实际 V3 checkpoint，具体步骤见 [Finetuning Guide](docs/FINETUNING_GUIDE.md)。
权重格式、model card、license 和使用限制以发布仓库为准。

### 仓库结构

- `rhinovla/`：核心 Python 包，包含模型框架、Qwen3-VL 封装、Action Expert、LeRobot 数据加载和训练逻辑。
- `rhinovla/assets/qwen3_vl_processor/`：本地 Qwen3-VL processor / tokenizer 配置。
- `configs/training/`：训练配置，包括 Action Expert 微调、全参微调。
- `configs/data_mappings/`：LeRobot v3 原始 state/action 字段到 RhinoVLA V3 96D canonical slots 的 mapping。
- `datasets/example_lerobot_v3/`：可直接跑通 `demo_ae_finetune.yaml` 的最小 LeRobot v3 示例数据。
- `checkpoints/`：预训练权重。
- `scripts/train/`：训练启动脚本。
- `docs/`：微调说明文档。

## 🎬 演示 Demo

以下为项目既有演示，不作为本分支 V3 的验证结果。

### Galbot G1：指令跟随

RhinoVLA 运行在银河 G1 与辉羲 R1 芯片上，连续执行三次自然语言指令。

<div align="center">
  <video src="https://github.com/user-attachments/assets/5cc67652-53a8-49a0-a38c-08ae5fc38f68" width="100%" controls></video>
</div>

### AgiBot G2：长程任务

RhinoVLA 运行在智元 G2 与辉羲 R1 芯片上，通过一次长指令完成多步骤任务。

<div align="center">
  <video src="https://github.com/user-attachments/assets/04499b63-fefd-498d-8bd2-fdab28c1c2dd" width="100%" controls></video>
</div>

### AgiBot G1：双臂叠毛巾

RhinoVLA 运行在智元 G1 与辉羲 R1 芯片上，完成双臂柔性物体操作。

<div align="center">
  <video src="https://github.com/user-attachments/assets/de229167-6e2d-4e5c-849c-334d9d6800ab" width="100%" controls></video>
</div>

## 📦 Release Plan

- [x] 模型训练代码
- [x] V3 模型参数：见上方发布仓库。

## 📄 Citation

如果 RhinoVLA 对你的研究有帮助，请引用我们的技术报告：

```bibtex
@misc{intelligence2026rhinovlatechnicalreport,
      title={RhinoVLA Technical Report},
      author={Huixi Intelligence and Chen Zhang and Chenyang Zhou and Guanglei Ding and Guanghui He and Haibin Gao and Jiajia Chen and Jianyong Zhang and Lianyi Yu and Ningyi Xu and Ping Xu and Qingchen Li and Yingjun Hu and Yijia Zhang and Yuxi Liu},
      year={2026},
      eprint={2606.07383},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2606.07383},
}
```

## ⚖️ License

本项目采用 Apache-2.0 License。第三方组件和随仓库提供的第三方资源文件的许可证与来源见
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

## 💬 Contact

微信公众号：

<p align="left">
  <img src="assets/huixi_wechat_qrcode.png" alt="辉羲智能微信公众号二维码" width="360" />
</p>

## Acknowledgments

RhinoVLA 的视觉语言模型部分参考并使用了 [Qwen3-VL](https://github.com/QwenLM/Qwen3-VL)。
感谢相关团队开源高质量的多模态模型和 processor。
