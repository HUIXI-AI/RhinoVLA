<p align="center">
  <img src="assets/huixi_logo_cropped.png" alt="Huixi AI" height="72" />
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
  <a href="./README.md">中文</a> |
  <b>English</b> |
  <a href="https://arxiv.org/abs/2606.07383">Paper</a>
</p>

<p align="center">
  <b>A cross-embodiment Vision-Language-Action model for real-time robot control on edge AI chips</b>
</p>

RhinoVLA is a self-developed cross-embodiment VLA system from Huixi AI, designed for real-time robot control on edge devices. The technical report presents a model-chip-robot deployment stack built with RhinoVLA, the Huixi R1 chip, and real robots from AgiBot and Galbot.

The technical report records **11.69Hz** end-to-end inference on the Huixi R1 chip and cross-embodiment deployment results. These earlier results are not validation results for V3 on this branch.

## Latest News

- [2026-10-15] Released RhinoVLA-V3 with a 96D action/state space, adding slots for dual-arm end-effector (EEF) control and humanoid leg and foot joints, and pretrained on more robot data.

V2 is frozen. This branch supports V3 only and is incompatible with V2 weights.

## Key Features

- **Edge deployment interface**: provides an RPU backend wrapper; V3 requires a matching backend and artifact. See [deployment notes](docs/RPU_BACKEND_INFERENCE.md).

- **Algorithm-system co-optimization**: combines a Qwen3-VL Backbone, an Action Expert, and chip/runtime optimization for lower edge-side cost.

- **Cross-embodiment adaptation**: uses View Registry and a unified V3 96D state-action interface to handle visual-input and action-space differences.

<details>
<summary>Original technical report architecture (not this branch's V3 finetuning model)</summary>

<p align="center">
  <img src="assets/rhinovla_architecture.png" alt="RhinoVLA technical report architecture" width="95%" />
</p>
</details>

## 🚀 Quickstart

For training environment setup, data mappings, checkpoint paths, and launch
scripts, see the [Finetuning Guide](docs/FINETUNING_GUIDE.md).
For dataset preparation, see [Data Preparation](docs/DATA_PREPARATION.md).

### Pretrained weights

V3 release repositories: [ModelScope](https://www.modelscope.cn/models/huixiAI/RhinoVLA-V3) ·
[Hugging Face](https://huggingface.co/HuixiAI/RhinoVLA-V3).

After downloading, set `trainer.pretrained_checkpoint` to the actual V3 checkpoint;
see the [Finetuning Guide](docs/FINETUNING_GUIDE.md).
Consult the release repositories for weight formats, model cards, licenses, and usage restrictions.

### Repository layout

- `rhinovla/`: core Python package with model framework, Qwen3-VL wrapper, Action Expert, LeRobot dataloading, and training logic.
- `rhinovla/assets/qwen3_vl_processor/`: local Qwen3-VL processor / tokenizer configuration.
- `configs/training/`: training configs, including Action Expert finetuning and full-parameter finetuning.
- `configs/data_mappings/`: mappings from raw LeRobot v3 state/action fields to RhinoVLA V3 96D canonical slots.
- `datasets/example_lerobot_v3/`: minimal LeRobot v3 demo dataset for `demo_ae_finetune.yaml`.
- `checkpoints/`: pretrained weights.
- `scripts/train/`: training launch scripts.
- `docs/`: finetuning documentation.

## 🎬 Demo

These earlier project demonstrations are not validation results for V3 on this branch.

### Galbot G1: Instruction following

RhinoVLA runs on Galbot G1 with the Huixi R1 chip and executes three natural-language instructions.

<div align="center">
  <video src="https://github.com/user-attachments/assets/5cc67652-53a8-49a0-a38c-08ae5fc38f68" width="100%" controls></video>
</div>

### AgiBot G2: Long-horizon task

RhinoVLA runs on AgiBot G2 with the Huixi R1 chip and completes a multi-step task from one long instruction.

<div align="center">
  <video src="https://github.com/user-attachments/assets/04499b63-fefd-498d-8bd2-fdab28c1c2dd" width="100%" controls></video>
</div>

### AgiBot G1: Bimanual towel folding

RhinoVLA runs on AgiBot G1 with the Huixi R1 chip and completes a bimanual deformable-object task.

<div align="center">
  <video src="https://github.com/user-attachments/assets/de229167-6e2d-4e5c-849c-334d9d6800ab" width="100%" controls></video>
</div>

## 📦 Release Plan

- [x] Model training code
- [x] V3 model parameters: see the release repositories above.

## 📄 Citation

If RhinoVLA is helpful for your research, please cite our technical report:

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

This project is released under the Apache-2.0 License. Third-party components
and bundled third-party resource files are listed in
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

## 💬 Contact

WeChat Official Account:

<p align="left">
  <img src="assets/huixi_wechat_qrcode.png" alt="Huixi AI WeChat official account QR code" width="360" />
</p>

## Acknowledgments

RhinoVLA uses [Qwen3-VL](https://github.com/QwenLM/Qwen3-VL) compatible
configuration, tokenizer, and processor assets for its vision-language prefix
path.
