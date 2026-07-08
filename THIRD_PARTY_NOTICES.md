# Third-Party Notices

This file records third-party code and resource files that are bundled with or
otherwise redistributed with RhinoVLA. The RhinoVLA project code is released under
Apache-2.0; the components below retain their upstream licenses.

## Qwen3-VL

- Project: Qwen3-VL
- Source repository: https://github.com/QwenLM/Qwen3-VL
- Model card used for license confirmation: https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct
- License: Apache-2.0
- Used in this repository:
  - `rhinovla/assets/qwen3_vl_processor/*`
  - Qwen3-VL-compatible model construction and preprocessing paths in `rhinovla/model/modules/qwen.py`

RhinoVLA uses repository-local Qwen3-VL config, tokenizer, and processor
assets to build the model structure and image/text preprocessing path. These
assets do not contain the official Qwen3-VL model weights. RhinoVLA
checkpoints are training artifacts produced for this project.
