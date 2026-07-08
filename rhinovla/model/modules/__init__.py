from pathlib import Path

from transformers import AutoConfig

from rhinovla.model.modules.qwen import Qwen


def _resolve_qwen_model_type(base_vlm: str) -> str:
    path = Path(str(base_vlm))
    if not path.exists():
        raise FileNotFoundError(
            f"framework.qwenvl.base_vlm must be a local Qwen3-VL config/processor asset directory, got {base_vlm!r}"
        )
    cfg = AutoConfig.from_pretrained(path, local_files_only=True)
    return str(getattr(cfg, "model_type", ""))


def build_qwen(config):
    base_vlm = config.framework.qwenvl.base_vlm
    if _resolve_qwen_model_type(base_vlm) != "qwen3_vl":
        raise NotImplementedError(f"Only Qwen3-VL is supported, got {base_vlm!r}")
    return Qwen(config)


__all__ = ["Qwen", "build_qwen"]
