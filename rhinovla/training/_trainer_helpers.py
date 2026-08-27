# RhinoVLA Training Framework

# Standard Library
import argparse
import hashlib
import json
import os
import re
import shutil
import time
from collections import deque
from pathlib import Path
from typing import Any, Tuple

# Third-Party Libraries
import numpy as np
import torch
import torch.distributed as dist
try:
    import swanlab
    HAS_SWANLAB = True
except ImportError:
    HAS_SWANLAB = False
from accelerate import Accelerator, DeepSpeedPlugin
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs, set_seed
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_scheduler

# Local Modules
from rhinovla.dataloader import build_dataloader
from rhinovla.model.framework import build_framework
from rhinovla.training.trainer_utils.config_tracker import AccessTrackedConfig, wrap_config
from rhinovla.training.trainer_utils.trainer_tools import TrainerUtils, build_param_lr_groups, normalize_dotlist_args, resize_images

def _finite_float_or_none(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None


def _format_optional_metric(value: Any, precision: int = 6) -> str:
    out = _finite_float_or_none(value)
    if out is None:
        return "NA"
    return f"{out:.{precision}f}"


def _truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _atomic_path(path: str | os.PathLike[str]) -> Path:
    final_path = Path(path)
    return final_path.with_name(f".{final_path.name}.{os.getpid()}.tmp")


def _atomic_replace_file(tmp_path: Path, final_path: str | os.PathLike[str]) -> None:
    os.replace(tmp_path, Path(final_path))


def _atomic_torch_save(obj: Any, path: str | os.PathLike[str]) -> None:
    final_path = Path(path)
    final_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = _atomic_path(final_path)
    try:
        torch.save(obj, tmp_path)
        _atomic_replace_file(tmp_path, final_path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def _atomic_safetensors_save(state_dict: dict[str, Any], path: str | os.PathLike[str]) -> None:
    from safetensors.torch import save_file

    final_path = Path(path)
    final_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = _atomic_path(final_path)
    try:
        save_file(state_dict, str(tmp_path))
        _atomic_replace_file(tmp_path, final_path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def _atomic_json_dump(payload: dict[str, Any], path: str | os.PathLike[str], *, indent: int = 2) -> None:
    final_path = Path(path)
    final_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = _atomic_path(final_path)
    try:
        with tmp_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=indent, ensure_ascii=False)
            f.write("\n")
        _atomic_replace_file(tmp_path, final_path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def _infer_peak_tflops_per_gpu(device_name: str) -> float | None:
    text = str(device_name).upper()
    # BF16/FP16 tensor-core peak, approximate. Prefer trainer.mfu_peak_tflops_per_gpu
    # when a run needs exact SKU-specific accounting.
    if "H200" in text:
        return 989.0
    if "H100" in text:
        return 989.0
    if "A100" in text or "A800" in text:
        return 312.0
    return None


def _rhino_tf_display_values(entry: dict[str, Any], values: np.ndarray) -> np.ndarray:
    """Convert Rhino canonical values to display units for TF plots only."""
    out = np.array(values, dtype=np.float32, copy=True)
    groups = entry.get("native_joint_groups") or []
    if not groups and any(key in entry for key in ("target_slots", "unit", "value_transform", "action_value_transform")):
        groups = [entry]
    for group in groups:
        if str(group.get("unit", "")) != "closed_ratio":
            continue
        transform = str(group.get("value_transform", group.get("action_value_transform", "")) or "")
        if transform != "linear_open_closed_to_closed_ratio":
            continue
        params = dict(group.get("value_transform_params", group.get("action_value_transform_params", {})) or {})
        raw_open = float(params.get("raw_open", -0.785))
        raw_closed = float(params.get("raw_closed", 0.0))
        slots = [int(slot) for slot in group.get("target_slots", [])]
        if not slots:
            continue
        out[..., slots] = raw_open - out[..., slots] * (raw_open - raw_closed)
    return out


def _short_text(text: Any, limit: int) -> str:
    text = " ".join(str(text).split())
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)] + "..."


def _mask_segment(mask: np.ndarray, start: int, end: int) -> str:
    """Human-readable active-slot segment, numbered from 1 inside the segment."""
    return "".join(
        str(i - start + 1) if i < len(mask) and float(mask[i]) > 0.5 else "-"
        for i in range(start, end)
    )


def _mask_fingers(mask: np.ndarray, base: int) -> str:
    return (
        f"T[{_mask_segment(mask, base + 0, base + 4)}] "
        f"I[{_mask_segment(mask, base + 4, base + 7)}] "
        f"M[{_mask_segment(mask, base + 7, base + 10)}] "
        f"R[{_mask_segment(mask, base + 10, base + 13)}] "
        f"L[{_mask_segment(mask, base + 13, base + 16)}]"
    )


def _format_72d_mask_lines(mask_row: Any) -> tuple[str, str, str, str, str]:
    """Format RhinoVLA 72D mask by semantic groups for GIF footers."""
    mask = np.asarray(mask_row, dtype=np.float32).reshape(-1)
    return (
        f"Arm    L[{_mask_segment(mask, 0, 7)}] R[{_mask_segment(mask, 7, 14)}] Grip[{_mask_segment(mask, 14, 16)}]",
        f"Hand L {_mask_fingers(mask, 16)}",
        f"Hand R {_mask_fingers(mask, 32)}",
        f"Body   Head[{_mask_segment(mask, 48, 51)}] Torso[{_mask_segment(mask, 51, 53)}] Leg[{_mask_segment(mask, 53, 55)}] Waist[{_mask_segment(mask, 55, 58)}]",
        f"Base   Vel[{_mask_segment(mask, 58, 61)}] Reserved[{_mask_segment(mask, 61, 72)}]",
    )


def _resize_down_keep_aspect(image, max_width: int = 720, max_height: int = 420):
    """Downscale only when needed. Never stretch to a fixed aspect ratio."""
    img = image.convert("RGB").copy()
    if img.width > max_width or img.height > max_height:
        try:
            from PIL import Image as PILImage

            resample = PILImage.Resampling.LANCZOS
        except Exception:  # noqa: BLE001
            resample = 1
        img.thumbnail((max_width, max_height), resample=resample)
    return img


def _render_sample_footer_frame(
    image,
    *,
    dataset_id: str,
    episode_index: int,
    frame_idx: int,
    raw_frame_count: int,
    instruction: str,
    overlay: str,
    mask_lines: tuple[str, ...],
):
    """Create a GIF frame with the original image aspect and a black info footer."""
    from PIL import Image, ImageDraw

    img = _resize_down_keep_aspect(image)
    canvas_w = max(img.width, 680)
    footer_h = 188
    canvas_h = img.height + footer_h
    canvas = Image.new("RGB", (canvas_w, canvas_h), (16, 16, 16))
    canvas.paste(img, ((canvas_w - img.width) // 2, 0))
    draw = ImageDraw.Draw(canvas)
    progress = 0.0
    if raw_frame_count > 1:
        progress = min(1.0, max(0.0, float(frame_idx) / float(raw_frame_count - 1)))
    y0 = img.height
    draw.rectangle([(0, y0), (canvas_w, canvas_h)], fill=(16, 16, 16))
    draw.rectangle([(0, y0), (int(canvas_w * progress), y0 + 5)], fill=(31, 185, 99))
    tag_line = f"{_short_text(dataset_id, 58)} | ep {episode_index} | f {frame_idx}/{max(0, raw_frame_count - 1)}"
    draw.text((10, y0 + 12), tag_line, fill=(255, 255, 255))
    draw.text((10, y0 + 34), overlay, fill=(210, 230, 255))
    y = y0 + 56
    for line in mask_lines[:5]:
        draw.text((10, y), _short_text(line, 112), fill=(175, 235, 175))
        y += 20
    draw.text((10, y + 2), _short_text(instruction, 110), fill=(235, 235, 235))
    return canvas


def build_accelerator(cfg) -> Accelerator:
    """Build the standard Accelerate CUDA/NCCL runtime.

    Keep startup on Accelerate's standard multi-GPU path. The sole DDP option
    is unused-parameter detection, required when selectively exercised Qwen
    branches are trainable.
    """
    grad_accum_steps = int(getattr(cfg.trainer, "gradient_accumulation_steps", 1))
    mixed_precision = "bf16" if getattr(cfg.trainer, "enable_mixed_precision_training", True) else "no"
    accelerator = Accelerator(
        mixed_precision=mixed_precision,
        gradient_accumulation_steps=grad_accum_steps,
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)],
    )
    accelerator.print(accelerator.state)
    return accelerator

# Sane Defaults
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Initialize logger
logger = get_logger(__name__)


def _configure_matplotlib_cjk_font(plt) -> None:
    """Prefer installed CJK fonts so TF/SwanLab figures render Chinese labels."""
    try:
        from matplotlib import font_manager

        preferred = [
            "Noto Sans CJK SC",
            "Noto Sans CJK JP",
            "WenQuanYi Micro Hei",
            "Microsoft YaHei",
            "SimHei",
            "Arial Unicode MS",
        ]
        installed = {f.name for f in font_manager.fontManager.ttflist}
        for name in preferred:
            if name in installed:
                plt.rcParams["font.sans-serif"] = [name] + list(plt.rcParams.get("font.sans-serif", []))
                break
        plt.rcParams["axes.unicode_minus"] = False
    except Exception:
        pass


def setup_directories(cfg) -> Path:
    """Create output directory and checkpoint directory."""
    cfg.output_dir = os.path.join(cfg.run_root_dir, cfg.run_id)
    output_dir = Path(cfg.output_dir)

    if not dist.is_initialized() or dist.get_rank() == 0:
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(output_dir / "checkpoints", exist_ok=True)

    return output_dir


def save_startup_config(cfg, output_dir: Path) -> None:
    """Save the exact merged config at run start for attribution.

    `config.yaml` is currently written by the access-tracker during checkpoint
    save, which means short/early runs can be hard to attribute if the source
    YAML is edited after launch.  This snapshot is written before model/data
    construction and is not access-filtered.
    """

    if dist.is_initialized() and dist.get_rank() != 0:
        return
    payload = _plain_config(cfg)
    OmegaConf.save(OmegaConf.create(payload), output_dir / "config_start.yaml", resolve=True)


def _plain_config(value: Any) -> Any:
    """Convert OmegaConf / tracked config nodes to regular Python containers."""
    if value is None:
        return None
    if isinstance(value, AccessTrackedConfig):
        return value.to_dict(resolve=True)
    if OmegaConf.is_config(value):
        return OmegaConf.to_container(value, resolve=True)
    return value


def _get_data_phase_schedule(cfg) -> list[dict[str, Any]]:
    vla_cfg = cfg.datasets.vla_data
    schedule = vla_cfg.get("phase_schedule", None)
    if schedule is None:
        schedule = vla_cfg.get("data_phase_schedule", None)
    schedule = _plain_config(schedule) or []
    if isinstance(schedule, dict):
        schedule = schedule.get("phases", [])
    phases: list[dict[str, Any]] = []
    for i, phase in enumerate(schedule):
        phase = dict(_plain_config(phase) or {})
        phase.setdefault("name", f"phase_{i}")
        phase["start_step"] = int(phase.get("start_step", 0))
        phases.append(phase)
    phases.sort(key=lambda x: x["start_step"])
    return phases


def _select_data_phase(phases: list[dict[str, Any]], step: int) -> tuple[int | None, dict[str, Any] | None]:
    selected_idx: int | None = None
    selected_phase: dict[str, Any] | None = None
    for idx, phase in enumerate(phases):
        if step >= int(phase.get("start_step", 0)):
            selected_idx = idx
            selected_phase = phase
        else:
            break
    return selected_idx, selected_phase


def _apply_data_phase(cfg, phase: dict[str, Any] | None) -> None:
    if not phase:
        return
    reserved = {"name", "start_step", "description", "note", "vla_data", "overrides"}
    updates = phase.get("vla_data", None)
    if updates is None:
        updates = phase.get("overrides", None)
    if updates is None:
        updates = {k: v for k, v in phase.items() if k not in reserved}
    updates = dict(_plain_config(updates) or {})
    for key, value in updates.items():
        setattr(cfg.datasets.vla_data, key, value)
    cfg.datasets.vla_data.active_phase_name = str(phase.get("name", ""))
    cfg.datasets.vla_data.active_phase_start_step = int(phase.get("start_step", 0))


def _auto_parallel_index_build(cfg) -> None:
    """No-op in the native72 path."""
    return


def prepare_data(cfg, accelerator, output_dir, *, apply_initial_phase: bool = True, qwen_processor=None) -> DataLoader:
    """Prepare VLA training data."""
    if apply_initial_phase:
        phases = _get_data_phase_schedule(cfg)
        _, phase = _select_data_phase(phases, step=0)
        _apply_data_phase(cfg, phase)
    data_mix = getattr(cfg.datasets.vla_data, "data_mix", cfg.datasets.vla_data.get("dataset_py", "unknown"))
    logger.info(f"Creating VLA Dataset with Mixture `{data_mix}`")
    build_t0 = time.perf_counter()
    _auto_parallel_index_build(cfg)
    vla_train_dataloader = build_dataloader(
        cfg=cfg,
        dataset_py=cfg.datasets.vla_data.dataset_py,
        qwen_processor=qwen_processor,
    )
    build_sec = time.perf_counter() - build_t0
    if not dist.is_initialized() or dist.get_rank() == 0:
        logger.info(f"Built VLA dataloader in {build_sec:.2f}s")
        ds = getattr(vla_train_dataloader, "dataset", None)
        logger.info(
            f"Train dataset stats={getattr(ds, 'dataset_stats', {})} "
            f"balance={getattr(ds, 'dataset_balance_stats', {})}"
        )
        try:
            profile_path = Path(output_dir) / "dataset_index_profile.json"
            profile_path.parent.mkdir(parents=True, exist_ok=True)
            profile_payload = {
                "build_dataloader_sec": float(build_sec),
                "index_profile": getattr(ds, "index_profile", {}),
                "dataset_stats": getattr(ds, "dataset_stats", {}),
                "dataset_balance_stats": getattr(ds, "dataset_balance_stats", {}),
            }
            profile_path.write_text(json.dumps(profile_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Failed to write dataset index profile: {exc}")

    accelerator.dataloader_config.dispatch_batches = False
    # Removed dist.barrier() - unnecessary cross-card synchronization
    return vla_train_dataloader


def setup_optimizer_and_scheduler(model, cfg) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
    """Set optimizer and scheduler."""
    param_groups = build_param_lr_groups(model=model, cfg=cfg)
    optimizer = torch.optim.AdamW(
        param_groups,
        lr=cfg.trainer.learning_rate.base,
        betas=tuple(cfg.trainer.optimizer.betas),
        weight_decay=cfg.trainer.optimizer.weight_decay,
        eps=cfg.trainer.optimizer.eps,
    )

    if dist.is_initialized() and dist.get_rank() == 0:
        for group in optimizer.param_groups:
            logger.info(f"LR Group {group['name']}: lr={group['lr']}, num_params={len(group['params'])}")

    scheduler_total_steps = int(
        getattr(cfg.trainer, "lr_scheduler_total_steps", cfg.trainer.max_train_steps)
        or cfg.trainer.max_train_steps
    )
    lr_scheduler = get_scheduler(
        name=cfg.trainer.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=cfg.trainer.num_warmup_steps,
        num_training_steps=scheduler_total_steps,
        scheduler_specific_kwargs=getattr(cfg.trainer, "scheduler_specific_kwargs", {}),
    )

    return optimizer, lr_scheduler


__all__ = [
    "logger",
    "_finite_float_or_none",
    "_format_optional_metric",
    "_truthy",
    "_atomic_path",
    "_atomic_replace_file",
    "_atomic_torch_save",
    "_atomic_safetensors_save",
    "_atomic_json_dump",
    "_infer_peak_tflops_per_gpu",
    "_rhino_tf_display_values",
    "_short_text",
    "_mask_segment",
    "_mask_fingers",
    "_format_72d_mask_lines",
    "_resize_down_keep_aspect",
    "_render_sample_footer_frame",
    "build_accelerator",
    "_configure_matplotlib_cjk_font",
    "setup_directories",
    "save_startup_config",
    "_plain_config",
    "_get_data_phase_schedule",
    "_select_data_phase",
    "_apply_data_phase",
    "_auto_parallel_index_build",
    "prepare_data",
    "setup_optimizer_and_scheduler",
]
