"""Native72 visual logging.

The training path is intentionally narrow: VLM prompt cards, teacher-forcing
error figures, and action-expert attention figures generated from actual model
inputs.
"""

from __future__ import annotations

import math
import os
import textwrap
import io
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from accelerate.logging import get_logger
from PIL import Image, ImageDraw, ImageFont

try:
    import swanlab

    HAS_SWANLAB = True
except ImportError:  # pragma: no cover - exercised on hosts without swanlab
    swanlab = None
    HAS_SWANLAB = False


logger = get_logger(__name__)
_RESAMPLE_BILINEAR = getattr(Image, "Resampling", Image).BILINEAR
_RESAMPLE_BICUBIC = getattr(Image, "Resampling", Image).BICUBIC
_PARENTHETICAL_TITLE_RE = re.compile(r"\s*\([^)]*\)")
ATTENTION_HEATMAP_ALPHA_MAX = 0.48
ATTENTION_HEATMAP_LIGHTEN = 0.16


def _cfg_get(container: Any, key: str) -> Any:
    if container is None:
        return None
    if isinstance(container, dict):
        return container.get(key)
    try:
        value = container.get(key)
    except Exception:  # noqa: BLE001
        value = None
    if value is not None:
        return value
    return getattr(container, key, None)


def _as_positive_int(value: Any) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _config_int(config: Any, key: str, env_key: str, default: int) -> int:
    env_value = _as_positive_int(os.environ.get(env_key))
    if env_value is not None:
        return env_value
    for container in (_cfg_get(config, "trainer"), config):
        cfg_value = _as_positive_int(_cfg_get(container, key))
        if cfg_value is not None:
            return cfg_value
    return max(1, int(default))


def _indexed_viz_key(base_key: str, index: int, total: int) -> str:
    if total <= 1:
        return base_key
    return f"{base_key}_{index:02d}"


def _indexed_viz_path(stem: str, step: int, index: int, total: int) -> Path:
    if total <= 1:
        return Path(f"{stem}_step_{step}.png")
    return Path(f"{stem}_step_{step}_sample_{index:02d}.png")


def split_qwen_image_token_attention(
    *,
    input_ids: np.ndarray,
    token_weights: np.ndarray,
    image_grid_thw: np.ndarray,
    image_token_id: int,
    merge_size: int,
) -> list[np.ndarray]:
    """Split flat image-token weights into per-image Qwen-VL grids.

    Qwen-VL stores image pad tokens in the same order as ``image_grid_thw``.
    After patch merging, each image contributes ``t*h*w / merge_size**2`` tokens
    and its spatial attention grid is ``h/merge_size`` by ``w/merge_size``.
    """

    ids = np.asarray(input_ids)
    weights = np.asarray(token_weights, dtype=np.float32).reshape(-1)
    grids = np.asarray(image_grid_thw, dtype=np.int64)
    if grids.ndim == 1:
        grids = grids[None, :]
    positions = np.flatnonzero(ids == int(image_token_id)).tolist()
    if len(weights) < ids.shape[0]:
        raise ValueError(f"token_weights length {len(weights)} < input_ids length {ids.shape[0]}")

    heatmaps: list[np.ndarray] = []
    cursor = 0
    for grid in grids:
        t, h, w = [int(x) for x in grid.tolist()]
        if merge_size <= 0:
            raise ValueError(f"merge_size must be positive, got {merge_size}")
        token_count = (t * h * w) // (merge_size**2)
        gh, gw = h // merge_size, w // merge_size
        pos = positions[cursor : cursor + token_count]
        cursor += token_count
        if len(pos) != gh * gw:
            raise ValueError(
                f"image grid {grid.tolist()} expects {gh * gw} tokens, got {len(pos)} image tokens"
            )
        heatmaps.append(weights[pos].reshape(gh, gw).astype(np.float32))
    return heatmaps


def _patched_sdpa_factory(captured: list[torch.Tensor]):
    def patched_sdpa(
        query,
        key,
        value,
        attn_mask=None,
        dropout_p=0.0,
        is_causal=False,
        scale=None,
        enable_gqa=False,
        **kwargs,
    ):
        head_dim = query.shape[-1]
        attn_scale = scale if scale is not None else (1.0 / math.sqrt(float(head_dim)))
        if query.shape[1] != key.shape[1]:
            if query.shape[1] % key.shape[1] != 0:
                raise ValueError(f"GQA mismatch: q heads {query.shape[1]} % k heads {key.shape[1]} != 0")
            repeat = query.shape[1] // key.shape[1]
            key = key.repeat_interleave(repeat, dim=1)
            value = value.repeat_interleave(repeat, dim=1)

        scores = (query @ key.transpose(-2, -1)) * attn_scale
        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                scores = scores.masked_fill(~attn_mask, float("-inf"))
            else:
                scores = scores + attn_mask
        if is_causal:
            q_len, k_len = query.shape[-2], key.shape[-2]
            causal = torch.ones((q_len, k_len), dtype=torch.bool, device=query.device).tril(
                diagonal=k_len - q_len
            )
            scores = scores.masked_fill(~causal, float("-inf"))
        weights = scores.softmax(dim=-1)
        if dropout_p > 0:
            weights = F.dropout(weights, p=dropout_p)
        captured.append(weights.detach())
        return weights @ value

    return patched_sdpa


def _mean_prefix_token_attention(captured: list[torch.Tensor], prefix_len: int) -> np.ndarray:
    per_layer: list[torch.Tensor] = []
    for weights in captured:
        if weights.ndim != 4 or weights.shape[0] != 1:
            continue
        if weights.shape[-1] < prefix_len:
            continue
        # Action Expert suffix queries are short: one optional state token plus H action tokens.
        if weights.shape[-2] > 64:
            continue
        per_layer.append(weights[0, :, :, :prefix_len].float().mean(dim=0).mean(dim=0).cpu())
    if not per_layer:
        raise RuntimeError(f"no action-expert attention tensors captured; total tensors={len(captured)}")
    return torch.stack(per_layer, dim=0).mean(dim=0).numpy().astype(np.float32)


def _font(size: int = 18):
    for path in (
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            pass
    return ImageFont.load_default()


def _matplotlib_cjk_fontproperties():
    try:
        from matplotlib.font_manager import FontProperties

        for path in (
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.otf",
        ):
            if Path(path).exists():
                return FontProperties(fname=path)
    except Exception:  # noqa: BLE001
        return None
    return None


def _text_width(text: str, font) -> float:
    try:
        return float(font.getlength(text))
    except Exception:  # noqa: BLE001
        bbox = font.getbbox(text)
        return float(bbox[2] - bbox[0])


def _line_height(font, spacing: int = 4) -> int:
    bbox = font.getbbox("Ag")
    return int(bbox[3] - bbox[1] + spacing)


def _wrap_text_pixels(text: str, font, max_width: int) -> list[str]:
    """Wrap text by rendered pixel width, including CJK text without spaces."""

    wrapped: list[str] = []
    for raw_line in str(text).splitlines() or [""]:
        if raw_line == "":
            wrapped.append("")
            continue
        line = ""
        for char in raw_line:
            candidate = line + char
            if not line or _text_width(candidate, font) <= max_width:
                line = candidate
                continue
            wrapped.append(line)
            line = char
        wrapped.append(line)
    return wrapped


def _draw_text_lines(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    lines: list[str],
    *,
    font,
    fill: tuple[int, int, int],
    spacing: int = 4,
) -> int:
    x, y = xy
    line_h = _line_height(font, spacing)
    for line in lines:
        draw.text((x, y), line, fill=fill, font=font)
        y += line_h
    return y


def compact_prompt_from_input_ids(
    input_ids: np.ndarray | list[int],
    tokenizer: Any,
    image_token_id: int,
) -> str:
    """Decode a Qwen-VL prompt while compacting repeated image tokens.

    Processor-expanded prompts can contain hundreds of ``<|image_pad|>`` tokens
    per view. SwanLab then clips the rendered sample. This helper keeps the real
    token order but replaces each contiguous image-token run with one readable
    placeholder.
    """

    ids = np.asarray(input_ids).reshape(-1).astype(np.int64).tolist()
    pieces: list[str] = []
    image_idx = 0
    cursor = 0
    while cursor < len(ids):
        if ids[cursor] == int(image_token_id):
            end = cursor + 1
            while end < len(ids) and ids[end] == int(image_token_id):
                end += 1
            image_idx += 1
            pieces.append(f"<image_pad#{image_idx} tokens={end - cursor}>")
            cursor = end
            continue
        end = cursor + 1
        while end < len(ids) and ids[end] != int(image_token_id):
            end += 1
        pieces.append(tokenizer.decode(ids[cursor:end], skip_special_tokens=False))
        cursor = end
    return "".join(pieces)


def compact_chat_template_prompt(prompt: str) -> str:
    """Replace each chat-template image tag with a numbered placeholder."""

    image_idx = 0

    def _replace(_match: re.Match[str]) -> str:
        nonlocal image_idx
        image_idx += 1
        return f"<|vision_start|><image_pad#{image_idx}><|vision_end|>"

    return re.sub(r"<\|vision_start\|>\s*<\|image_pad\|>\s*<\|vision_end\|>", _replace, str(prompt))


def prompt_mode_label() -> str:
    return "compact View Registry prompt: [view_role|modality] before each image"


def _image_from_any(value: Any) -> Image.Image:
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    arr = np.asarray(value)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr).convert("RGB")


def _heatmap_to_rgb(heatmap: np.ndarray) -> Image.Image:
    arr = np.asarray(heatmap, dtype=np.float32)
    if arr.size == 0:
        arr = np.zeros((1, 1), dtype=np.float32)
    arr = arr - float(np.nanmin(arr))
    denom = float(np.nanmax(arr))
    if denom > 1e-8:
        arr = arr / denom
    arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
    try:
        import matplotlib.cm as cm

        rgb = (cm.get_cmap("turbo")(arr)[..., :3] * 255.0).astype(np.uint8)
    except Exception:  # noqa: BLE001
        rgb = np.stack(
            [
                np.clip(2.0 * arr - 0.2, 0.0, 1.0),
                np.clip(1.8 - np.abs(arr - 0.55) * 3.0, 0.0, 1.0),
                np.clip(1.2 - 2.0 * arr, 0.0, 1.0),
            ],
            axis=-1,
        )
        rgb = (rgb * 255.0).astype(np.uint8)
    return Image.fromarray(rgb, mode="RGB")


def _overlay_heatmap(image: Image.Image, heatmap: np.ndarray, alpha: float = 0.5) -> Image.Image:
    base = image.convert("RGB")
    hm = _heatmap_to_rgb(heatmap).resize(base.size, _RESAMPLE_BILINEAR)
    return Image.blend(base, hm, alpha=float(alpha))


_CAM_DISPLAY_ORDER = ("left", "head", "right")
_CAM_TITLE = {"head": "head", "left": "hand_left", "right": "hand_right"}
_RHINO72_DIM_LABELS = {
    **{i: f"left_arm_{i}" for i in range(7)},
    **{7 + i: f"right_arm_{i}" for i in range(7)},
    14: "left_gripper",
    15: "right_gripper",
    **{16 + i: f"left_hand_thumb_{i}" for i in range(4)},
    **{20 + i: f"left_hand_index_{i}" for i in range(3)},
    **{23 + i: f"left_hand_middle_{i}" for i in range(3)},
    **{26 + i: f"left_hand_ring_{i}" for i in range(3)},
    **{29 + i: f"left_hand_little_{i}" for i in range(3)},
    **{32 + i: f"right_hand_thumb_{i}" for i in range(4)},
    **{36 + i: f"right_hand_index_{i}" for i in range(3)},
    **{39 + i: f"right_hand_middle_{i}" for i in range(3)},
    **{42 + i: f"right_hand_ring_{i}" for i in range(3)},
    **{45 + i: f"right_hand_little_{i}" for i in range(3)},
    48: "head_roll",
    49: "head_pitch",
    50: "head_yaw",
    51: "torso_pitch",
    52: "torso_lift",
    53: "folding_lift_leg_0",
    54: "folding_lift_leg_1",
    55: "waist_roll",
    56: "waist_pitch",
    57: "waist_yaw",
    58: "base_vx",
    59: "base_vy",
    60: "base_wz",
    **{61 + i: f"reserved_{i}" for i in range(11)},
}


def _tf_error_active_slots_and_labels(mask: np.ndarray, *, action_dim: int) -> tuple[list[int], list[str]]:
    mask_np = np.asarray(mask, dtype=np.float32)
    if mask_np.ndim == 2:
        mask_np = mask_np[0]
    mask_np = mask_np.reshape(-1)[: int(action_dim)]
    active = np.flatnonzero(mask_np > 0.5).astype(int).tolist()
    if not active:
        active = list(range(int(action_dim)))
    labels = [_RHINO72_DIM_LABELS.get(int(slot), f"slot_{int(slot)}") for slot in active]
    return active, labels


def _denormalize_tf_error_actions(
    values: np.ndarray,
    *,
    action_mean: np.ndarray,
    action_std: np.ndarray,
) -> np.ndarray:
    """Convert TF-error values from model-normalized to native action units."""
    values_np = np.asarray(values, dtype=np.float32)
    mean_np = np.asarray(action_mean, dtype=np.float32).reshape(-1)
    std_np = np.asarray(action_std, dtype=np.float32).reshape(-1)
    if values_np.shape[-1] != mean_np.shape[0] or mean_np.shape != std_np.shape:
        raise ValueError(
            "TF-error action stats do not match values: "
            f"values={values_np.shape}, mean={mean_np.shape}, std={std_np.shape}"
        )
    return (values_np * std_np + mean_np).astype(np.float32)


_VIZ_NORM_KEYS = ("state_mean", "state_std", "action_mean", "action_std")


def _mixture_child_index_for_viz_sample(
    dataset: Any,
    sample: Mapping[str, Any],
) -> int | None:
    """Resolve the child that owns a mixture sample's raw episode rows."""
    children = getattr(dataset, "datasets", None)
    if children is None:
        return None
    raw_index = sample.get("_mixture_dataset_index")
    if raw_index is None:
        raise ValueError("mixture visualization sample is missing dataset child index")
    if isinstance(raw_index, bool):
        raise ValueError(f"invalid mixture dataset child index {raw_index!r}")
    if isinstance(raw_index, (int, np.integer)):
        child_index = int(raw_index)
    else:
        raise ValueError(f"invalid mixture dataset child index {raw_index!r}")

    if child_index < 0 or child_index >= len(children):
        raise ValueError(f"mixture dataset child index {child_index} is out of range")
    return child_index


def _viz_dataset_for_sample(dataset: Any, sample: Mapping[str, Any]) -> Any:
    """Route a mixture sample to the child owning its raw episode rows."""

    child_index = _mixture_child_index_for_viz_sample(dataset, sample)
    return dataset if child_index is None else dataset.datasets[child_index]


def _norm_stats_for_viz_sample(dataset: Any) -> dict[str, np.ndarray]:
    """Resolve the global norm shared by all visualization samples."""
    resolved: dict[str, np.ndarray] = {}
    for name in _VIZ_NORM_KEYS:
        value = getattr(dataset, name, None)
        if value is None:
            raise ValueError(f"visualization norm is missing {name}")
        array = np.asarray(value, dtype=np.float32).reshape(-1)
        if array.size == 0 or not np.all(np.isfinite(array)):
            raise ValueError(f"visualization norm {name} must be finite and non-empty")
        resolved[name] = array

    if resolved["state_mean"].shape != resolved["state_std"].shape:
        raise ValueError("visualization state norm mean/std shapes do not match")
    if resolved["action_mean"].shape != resolved["action_std"].shape:
        raise ValueError("visualization action norm mean/std shapes do not match")
    if np.any(resolved["state_std"] <= 0) or np.any(resolved["action_std"] <= 0):
        raise ValueError("visualization norm std values must be positive")
    return resolved


def _delta_action_slots_for_viz_sample(dataset: Any, sample: Mapping[str, Any]) -> list[int]:
    """Return action semantics from the mixture child that produced a sample."""

    child = _viz_dataset_for_sample(dataset, sample)
    if child is dataset:
        return [int(slot) for slot in getattr(dataset, "delta_action_slots", [])]
    if not hasattr(child, "delta_action_slots"):
        raise ValueError("mixture visualization child is missing delta_action_slots")
    return [int(slot) for slot in child.delta_action_slots]


def _tf_error_actions_to_absolute_units(
    *,
    gt_action: np.ndarray,
    gt_reference_states: np.ndarray,
    pred_segments: list[tuple[int, np.ndarray]],
    delta_slots: Sequence[int],
) -> tuple[np.ndarray, list[tuple[int, np.ndarray]]]:
    """Restore comparable absolute GT/actions for TF-error visualization.

    Episode GT rows each use their own row state as the delta reference. Each
    prediction chunk instead uses the state at that chunk's start ``t0``.
    Gripper and action-only command slots are absent from ``delta_slots`` and
    therefore remain in their original native units.
    """
    gt_abs = np.asarray(gt_action, dtype=np.float32).copy()
    states = np.asarray(gt_reference_states, dtype=np.float32)
    if gt_abs.ndim != 2:
        raise ValueError(f"TF-error GT action must be 2D, got {gt_abs.shape}")
    if states.ndim == 1:
        states = np.broadcast_to(states[None, :], gt_abs.shape)
    if states.ndim != 2 or states.shape != gt_abs.shape:
        raise ValueError(
            f"TF-error GT reference states {states.shape} must match actions {gt_abs.shape}"
        )
    slots = np.asarray(sorted({int(slot) for slot in delta_slots}), dtype=np.int64)
    if slots.size:
        max_slot = int(slots.max())
        if int(slots.min()) < 0 or max_slot >= gt_abs.shape[1]:
            raise ValueError(f"TF-error delta slots {slots.tolist()} outside dim {gt_abs.shape[1]}")
        gt_abs[:, slots] += states[:, slots]

    pred_abs_segments: list[tuple[int, np.ndarray]] = []
    for t0, segment in pred_segments:
        start = int(t0)
        if start < 0 or start >= len(states):
            raise ValueError(f"TF-error prediction start {start} outside {len(states)} GT rows")
        segment_abs = np.asarray(segment, dtype=np.float32).copy()
        if segment_abs.ndim != 2 or segment_abs.shape[1] != gt_abs.shape[1]:
            raise ValueError(
                f"TF-error prediction segment {segment_abs.shape} must have dim {gt_abs.shape[1]}"
            )
        if slots.size:
            segment_abs[:, slots] += states[start, slots][None, :]
        pred_abs_segments.append((start, segment_abs))
    return gt_abs, pred_abs_segments


def _cv2_module():
    try:
        import cv2

        return cv2
    except Exception:  # noqa: BLE001
        return None


def _canonical_view_key(label: Any, fallback_index: int = 0) -> str:
    text = str(label or "").lower()
    if "left" in text:
        return "left"
    if "right" in text:
        return "right"
    if "head" in text or "top" in text or "front" in text or "chest" in text:
        return "head"
    return f"view{fallback_index}"


def _normalize_attn(attn: np.ndarray) -> np.ndarray:
    arr = np.asarray(attn, dtype=np.float32)
    if arr.size == 0:
        return arr
    lo = float(np.nanmin(arr))
    hi = float(np.nanmax(arr))
    if hi > lo:
        arr = (arr - lo) / (hi - lo + 1e-9)
    else:
        arr = np.zeros_like(arr, dtype=np.float32)
    return np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)


def _token_grid_lines_for_image(
    *,
    image_size: tuple[int, int],
    token_grid: tuple[int, int],
) -> tuple[list[int], list[int]]:
    width, height = int(image_size[0]), int(image_size[1])
    grid_h, grid_w = int(token_grid[0]), int(token_grid[1])
    if width <= 0 or height <= 0 or grid_w <= 0 or grid_h <= 0:
        return [], []
    xs = [int(round(k * width / grid_w)) for k in range(grid_w + 1)]
    ys = [int(round(k * height / grid_h)) for k in range(grid_h + 1)]
    xs = sorted(set(max(0, min(width - 1, x)) for x in xs))
    ys = sorted(set(max(0, min(height - 1, y)) for y in ys))
    return xs, ys


def _draw_token_grid(
    rgb: np.ndarray,
    grid_lines: tuple[list[int], list[int]] | None,
) -> np.ndarray:
    if not grid_lines:
        return rgb
    out = np.asarray(rgb).copy()
    height, width = out.shape[:2]
    xs, ys = grid_lines
    mask = np.zeros((height, width), dtype=bool)
    mask[:, [max(0, min(width - 1, int(x))) for x in xs]] = True
    mask[[max(0, min(height - 1, int(y))) for y in ys], :] = True
    out[mask] = (0.78 * out[mask].astype(np.float32) + 0.22 * 255.0).clip(0, 255).astype(np.uint8)
    return out


def _overlay_reference_cell(
    img: Image.Image,
    attn: np.ndarray | None,
    label: str,
    *,
    cell_w: int | None = None,
    token_grid_lines: tuple[list[int], list[int]] | None = None,
) -> Image.Image:
    rgb = np.asarray(img.convert("RGB"))
    height, width = rgb.shape[:2]
    if attn is None:
        overlay = rgb
    else:
        arr = _normalize_attn(attn)
        cv2 = _cv2_module()
        if cv2 is not None:
            up = cv2.resize(arr, (width, height), interpolation=cv2.INTER_CUBIC)
            heat = cv2.applyColorMap((up * 255).clip(0, 255).astype(np.uint8), cv2.COLORMAP_JET)
            heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
        else:
            up = np.asarray(Image.fromarray(arr).resize((width, height), _RESAMPLE_BICUBIC)).clip(0.0, 1.0)
            heat = np.asarray(_heatmap_to_rgb(up))
        heat = (
            (1.0 - ATTENTION_HEATMAP_LIGHTEN) * heat.astype(np.float32)
            + ATTENTION_HEATMAP_LIGHTEN * 255.0
        ).clip(0, 255)
        alpha = (ATTENTION_HEATMAP_ALPHA_MAX * np.sqrt(up.clip(0.0, 1.0)))[..., None]
        overlay = ((1.0 - alpha) * rgb.astype(np.float32) + alpha * heat).clip(0, 255).astype(np.uint8)
    overlay = _draw_token_grid(overlay, token_grid_lines)
    header_h = 36
    canvas = Image.new("RGB", (width, height + header_h), (0, 0, 0))
    canvas.paste(Image.fromarray(overlay), (0, header_h))
    ImageDraw.Draw(canvas).text((8, 8), label, fill=(255, 255, 255), font=_font(15))
    if cell_w and canvas.width != cell_w:
        out = Image.new("RGB", (cell_w, canvas.height), (32, 32, 32))
        out.paste(canvas, ((cell_w - canvas.width) // 2, 0))
        return out
    return canvas


def _hstack(cells: list[Image.Image]) -> Image.Image:
    height = max(c.height for c in cells)
    width = sum(c.width for c in cells)
    out = Image.new("RGB", (width, height), (32, 32, 32))
    x = 0
    for cell in cells:
        out.paste(cell, (x, 0))
        x += cell.width
    return out


def _vstack(rows: list[Image.Image]) -> Image.Image:
    width = max(r.width for r in rows)
    height = sum(r.height for r in rows)
    out = Image.new("RGB", (width, height), (32, 32, 32))
    y = 0
    for row in rows:
        out.paste(row, ((width - row.width) // 2, y))
        y += row.height
    return out


def build_vlm_prompt_sample_panel(
    *,
    images: list[Any],
    labels: list[str],
    prompt: str | None = None,
    chat_prompt: str | None = None,
    tokenized_prompt: str | None = None,
    prompt_mode: str | None = None,
    title: str = "vlm_prompt/sample",
) -> Image.Image:
    """Render model-input images plus the compact tokenized VLM prompt."""

    source_images = [_image_from_any(img) for img in images]
    n = max(1, len(source_images))
    labels = (list(labels) + [f"view{i}" for i in range(len(labels), n)])[:n]
    margin = 28
    gap = 18
    col_w = 520 if n >= 3 else 640
    thumb_h = 330
    title_h = 66
    label_h = 28
    prompt_pad = 18
    width = max(margin * 2 + col_w * n + gap * (n - 1), 1100)
    prompt_font = _font(17)
    title_font = _font(26)
    label_font = _font(17)
    mono_font = _font(15)
    if chat_prompt is None:
        chat_prompt = prompt or ""
    if tokenized_prompt is None:
        tokenized_prompt = prompt or chat_prompt
    mode_line = f"mode: {prompt_mode}" if prompt_mode else "mode: unknown"
    token_lines = _wrap_text_pixels(str(tokenized_prompt), mono_font, width - margin * 2 - prompt_pad * 2)
    section_title_h = _line_height(prompt_font, 4) + 8
    line_h = _line_height(mono_font, 5)
    prompt_h = (
        prompt_pad * 2
        + section_title_h
        + max(1, len(token_lines)) * line_h
    )
    height = title_h + label_h + thumb_h + 24 + prompt_h + margin

    canvas = Image.new("RGB", (width, height), (248, 250, 252))
    draw = ImageDraw.Draw(canvas)
    draw.text((margin, 20), title, fill=(24, 28, 33), font=title_font)
    draw.text((margin + 310, 26), mode_line, fill=(82, 94, 106), font=prompt_font)

    x = margin
    top_y = title_h + label_h
    for image, label in zip(source_images, labels):
        draw.text((x, title_h), f"{label} | {image.width}x{image.height}", fill=(93, 64, 0), font=label_font)
        thumb = _fit_image(image, (col_w, thumb_h), fill=(230, 234, 238))
        canvas.paste(thumb, (x, top_y))
        draw.rectangle((x, top_y, x + col_w - 1, top_y + thumb_h - 1), outline=(184, 193, 204), width=1)
        x += col_w + gap

    prompt_y = top_y + thumb_h + 24
    draw.rectangle(
        (margin, prompt_y, width - margin, prompt_y + prompt_h),
        fill=(255, 255, 255),
        outline=(206, 214, 224),
        width=1,
    )
    draw.text(
        (margin + prompt_pad, prompt_y + prompt_pad - 2),
        "tokenized prompt after processor (compact image-token runs)",
        fill=(82, 94, 106),
        font=prompt_font,
    )
    text_y = prompt_y + prompt_pad + section_title_h
    _draw_text_lines(
        draw,
        (margin + prompt_pad, text_y),
        token_lines,
        font=mono_font,
        fill=(24, 28, 33),
        spacing=5,
    )
    return canvas


def _fit_image(image: Image.Image, size: tuple[int, int], fill: tuple[int, int, int] = (5, 6, 7)) -> Image.Image:
    image = image.convert("RGB")
    box_w, box_h = size
    scale = min(box_w / max(1, image.width), box_h / max(1, image.height))
    new_size = (max(1, int(round(image.width * scale))), max(1, int(round(image.height * scale))))
    resized = image.resize(new_size, _RESAMPLE_BICUBIC)
    canvas = Image.new("RGB", size, fill)
    canvas.paste(resized, ((box_w - resized.width) // 2, (box_h - resized.height) // 2))
    return canvas


def build_action_expert_attention_panel(
    *,
    model_images: list[Any] | None = None,
    raw_images: list[Any] | None = None,
    source_images: list[Any] | None = None,
    heatmaps: list[np.ndarray],
    labels: list[str] | None = None,
    tag: str = "RhinoVLA",
    run_id: str = "native72",
    step: int | str = 0,
    framework: str = "RhinoVLA",
    ckpt_path: str = "",
    subtitle: str = "",
) -> Image.Image:
    """Standard RhinoVLA action-expert attention visualization.

    The top row uses raw-resolution views and the bottom row uses the dynamic
    model-input geometry. Both preserve image aspect ratio without padding.
    """

    if model_images is None:
        model_images = source_images
    if model_images is None:
        raise TypeError("build_action_expert_attention_panel requires model_images or source_images")
    model = [_image_from_any(img) for img in list(model_images)[: len(heatmaps)]]
    raw = [_image_from_any(img) for img in list(raw_images or model)[: len(heatmaps)]]
    labels = list(labels or [f"view{i}" for i in range(len(model))])[: len(model)]
    if len(raw) < len(model):
        raw.extend(model[len(raw) :])

    entries: dict[str, dict[str, Any]] = {}
    for idx, (label, model_img, raw_img, heatmap) in enumerate(zip(labels, model, raw, heatmaps)):
        key = _canonical_view_key(label, idx)
        entries[key] = {
            "label": str(label),
            "title": _CAM_TITLE.get(key, str(label)),
            "model": model_img,
            "raw": raw_img,
            "attn": np.asarray(heatmap, dtype=np.float32),
        }

    ordered_keys = [key for key in _CAM_DISPLAY_ORDER if key in entries]
    ordered_keys += [key for key in entries if key not in ordered_keys]
    raw_cells: list[Image.Image] = []
    model_cells: list[Image.Image] = []
    for key in ordered_keys:
        entry = entries[key]
        attn = entry["attn"]
        model_img = entry["model"]
        raw_img = entry["raw"]
        token_grid = tuple(int(x) for x in np.asarray(attn).shape[:2])
        attn_content = attn
        raw_grid_lines = _token_grid_lines_for_image(
            image_size=raw_img.size,
            token_grid=token_grid,
        )
        model_grid_lines = _token_grid_lines_for_image(image_size=model_img.size, token_grid=token_grid)
        raw_cells.append(
            _overlay_reference_cell(
                raw_img,
                attn_content,
                f"{tag} {entry['title']} raw {raw_img.width}x{raw_img.height} grid={token_grid}",
                token_grid_lines=raw_grid_lines,
            )
        )
        model_cells.append(
            _overlay_reference_cell(
                model_img,
                attn,
                f"{tag} {entry['title']} model-input {model_img.width}x{model_img.height} grid={token_grid}",
                token_grid_lines=model_grid_lines,
            )
        )

    title_h = 54
    body = _vstack([_hstack(raw_cells), _hstack(model_cells)])
    canvas = Image.new("RGB", (body.width, body.height + title_h), (24, 24, 24))
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (12, 8),
        f"{tag} {run_id} step={step} | {framework} | dynamic-aspect | compact-prompt",
        fill=(255, 255, 255),
        font=_font(18),
    )
    second = f"ckpt: {ckpt_path}" if ckpt_path else str(subtitle or "")
    if second:
        draw.text((12, 31), second, fill=(220, 220, 220), font=_font(13))
    canvas.paste(body, (0, title_h))
    return canvas


def _collect_subtask_bounds(payloads: list[dict[str, Any]]) -> list[tuple[int, int, str]]:
    if not payloads:
        return []
    bounds: list[tuple[int, int, str]] = []
    cur_start = 0
    cur_key = payloads[0].get("subtask", "unknown")
    for idx in range(1, len(payloads)):
        key = payloads[idx].get("subtask", "unknown")
        if key != cur_key:
            bounds.append((cur_start, idx, str(cur_key)))
            cur_start = idx
            cur_key = key
    bounds.append((cur_start, len(payloads), str(cur_key)))
    return bounds


def _collect_segment_bounds(payloads: list[dict[str, Any]]) -> list[tuple[int, int, str]]:
    if not payloads:
        return []
    bounds: list[tuple[int, int, str]] = []
    cur_start = 0
    cur_key = (
        payloads[0].get("segment_index", -1),
        payloads[0].get("segment_action_text", "") or payloads[0].get("subtask", "unknown"),
    )
    for idx in range(1, len(payloads)):
        key = (
            payloads[idx].get("segment_index", -1),
            payloads[idx].get("segment_action_text", "") or payloads[idx].get("subtask", "unknown"),
        )
        if key != cur_key:
            bounds.append((cur_start, idx, str(cur_key[1])))
            cur_start = idx
            cur_key = key
    bounds.append((cur_start, len(payloads), str(cur_key[1])))
    return bounds


def _chunk_start_positions(num_frames: int, chunk_len: int, max_chunks: int = 0) -> list[int]:
    if num_frames <= 0:
        return []
    chunk_len = max(1, int(chunk_len))
    if num_frames <= chunk_len:
        return [0]
    positions = list(range(0, num_frames - chunk_len + 1, chunk_len))
    terminal = num_frames - chunk_len
    if terminal not in positions:
        positions.append(terminal)
    positions = sorted(set(int(p) for p in positions))
    if max_chunks > 0 and len(positions) > max_chunks:
        keep = np.linspace(0, len(positions) - 1, max_chunks, dtype=int).tolist()
        positions = [positions[int(i)] for i in sorted(set(int(i) for i in keep))]
        if terminal not in positions:
            positions[-1] = terminal
        positions = sorted(set(int(p) for p in positions))
    return positions


def _expand_terminal_display_indices(
    dataset: Any,
    valid_start_indices: list[int],
    chunk_len: int,
    max_frames: int,
) -> list[int]:
    """Expand legal chunk starts into contiguous episode frames for TF plots."""

    if not valid_start_indices:
        return []
    first = int(valid_start_indices[0])
    last_exclusive = int(valid_start_indices[-1]) + int(chunk_len)
    if max_frames > 0:
        last_exclusive = min(last_exclusive, first + int(max_frames))
    first_payload = dataset.get_row_payload(first)
    episode_index = int(first_payload.get("episode_index", -1))
    expected_frame = int(first_payload.get("frame_index", first))
    out: list[int] = []
    for global_i in range(first, last_exclusive):
        payload = dataset.get_row_payload(int(global_i))
        frame = int(payload.get("frame_index", global_i))
        if int(payload.get("episode_index", -999999)) != episode_index:
            break
        if out and frame != expected_frame:
            break
        out.append(int(global_i))
        expected_frame = frame + 1
    return out


def build_tf_error_episode_panel(
    *,
    frame_ids: np.ndarray,
    gt_action: np.ndarray,
    pred_segments: list[tuple[int, np.ndarray]],
    bounds: list[tuple[int, int, str]] | None = None,
    dataset_index: int | None = None,
    episode_index: int | None = None,
    episode_id: str = "",
    chunk_len: int = 30,
    dim_labels: list[str] | None = None,
    title_prefix: str = "TF chunk inference",
    units_label: str = "canonical units",
) -> Image.Image:
    """Plot episode-level TF-error with the same style as the reference HTML."""

    import matplotlib.pyplot as plt

    frame_np = np.asarray(frame_ids, dtype=np.int64).reshape(-1)
    gt_np = np.asarray(gt_action, dtype=np.float32)
    if gt_np.ndim != 2:
        raise ValueError(f"gt_action must be 2D, got {gt_np.shape}")
    if frame_np.shape[0] != gt_np.shape[0]:
        raise ValueError(f"frame_ids length {frame_np.shape[0]} != gt_action rows {gt_np.shape[0]}")
    num_dims = int(gt_np.shape[1])
    labels = list(dim_labels or [_RHINO72_DIM_LABELS.get(i, f"slot_{i}") for i in range(num_dims)])
    if len(labels) < num_dims:
        labels += [_RHINO72_DIM_LABELS.get(i, f"slot_{i}") for i in range(len(labels), num_dims)]
    bounds = list(bounds or [])

    palette = plt.cm.tab10.colors
    cjk_font = _matplotlib_cjk_fontproperties()
    fig, axes = plt.subplots(num_dims, 1, figsize=(14, max(10, num_dims * 1.05)), sharex=True)
    axes = np.atleast_1d(axes)
    num_frames = int(gt_np.shape[0])
    for dim in range(num_dims):
        ax = axes[dim]
        color = palette[dim % len(palette)]
        ax.plot(frame_np, gt_np[:, dim], color=color, alpha=0.40, linewidth=2.0, label="GT action")
        for t0, segment in pred_segments:
            seg = np.asarray(segment, dtype=np.float32)
            if seg.ndim != 2 or dim >= seg.shape[1]:
                continue
            local_ts = np.arange(int(t0), int(t0) + len(seg))
            mask = local_ts < num_frames
            if not np.any(mask):
                continue
            ax.plot(frame_np[local_ts[mask]], seg[: int(mask.sum()), dim], color=color, alpha=1.0, linewidth=1.0)
            ax.axvline(int(frame_np[int(t0)]), color="gray", alpha=0.15, linewidth=0.5)
            ax.scatter(
                [int(frame_np[int(t0)])],
                [gt_np[int(t0), dim]],
                color=color,
                s=14,
                zorder=3,
                edgecolor="black",
                linewidth=0.4,
            )
        for idx, (s0, s1, key) in enumerate(bounds):
            if idx > 0 and 0 <= s0 < num_frames:
                ax.axvline(int(frame_np[s0]), color="black", linewidth=1.6, alpha=0.85, zorder=4)
            if dim == 0 and len(bounds) > 1 and 0 <= s0 < s1 <= num_frames:
                ax.text(
                    int((frame_np[s0] + frame_np[s1 - 1]) / 2),
                    1.02,
                    str(key),
                    ha="center",
                    va="bottom",
                    fontsize=10,
                    fontweight="bold",
                    color="#222",
                    transform=ax.get_xaxis_transform(),
                    bbox=dict(boxstyle="round,pad=0.22", facecolor="#ffe082", edgecolor="#444", linewidth=0.6),
                    zorder=5,
                    clip_on=False,
                    fontproperties=cjk_font,
                )
        ax.set_ylabel(labels[dim], fontsize=8, rotation=0, ha="right", va="center", labelpad=36)
        ax.grid(alpha=0.2)

    title_y = 1.06 if len(bounds) > 1 else 1.02
    title_prefix = _PARENTHETICAL_TITLE_RE.sub("", str(title_prefix)).strip()
    fig.suptitle(
        f"{title_prefix} — dataset_index={dataset_index} episode_index={episode_index}, episode_id={episode_id}",
        y=title_y,
        fontsize=12,
    )
    axes[-1].set_xlabel("frame")
    legend_handles = [
        plt.Line2D([], [], color="gray", alpha=0.40, linewidth=2.0, label="GT action"),
        plt.Line2D([], [], color="gray", alpha=1.0, linewidth=1.0, label="Pred action chunk"),
        plt.Line2D(
            [],
            [],
            color="gray",
            marker="o",
            linestyle="",
            markersize=5,
            markeredgecolor="black",
            markeredgewidth=0.4,
            label="chunk start (= GT action[t0])",
        ),
    ]
    axes[0].legend(handles=legend_handles, loc="upper right", fontsize=9)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    image = Image.open(buf).convert("RGB")
    return image.copy()


class _VizMixin:
    """Native72 visual diagnostics for VLATrainer."""

    def _visual_dir(self) -> Path:
        path = Path(self.config.output_dir) / "visualizations"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _viz_count(self, key: str, env_key: str) -> int:
        return _config_int(getattr(self, "config", None), key, env_key, 1)

    def _viz_sample_at(self, dataset, idx: int) -> dict[str, Any]:
        sample = dict(dataset[idx])
        sample["_dataset_sample_index"] = idx
        source_dataset = _viz_dataset_for_sample(dataset, sample)
        local_index = int(sample.get("_mixture_local_index", idx))
        valid = getattr(source_dataset, "valid_global_indices", None)
        if valid is not None and local_index < len(valid):
            sample["_global_row_index"] = int(valid[local_index])
        return sample

    def _viz_samples(
        self,
        count: int = 1,
        *,
        unique_episode: bool = False,
    ) -> list[dict[str, Any]]:
        dataset = getattr(self.vla_train_dataloader, "dataset", None)
        if dataset is None or len(dataset) == 0:
            return []
        try:
            start_idx = int(os.environ.get("RHINOVLA_VIZ_DATASET_INDEX", "0"))
        except ValueError:
            start_idx = 0
        start_idx = max(0, min(start_idx, len(dataset) - 1))
        wanted = max(1, int(count))
        samples: list[dict[str, Any]] = []
        seen_episodes: set[tuple[int | None, int]] = set()
        indices = (
            np.random.default_rng(int(self.config.seed) + int(self.completed_steps)).permutation(len(dataset))
            if unique_episode
            else ((start_idx + offset) % len(dataset) for offset in range(len(dataset)))
        )
        for idx in indices:
            idx = int(idx)
            sample = self._viz_sample_at(dataset, idx)
            if unique_episode:
                global_row = sample.get("_global_row_index")
                episode_index = None
                source_dataset = _viz_dataset_for_sample(dataset, sample)
                if global_row is not None and hasattr(source_dataset, "get_row_payload"):
                    try:
                        episode_index = int(
                            source_dataset.get_row_payload(int(global_row)).get("episode_index", idx)
                        )
                    except Exception:  # noqa: BLE001
                        episode_index = idx
                child_index = _mixture_child_index_for_viz_sample(dataset, sample)
                episode_key = (child_index, episode_index)
                if episode_key in seen_episodes:
                    continue
                if episode_index is not None:
                    seen_episodes.add(episode_key)
            samples.append(sample)
            if len(samples) >= wanted:
                break
        return samples

    def _viz_sample(self) -> dict[str, Any] | None:
        samples = self._viz_samples(1)
        return samples[0] if samples else None

    def _unwrap_model(self):
        return self.accelerator.unwrap_model(self.model)

    def _qwen_inputs_for_sample(self, raw_model, sample: dict[str, Any]):
        batch_images, instructions, view_roles, view_modalities = raw_model._prepare_batch([sample])
        qwen_inputs = raw_model.qwen.build_qwenvl_inputs(
            images=batch_images,
            instructions=instructions,
            view_roles=view_roles,
            view_modalities=view_modalities,
        )
        return qwen_inputs, batch_images, instructions

    def _prompt_payload_for_sample(self, raw_model, sample: dict[str, Any]):
        batch_images, instructions, view_roles, view_modalities = raw_model._prepare_batch([sample])
        messages = raw_model.qwen._build_qwenvl_messages(
            images=batch_images,
            instructions=instructions,
            view_roles=view_roles,
            view_modalities=view_modalities,
        )
        chat_prompt = raw_model.qwen.processor.apply_chat_template(
            messages[0],
            tokenize=False,
            add_generation_prompt=True,
        )
        qwen_inputs = raw_model.qwen.build_qwenvl_inputs(
            images=batch_images,
            instructions=instructions,
            view_roles=view_roles,
            view_modalities=view_modalities,
        )
        return qwen_inputs, batch_images, instructions, compact_chat_template_prompt(chat_prompt)

    def _decode_prompt(self, raw_model, sample: dict[str, Any]) -> str:
        qwen_inputs, _, _ = self._qwen_inputs_for_sample(raw_model, sample)
        tokenizer = raw_model.qwen.processor.tokenizer
        image_token_id = int(tokenizer.convert_tokens_to_ids("<|image_pad|>"))
        return compact_prompt_from_input_ids(
            qwen_inputs["input_ids"][0].detach().cpu().numpy(),
            tokenizer,
            image_token_id,
        )

    def _run_id_for_viz(self) -> str:
        return str(getattr(self.config, "run_id", "native72"))

    def _ckpt_path_for_viz(self) -> str:
        try:
            return str(self.config.trainer.get("pretrained_checkpoint", "") or "")
        except Exception:  # noqa: BLE001
            return ""

    def _raw_images_for_sample(self, sample: dict[str, Any]) -> list[Image.Image] | None:
        dataset = getattr(self.vla_train_dataloader, "dataset", None)
        global_row = sample.get("_global_row_index")
        if dataset is None or global_row is None:
            return None
        dataset = _viz_dataset_for_sample(dataset, sample)
        if not hasattr(dataset, "decode_frame"):
            return None
        try:
            return [_image_from_any(img) for img in dataset.decode_frame(int(global_row))]
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"raw frame decode for attention skipped: {exc}")
            return None

    def _viz_sample_identity(self, sample: dict[str, Any]) -> str:
        dataset = getattr(self.vla_train_dataloader, "dataset", None)
        source_id = str(sample.get("_mixture_source_id", "single"))
        global_row = sample.get("_global_row_index")
        episode = "?"
        frame = "?"
        if dataset is not None and global_row is not None:
            source_dataset = _viz_dataset_for_sample(dataset, sample)
            if hasattr(source_dataset, "get_row_payload"):
                try:
                    payload = source_dataset.get_row_payload(int(global_row))
                    episode = str(payload.get("episode_index", "?"))
                    frame = str(payload.get("frame_index", "?"))
                except Exception:  # noqa: BLE001
                    pass
        prompt = str(sample.get("lang", "")).replace("\n", " ").strip()
        return (
            f"source={source_id}, episode={episode}, frame={frame}, "
            f"row={global_row}, prompt={prompt}"
        )

    def _log_swanlab_image(self, key: str, path: Path, *, step: int) -> None:
        if not (HAS_SWANLAB and self._swanlab_enabled):
            return
        try:
            swanlab.log({key: swanlab.Image(str(path))}, step=step)
        except Exception as exc:  # noqa: BLE001
            failures = int(getattr(self, "_swanlab_image_log_failures", 0)) + 1
            self._swanlab_image_log_failures = failures
            if failures == 1 or failures % 20 == 0:
                logger.warning(f"SwanLab image upload failed (count={failures}): {exc}")

    def _episode_tf_error_panel(self, raw_model, sample: dict[str, Any]) -> Image.Image | None:
        dataset = getattr(self.vla_train_dataloader, "dataset", None)
        global_row = sample.get("_global_row_index")
        if dataset is None or global_row is None:
            return None
        dataset = _viz_dataset_for_sample(dataset, sample)
        if not all(hasattr(dataset, name) for name in ("get_row_payload", "decode_frame")):
            return None

        payload0 = dataset.get_row_payload(int(global_row))
        episode_index = int(payload0.get("episode_index", -1))
        valid = list(getattr(dataset, "valid_global_indices", []) or [])
        valid_start_indices = [
            int(row)
            for row in valid
            if int(dataset.get_row_payload(int(row)).get("episode_index", -999999)) == episode_index
        ]
        valid_start_indices = sorted(set(valid_start_indices))
        if not valid_start_indices:
            return None
        chunk_len = int(getattr(raw_model, "action_horizon", sample.get("action", np.zeros((30, 1))).shape[0]))
        # A TF episode report is intentionally complete: never truncate its
        # chronological frame range, even when the episode is longer than the
        # historical visualization cap.
        row_indices = _expand_terminal_display_indices(dataset, valid_start_indices, chunk_len, 0)
        if not row_indices:
            return None

        payloads = [dataset.get_row_payload(row) for row in row_indices]
        frame_ids = np.asarray([int(p.get("frame_index", i)) for i, p in enumerate(payloads)], dtype=np.int64)
        action_raw = np.stack([np.asarray(p["action_raw"], dtype=np.float32) for p in payloads], axis=0)
        state_raw_rows = np.stack([np.asarray(p["state_raw"], dtype=np.float32) for p in payloads], axis=0)
        first_mask = np.asarray(sample.get("action_mask", np.ones((1, action_raw.shape[-1]))), dtype=np.float32)
        if first_mask.ndim == 2:
            first_mask = first_mask[0]
        active, dim_labels = _tf_error_active_slots_and_labels(first_mask, action_dim=action_raw.shape[-1])

        norm_stats = _norm_stats_for_viz_sample(dataset)
        state_mean = norm_stats["state_mean"]
        state_std = norm_stats["state_std"]
        action_mean = norm_stats["action_mean"]
        action_std = norm_stats["action_std"]
        state_mask = np.asarray(sample.get("state_mask", np.ones((1, state_mean.shape[0]))), dtype=np.float32)
        action_mask = np.asarray(sample.get("action_mask", np.ones((chunk_len, action_mean.shape[0]))), dtype=np.float32)
        if action_mask.ndim == 1:
            action_mask = np.broadcast_to(action_mask[None, :], (chunk_len, action_mean.shape[0])).copy()
        pred_segments: list[tuple[int, np.ndarray]] = []
        fallback_prompt = str(sample.get("lang") or payload0.get("subtask_prompt") or "")
        fixed_instruction = getattr(dataset, "fixed_instruction", None)
        camera_keys = list(getattr(dataset, "camera_source_keys", []))
        row_positions = {int(row): pos for pos, row in enumerate(row_indices)}
        valid_chunk_positions = {
            row_positions[int(row)]
            for row in valid_start_indices
            if int(row) in row_positions
        }
        chunk_positions = [
            position
            for position in _chunk_start_positions(len(row_indices), chunk_len)
            if position in valid_chunk_positions
        ]
        config = getattr(self, "config", None)
        datasets_cfg = getattr(config, "datasets", None)
        vla_cfg = getattr(datasets_cfg, "vla_data", {})
        batch_size = int(vla_cfg.get("per_device_batch_size", 1))
        if batch_size <= 0:
            raise ValueError(f"per_device_batch_size must be positive, got {batch_size}")
        pending_examples: list[dict[str, Any]] = []
        pending_positions: list[int] = []

        def flush_pending() -> None:
            if not pending_examples:
                return
            with torch.no_grad():
                pred = raw_model.predict_action(pending_examples)
            pred_np_batch = pred.detach().float().cpu().numpy()
            for position, pred_np in zip(pending_positions, pred_np_batch):
                pred_raw = _denormalize_tf_error_actions(
                    pred_np,
                    action_mean=action_mean,
                    action_std=action_std,
                )
                pred_segments.append((int(position), pred_raw))
            pending_examples.clear()
            pending_positions.clear()

        for t0 in chunk_positions:
            row = int(row_indices[t0])
            payload = payloads[t0]
            raw_images = dataset.decode_frame(row)
            roles = (
                getattr(dataset, "view_roles", None)
                or getattr(dataset, "camera_order", None)
                or sample.get("view_roles")
                or []
            )
            model_images = []
            for image_index, image in enumerate(raw_images):
                if camera_keys and hasattr(dataset, "_select_image"):
                    model_image = dataset._select_image(image)  # noqa: SLF001
                else:
                    model_image = _image_from_any(image)
                if hasattr(dataset, "_apply_top_head_crop"):
                    role = roles[image_index] if image_index < len(roles) else None
                    model_image = dataset._apply_top_head_crop(model_image, role)  # noqa: SLF001
                model_images.append(model_image)
            state_raw = np.asarray(payload["state_raw"], dtype=np.float32)
            state_norm = (state_raw - state_mean) / state_std
            example = {
                "image": model_images,
                # Fixed-instruction recipes must reuse the exact training
                # prompt for every chunk; legacy segment-prompt recipes retain
                # their per-row instruction semantics.
                "lang": (
                    fallback_prompt
                    if fixed_instruction
                    else str(payload.get("subtask_prompt") or fallback_prompt)
                ),
                "state": state_norm[None, :],
                "state_mask": state_mask,
                "action": np.zeros((chunk_len, action_mean.shape[0]), dtype=np.float32),
                "action_mask": action_mask,
                "view_roles": sample.get("view_roles"),
                "view_modalities": sample.get("view_modalities"),
            }
            pending_examples.append(example)
            pending_positions.append(int(t0))
            if len(pending_examples) >= batch_size:
                flush_pending()
        flush_pending()

        if not pred_segments:
            return None
        gt_action_abs, pred_segments_abs = _tf_error_actions_to_absolute_units(
            gt_action=action_raw,
            gt_reference_states=state_raw_rows,
            pred_segments=pred_segments,
            delta_slots=_delta_action_slots_for_viz_sample(dataset, sample),
        )
        gt_action = gt_action_abs[:, active]
        pred_segments = [(t0, segment[:, active]) for t0, segment in pred_segments_abs]
        bounds = _collect_segment_bounds(payloads)
        if len(bounds) <= 1:
            bounds = _collect_subtask_bounds(payloads)
        return build_tf_error_episode_panel(
            frame_ids=frame_ids,
            gt_action=gt_action,
            pred_segments=pred_segments,
            bounds=bounds,
            dataset_index=sample.get("_dataset_sample_index"),
            episode_index=episode_index,
            episode_id=str(payload0.get("original_episode_id", "")),
            chunk_len=chunk_len,
            dim_labels=dim_labels,
            units_label="absolute action units",
        )

    def _log_vlm_prompt_cards(self, dataset=None, step: int | None = None):
        if not self.accelerator.is_main_process:
            return
        samples = self._viz_samples(
            self._viz_count("viz_prompt_num_samples", "RHINOVLA_VIZ_PROMPT_NUM_SAMPLES")
        )
        if not samples:
            return
        raw_model = self._unwrap_model()
        total = len(samples)
        visual_dir = self._visual_dir()
        for sample_idx, sample in enumerate(samples):
            try:
                qwen_inputs, batch_images, _instructions, chat_prompt = self._prompt_payload_for_sample(
                    raw_model,
                    sample,
                )
                tokenizer = raw_model.qwen.processor.tokenizer
                image_token_id = int(tokenizer.convert_tokens_to_ids("<|image_pad|>"))
                tokenized_prompt = compact_prompt_from_input_ids(
                    qwen_inputs["input_ids"][0].detach().cpu().numpy(),
                    tokenizer,
                    image_token_id,
                )
                images = [_image_from_any(img) for img in sample.get("image", [])[:3]]
                labels = list(sample.get("view_roles") or ["view0", "view1", "view2"])[: len(images)]
                panel = build_vlm_prompt_sample_panel(
                    images=[_image_from_any(img) for img in batch_images[0][:3]] or images,
                    labels=labels,
                    chat_prompt=chat_prompt,
                    tokenized_prompt=tokenized_prompt,
                    prompt_mode=prompt_mode_label(),
                )
                out_path = visual_dir / _indexed_viz_path(
                    "vlm_prompt",
                    self.completed_steps,
                    sample_idx,
                    total,
                )
                panel.save(out_path)
                self._log_swanlab_image(
                    _indexed_viz_key("vlm_prompt/sample", sample_idx, total),
                    out_path,
                    step=step or self._swanlab_step(),
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"VLM prompt visualization skipped for sample {sample_idx}: {exc}")

    def _log_sample_visualization(self):
        self._log_vlm_prompt_cards(self.vla_train_dataloader.dataset, self._swanlab_step(0))

    def _log_tf_error_curves(self, use_ema: bool = False):
        if use_ema:
            return
        if not self.accelerator.is_main_process:
            return
        samples = self._viz_samples(
            self._viz_count("viz_tf_error_num_episodes", "RHINOVLA_VIZ_TF_NUM_EPISODES"),
            unique_episode=True,
        )
        if not samples:
            return
        raw_model = None
        was_training = False
        try:
            raw_model = self._unwrap_model()
            was_training = bool(raw_model.training)
            raw_model.eval()
            total = len(samples)
            visual_dir = self._visual_dir()
            for sample_idx, sample in enumerate(samples):
                try:
                    torch.manual_seed(0)
                    panel = self._episode_tf_error_panel(raw_model, sample)
                    if panel is None:
                        logger.warning(
                            "TF-error visualization skipped for sample %s: complete episode unavailable",
                            sample_idx,
                        )
                        continue
                    out_path = visual_dir / _indexed_viz_path(
                        "tf_error",
                        self.completed_steps,
                        sample_idx,
                        total,
                    )
                    panel.save(out_path)
                    self._log_swanlab_image(
                        _indexed_viz_key("tf_error/native72", sample_idx, total),
                        out_path,
                        step=self._swanlab_step(),
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.warning(f"TF-error visualization skipped for sample {sample_idx}: {exc}")
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"TF-error visualization skipped: {exc}")
        finally:
            if raw_model is not None and was_training:
                raw_model.train()

    def _log_attention_figure(self):
        if not self.accelerator.is_main_process:
            return
        samples = self._viz_samples(
            self._viz_count("viz_attention_num_samples", "RHINOVLA_VIZ_ATTENTION_NUM_SAMPLES"),
            unique_episode=True,
        )
        if not samples:
            return
        raw_model = None
        was_training = False
        try:
            raw_model = self._unwrap_model()
            was_training = bool(raw_model.training)
            raw_model.eval()
            total = len(samples)
            visual_dir = self._visual_dir()
            for sample_idx, sample in enumerate(samples):
                try:
                    qwen_inputs, batch_images, _ = self._qwen_inputs_for_sample(raw_model, sample)
                    processor = raw_model.qwen.processor
                    image_token_id = int(processor.tokenizer.convert_tokens_to_ids("<|image_pad|>"))
                    merge_size = int(getattr(processor.image_processor, "merge_size", 2))
                    input_ids = qwen_inputs["input_ids"][0].detach().cpu().numpy()
                    grid = qwen_inputs["image_grid_thw"].detach().cpu().numpy()

                    captured: list[torch.Tensor] = []
                    original_sdpa = F.scaled_dot_product_attention
                    F.scaled_dot_product_attention = _patched_sdpa_factory(captured)
                    try:
                        torch.manual_seed(0)
                        with torch.no_grad():
                            raw_model.predict_action([sample], num_steps=1)
                    finally:
                        F.scaled_dot_product_attention = original_sdpa

                    token_weights = _mean_prefix_token_attention(captured, prefix_len=len(input_ids))
                    heatmaps = split_qwen_image_token_attention(
                        input_ids=input_ids,
                        token_weights=token_weights,
                        image_grid_thw=grid,
                        image_token_id=image_token_id,
                        merge_size=merge_size,
                    )
                    model_images = [_image_from_any(img) for img in batch_images[0][: len(heatmaps)]]
                    raw_images = self._raw_images_for_sample(sample) or model_images
                    labels = list(sample.get("view_roles") or ["view0", "view1", "view2"])[: len(model_images)]
                    panel = build_action_expert_attention_panel(
                        raw_images=raw_images[: len(heatmaps)],
                        model_images=model_images,
                        labels=labels,
                        heatmaps=heatmaps,
                        tag="RhinoVLA",
                        run_id=self._run_id_for_viz(),
                        step=self.completed_steps,
                        framework=raw_model.__class__.__name__,
                        ckpt_path=self._ckpt_path_for_viz(),
                        subtitle=(
                            f"{self._viz_sample_identity(sample)} | captured={len(captured)} tensors, "
                            f"image_grids={grid.tolist()}, merge={merge_size}"
                        ),
                    )
                    out_path = visual_dir / _indexed_viz_path(
                        "attention",
                        self.completed_steps,
                        sample_idx,
                        total,
                    )
                    panel.save(out_path)
                    self._log_swanlab_image(
                        _indexed_viz_key("attention/action_expert", sample_idx, total),
                        out_path,
                        step=self._swanlab_step(),
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.warning(f"Attention visualization skipped for sample {sample_idx}: {exc}")
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Attention visualization skipped: {exc}")
        finally:
            if raw_model is not None and was_training:
                raw_model.train()
