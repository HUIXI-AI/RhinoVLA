"""Native LeRobot -> RhinoVLA 72D adapter dataset.

Wraps the official `LeRobotDataset` and reformats samples for the RhinoVLA
72D training batch contract.  The mapping YAML is the single source of truth
for source columns, view list, state/action source indices, target 72D slots,
and mask-active slots.

Per `docs/LEROBOT_DATA_PIPELINE.md`:
- Sample clock is the canonical 30Hz row index. Action chunks are built by
  slicing rows `[i : i + H]`. Chunks must not cross episode boundaries.
- `valid_chunk_filter` chooses the quality gate before training:
  `episode_only` only checks episode containment, `valid_chunk_start` trusts a
  per-row `valid_chunk_start` column written by the converter, and `valid_intervals`
  checks full chunk containment inside merged valid intervals from sidecar
  parquet. The adapter pre-filters kept starts into `valid_global_indices`.
- High-frequency raw streams (`raw/sensor_*`, `raw/target_*`) are not exposed
  to the trainer in the first version.
- Image resize is applied here (mapping `image_size`), not at convert time.
"""
from __future__ import annotations

import json
from importlib import import_module
import os
from bisect import bisect_right
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image
from qwen_vl_utils.vision_process import smart_resize as qwen_smart_resize
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Sampler, WeightedRandomSampler


RHINO72_DIM = 72
ACTION_TYPE_ABSOLUTE = "absolute_joint_position"
ACTION_TYPE_DELTA = "delta_from_current_state"
ACTION_TYPES = (ACTION_TYPE_ABSOLUTE, ACTION_TYPE_DELTA)
_IMAGE_SHORT_SIDE = 256
_IMAGE_RESIZE_FACTOR = 32
_IMAGE_MAX_LONG_SIDE = 512


@dataclass(frozen=True)
class ViewSpec:
    key: str
    role: str
    modality: str = "rgb"
    required: bool = True


@dataclass(frozen=True)
class SlotSpec:
    source_key: str
    source_indices: List[int]
    target_slots: List[int]
    active_slots: List[int]
    target_dim: int = RHINO72_DIM
    value_transforms: List[str] | None = None


@dataclass(frozen=True)
class Native72Mapping:
    raw: dict[str, Any]
    dataset_id: str
    root: str
    repo_id: str
    fps: int
    action_horizon: int
    image_size: tuple[int, int]
    views: List[ViewSpec]
    state: SlotSpec
    action: SlotSpec
    instruction: dict[str, Any]
    sampling: dict[str, Any]
    loss: dict[str, Any]


def _build_qwen_cpu_inputs(*args, **kwargs):
    """Lazily import the shared CPU-only Qwen preprocessing routine."""
    from rhinovla.model.modules.qwen import build_qwenvl_cpu_inputs

    return build_qwenvl_cpu_inputs(*args, **kwargs)


@dataclass
class QwenCPUPreprocessCollator:
    """Build one CPU Qwen batch in a DataLoader worker, preserving model input parity."""

    processor: Any
    allowed_roles: tuple[str, ...]
    allowed_modalities: tuple[str, ...]
    cot_prompt: str | None
    cot_prompt_present: bool

    def __call__(self, batch: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not batch:
            return batch
        batch[0]["_qwen_inputs_cpu"] = _build_qwen_cpu_inputs(
            self.processor,
            [example["image"] for example in batch],
            [example["lang"] for example in batch],
            view_roles=[example.get("view_roles") for example in batch],
            view_modalities=[example.get("view_modalities") for example in batch],
            allowed_roles=self.allowed_roles,
            allowed_modalities=self.allowed_modalities,
            cot_prompt=self.cot_prompt,
            cot_prompt_present=self.cot_prompt_present,
        )
        return batch


def build_qwen_cpu_preprocess_collator(vla_cfg: Mapping[str, Any], *, processor: Any) -> QwenCPUPreprocessCollator:
    """Create a worker collator with exactly the model's Qwen prompt settings."""
    from rhinovla.model.modules.qwen import _DEFAULT_VIEW_MODALITY_VOCAB, _DEFAULT_VIEW_ROLE_VOCAB

    return QwenCPUPreprocessCollator(
        processor=processor,
        allowed_roles=tuple(str(value) for value in vla_cfg.get("view_role_vocab", _DEFAULT_VIEW_ROLE_VOCAB)),
        allowed_modalities=tuple(
            str(value) for value in vla_cfg.get("view_modality_vocab", _DEFAULT_VIEW_MODALITY_VOCAB)
        ),
        cot_prompt=str(vla_cfg.get("CoT_prompt", "")) if "CoT_prompt" in vla_cfg else None,
        cot_prompt_present="CoT_prompt" in vla_cfg,
    )


def qwen16_reference_slots() -> List[int]:
    """T3 16D order -> RhinoVLA first-16 slot layout.

    T3 16D order is left arm, left gripper, right arm, right gripper.
    RhinoVLA first-16 slots are left arm, right arm, left/right grippers.
    """
    return [0, 1, 2, 3, 4, 5, 6, 14, 7, 8, 9, 10, 11, 12, 13, 15]

# LeRobot is an optional runtime dependency (only available in inference /
# training envs that have it installed). Defer the import so that environments
# without LeRobot can still import this module for code inspection.
#
# LeRobot reorganized its package layout between 0.1.x (`lerobot.common.datasets`)
# and 0.5.x (`lerobot.datasets`). The dataset *on disk* (v2.1) is the same so
# we can read either version's writer output; the import path is the only
# thing that differs at runtime.
def _lazy_lerobot_dataset_cls():
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset  # 0.5.x  # noqa: WPS433
    except ModuleNotFoundError:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset  # 0.1.x  # noqa: WPS433
    return LeRobotDataset


def _is_lerobot_frame_timestamp_error(exc: Exception) -> bool:
    """Return whether *exc* is the optional LeRobot timestamp exception.

    LeRobot has shipped both module layouts.  Resolve its concrete class only
    on the exceptional path, rather than recognizing arbitrary same-named
    exceptions or making LeRobot a module-import dependency.
    """

    exception_types = []
    for module_name in (
        "lerobot.datasets.video_utils",
        "lerobot.common.datasets.video_utils",
    ):
        try:
            exception_type = getattr(import_module(module_name), "FrameTimestampError")
        except (ModuleNotFoundError, AttributeError):
            continue
        if isinstance(exception_type, type) and issubclass(exception_type, Exception):
            exception_types.append(exception_type)
    return bool(exception_types) and isinstance(exc, tuple(exception_types))


def wrap_lerobot_pyav_seek_decoder(decoder, *, backoff_s: float = 1.0):
    """Recover PyAV keyframe-only seek failures with a discarded anchor frame.

    A pre-emptive anchor is incorrect: depending on the source GOP, it can
    itself move torchvision's PyAV reader to a later keyframe.  Preserve the
    direct request first, and retry with one earlier frame only for LeRobot's
    ``FrameTimestampError``.  The retry frame is discarded, so the returned
    tensor stays aligned exactly to the requested timestamps.
    """

    def decode(video_path, timestamps, tolerance_s, backend=None):
        requested = list(timestamps)
        if backend != "pyav" or not requested:
            return decoder(video_path, requested, tolerance_s, backend)
        try:
            return decoder(video_path, requested, tolerance_s, backend)
        except Exception as exc:
            if not _is_lerobot_frame_timestamp_error(exc):
                raise
            first_timestamp = min(float(timestamp) for timestamp in requested)
            anchor = max(0.0, first_timestamp - float(backoff_s))
            if anchor >= first_timestamp:
                raise
            frames = decoder(video_path, [anchor, *requested], tolerance_s, backend)
            if len(frames) != len(requested) + 1:
                raise RuntimeError("PyAV seek-anchor decoder returned an unexpected frame count")
            return frames[1:]

    return decode


def _install_lerobot_pyav_seek_workaround() -> None:
    """Install the process-local exact-timestamp workaround once."""

    try:
        from lerobot.datasets import dataset_reader  # noqa: WPS433
    except ModuleNotFoundError:
        return
    if getattr(dataset_reader, "_rhinovla_pyav_seek_workaround", False):
        return
    dataset_reader.decode_video_frames = wrap_lerobot_pyav_seek_decoder(dataset_reader.decode_video_frames)
    dataset_reader._rhinovla_pyav_seek_workaround = True


class _TorchCodecPathSource:
    """No-op cache companion for a path-backed TorchCodec decoder."""

    def close(self) -> None:
        return None


TORCHCODEC_EXACT_DECODER_CACHE_MAX_SIZE = 32


def _close_torchcodec_cached_decoder(decoder: Any, source: Any) -> None:
    """Release an evicted path decoder when the backend exposes a closer."""
    close_decoder = getattr(decoder, "close", None)
    if callable(close_decoder):
        close_decoder()
    close_source = getattr(source, "close", None)
    if callable(close_source):
        close_source()


def get_torchcodec_exact_cached_decoder(
    cache,
    video_path,
    decoder_cls,
    *,
    max_size: int = TORCHCODEC_EXACT_DECODER_CACHE_MAX_SIZE,
):
    """Cache an exact-seek TorchCodec decoder for a local video path.

    LeRobot 0.5's default cache passes an fsspec file object to TorchCodec with
    approximate seeking.  For the local LeRobot releases used here that can
    select the wrong 30 Hz frame despite valid PTS, use the filesystem path and
    TorchCodec's exact frame-index seeking instead.
    """

    if max_size < 1:
        raise ValueError(f"TorchCodec decoder cache size must be positive, got {max_size}")
    key = str(video_path)
    with cache._lock:
        cached = cache._cache.pop(key, None)
        if cached is not None:
            cache._cache[key] = cached
            return cached[0]
        while len(cache._cache) >= max_size:
            oldest_key = next(iter(cache._cache))
            evicted_decoder, evicted_source = cache._cache.pop(oldest_key)
            _close_torchcodec_cached_decoder(evicted_decoder, evicted_source)
        decoder = decoder_cls(key, seek_mode="exact")
        cache._cache[key] = (decoder, _TorchCodecPathSource())
        return decoder


def _install_lerobot_torchcodec_exact_decoder() -> None:
    """Use exact local-path TorchCodec decoders in LeRobot's shared cache."""

    try:
        from lerobot.datasets import video_utils  # noqa: WPS433
        from torchcodec.decoders import VideoDecoder  # noqa: WPS433
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "video_backend=torchcodec requires importable TorchCodec in the training environment"
        ) from exc
    cache_cls = video_utils.VideoDecoderCache
    if getattr(cache_cls, "_rhinovla_exact_path_decoder", False):
        return
    video_utils._default_decoder_cache.clear()

    def get_decoder(cache, video_path):
        return get_torchcodec_exact_cached_decoder(cache, video_path, VideoDecoder)

    cache_cls.get_decoder = get_decoder
    cache_cls._rhinovla_exact_path_decoder = True


def _load_mapping(path: str | Path) -> dict[str, Any]:
    cfg = OmegaConf.load(str(path))
    return OmegaConf.to_container(cfg, resolve=True)  # type: ignore[return-value]


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


VALID_CHUNK_FILTER_MODES = ("episode_only", "valid_chunk_start", "valid_intervals")


def _normalize_valid_chunk_filter(value: Any) -> str:
    """Return one of the three supported valid chunk filtering modes.

    Empty/missing config defaults to `episode_only`, meaning "do not apply
    data-quality filtering": the loader only prevents an H-step action chunk
    from crossing the episode boundary. All other strings are rejected on
    purpose so the mapping contract stays explicit and easy to audit.
    """
    raw = "episode_only" if value is None else str(value).strip().lower()
    if raw == "":
        raw = "episode_only"
    if raw not in VALID_CHUNK_FILTER_MODES:
        allowed = ", ".join(VALID_CHUNK_FILTER_MODES)
        raise ValueError(f"sampling.valid_chunk_filter must be one of: {allowed}; got {value!r}")
    return raw


def _normalize_action_type(value: Any) -> str:
    raw = str(value or ACTION_TYPE_ABSOLUTE).strip().lower()
    if raw not in ACTION_TYPES:
        raise ValueError(f"action_type must be one of {ACTION_TYPES}, got {value!r}")
    return raw


def _resolve_action_delta_slots(
    *,
    action_type: str,
    active_state_slots: Sequence[int],
    active_action_slots: Sequence[int],
    absolute_action_slots: Sequence[int],
) -> list[int]:
    """Resolve the generic action representation in Rhino72 target slots.

    In delta mode every active action slot with a same-slot state becomes a
    delta except explicitly absolute gripper opening slots. Action-only command
    slots (for example a velocity command) keep their native command semantics.
    """
    mode = _normalize_action_type(action_type)
    action_slots = sorted({int(slot) for slot in active_action_slots})
    if mode == ACTION_TYPE_ABSOLUTE:
        return []

    state_slots = {int(slot) for slot in active_state_slots}
    absolute_slots = {int(slot) for slot in absolute_action_slots}
    bad_absolute = sorted(slot for slot in absolute_slots if slot < 0 or slot >= RHINO72_DIM)
    if bad_absolute:
        raise ValueError(f"absolute_action_slots out of Rhino72 range: {bad_absolute}")
    return [
        slot
        for slot in action_slots
        if slot not in absolute_slots and slot in state_slots
    ]


def _action_to_model_space(
    action_abs: Any,
    current_state: Any,
    *,
    delta_slots: Sequence[int],
) -> np.ndarray:
    """Return absolute actions or `action - chunk_start_state` on selected slots."""
    action_model = np.asarray(action_abs, dtype=np.float32).copy()
    state = np.asarray(current_state, dtype=np.float32).reshape(-1)
    slots = np.asarray([int(slot) for slot in delta_slots], dtype=np.int64)
    if slots.size == 0:
        return action_model
    if action_model.ndim not in (1, 2):
        raise ValueError(f"action must be 1D or 2D, got shape {action_model.shape}")
    max_slot = int(slots.max())
    if action_model.shape[-1] <= max_slot or state.shape[0] <= max_slot:
        raise ValueError(
            f"delta slot {max_slot} is outside action/state shapes "
            f"{action_model.shape}/{state.shape}"
        )
    if action_model.ndim == 1:
        action_model[slots] -= state[slots]
    else:
        action_model[:, slots] -= state[None, slots]
    return action_model


def _select_action_norm_stats(
    norm: dict[str, Any],
    *,
    action_type: str,
    delta_slots: Sequence[int],
    absolute_action_slots: Sequence[int],
) -> dict[str, Any]:
    """Select one action space from a combined norm file.

    Legacy files containing only `action_mean/std` remain valid for absolute
    training. Delta training requires explicit delta fields and matching
    metadata so an absolute norm can never be used accidentally.
    """
    mode = _normalize_action_type(action_type)
    if mode == ACTION_TYPE_ABSOLUTE:
        has_abs_mean = "action_abs_mean" in norm
        has_abs_std = "action_abs_std" in norm
        if has_abs_mean != has_abs_std:
            raise ValueError(
                "combined norm must contain both action_abs_mean and action_abs_std"
            )
        mean_key = "action_abs_mean" if has_abs_mean else "action_mean"
        std_key = "action_abs_std" if has_abs_std else "action_std"
    else:
        mean_key = "action_delta_mean"
        std_key = "action_delta_std"
        missing = [key for key in (mean_key, std_key) if key not in norm]
        if missing:
            raise ValueError(
                f"delta_from_current_state requires combined norm fields {missing}; "
                "regenerate norm.json with datasets/compute_norm_json.py"
            )
        meta = norm.get("_meta", {}) or {}
        expected_delta = sorted(int(slot) for slot in delta_slots)
        actual_delta = sorted(int(slot) for slot in meta.get("delta_from_state_slots", []))
        if actual_delta != expected_delta:
            raise ValueError(
                "norm delta_from_state_slots do not match loader mapping: "
                f"norm={actual_delta}, loader={expected_delta}"
            )
        expected_absolute = sorted(int(slot) for slot in absolute_action_slots)
        actual_absolute = sorted(int(slot) for slot in meta.get("absolute_action_slots", []))
        if actual_absolute != expected_absolute:
            raise ValueError(
                "norm absolute_action_slots do not match loader mapping: "
                f"norm={actual_absolute}, loader={expected_absolute}"
            )
    for key in ("state_mean", "state_std", mean_key, std_key):
        if key not in norm:
            raise ValueError(f"norm file missing required field {key!r}")
    return {
        "state_mean": norm["state_mean"],
        "state_std": norm["state_std"],
        "action_mean": norm[mean_key],
        "action_std": norm[std_key],
    }


def _flatten_dim_list(spec: Any) -> List[int]:
    """`{left_arm: [0..6], left_gripper: 7, right_arm: [..], right_gripper: 15}`
    → `[0..6, 7, 8..14, 15]`. Keeps mapping-yaml as the source of truth.
    """
    flat: List[int] = []
    for key in ("left_arm", "left_gripper", "right_arm", "right_gripper"):
        if key not in spec:
            raise KeyError(f"mapping `dims` missing key '{key}'")
        v = spec[key]
        if isinstance(v, (list, tuple)):
            flat.extend(int(x) for x in v)
        else:
            flat.append(int(v))
    return flat


def _select_mapping_dataset(raw: dict[str, Any], dataset_id: str | None) -> dict[str, Any]:
    datasets = raw.get("datasets", None)
    if not datasets:
        return raw
    selected = None
    if dataset_id:
        for item in datasets:
            if str(item.get("dataset_id", item.get("id", ""))) == str(dataset_id):
                selected = dict(item)
                break
        if selected is None:
            raise KeyError(f"dataset_id={dataset_id!r} not found in mapping datasets")
    else:
        selected = dict(datasets[0])
    merged = dict(raw)
    merged.update(selected)
    merged["id"] = selected.get("dataset_id", selected.get("id", raw.get("id", "")))
    return merged


def _views_from_mapping(mapping: dict[str, Any]) -> List[ViewSpec]:
    views = mapping.get("views", None)
    if views:
        return [
            ViewSpec(
                key=str(view["key"]),
                role=str(view.get("role", view.get("name", view["key"]))),
                modality=str(view.get("modality", "rgb")),
                required=bool(view.get("required", True)),
            )
            for view in views
        ]
    camera_mapping = mapping.get("camera_mapping", {}) or {}
    return [
        ViewSpec(key=str(source_key), role=str(alias), modality="rgb", required=True)
        for alias, source_key in camera_mapping.items()
    ]


def _legacy_slot_spec(mapping: dict[str, Any], kind: str) -> SlotSpec:
    cfg = mapping[f"{kind}_mapping"]
    source_indices = _flatten_dim_list(cfg["dims"])
    target_dim = int(cfg.get("target_dim", len(source_indices)))
    target_slots = list(cfg.get("target_slots", range(target_dim)))
    active_slots = list(mapping.get(f"active_{kind}_slots", target_slots))
    return SlotSpec(
        source_key=str(cfg["source_key"]),
        source_indices=[int(x) for x in source_indices],
        target_slots=[int(x) for x in target_slots],
        active_slots=[int(x) for x in active_slots],
        target_dim=int(mapping.get("target_dim", target_dim)),
        value_transforms=["identity"] * len(target_slots),
    )


def _native_slot_spec(mapping: dict[str, Any], kind: str) -> SlotSpec:
    groups = list(mapping.get("native_joint_groups", []) or [])
    if not groups:
        return _legacy_slot_spec(mapping, kind)

    source_key = str(mapping.get(f"{kind}_source", "observation.state" if kind == "state" else "action"))
    source_indices: list[int] = []
    target_slots: list[int] = []
    transforms: list[str] = []
    for group in groups:
        if kind == "state" and str(group.get("state_source", source_key)) == "none":
            continue
        indices_key = "state_source_indices" if kind == "state" else "action_source_indices"
        fields_key = "source_fields" if kind == "state" else "action_source_fields"
        indices = group.get(indices_key, [])
        if indices is None:
            indices = []
        slots = group.get("target_slots", [])
        if len(indices) != len(slots):
            raise ValueError(
                f"native_joint_groups.{group.get('slot_group', '<unknown>')} {indices_key} "
                f"length {len(indices)} != target_slots length {len(slots)}"
            )
        if not indices and group.get(fields_key):
            raise ValueError(
                f"native_joint_groups.{group.get('slot_group', '<unknown>')} has {fields_key} "
                f"but no {indices_key}"
            )
        transform = str(group.get("value_transform", "identity"))
        source_indices.extend(int(x) for x in indices)
        target_slots.extend(int(x) for x in slots)
        transforms.extend([transform] * len(slots))

    active_key = f"active_{kind}_slots"
    active_slots = [int(x) for x in mapping.get(active_key, target_slots)]
    return SlotSpec(
        source_key=source_key,
        source_indices=source_indices,
        target_slots=target_slots,
        active_slots=active_slots,
        target_dim=int(mapping.get("target_dim", RHINO72_DIM)),
        value_transforms=transforms,
    )


def load_lerobot_mapping(path: str | Path, dataset_id: str | None = None) -> Native72Mapping:
    raw = _load_mapping(path)
    mapping = _select_mapping_dataset(raw, dataset_id)
    views = _views_from_mapping(mapping)
    state = _native_slot_spec(mapping, "state")
    action = _native_slot_spec(mapping, "action")
    if state.target_dim != RHINO72_DIM or action.target_dim != RHINO72_DIM:
        raise ValueError(
            f"RhinoVLA native dataloader requires 72D target dims, got "
            f"state={state.target_dim}, action={action.target_dim}"
        )
    root = str(mapping.get("root", ""))
    repo_id = str(mapping.get("repo_id", mapping.get("dataset_id", mapping.get("id", ""))))
    image_size = mapping.get("image_size", [256, 256])
    return Native72Mapping(
        raw=mapping,
        dataset_id=str(mapping.get("id", mapping.get("dataset_id", ""))),
        root=root,
        repo_id=repo_id,
        fps=int(mapping.get("fps", 30)),
        action_horizon=int(mapping.get("action_horizon", 30)),
        image_size=(int(image_size[0]), int(image_size[1])),
        views=views,
        state=state,
        action=action,
        instruction=dict(mapping.get("instruction", {}) or {}),
        sampling=dict(mapping.get("sampling", {}) or {}),
        loss=dict(mapping.get("loss", {}) or {}),
    )


def _apply_value_transform(values: np.ndarray, transforms: Sequence[str] | None) -> np.ndarray:
    out = np.asarray(values, dtype=np.float32).copy()
    if not transforms:
        return out
    if out.shape[-1] != len(transforms):
        raise ValueError(f"value transform length {len(transforms)} does not match values shape {out.shape}")
    for idx, transform in enumerate(transforms):
        if transform in ("identity", "", None):
            continue
        if transform == "one_minus_raw":
            out[..., idx] = 1.0 - out[..., idx]
        else:
            raise ValueError(f"unsupported value_transform={transform!r}")
    return out


def pack_source_to_rhino72(
    values: Any,
    *,
    source_indices: Sequence[int],
    target_slots: Sequence[int],
    fill_value: float = 0.0,
    value_transforms: Sequence[str] | None = None,
) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if not source_indices:
        out_shape = (*arr.shape[:-1], RHINO72_DIM)
        return np.full(out_shape, fill_value, dtype=np.float32)
    max_source = max(int(x) for x in source_indices)
    if arr.shape[-1] <= max_source:
        raise ValueError(f"source shape {arr.shape} does not contain source index {max_source}")
    selected = arr[..., list(source_indices)]
    selected = _apply_value_transform(selected, value_transforms)
    out = np.full((*arr.shape[:-1], RHINO72_DIM), fill_value, dtype=np.float32)
    out[..., list(target_slots)] = selected
    return out


def expand_stats_to_rhino72(
    values: Any,
    *,
    spec: SlotSpec,
    fill_value: float,
    transform_values: bool = True,
) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if arr.shape == (RHINO72_DIM,):
        return arr.copy()
    if arr.ndim != 1:
        raise ValueError(f"stats must be 1D, got shape {arr.shape}")
    if arr.shape[0] > max(spec.source_indices, default=-1):
        selected = arr[list(spec.source_indices)]
    elif arr.shape[0] == len(spec.target_slots):
        selected = arr
    else:
        raise ValueError(
            f"cannot expand stats shape {arr.shape} into 72D: need 72, "
            f"> max source index {max(spec.source_indices, default=-1)}, "
            f"or {len(spec.target_slots)} active values"
        )
    if transform_values:
        selected = _apply_value_transform(selected, spec.value_transforms)
    out = np.full((RHINO72_DIM,), fill_value, dtype=np.float32)
    out[list(spec.target_slots)] = selected
    return out


def mask_from_spec(spec: SlotSpec) -> np.ndarray:
    mask = np.zeros((RHINO72_DIM,), dtype=np.float32)
    mask[list(spec.active_slots)] = 1.0
    return mask


def _tensor_to_pil(img_tensor: torch.Tensor) -> Image.Image:
    """LeRobot returns CHW float32 in [0,1]. Convert to PIL RGB uint8."""
    if img_tensor.dim() != 3:
        raise ValueError(f"unexpected image tensor shape: {tuple(img_tensor.shape)}")
    if img_tensor.shape[0] == 3:
        chw = img_tensor
    elif img_tensor.shape[-1] == 3:
        chw = img_tensor.permute(2, 0, 1)
    else:
        raise ValueError(f"unsupported image tensor channel layout: {tuple(img_tensor.shape)}")
    arr = (chw.clamp(0, 1) * 255.0).byte().permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(arr, mode="RGB")


def _scalar_py(value: Any) -> Any:
    """Return a Python scalar from LeRobot/HF scalar variants.

    LeRobot v2 scalar columns usually arrive as plain scalars or shape-(1,)
    tensors. LeRobot v3 raw datasets may store numeric scalars as shape-(1,1)
    arrays to satisfy the current writer/HF encoding combination.
    """
    if torch.is_tensor(value):
        if value.numel() == 0:
            return None
        return value.reshape(-1)[0].item()
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return None
        return value.reshape(-1)[0].item()
    if isinstance(value, (list, tuple)):
        if not value:
            return None
        return _scalar_py(value[0])
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return value
    return value


def _column_to_1d_array(column: Any, dtype: Any) -> np.ndarray:
    if isinstance(column, list):
        return np.asarray([_scalar_py(v) for v in column], dtype=dtype).reshape(-1)
    return np.asarray(column, dtype=dtype).reshape(len(column), -1)[:, 0]


def _hf_column_to_1d_array(hf_dataset: Any, key: str, dtype: Any) -> np.ndarray:
    """Read a scalar HF column without triggering row-wise torch transforms.

    `datasets.Dataset.__getitem__(column_name)` applies the active transform to
    every row. That is acceptable for small v2 datasets but becomes minutes of
    CPU time for v3 raw datasets with many parquet files. The Arrow table keeps
    the same values and is the right path for init-time indexing columns.
    """
    data = getattr(hf_dataset, "data", None)
    if data is not None:
        try:
            column = data.column(key)
            try:
                return np.asarray(column.to_numpy(zero_copy_only=False), dtype=dtype).reshape(-1)
            except Exception:
                return np.asarray([_scalar_py(v) for v in column.to_pylist()], dtype=dtype).reshape(-1)
        except Exception:
            pass
    return _column_to_1d_array(hf_dataset[key], dtype)


def _hf_vector_column_to_array(hf_dataset: Any, key: str, dtype: Any) -> np.ndarray:
    """Read a vector HF column through Arrow for stats computation."""
    data = getattr(hf_dataset, "data", None)
    if data is not None:
        try:
            column = data.column(key)
            return np.asarray(column.to_pylist(), dtype=dtype).reshape(len(hf_dataset), -1)
        except Exception:
            pass
    return np.stack(hf_dataset[key]).astype(dtype)


def _load_weight_override_table(path: str | Path):
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        import pandas as pd  # noqa: WPS433
        return pd.read_parquet(path)
    if suffix == ".csv":
        import pandas as pd  # noqa: WPS433
        return pd.read_csv(path)
    raise ValueError(f"unsupported sample_weight_override_path suffix: {path}")


class Native72MixtureDataset(ConcatDataset):
    """Concatenate native72 datasets that share one normalization contract."""

    def __init__(self, datasets: Sequence[Dataset], source_ids: Sequence[str]) -> None:
        if not datasets:
            raise ValueError("Native72MixtureDataset requires at least one dataset")
        if len(datasets) != len(source_ids):
            raise ValueError(
                f"source_ids count {len(source_ids)} must match dataset count {len(datasets)}"
            )
        self.source_ids = [str(source_id) for source_id in source_ids]
        if any(not source_id for source_id in self.source_ids):
            raise ValueError("mixture source_ids must be non-empty")
        super().__init__(list(datasets))
        primary = self.datasets[0]
        for name in ("state_mean", "state_std", "action_mean", "action_std"):
            expected = np.asarray(getattr(primary, name), dtype=np.float32)
            if any(
                not np.array_equal(expected, np.asarray(getattr(child, name), dtype=np.float32))
                for child in self.datasets[1:]
            ):
                raise ValueError(f"all mixture sources must share the same {name}")
            setattr(self, name, expected)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(f"mixture dataset index {index} out of range")
        dataset_index = bisect_right(self.cumulative_sizes, index)
        previous_size = 0 if dataset_index == 0 else self.cumulative_sizes[dataset_index - 1]
        local_index = index - previous_size
        sample = self.datasets[dataset_index][local_index]
        if not isinstance(sample, Mapping):
            raise TypeError(
                f"native72 child dataset {dataset_index} returned {type(sample)}, expected a mapping"
            )
        result = dict(sample)
        result["_mixture_dataset_index"] = dataset_index
        result["_mixture_local_index"] = local_index
        result["_mixture_source_id"] = self.source_ids[dataset_index]
        return result


class DeterministicMixtureSampler(Sampler[int]):
    """Deterministic replacement sampler with two-level mixture weighting.

    Dataset weights choose the source.  A source's local sample weights are
    normalized *inside that source*, preventing a large dataset or large raw
    weight scale from silently changing the requested source proportions.
    """

    def __init__(
        self,
        dataset: Native72MixtureDataset,
        dataset_weights: Sequence[float],
        epoch_samples: int,
        seed: int,
    ) -> None:
        if not isinstance(dataset, Native72MixtureDataset):
            raise TypeError("DeterministicMixtureSampler requires Native72MixtureDataset")
        if len(dataset_weights) != len(dataset.datasets):
            raise ValueError(
                f"dataset_weights count {len(dataset_weights)} must match "
                f"dataset count {len(dataset.datasets)}"
            )
        source_weights = torch.as_tensor(dataset_weights, dtype=torch.float64)
        if source_weights.numel() == 0 or not torch.isfinite(source_weights).all():
            raise ValueError("mixture dataset weights must be finite and non-empty")
        if torch.any(source_weights <= 0):
            raise ValueError("mixture dataset weights must be strictly positive")
        self.epoch_samples = int(epoch_samples)
        if self.epoch_samples <= 0:
            raise ValueError(f"epoch_samples must be positive, got {epoch_samples}")
        self.dataset = dataset
        self.seed = int(seed)
        self.epoch = 0
        self._resume_position = 0

        source_weights = source_weights / source_weights.sum()
        probability_parts: list[torch.Tensor] = []
        for dataset_index, (child, source_weight) in enumerate(
            zip(dataset.datasets, source_weights)
        ):
            if len(child) <= 0:
                raise ValueError(f"mixture child dataset {dataset_index} is empty")
            sample_weights_fn = getattr(child, "sample_weights", None)
            if callable(sample_weights_fn):
                local_weights = torch.as_tensor(sample_weights_fn(), dtype=torch.float64)
                if local_weights.ndim != 1 or local_weights.numel() != len(child):
                    raise ValueError(
                        f"child dataset {dataset_index} sample_weights shape "
                        f"{tuple(local_weights.shape)} does not match length {len(child)}"
                    )
            else:
                local_weights = torch.ones(len(child), dtype=torch.float64)
            if not torch.isfinite(local_weights).all() or torch.any(local_weights < 0):
                raise ValueError(
                    f"child dataset {dataset_index} sample_weights must be finite and non-negative"
                )
            local_sum = local_weights.sum()
            if float(local_sum) <= 0:
                raise ValueError(f"child dataset {dataset_index} sample_weights sum must be positive")
            probability_parts.append(source_weight * local_weights / local_sum)
        self.probabilities = torch.cat(probability_parts)

    def set_epoch(self, epoch: int) -> None:
        epoch = int(epoch)
        if epoch != self.epoch:
            self._resume_position = 0
        self.epoch = epoch

    def set_resume_position(self, position: int) -> None:
        position = int(position)
        if position < 0 or position > self.epoch_samples:
            raise ValueError(
                f"resume position must be in [0, {self.epoch_samples}], got {position}"
            )
        self._resume_position = position

    def __iter__(self) -> Iterator[int]:
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        indices = torch.multinomial(
            self.probabilities,
            self.epoch_samples,
            replacement=True,
            generator=generator,
        )
        return iter(indices[self._resume_position :].tolist())

    def __len__(self) -> int:
        return self.epoch_samples - self._resume_position


class LeRobotNative72Dataset(Dataset):
    """Map LeRobotDataset rows into native RhinoVLA 72D training samples."""

    def __init__(
        self,
        mapping_path: str | Path,
        action_horizon: int = 30,
        stats_override: Optional[Dict[str, np.ndarray]] = None,
        episodes: Optional[Sequence[int]] = None,
        image_aug_profile: str = "none",
        image_aug_size: int = 256,
        image_augmentation: Optional[Dict[str, Any]] = None,
        top_head_crop_size: Optional[Sequence[int]] = None,
        video_backend: str = "pyav",
        fixed_instruction: Optional[str] = None,
        sample_weight_override_path: Optional[str] = None,
        sample_weight_override_column: str = "sample_weight",
        sample_weight_override_mode: str = "replace",
        camera_dropout: Optional[Dict[str, Any]] = None,
        mapping_dataset_id: Optional[str] = None,
        action_type: str = ACTION_TYPE_ABSOLUTE,
    ) -> None:
        self.native_mapping = load_lerobot_mapping(mapping_path, dataset_id=mapping_dataset_id)
        self.mapping = self.native_mapping.raw
        if self.mapping.get("format", "lerobot") != "lerobot":
            raise ValueError(f"mapping format must be 'lerobot', got {self.mapping.get('format')!r}")
        if not self.native_mapping.root:
            raise ValueError(f"mapping {mapping_path} does not define a LeRobot root")

        self.action_horizon = int(action_horizon)
        # Camera order is fixed by mapping. Keep insertion order. New mappings
        # may declare arbitrary views; legacy mappings can still use
        # camera_mapping aliases.
        camera_mapping = self.mapping.get("camera_mapping", {}) or {}
        if camera_mapping:
            self.camera_order = [str(k) for k in camera_mapping.keys()]
            self.camera_source_keys = [str(camera_mapping[k]) for k in camera_mapping.keys()]
        else:
            self.camera_order = [view.role for view in self.native_mapping.views]
            self.camera_source_keys = [view.key for view in self.native_mapping.views]
        self._camera_alias_by_source = dict(zip(self.camera_source_keys, self.camera_order))
        if top_head_crop_size is None:
            self.top_head_crop_size: tuple[int, int] | None = None
        else:
            crop_size = list(top_head_crop_size)
            if len(crop_size) != 2:
                raise ValueError(
                    f"top_head_crop_size must be [positive_width, positive_height], got {crop_size}"
                )
            crop_width, crop_height = (int(value) for value in crop_size)
            if crop_width <= 0 or crop_height <= 0:
                raise ValueError(
                    f"top_head_crop_size must be [positive_width, positive_height], got {crop_size}"
                )
            self.top_head_crop_size = (crop_width, crop_height)
        cd_cfg = camera_dropout or {}
        self.camera_dropout_enabled = bool(cd_cfg.get("enabled", False))
        self.camera_dropout_p = float(cd_cfg.get("p", 0.0))
        self.camera_dropout_replace = str(cd_cfg.get("replace", "mean"))
        self.camera_dropout_candidates = list(cd_cfg.get("cameras", ["head_rgb", "left_rgb"]))
        self.camera_dropout_never_drop = set(cd_cfg.get("never_drop", ["right_rgb"]))
        if self.camera_dropout_enabled:
            if not (0.0 <= self.camera_dropout_p <= 1.0):
                raise ValueError(f"camera_dropout.p must be in [0,1], got {self.camera_dropout_p}")
            if self.camera_dropout_replace != "mean":
                raise ValueError("camera_dropout.replace currently supports only 'mean'")
            unknown = set(self.camera_dropout_candidates) - set(self.camera_order)
            if unknown:
                raise ValueError(f"camera_dropout.cameras contains unknown cameras: {sorted(unknown)}")
            protected = set(self.camera_dropout_candidates) & self.camera_dropout_never_drop
            if protected:
                raise ValueError(f"camera_dropout cameras also marked never_drop: {sorted(protected)}")
            print(
                f"[LeRobotAdapter] camera_dropout enabled p={self.camera_dropout_p} "
                f"candidates={self.camera_dropout_candidates} never_drop={sorted(self.camera_dropout_never_drop)} "
                f"replace={self.camera_dropout_replace}",
                flush=True,
            )

        # Native 72D slot mapping. No extra 72D index cache is used: __getitem__
        # reads the raw source row/chunk and packs it directly according to
        # these source-index -> target-slot arrays, matching the old 16D loader
        # execution style.
        self.state_spec = self.native_mapping.state
        self.action_spec = self.native_mapping.action
        self.state_source_key = self.state_spec.source_key
        self.action_source_key = self.action_spec.source_key
        self.state_active_dims = list(self.state_spec.source_indices)
        self.action_active_dims = list(self.action_spec.source_indices)
        self.action_type = _normalize_action_type(action_type)
        active_action_slot_set = {int(slot) for slot in self.action_spec.active_slots}
        self.absolute_action_slots = sorted(
            {
                int(slot)
                for slot in self.mapping.get("absolute_action_slots", [])
                if int(slot) in active_action_slot_set
            }
        )
        self.delta_action_slots = _resolve_action_delta_slots(
            action_type=self.action_type,
            active_state_slots=self.state_spec.active_slots,
            active_action_slots=self.action_spec.active_slots,
            absolute_action_slots=self.absolute_action_slots,
        )
        print(
            f"[LeRobotAdapter] action_type={self.action_type} "
            f"delta_from_state_slots={self.delta_action_slots} "
            f"absolute_action_slots={self.absolute_action_slots}",
            flush=True,
        )

        # Sample weight + valid chunk start columns (converter populates them).
        sampling_cfg = self.mapping.get("sampling", {})
        self.sample_weight_key = sampling_cfg.get("sample_weight_key", "sample_weight")
        self.valid_chunk_key = sampling_cfg.get("valid_chunk_start_key", "valid_chunk_start")
        # valid_chunk_filter has exactly three modes:
        #
        # 1. `episode_only` (default): no data-quality filtering. A start row i
        #    is kept when the action window [i, i + H) stays inside the same
        #    episode. It does not require `valid_chunk_start`, segment sidecars,
        #    subtask consistency, timestamp-offset checks, or error-frame labels.
        #
        # 2. `valid_chunk_start`: frame/start-level filtering. The converter must
        #    provide a per-row `valid_chunk_start` column. The loader trusts that
        #    column as the training mask, so any stricter checks must be encoded
        #    during conversion, for example frame contiguity, subtask consistency,
        #    camera or sensor offset tolerances, and target availability.
        #
        # 3. `valid_intervals`: interval-level filtering. The loader reads
        #    meta/episode_valid_intervals.parquet, merges overlapping/touching
        #    valid intervals per episode, then keeps a start row only if its full
        #    action window [frame, frame + H) is contained in one merged interval.
        #
        # Keep this list intentionally small. Extra aliases make configs harder
        # to reason about, so unknown values fail fast.
        self.valid_chunk_filter = _normalize_valid_chunk_filter(sampling_cfg.get("valid_chunk_filter", "episode_only"))
        self.valid_intervals_path = sampling_cfg.get("valid_intervals_path", None)
        self.segment_table_path = sampling_cfg.get("segment_table_path", None)
        self.segment_weight_by_skill = dict(sampling_cfg.get("segment_weight_by_skill", {}) or {})
        self.segment_weight_by_action_text = dict(sampling_cfg.get("segment_weight_by_action_text", {}) or {})
        self.segment_weight_default = float(sampling_cfg.get("segment_weight_default", 1.0))
        self._segment_skill_full: Optional[np.ndarray] = None
        self._segment_action_text_full: Optional[np.ndarray] = None
        self._segment_index_full: Optional[np.ndarray] = None
        self._segment_weight_multiplier_full: Optional[np.ndarray] = None

        # Instruction lookup: subtask_prompt then fallbacks.
        instruction_cfg = self.mapping.get("instruction", {})
        self.fixed_instruction = fixed_instruction or instruction_cfg.get("fixed_instruction", None)
        self.instruction_source = instruction_cfg.get("source_key", "subtask_prompt")
        self.instruction_fallbacks = list(instruction_cfg.get("fallback_keys", []))
        # Derive the instruction from the human annotation segment instead of the
        # per-frame `subtask_prompt`. The latter is tagged from the gripper
        # close-open MIDPOINT, which lags the annotation boundary by ~50 frames,
        # so the frames right after each object hand-off carry the PREVIOUS
        # object's prompt (object-mismatch on a double-digit % of chunk starts;
        # see meta/alignment_audit_summary_fixed.json). `segment_action_text`
        # comes from the same segment table that defines chunk validity in
        # `valid_intervals` mode, so resolving the prompt from it makes the
        # instruction boundary == the validity boundary → 0 object-mismatch.
        # Map keys are matched as substrings of action_text (object name).
        self.prompt_from_segment = bool(instruction_cfg.get("prompt_from_segment", False))
        self.action_text_to_prompt = dict(instruction_cfg.get("action_text_to_prompt", {}) or {})
        if self.prompt_from_segment and not self.action_text_to_prompt:
            raise ValueError(
                "instruction.prompt_from_segment=true requires "
                "instruction.action_text_to_prompt (action_text substring -> prompt)"
            )

        # Loss mask (optional, broadcast at training time). Accept either a
        # full 72D mask or active-slot values in mapping order.
        loss_cfg = self.mapping.get("loss", {}) or {}
        loss_mask_cfg = loss_cfg.get("action_loss_mask", None)
        if loss_mask_cfg is None:
            self.action_loss_mask = mask_from_spec(self.action_spec)
        else:
            raw_loss_mask = np.asarray(loss_mask_cfg, dtype=np.float32)
            if raw_loss_mask.shape == (RHINO72_DIM,):
                self.action_loss_mask = raw_loss_mask
            elif raw_loss_mask.shape == (len(self.action_spec.target_slots),):
                self.action_loss_mask = np.zeros((RHINO72_DIM,), dtype=np.float32)
                self.action_loss_mask[list(self.action_spec.target_slots)] = raw_loss_mask
            else:
                raise ValueError(
                    f"loss.action_loss_mask shape {raw_loss_mask.shape} must be "
                    f"({RHINO72_DIM},) or ({len(self.action_spec.target_slots)},)"
                )

        # ----- Build the underlying LeRobotDataset -----
        # IMPORTANT: we deliberately do NOT pass `episodes=...` to LeRobot.
        # LeRobot 0.1.0's `_get_query_indices` indexes `episode_data_index[ep_idx]`
        # using the ORIGINAL episode_index stored in each row, but
        # `episode_data_index` is only sized for the filtered episode set.
        # For example, episodes=[2] -> episode_data_index has 1 entry but
        # rows still carry episode_index=2, so `__getitem__` raises
        # IndexError. We load the whole dataset and filter at this layer by
        # intersecting `valid_chunk_start==1` with the requested episode ids.
        if video_backend == "pyav":
            _install_lerobot_pyav_seek_workaround()
        elif video_backend == "torchcodec":
            _install_lerobot_torchcodec_exact_decoder()
        LeRobotDataset = _lazy_lerobot_dataset_cls()
        delta_timestamps = {
            self.action_source_key: [k / float(self.native_mapping.fps) for k in range(self.action_horizon)]
        }
        self.episodes_filter = [int(x) for x in episodes] if episodes else None

        self.lerobot = LeRobotDataset(
            repo_id=self.native_mapping.repo_id,
            root=self.native_mapping.root,
            episodes=None,  # see comment above — adapter-layer filter instead
            delta_timestamps=delta_timestamps,
            video_backend=video_backend,
            tolerance_s=1.0 / float(self.native_mapping.fps) - 1e-4,
        )

        self._task_str_by_index: dict[int, str] = {}
        tasks_path = Path(self.native_mapping.root) / "meta" / "tasks.parquet"
        if tasks_path.exists():
            import pandas as pd  # noqa: WPS433

            tasks_pq = pd.read_parquet(tasks_path)
            if "task_index" in tasks_pq.columns:
                self._task_str_by_index = {
                    int(ti): str(task)
                    for task, ti in zip(tasks_pq.index, tasks_pq["task_index"])
                }

        # Avoid decoding video features not declared in camera_mapping.
        try:
            mapping_video_keys = set(self.camera_source_keys)
            current_video_keys = list(self.lerobot.meta.video_keys)
            dropped = [vk for vk in current_video_keys if vk not in mapping_video_keys]
            for vk in dropped:
                self.lerobot.meta.features.pop(vk, None)
            if dropped:
                kept = [vk for vk in self.lerobot.meta.video_keys if vk in mapping_video_keys]
                print(
                    f"[LeRobotAdapter] pruned video_keys, dropped: {dropped}, "
                    f"kept: {kept}",
                    flush=True,
                )
        except Exception as e:
            print(f"[LeRobotAdapter] video_keys prune failed (non-fatal): {e!r}", flush=True)

        # ----- Pre-filter valid chunk starts via parquet column lookup -----
        # We avoid loading frames here — only need the integer columns from
        # each episode's parquet. LeRobot's hf_dataset exposes them directly.
        ep_arr = _hf_column_to_1d_array(self.lerobot.hf_dataset, "episode_index", np.int64)

        if self.valid_chunk_filter == "valid_chunk_start":
            # valid_chunk_start: use the converter's frame/start-level boolean
            # mask as-is. The current row is a valid training start only when
            # valid_chunk_start == 1. This is the only mode that requires the
            # dataset parquet to contain `self.valid_chunk_key`.
            valid_mask = _hf_column_to_1d_array(self.lerobot.hf_dataset, self.valid_chunk_key, np.int64)
        elif self.valid_chunk_filter == "episode_only":
            # episode_only: do not use validity annotations. Row i is valid iff
            # the H-frame action chunk [i, i + H) stays within one episode. This
            # keeps plain jsonl/LeRobot conversions trainable even when no extra
            # quality columns or sidecars exist.
            n_rows = len(ep_arr)
            valid_mask = np.zeros(n_rows, dtype=np.int64)
            if n_rows >= self.action_horizon:
                # Broadcast check: for an H-step contiguous table, same
                # episode_index at the first and last rows is enough to prove
                # the chunk does not cross an episode boundary.
                last_idx = n_rows - self.action_horizon + 1
                same_episode = ep_arr[:last_idx] == ep_arr[self.action_horizon - 1:]
                valid_mask[:last_idx] = same_episode.astype(np.int64)
        else:
            # valid_intervals: use interval-level validity from the sidecar. A
            # start row is valid only if the entire H-frame chunk lies inside a
            # merged valid interval for that episode. This removes chunks that
            # overlap annotated invalid/mistake/error regions while still
            # allowing chunks to cross touching action-segment boundaries.
            frame_arr = _hf_column_to_1d_array(self.lerobot.hf_dataset, "frame_index", np.int64)
            valid_mask = self._valid_intervals_mask(ep_arr, frame_arr)

        if self.episodes_filter is not None:
            episode_mask = np.isin(ep_arr, np.asarray(self.episodes_filter, dtype=np.int64))
            valid_mask = valid_mask * episode_mask.astype(np.int64)

        self.valid_global_indices: List[int] = np.nonzero(valid_mask == 1)[0].tolist()
        if not self.valid_global_indices:
            raise RuntimeError(
                f"no valid chunk starts under filter={self.valid_chunk_filter} "
                f"(episodes_filter={self.episodes_filter}) in {self.mapping['root']}"
            )
        # Log the filtering mode applied to this dataset.
        ep_summary = (
            f"all {len(set(ep_arr.tolist()))}"
            if self.episodes_filter is None else f"{len(self.episodes_filter)}"
        )
        print(
            f"[LeRobotAdapter] mapping={self.mapping['id']} "
            f"valid_chunk_filter={self.valid_chunk_filter} "
            f"episodes={ep_summary} valid_chunk_starts={len(self.valid_global_indices)} "
            f"action_horizon={self.action_horizon} root={self.mapping['root']}",
            flush=True,
        )

        # ----- Sample weights -----
        column_names = set(getattr(self.lerobot.hf_dataset, "column_names", []) or [])
        if self.sample_weight_key in column_names:
            self._sample_weights_full = _hf_column_to_1d_array(
                self.lerobot.hf_dataset,
                self.sample_weight_key,
                np.float64,
            )
        else:
            self._sample_weights_full = np.ones(len(self.lerobot.hf_dataset), dtype=np.float64)
        if self._segment_weight_multiplier_full is not None:
            self._sample_weights_full = self._sample_weights_full * self._segment_weight_multiplier_full
            valid_m = self._segment_weight_multiplier_full[self.valid_global_indices]
            print(
                f"[LeRobotAdapter] segment weights applied: "
                f"skill_keys={sorted(self.segment_weight_by_skill)} "
                f"action_text_keys={sorted(self.segment_weight_by_action_text)} "
                f"valid_multiplier_min={valid_m.min():.4f} "
                f"valid_multiplier_mean={valid_m.mean():.4f} "
                f"valid_multiplier_max={valid_m.max():.4f}",
                flush=True,
            )
        if sample_weight_override_path:
            override_mode = str(sample_weight_override_mode).lower()
            if override_mode not in {"replace", "multiply"}:
                raise ValueError(
                    "sample_weight_override_mode must be 'replace' or 'multiply', "
                    f"got {sample_weight_override_mode!r}"
                )
            weight_df = _load_weight_override_table(sample_weight_override_path)
            required_cols = {"episode_index", "subtask_id", sample_weight_override_column}
            missing_cols = required_cols - set(weight_df.columns)
            if missing_cols:
                raise KeyError(
                    f"sample weight override file {sample_weight_override_path} "
                    f"missing columns: {sorted(missing_cols)}"
                )
            if len(weight_df) != len(weight_df[["episode_index", "subtask_id"]].drop_duplicates()):
                weight_df = weight_df.drop_duplicates(["episode_index", "subtask_id"])
            override_by_key = {
                (int(row.episode_index), int(row.subtask_id)): float(getattr(row, sample_weight_override_column))
                for row in weight_df[["episode_index", "subtask_id", sample_weight_override_column]].itertuples(index=False)
            }
            subtask_arr = _hf_column_to_1d_array(self.lerobot.hf_dataset, "subtask_id", np.int64)
            override = np.asarray(
                [override_by_key.get((int(ep), int(st)), 1.0) for ep, st in zip(ep_arr, subtask_arr)],
                dtype=np.float64,
            )
            if override_mode == "multiply":
                self._sample_weights_full = self._sample_weights_full * override
            else:
                self._sample_weights_full = override
            valid_weights = self._sample_weights_full[self.valid_global_indices]
            print(
                f"[LeRobotAdapter] sample_weight_override={sample_weight_override_path} "
                f"column={sample_weight_override_column} mode={override_mode} "
                f"keys={len(override_by_key)} valid_weight_min={valid_weights.min():.4f} "
                f"valid_weight_mean={valid_weights.mean():.4f} valid_weight_max={valid_weights.max():.4f}",
                flush=True,
            )

        # ----- Image augmentation -----
        self._image_aug = None
        self._role_aug_target_role = None
        self._role_aug_crop_ratio = 1.0
        self._role_aug_rotate_degrees = 0.0
        self._role_aug_size = int(image_aug_size)
        aug_cfg = dict(image_augmentation or {})
        aug_enabled = bool(aug_cfg.get("enabled", False))
        aug_profile = str(aug_cfg.get("profile", image_aug_profile) or "none") if aug_enabled else image_aug_profile
        if aug_profile and aug_profile != "none":
            from torchvision import transforms  # noqa: WPS433
            valid_profiles = {
                "safe_photometric",
                "safe_photometric_tiny_crop",
                "strong_light_safe_color",
            }
            if aug_profile not in valid_profiles:
                raise ValueError(f"Unknown image_aug_profile '{aug_profile}'. Valid: {sorted(valid_profiles)}")
            aug_list = []
            if aug_profile == "safe_photometric_tiny_crop":
                aug_list.append(transforms.RandomResizedCrop(image_aug_size, scale=(0.97, 1.0), ratio=(0.98, 1.02)))
            if aug_profile == "strong_light_safe_color":
                jitter = transforms.ColorJitter(
                    brightness=float(aug_cfg.get("brightness", 0.30)),
                    contrast=float(aug_cfg.get("contrast", 0.25)),
                    saturation=float(aug_cfg.get("saturation", 0.12)),
                    hue=float(aug_cfg.get("hue", 0.01)),
                )
            else:
                jitter = transforms.ColorJitter(
                    brightness=float(aug_cfg.get("brightness", 0.15)),
                    contrast=float(aug_cfg.get("contrast", 0.15)),
                    saturation=float(aug_cfg.get("saturation", 0.08)),
                    hue=float(aug_cfg.get("hue", 0.02)),
                )
            blur_sigma = aug_cfg.get("blur_sigma", [0.1, 0.5])
            aug_list.extend([
                jitter,
                transforms.RandomApply(
                    [
                        transforms.GaussianBlur(
                            kernel_size=int(aug_cfg.get("blur_kernel_size", 3)),
                            sigma=(float(blur_sigma[0]), float(blur_sigma[1])),
                        )
                    ],
                    p=float(aug_cfg.get("blur_p", 0.2)),
                ),
            ])
            self._image_aug = transforms.Compose(aug_list)
        if aug_enabled:
            crop_ratio = float(aug_cfg.get("top_head_crop_ratio", 1.0) or 1.0)
            rotate_degrees = float(aug_cfg.get("top_head_rotate_degrees", 0.0) or 0.0)
            if crop_ratio < 1.0 or rotate_degrees > 0.0:
                if not (0.0 < crop_ratio <= 1.0):
                    raise ValueError(f"top_head_crop_ratio must be in (0, 1], got {crop_ratio}")
                if rotate_degrees < 0.0:
                    raise ValueError(f"top_head_rotate_degrees must be >= 0, got {rotate_degrees}")
                self._role_aug_target_role = str(aug_cfg.get("top_head_target_role", "top_head") or "top_head")
                self._role_aug_crop_ratio = crop_ratio
                self._role_aug_rotate_degrees = rotate_degrees

        # ----- Normalization stats -----
        self.action_mean, self.action_std, self.state_mean, self.state_std = self._resolve_stats(stats_override)
        self._validate_stats()

    def _resolve_sidecar_path(self, configured_path: Optional[str], default_name: str) -> Path:
        if configured_path:
            return Path(configured_path)
        return Path(self.native_mapping.root) / "meta" / default_name

    def _valid_intervals_mask(self, ep_arr: np.ndarray, frame_arr: np.ndarray) -> np.ndarray:
        """Return H-window starts fully contained in merged valid intervals.

        Sidecars are built from manifest label_info.action_config. The interval
        table already removes invalid/mistake/error regions and merges adjacent
        valid action segments, so chunk windows may cross semantic action_text
        boundaries as long as they stay inside one merged valid interval.
        Segment metadata and optional segment-specific weights are read from the
        original unmerged segment table based on the chunk start frame.
        """
        import pandas as pd  # noqa: WPS433

        interval_path = self._resolve_sidecar_path(
            self.valid_intervals_path,
            "episode_valid_intervals.parquet",
        )
        if not interval_path.exists():
            raise FileNotFoundError(
                f"valid_intervals requires valid intervals sidecar: {interval_path}. "
                "Build it from the manifest label_info.action_config first."
            )
        intervals = pd.read_parquet(interval_path)
        required_interval_cols = {"episode_index", "start_frame", "end_frame"}
        missing = required_interval_cols - set(intervals.columns)
        if missing:
            raise KeyError(f"{interval_path} missing columns: {sorted(missing)}")
        if "episode_id" in intervals.columns:
            self._episode_id_by_index = {
                int(row.episode_index): str(row.episode_id)
                for row in intervals[["episode_index", "episode_id"]].drop_duplicates().itertuples(index=False)
            }

        n_rows = len(ep_arr)
        valid_mask = np.zeros(n_rows, dtype=np.int64)
        # Merge contiguous/overlapping intervals per episode into one big valid run
        # BEFORE testing chunk containment: a chunk that spans two
        # touching valid intervals must be kept. Half-open [start_frame, end_frame),
        # so intervals touch when next.start == prev.end. Idempotent if the sidecar
        # is already merged.
        raw_by_episode: dict[int, list[tuple[int, int]]] = {}
        for row in intervals[["episode_index", "start_frame", "end_frame"]].itertuples(index=False):
            s = int(row.start_frame)
            e = int(row.end_frame)
            if e <= s:
                continue
            raw_by_episode.setdefault(int(row.episode_index), []).append((s, e))
        for ep, ep_intervals in raw_by_episode.items():
            merged: list[list[int]] = []
            for s, e in sorted(ep_intervals):
                if not merged or s > merged[-1][1]:
                    merged.append([s, e])
                else:
                    merged[-1][1] = max(merged[-1][1], e)
            idx = np.nonzero(ep_arr == ep)[0]
            if idx.size == 0:
                continue
            frames = frame_arr[idx]
            for s, e in merged:
                if e - s < self.action_horizon:
                    continue
                keep = (frames >= s) & ((frames + self.action_horizon) <= e)
                valid_mask[idx[keep]] = 1

        segment_path = self._resolve_sidecar_path(self.segment_table_path, "episode_segments.parquet")
        self._segment_skill_full = np.full(n_rows, "", dtype=object)
        self._segment_action_text_full = np.full(n_rows, "", dtype=object)
        self._segment_index_full = np.full(n_rows, -1, dtype=np.int64)
        self._segment_weight_multiplier_full = np.full(n_rows, self.segment_weight_default, dtype=np.float64)
        if segment_path.exists():
            segments = pd.read_parquet(segment_path)
            required_segment_cols = {"episode_index", "segment_index", "start_frame", "end_frame", "skill", "action_text"}
            missing = required_segment_cols - set(segments.columns)
            if missing:
                raise KeyError(f"{segment_path} missing columns: {sorted(missing)}")
            for row in segments[
                ["episode_index", "segment_index", "start_frame", "end_frame", "skill", "action_text"]
            ].itertuples(index=False):
                ep = int(row.episode_index)
                start = int(row.start_frame)
                end = int(row.end_frame)
                idx = np.nonzero(ep_arr == ep)[0]
                if idx.size == 0:
                    continue
                frames = frame_arr[idx]
                hit = (frames >= start) & (frames < end)
                hit_idx = idx[hit]
                if hit_idx.size == 0:
                    continue
                skill = str(row.skill)
                action_text = str(row.action_text)
                self._segment_skill_full[hit_idx] = skill
                self._segment_action_text_full[hit_idx] = action_text
                self._segment_index_full[hit_idx] = int(row.segment_index)
                multiplier = float(self.segment_weight_by_skill.get(skill, 1.0))
                multiplier *= float(self.segment_weight_by_action_text.get(action_text, 1.0))
                self._segment_weight_multiplier_full[hit_idx] = self.segment_weight_default * multiplier
        elif self.segment_weight_by_skill or self.segment_weight_by_action_text:
            raise FileNotFoundError(
                f"segment weights require segment table sidecar: {segment_path}"
            )

        valid_count = int(valid_mask.sum())
        if valid_count == 0:
            raise RuntimeError(f"valid_intervals found zero valid starts from {interval_path}")
        print(
            f"[LeRobotAdapter] valid_intervals sidecar={interval_path} "
            f"intervals={len(intervals)} valid_chunk_starts={valid_count}",
            flush=True,
        )
        return valid_mask

    # ------------------------------------------------------------------ stats
    def _resolve_stats(
        self, stats_override: Optional[Dict[str, np.ndarray]]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if stats_override:
            selected_stats = _select_action_norm_stats(
                stats_override,
                action_type=self.action_type,
                delta_slots=self.delta_action_slots,
                absolute_action_slots=self.absolute_action_slots,
            )
            return (
                expand_stats_to_rhino72(selected_stats["action_mean"], spec=self.action_spec, fill_value=0.0),
                expand_stats_to_rhino72(
                    selected_stats["action_std"],
                    spec=self.action_spec,
                    fill_value=1.0,
                    transform_values=False,
                ),
                expand_stats_to_rhino72(selected_stats["state_mean"], spec=self.state_spec, fill_value=0.0),
                expand_stats_to_rhino72(
                    selected_stats["state_std"],
                    spec=self.state_spec,
                    fill_value=1.0,
                    transform_values=False,
                ),
            )

        if self.action_type == ACTION_TYPE_DELTA:
            raise ValueError(
                "delta_from_current_state requires a combined norm_stats_path; "
                "relative action stats cannot fall back to raw absolute-action statistics"
            )

        # Otherwise compute over the full dataset's native source columns and
        # expand into 72D. Production configs normally provide norm.json and do
        # not use this fallback.
        states_src = _hf_vector_column_to_array(self.lerobot.hf_dataset, self.state_source_key, np.float32)
        actions_src = _hf_vector_column_to_array(self.lerobot.hf_dataset, self.action_source_key, np.float32)
        states = pack_source_to_rhino72(
            states_src,
            source_indices=self.state_spec.source_indices,
            target_slots=self.state_spec.target_slots,
            fill_value=0.0,
            value_transforms=self.state_spec.value_transforms,
        )
        actions = pack_source_to_rhino72(
            actions_src,
            source_indices=self.action_spec.source_indices,
            target_slots=self.action_spec.target_slots,
            fill_value=0.0,
            value_transforms=self.action_spec.value_transforms,
        )
        state_mean = states.mean(axis=0).astype(np.float32)
        state_std = np.maximum(states.std(axis=0), 1e-6).astype(np.float32)
        state_std[np.asarray(mask_from_spec(self.state_spec)) < 0.5] = 1.0
        action_mean = actions.mean(axis=0).astype(np.float32)
        action_std = np.maximum(actions.std(axis=0), 1e-6).astype(np.float32)
        action_std[np.asarray(mask_from_spec(self.action_spec)) < 0.5] = 1.0
        return action_mean, action_std, state_mean, state_std

    def _validate_stats(self) -> None:
        for name, arr in (("action_mean", self.action_mean), ("action_std", self.action_std),
                          ("state_mean", self.state_mean), ("state_std", self.state_std)):
            if arr.shape != (RHINO72_DIM,):
                raise ValueError(f"norm stat '{name}' shape {arr.shape} != ({RHINO72_DIM},)")

    # --------------------------- viz/debug helpers (no augmentation/norm) --
    def decode_frame(self, global_row_idx: int) -> List[Image.Image]:
        """Decode the three canonical camera frames at an absolute parquet row.

        Use the episode-local `timestamp` column instead of deriving time from
        the global row index. Decode through the stateless helper because the
        dataset's stateful video query is unsafe alongside dataloader workers.
        """
        row = self.lerobot.hf_dataset[global_row_idx]
        ts_val = row["timestamp"]
        t = float(ts_val.item() if hasattr(ts_val, "item") else ts_val)
        ep_idx_val = row["episode_index"]
        ep_idx = int(ep_idx_val.item() if hasattr(ep_idx_val, "item") else ep_idx_val)

        try:
            from lerobot.datasets.video_utils import decode_video_frames  # 0.5.x
        except ModuleNotFoundError:
            from lerobot.common.datasets.video_utils import decode_video_frames  # 0.1.x

        root = Path(self.native_mapping.root)
        video_backend = getattr(self.lerobot, "video_backend", None) or "pyav"
        tolerance_s = float(self.lerobot.tolerance_s)
        out: List[Image.Image] = []
        for cam in self.camera_source_keys:
            video_path = root / self.lerobot.meta.get_video_file_path(ep_idx, cam)
            frames = decode_video_frames(str(video_path), [t], tolerance_s, video_backend)
            tensor = frames if hasattr(frames, "dim") else frames
            if tensor.dim() == 4:
                tensor = tensor.squeeze(0)
            out.append(_tensor_to_pil(tensor))
        return out

    def _derive_subtask(self, row: dict[str, Any]) -> tuple[int, str, str]:
        if "subtask_id" in row and "subtask" in row:
            st_id = int(_scalar_py(row["subtask_id"]))
            st = str(_scalar_py(row["subtask"]))
            prompt = str(_scalar_py(row.get("subtask_prompt", st)))
            return st_id, st, prompt
        task_index = int(_scalar_py(row.get("task_index", 0)) or 0)
        task = self._task_str_by_index.get(task_index, "")
        # Reserve subtask_id=0 for the trainer's unknown sentinel.
        return task_index + 1, task, task

    def get_row_payload(self, global_row_idx: int) -> Dict[str, Any]:
        """Return the per-row scalar/vector metadata we commonly need outside
        `__getitem__` (viz, TF eval) without re-running the whole normalization
        + video pipeline."""
        row = self.lerobot.hf_dataset[global_row_idx]
        state_src = np.asarray(row[self.state_source_key], dtype=np.float32)
        action_src = np.asarray(row[self.action_source_key], dtype=np.float32)
        state_raw = pack_source_to_rhino72(
            state_src,
            source_indices=self.state_spec.source_indices,
            target_slots=self.state_spec.target_slots,
            fill_value=0.0,
            value_transforms=self.state_spec.value_transforms,
        )
        action_abs_raw = pack_source_to_rhino72(
            action_src,
            source_indices=self.action_spec.source_indices,
            target_slots=self.action_spec.target_slots,
            fill_value=0.0,
            value_transforms=self.action_spec.value_transforms,
        )
        action_raw = _action_to_model_space(
            action_abs_raw,
            state_raw,
            delta_slots=self.delta_action_slots,
        )
        ep_idx = int(_scalar_py(row["episode_index"]))
        original_episode_id = str(_scalar_py(row.get("original_episode_id", "")))
        if not original_episode_id:
            original_episode_id = str(getattr(self, "_episode_id_by_index", {}).get(ep_idx, ""))
        st_id, st, st_prompt = self._derive_subtask(row)
        segment_skill = ""
        segment_action_text = ""
        segment_index = -1
        if self._segment_skill_full is not None:
            segment_skill = str(self._segment_skill_full[global_row_idx])
        if self._segment_action_text_full is not None:
            segment_action_text = str(self._segment_action_text_full[global_row_idx])
        if self._segment_index_full is not None:
            segment_index = int(self._segment_index_full[global_row_idx])
        return {
            "global_row_idx": int(global_row_idx),
            "episode_index": ep_idx,
            "frame_index": int(_scalar_py(row["frame_index"])),
            "subtask_id": st_id,
            "subtask": st,
            "subtask_prompt": st_prompt,
            "state_raw": state_raw,
            "action_raw": action_raw,
            "original_episode_id": original_episode_id,
            "original_frame_index": int(_scalar_py(row.get("original_frame_index", 0))),
            "segment_skill": segment_skill,
            "segment_action_text": segment_action_text,
            "segment_index": segment_index,
        }

    def iter_row_payload(self) -> Iterable[Dict[str, Any]]:
        """Yield per-row metadata for every row in the underlying hf_dataset
        that belongs to this split (respects `episodes_filter`).

        Filtering prevents visualization and TF-error selection from crossing
        between train and validation splits.

        Heavier than reading raw columns directly but returns a uniform
        structure callers can reason about. No vector columns (state / action)
        are loaded — call `get_row_payload()` for those.
        """
        n_rows = len(self.lerobot.hf_dataset)
        columns = set(getattr(self.lerobot.hf_dataset, "column_names", []) or [])
        ep_col = self.lerobot.hf_dataset["episode_index"]
        frame_col = self.lerobot.hf_dataset["frame_index"]
        subtask_id_col = self.lerobot.hf_dataset["subtask_id"] if "subtask_id" in columns else None
        subtask_col = self.lerobot.hf_dataset["subtask"] if "subtask" in columns else None
        prompt_col = self.lerobot.hf_dataset["subtask_prompt"] if "subtask_prompt" in columns else None
        task_index_col = self.lerobot.hf_dataset["task_index"] if "task_index" in columns else None
        orig_id_col = self.lerobot.hf_dataset["original_episode_id"] if "original_episode_id" in columns else None
        orig_frame_col = self.lerobot.hf_dataset["original_frame_index"] if "original_frame_index" in columns else None

        allowed = set(self.episodes_filter) if self.episodes_filter is not None else None
        valid_rows = getattr(self, "valid_global_indices", None)
        row_iter = valid_rows if valid_rows is not None else range(n_rows)
        for i in row_iter:
            ep = int(_scalar_py(ep_col[i]))
            if allowed is not None and ep not in allowed:
                continue
            original_episode_id = str(_scalar_py(orig_id_col[i])) if orig_id_col is not None else ""
            if not original_episode_id:
                original_episode_id = str(getattr(self, "_episode_id_by_index", {}).get(ep, ""))
            if subtask_id_col is not None and subtask_col is not None:
                st_id = int(_scalar_py(subtask_id_col[i]))
                st = str(_scalar_py(subtask_col[i]))
                st_prompt = str(_scalar_py(prompt_col[i])) if prompt_col is not None else st
            else:
                ti = int(_scalar_py(task_index_col[i])) if task_index_col is not None else 0
                st_id = ti + 1
                st = self._task_str_by_index.get(ti, "")
                st_prompt = st
            yield {
                "global_row_idx": i,
                "episode_index": ep,
                "frame_index": int(_scalar_py(frame_col[i])),
                "subtask_id": st_id,
                "subtask": st,
                "subtask_prompt": st_prompt,
                "original_episode_id": original_episode_id,
                "original_frame_index": int(_scalar_py(orig_frame_col[i])) if orig_frame_col is not None else int(_scalar_py(frame_col[i])),
                "segment_skill": str(self._segment_skill_full[i]) if self._segment_skill_full is not None else "",
                "segment_action_text": str(self._segment_action_text_full[i]) if self._segment_action_text_full is not None else "",
                "segment_index": int(self._segment_index_full[i]) if self._segment_index_full is not None else -1,
            }

    # ------------------------------------------------------------------ misc
    def sample_weights(self) -> np.ndarray:
        return self._sample_weights_full[self.valid_global_indices]

    def __len__(self) -> int:
        return len(self.valid_global_indices)

    # --------------------------------------------------------------- getitem
    def _resolve_instruction(self, raw_sample: dict[str, Any]) -> str:
        if self.fixed_instruction:
            return self.fixed_instruction
        # Prefer the annotation-segment-derived prompt (boundary-correct). Falls
        # through to subtask_prompt/fallbacks if the segment text is absent
        # (e.g. non-valid_intervals mode) or matches no known object.
        if self.prompt_from_segment:
            seg_text = raw_sample.get("segment_action_text", "")
            if isinstance(seg_text, str) and seg_text:
                for needle, prompt in self.action_text_to_prompt.items():
                    if needle in seg_text:
                        return prompt
        for key in (self.instruction_source, *self.instruction_fallbacks):
            if key not in raw_sample:
                continue
            val = raw_sample[key]
            if isinstance(val, torch.Tensor):
                continue  # numeric tensor isn't a valid instruction
            if isinstance(val, str) and val:
                return val
        _st_id, _st, prompt = self._derive_subtask(raw_sample)
        if prompt:
            return prompt
        raise KeyError(
            f"no usable instruction in sample: tried {[self.instruction_source, *self.instruction_fallbacks]}"
        )

    def _select_image(self, tensor_or_pil: Any) -> Image.Image:
        if isinstance(tensor_or_pil, torch.Tensor):
            img = _tensor_to_pil(tensor_or_pil)
        elif isinstance(tensor_or_pil, Image.Image):
            img = tensor_or_pil
        else:
            raise TypeError(f"unexpected image type from LeRobot: {type(tensor_or_pil)}")
        sw, sh = img.size
        if sw <= 0 or sh <= 0:
            raise ValueError(f"image dimensions must be positive, got {(sw, sh)}")
        scale = float(_IMAGE_SHORT_SIDE) / float(min(sw, sh))
        target_h, target_w = qwen_smart_resize(
            float(sh) * scale,
            float(sw) * scale,
            factor=_IMAGE_RESIZE_FACTOR,
            min_pixels=_IMAGE_SHORT_SIDE * _IMAGE_SHORT_SIDE,
            max_pixels=_IMAGE_SHORT_SIDE * _IMAGE_MAX_LONG_SIDE,
        )
        return img.convert("RGB").resize((int(target_w), int(target_h)), Image.BICUBIC)

    def _apply_role_image_aug(self, image: Image.Image, role: Optional[str]) -> Image.Image:
        if self._role_aug_target_role is None or str(role or "") != self._role_aug_target_role:
            return image
        if self._role_aug_crop_ratio < 1.0:
            width, height = image.size
            crop_w = max(1, min(width, int(round(width * self._role_aug_crop_ratio))))
            crop_h = max(1, min(height, int(round(height * self._role_aug_crop_ratio))))
            rng = np.random.default_rng()
            left = int(rng.integers(0, width - crop_w + 1)) if crop_w < width else 0
            top = int(rng.integers(0, height - crop_h + 1)) if crop_h < height else 0
            image = image.crop((left, top, left + crop_w, top + crop_h)).resize(
                (self._role_aug_size, self._role_aug_size),
                Image.Resampling.BILINEAR,
            )
        if self._role_aug_rotate_degrees > 0.0:
            angle = float(np.random.uniform(-self._role_aug_rotate_degrees, self._role_aug_rotate_degrees))
            image = image.rotate(angle, resample=Image.Resampling.BILINEAR, fillcolor=(0, 0, 0))
        return image

    def _apply_top_head_crop(self, image: Image.Image, role: Optional[str]) -> Image.Image:
        """Apply the configured deterministic center crop to the top-head view."""
        crop_size = self.top_head_crop_size
        if crop_size is None or str(role or "") != "top_head":
            return image
        image_width, image_height = image.size
        crop_width, crop_height = crop_size
        if crop_width > image_width or crop_height > image_height:
            raise ValueError(
                f"top_head crop exceeds resized image: image={image.size}, crop={crop_size}"
            )
        left = (image_width - crop_width) // 2
        top = (image_height - crop_height) // 2
        return image.crop((left, top, left + crop_width, top + crop_height))

    @staticmethod
    def _mean_image(img: Image.Image) -> Image.Image:
        arr = np.asarray(img.convert("RGB"), dtype=np.float32)
        color = tuple(np.clip(arr.mean(axis=(0, 1)), 0, 255).astype(np.uint8).tolist())
        return Image.new("RGB", img.size, color)

    def _apply_camera_dropout(self, pil_images: List[Image.Image]) -> tuple[List[Image.Image], str]:
        if not self.camera_dropout_enabled or self.camera_dropout_p <= 0:
            return pil_images, ""
        if np.random.random() >= self.camera_dropout_p:
            return pil_images, ""

        eligible = [
            idx
            for idx, alias in enumerate(self.camera_order)
            if alias in self.camera_dropout_candidates and alias not in self.camera_dropout_never_drop
        ]
        if not eligible:
            return pil_images, ""

        drop_idx = int(np.random.choice(eligible))
        dropped = self.camera_order[drop_idx]
        out = list(pil_images)
        out[drop_idx] = self._mean_image(out[drop_idx])
        return out, dropped

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        global_i = self.valid_global_indices[idx]
        sample = self.lerobot[global_i]

        # No-pad invariant: every valid_chunk_filter mode must keep only starts
        # whose H-step action chunk fits inside the episode.
        pad_key = f"{self.action_source_key}_is_pad"
        if pad_key in sample:
            pad = sample[pad_key]
            if torch.is_tensor(pad) and pad.any():
                raise RuntimeError(
                    f"adapter idx {idx} (global {global_i}) produced a padded action chunk despite "
                    f"valid_chunk_filter={self.valid_chunk_filter}; converter and loader are out of sync."
                )

        state_raw_source = sample[self.state_source_key].cpu().numpy().astype(np.float32)
        action_raw_source = sample[self.action_source_key].cpu().numpy().astype(np.float32)
        if action_raw_source.ndim != 2 or action_raw_source.shape[0] != self.action_horizon:
            raise RuntimeError(
                f"action chunk shape {action_raw_source.shape} != "
                f"(action_horizon={self.action_horizon}, <source_dim>)"
            )

        state_raw = pack_source_to_rhino72(
            state_raw_source,
            source_indices=self.state_spec.source_indices,
            target_slots=self.state_spec.target_slots,
            fill_value=0.0,
            value_transforms=self.state_spec.value_transforms,
        )
        action_abs_raw = pack_source_to_rhino72(
            action_raw_source,
            source_indices=self.action_spec.source_indices,
            target_slots=self.action_spec.target_slots,
            fill_value=0.0,
            value_transforms=self.action_spec.value_transforms,
        )
        action_raw = _action_to_model_space(
            action_abs_raw,
            state_raw,
            delta_slots=self.delta_action_slots,
        )
        normalized_state = (state_raw - self.state_mean) / self.state_std
        normalized_action = (action_raw - self.action_mean[None, :]) / self.action_std[None, :]
        action_mask = np.broadcast_to(mask_from_spec(self.action_spec), normalized_action.shape).copy()
        state_mask = np.broadcast_to(mask_from_spec(self.state_spec), normalized_state[None, :].shape).copy()

        roles = getattr(self, "view_roles", None) or self.camera_order
        pil_images = [
            self._apply_top_head_crop(
                self._select_image(sample[key]),
                roles[index] if index < len(roles) else None,
            )
            for index, key in enumerate(self.camera_source_keys)
        ]
        if self._image_aug is not None:
            pil_images = [self._image_aug(img) for img in pil_images]
        if self._role_aug_target_role is not None:
            pil_images = [
                self._apply_role_image_aug(img, roles[i] if i < len(roles) else None)
                for i, img in enumerate(pil_images)
            ]
        pil_images, dropped_camera = self._apply_camera_dropout(pil_images)

        segment_skill = ""
        segment_action_text = ""
        segment_index = -1
        if self._segment_skill_full is not None:
            segment_skill = str(self._segment_skill_full[global_i])
            sample["segment_skill"] = segment_skill
        if self._segment_action_text_full is not None:
            segment_action_text = str(self._segment_action_text_full[global_i])
            sample["segment_action_text"] = segment_action_text
        if self._segment_index_full is not None:
            segment_index = int(self._segment_index_full[global_i])
            sample["segment_index"] = segment_index

        instruction = self._resolve_instruction(sample)
        ep_id_raw = sample.get("original_episode_id", "") if isinstance(sample, dict) else ""
        if not isinstance(ep_id_raw, str):
            ep_id_raw = str(ep_id_raw)
        if not ep_id_raw:
            ep_idx = int(_scalar_py(sample.get("episode_index", -1))) if isinstance(sample, dict) else -1
            ep_id_raw = str(getattr(self, "_episode_id_by_index", {}).get(ep_idx, ""))

        weight = float(self._sample_weights_full[global_i])

        return {
            "lang": instruction,
            "image": pil_images,
            "view_roles": getattr(self, "view_roles", None),
            "view_modalities": getattr(self, "view_modalities", None),
            "action": normalized_action,
            "action_raw": action_raw,
            "action_mask": action_mask,
            "state": normalized_state[None, :],
            "state_raw": state_raw[None, :],
            "state_mask": state_mask,
            "supervision_weight": action_mask.copy(),
            "_episode_id": ep_id_raw,
            "_frame_index": int(sample["frame_index"].item()) if torch.is_tensor(sample.get("frame_index")) else int(global_i),
            "_original_frame_index": int(_scalar_py(sample.get("original_frame_index", global_i))),
            "_dataset_id": self.mapping["id"],
            "_sample_weight": weight,
            "_segment_skill": segment_skill,
            "_segment_action_text": segment_action_text,
            "_segment_index": segment_index,
            "_action_loss_mask": self.action_loss_mask,
            "_camera_order": list(self.camera_order),
            "_camera_dropout": dropped_camera,
        }


# ---------------------------------------------------------------------------
# `build_dataloader` entry point.
# ---------------------------------------------------------------------------
def _norm_payload_to_dict(value: Any) -> dict[str, Any]:
    if OmegaConf.is_config(value):
        return OmegaConf.to_container(value, resolve=True)  # type: ignore[return-value]
    return dict(value)


def _resolve_lerobot_stats(vla_cfg, split: str) -> Optional[Dict[str, Any]]:
    """Return configured normalization stats, or None when fallback is allowed.

    Raises a fatal error if `norm_stats_path` is configured but the file is
    missing AND `allow_norm_stats_fallback` is not set to true. Native72
    training must not silently drift away from the normalization file.
    """
    norm_stats_path = vla_cfg.get("norm_stats_path", None)
    if norm_stats_path:
        path = Path(norm_stats_path)
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        if not bool(vla_cfg.get("allow_norm_stats_fallback", False)):
            raise FileNotFoundError(
                f"norm_stats_path {path} not found and allow_norm_stats_fallback "
                f"is not set. Normalization stats must exist before a "
                f"LeRobot train/{split} run can start; otherwise the action "
                f"space drifts. Provide the expected stats json or set "
                f"allow_norm_stats_fallback: true to explicitly accept the drift."
            )
        # If an inline stats payload is available (val path sets this from the
        # train loader's already-resolved stats), prefer it over a per-dataset
        # recompute — that keeps train/val on the same canonical space.
        if vla_cfg.get("norm_stats", None):
            return _norm_payload_to_dict(vla_cfg.norm_stats)
        print(
            f"[WARN] norm_stats_path {path} missing but "
            f"allow_norm_stats_fallback=true — computing per-dataset stats.",
            flush=True,
        )
        return None
    if vla_cfg.get("norm_stats", None):
        return _norm_payload_to_dict(vla_cfg.norm_stats)
    return None


def _resolve_episode_filter(vla_cfg, split: str) -> Optional[list[int]]:
    """Load `meta/{split}_episodes.json` written by the converter, or honour
    an explicit `val_episodes` / `train_episodes` list from the config.

    Returns None when no split file is present and no inline list is set
    (the caller treats that as "use all episodes"). For the train split we
    also tolerate missing metadata because early smoke runs may skip the
    split step.
    """
    inline_key = f"{split}_episodes"
    inline = vla_cfg.get(inline_key, None)
    if inline:
        return [int(x) for x in inline]

    mapping_path = Path(vla_cfg.get("mapping_path"))
    mapping = load_lerobot_mapping(
        mapping_path,
        dataset_id=vla_cfg.get("mapping_dataset_id", None),
    )
    root = Path(mapping.root)
    split_file = root / "meta" / f"{split}_episodes.json"
    if split_file.exists():
        with open(split_file, "r", encoding="utf-8") as f:
            payload = json.load(f)
        return [int(x) for x in payload.get("episode_indices", [])]

    # Val must be an explicit list — silently training on all episodes as val
    # would produce misleading eval metrics.
    if split == "val":
        return None
    return None


def validate_native72_mixture_config(
    vla_cfg: Any,
    *,
    split: str,
    global_batch_size: int,
) -> list[dict[str, Any]]:
    """Validate a native72 mixture before constructing any video datasets."""
    mixture_cfg = vla_cfg.get("mixture", None)
    if not mixture_cfg:
        raise ValueError("vla_data.mixture is required for native72 mixture validation")
    entries_cfg = mixture_cfg.get("datasets", None)
    if not entries_cfg:
        raise ValueError("vla_data.mixture.datasets must contain at least one dataset")

    required_fields = {
        "id",
        "weight",
        "mapping_path",
        "mapping_dataset_id",
        "action_type",
    }
    entries: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    first_entry = entries_cfg[0]
    norm_value = vla_cfg.get("norm_stats_path", None) or first_entry.get("norm_stats_path", None)
    norm_path = Path(str(norm_value or ""))
    if not norm_path.is_file():
        raise FileNotFoundError(
            f"vla_data.norm_stats_path must name the shared mixture norm file: {norm_path}"
        )
    with norm_path.open("r", encoding="utf-8") as handle:
        norm = json.load(handle)
    for entry_index, entry_cfg in enumerate(entries_cfg):
        if OmegaConf.is_config(entry_cfg):
            entry = OmegaConf.to_container(entry_cfg, resolve=True)
        else:
            entry = dict(entry_cfg)
        if not isinstance(entry, dict):
            raise ValueError(f"mixture dataset entry {entry_index} must be a mapping")
        missing = sorted(required_fields - set(entry))
        if missing:
            raise KeyError(
                f"mixture dataset entry {entry_index} missing required field(s): {', '.join(missing)}"
            )

        dataset_id = str(entry["id"])
        if not dataset_id:
            raise ValueError(f"mixture dataset entry {entry_index} id must be non-empty")
        if dataset_id in seen_ids:
            raise ValueError(f"mixture dataset ids must be unique; duplicate {dataset_id!r}")
        seen_ids.add(dataset_id)

        weight = float(entry["weight"])
        if not np.isfinite(weight) or weight <= 0:
            raise ValueError(
                f"mixture dataset {dataset_id!r} weight must be finite and strictly positive"
            )
        entry["weight"] = weight
        entry["action_type"] = _normalize_action_type(entry["action_type"])

        mapping_path = Path(str(entry["mapping_path"]))
        source_norm = entry.get("norm_stats_path", norm_path)
        if Path(str(source_norm)).resolve() != norm_path.resolve():
            raise ValueError(
                f"mixture source {dataset_id!r} must use shared norm {norm_path}, got {source_norm}"
            )
        entry["norm_stats_path"] = str(norm_path)
        if not mapping_path.is_file():
            raise FileNotFoundError(
                f"mixture dataset {dataset_id!r} mapping_path does not exist: {mapping_path}"
            )

        mapping_dataset_id = str(entry["mapping_dataset_id"])
        try:
            mapping = load_lerobot_mapping(mapping_path, dataset_id=mapping_dataset_id)
        except KeyError as exc:
            raise KeyError(
                f"mixture dataset {dataset_id!r} mapping_dataset_id "
                f"{mapping_dataset_id!r} was not found in {mapping_path}"
            ) from exc
        if mapping.dataset_id != dataset_id or mapping_dataset_id != dataset_id:
            raise ValueError(
                f"mixture id/mapping_dataset_id must match selected mapping dataset id: "
                f"id={dataset_id!r}, mapping_dataset_id={mapping_dataset_id!r}, "
                f"selected={mapping.dataset_id!r}"
            )
        dataset_root = Path(mapping.root)
        if not dataset_root.exists():
            raise FileNotFoundError(
                f"mixture dataset {dataset_id!r} root does not exist: {dataset_root}"
            )
        active_action_slots = {int(slot) for slot in mapping.action.active_slots}
        absolute_action_slots = sorted(
            int(slot)
            for slot in mapping.raw.get("absolute_action_slots", [])
            if int(slot) in active_action_slots
        )
        delta_slots = _resolve_action_delta_slots(
            action_type=entry["action_type"],
            active_state_slots=mapping.state.active_slots,
            active_action_slots=mapping.action.active_slots,
            absolute_action_slots=absolute_action_slots,
        )
        selected_norm = _select_action_norm_stats(
            norm,
            action_type=entry["action_type"],
            delta_slots=delta_slots,
            absolute_action_slots=absolute_action_slots,
        )
        for stat_name, values in selected_norm.items():
            array = np.asarray(values, dtype=np.float64)
            if array.ndim != 1 or not np.isfinite(array).all():
                raise ValueError(
                    f"mixture dataset {dataset_id!r} norm {stat_name} must be a finite 1D array"
                )
            if stat_name.endswith("_std") and np.any(array <= 0):
                raise ValueError(
                    f"mixture dataset {dataset_id!r} norm {stat_name} must be strictly positive"
                )
        entries.append(entry)

    if split == "train":
        epoch_samples = int(mixture_cfg.get("epoch_samples", 0))
        batch_size = int(global_batch_size)
        if epoch_samples <= 0:
            raise ValueError(f"mixture epoch_samples must be positive, got {epoch_samples}")
        if batch_size <= 0:
            raise ValueError(f"global_batch_size must be positive, got {global_batch_size}")
        if epoch_samples % batch_size != 0:
            raise ValueError(
                f"mixture epoch_samples={epoch_samples} must be divisible by "
                f"global_batch_size={batch_size}"
            )
    return entries


def _build_mixture_dataloader(cfg: Any, vla_cfg: Any, split: str, **kwargs: Any) -> DataLoader | None:
    qwen_processor = kwargs.get("qwen_processor")
    per_device_batch_size = int(vla_cfg.get("per_device_batch_size", 1))
    if per_device_batch_size <= 0:
        raise ValueError(f"per_device_batch_size must be positive, got {per_device_batch_size}")
    world_size = int(kwargs.get("num_processes", os.environ.get("WORLD_SIZE", 1)))
    global_batch_size = int(
        kwargs.get("global_batch_size", per_device_batch_size * world_size)
    )
    entries = validate_native72_mixture_config(
        vla_cfg,
        split=split,
        global_batch_size=global_batch_size,
    )

    if OmegaConf.is_config(vla_cfg):
        base_cfg = OmegaConf.to_container(vla_cfg, resolve=True)
    else:
        base_cfg = dict(vla_cfg)
    if not isinstance(base_cfg, dict):
        raise ValueError("vla_data config must be a mapping")
    base_cfg.pop("mixture", None)

    datasets: list[LeRobotNative72Dataset] = []
    source_ids: list[str] = []
    dataset_weights: list[float] = []
    for entry in entries:
        entry_cfg = OmegaConf.merge(OmegaConf.create(base_cfg), OmegaConf.create(entry))
        stats_override = _resolve_lerobot_stats(entry_cfg, split=split)
        episodes_filter = _resolve_episode_filter(entry_cfg, split=split)
        if split == "val" and not episodes_filter:
            continue

        image_size = entry_cfg.get("image_size", [256, 256])
        image_size = list(image_size) if hasattr(image_size, "__iter__") else [int(image_size)]
        aug_profile = str(entry_cfg.get("image_aug_profile", "none")) if split == "train" else "none"
        dataset = LeRobotNative72Dataset(
            mapping_path=entry_cfg.mapping_path,
            action_horizon=int(entry_cfg.get("action_horizon", 30)),
            stats_override=stats_override,
            episodes=episodes_filter,
            image_aug_profile=aug_profile,
            image_aug_size=int(image_size[0]),
            image_augmentation=dict(entry_cfg.get("image_augmentation", {})) if split == "train" else None,
            top_head_crop_size=entry_cfg.get("top_head_crop_size", None),
            video_backend=str(entry_cfg.get("video_backend", "pyav")),
            fixed_instruction=entry_cfg.get("fixed_instruction", None),
            sample_weight_override_path=(
                entry_cfg.get("sample_weight_override_path", None) if split == "train" else None
            ),
            sample_weight_override_column=str(
                entry_cfg.get("sample_weight_override_column", "sample_weight")
            ),
            camera_dropout=entry_cfg.get("camera_dropout", None) if split == "train" else None,
            sample_weight_override_mode=str(
                entry_cfg.get("sample_weight_override_mode", "replace")
            ),
            mapping_dataset_id=entry_cfg.mapping_dataset_id,
            action_type=str(entry_cfg.action_type),
        )
        view_roles = entry_cfg.get("view_roles", None)
        view_modalities = entry_cfg.get("view_modalities", None)
        dataset.view_roles = [str(value) for value in view_roles] if view_roles else None
        dataset.view_modalities = [str(value) for value in view_modalities] if view_modalities else None
        datasets.append(dataset)
        source_ids.append(str(entry["id"]))
        dataset_weights.append(float(entry["weight"]))

    if not datasets:
        return None
    dataset = Native72MixtureDataset(datasets=datasets, source_ids=source_ids)

    if split == "train":
        primary = datasets[0]
        vla_cfg.state_mean = primary.state_mean.tolist()
        vla_cfg.state_std = primary.state_std.tolist()
        vla_cfg.action_mean = primary.action_mean.tolist()
        vla_cfg.action_std = primary.action_std.tolist()
        vla_cfg.action_type = primary.action_type
        vla_cfg.action_delta_from_state_dims = list(primary.delta_action_slots)
        vla_cfg.absolute_action_slots = list(primary.absolute_action_slots)

    sampler = None
    shuffle = split == "val"
    if split == "train":
        mixture_cfg = vla_cfg.mixture
        sampler = DeterministicMixtureSampler(
            dataset=dataset,
            dataset_weights=dataset_weights,
            epoch_samples=int(mixture_cfg.epoch_samples),
            seed=int(mixture_cfg.get("seed", 0)),
        )
        shuffle = False

    num_workers = int(vla_cfg.get("num_workers", 1))
    preprocess_qwen = bool(vla_cfg.get("preprocess_qwen_in_dataloader", False))
    if preprocess_qwen and qwen_processor is not None:
        collate_fn = build_qwen_cpu_preprocess_collator(vla_cfg, processor=qwen_processor)
    elif preprocess_qwen and split == "train":
        raise ValueError("preprocess_qwen_in_dataloader=true requires the training Qwen processor")
    else:
        collate_fn = lambda batch: batch
    dl_kwargs = {
        "batch_size": per_device_batch_size,
        "shuffle": shuffle,
        "sampler": sampler,
        "num_workers": num_workers,
        "collate_fn": collate_fn,
        "pin_memory": bool(vla_cfg.get("pin_memory", True)),
    }
    if num_workers > 0:
        dl_kwargs["persistent_workers"] = bool(vla_cfg.get("persistent_workers", True))
        dl_kwargs["prefetch_factor"] = int(vla_cfg.get("prefetch_factor", 4))
    return DataLoader(dataset, **dl_kwargs)


def build_dataloader(
    cfg,
    dataset_py: str = "lerobot_native72",
    split: str = "train",
    *,
    qwen_processor: Any | None = None,
    **_kwargs,
) -> DataLoader | None:
    if dataset_py != "lerobot_native72":
        raise ValueError(f"Unsupported dataset_py: {dataset_py}")

    vla_cfg = cfg.datasets.vla_data
    if vla_cfg.get("mixture", None):
        return _build_mixture_dataloader(
            cfg,
            vla_cfg,
            split,
            qwen_processor=qwen_processor,
            **_kwargs,
        )
    mapping_path = vla_cfg.get("mapping_path", None)
    if not mapping_path:
        raise ValueError("vla_data.mapping_path is required for dataset_py=lerobot_native72")

    stats_override = _resolve_lerobot_stats(vla_cfg, split=split)
    episodes_filter = _resolve_episode_filter(vla_cfg, split=split)

    # For val: if a split file was not written by the converter and the config
    # doesn't supply val_episodes explicitly, surface None so the caller can
    # decide (train.py falls back to a warning instead of treating the full
    # dataset as val).
    if split == "val" and not episodes_filter:
        return None

    image_size = vla_cfg.get("image_size", [256, 256])
    if hasattr(image_size, "__iter__"):
        image_size = list(image_size)
    else:
        image_size = [int(image_size), int(image_size)]

    # Disable augmentation for val — we want deterministic metrics.
    aug_profile = str(vla_cfg.get("image_aug_profile", "none")) if split == "train" else "none"

    dataset = LeRobotNative72Dataset(
        mapping_path=mapping_path,
        action_horizon=int(vla_cfg.get("action_horizon", 30)),
        stats_override=stats_override,
        episodes=episodes_filter,
        image_aug_profile=aug_profile,
        image_aug_size=int(image_size[0]),
        image_augmentation=dict(vla_cfg.get("image_augmentation", {})) if split == "train" else None,
        top_head_crop_size=vla_cfg.get("top_head_crop_size", None),
        video_backend=str(vla_cfg.get("video_backend", "pyav")),
        fixed_instruction=vla_cfg.get("fixed_instruction", None),
        sample_weight_override_path=vla_cfg.get("sample_weight_override_path", None) if split == "train" else None,
        sample_weight_override_column=str(vla_cfg.get("sample_weight_override_column", "sample_weight")),
        camera_dropout=vla_cfg.get("camera_dropout", None) if split == "train" else None,
        sample_weight_override_mode=str(vla_cfg.get("sample_weight_override_mode", "replace")),
        mapping_dataset_id=vla_cfg.get("mapping_dataset_id", None),
        action_type=str(vla_cfg.get("action_type", ACTION_TYPE_ABSOLUTE)),
    )

    # Per-camera labels are attached to every example so the compact view prompt
    # can preserve the relationship between image order and camera role.
    _vr = vla_cfg.get("view_roles", None)
    _vm = vla_cfg.get("view_modalities", None)
    dataset.view_roles = [str(x) for x in _vr] if _vr else None
    dataset.view_modalities = [str(x) for x in _vm] if _vm else None

    # Train split seeds the canonical stats exported to config for eval/export.
    # Val uses the same stats (passed in via stats_override) — don't overwrite.
    if split == "train":
        vla_cfg.state_mean = dataset.state_mean.tolist()
        vla_cfg.state_std = dataset.state_std.tolist()
        vla_cfg.action_mean = dataset.action_mean.tolist()
        vla_cfg.action_std = dataset.action_std.tolist()
        vla_cfg.action_type = dataset.action_type
        vla_cfg.action_delta_from_state_dims = list(dataset.delta_action_slots)
        vla_cfg.absolute_action_slots = list(dataset.absolute_action_slots)

    sampler = None
    shuffle = True
    if split == "train" and bool(vla_cfg.get("weighted_sampler", False)):
        weights = torch.as_tensor(dataset.sample_weights(), dtype=torch.float64)
        sampler = WeightedRandomSampler(
            weights=weights,
            num_samples=len(dataset),
            replacement=True,
        )
        shuffle = False
    elif split == "val":
        # Deterministic-ish val iteration; shuffle still OK since we use it
        # for random eval batch selection.
        shuffle = True

    # LeRobot data throughput is video-decode-bound (pyav per-frame). JSONL
    # pipeline got 3.5s/step with 2 workers because it read pre-rendered JPGs;
    # LeRobot at workers=2 drops to 9.4s/step because a single worker has to
    # seek + decode mp4 for every `__getitem__`. `persistent_workers=True` +
    # `prefetch_factor=4` keeps decoders warm across epochs, and
    # `pin_memory=True` moves tensors to pinned host memory for faster
    # H2D copies on the training side.
    num_workers = int(vla_cfg.get("num_workers", 1))
    preprocess_qwen = bool(vla_cfg.get("preprocess_qwen_in_dataloader", False))
    if preprocess_qwen and qwen_processor is not None:
        collate_fn = build_qwen_cpu_preprocess_collator(vla_cfg, processor=qwen_processor)
    elif preprocess_qwen and split == "train":
        raise ValueError("preprocess_qwen_in_dataloader=true requires the training Qwen processor")
    else:
        collate_fn = lambda batch: batch

    dl_kwargs = {
        "batch_size": int(vla_cfg.get("per_device_batch_size", 1)),
        "shuffle": shuffle,
        "sampler": sampler,
        "num_workers": num_workers,
        "collate_fn": collate_fn,
        "pin_memory": bool(vla_cfg.get("pin_memory", True)),
    }
    if num_workers > 0:
        dl_kwargs["persistent_workers"] = bool(vla_cfg.get("persistent_workers", True))
        dl_kwargs["prefetch_factor"] = int(vla_cfg.get("prefetch_factor", 4))
    return DataLoader(dataset, **dl_kwargs)
