"""Normalization stats and native72 mapping helpers for RPU inference."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import yaml


RHINO72_DIM = 72


@dataclass(frozen=True)
class SlotMapping:
    source_indices: list[int]
    target_slots: list[int]
    value_transforms: list[str]
    active_slots: list[int]
    target_dim: int = RHINO72_DIM


@dataclass(frozen=True)
class NormStats:
    """Mean/std stats used to normalize raw state and denormalize actions."""

    state_mean: np.ndarray
    state_std: np.ndarray
    action_mean: np.ndarray
    action_std: np.ndarray
    active_state_slots: Optional[list[int]] = None
    active_action_slots: Optional[list[int]] = None

    @property
    def state_dim(self) -> int:
        return int(self.state_mean.reshape(-1).shape[0])

    @property
    def action_dim(self) -> int:
        return int(self.action_mean.reshape(-1).shape[0])

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        mapping_path: str | Path | None = None,
        mapping_dataset_id: str | None = None,
        unnorm_key: str | None = None,
    ) -> "NormStats":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        stats = cls.from_dict(data, unnorm_key=unnorm_key)
        if mapping_path is not None:
            stats = stats.expand_native72_mapping(mapping_path, mapping_dataset_id=mapping_dataset_id)
        return stats

    @classmethod
    def from_dict(cls, data: dict[str, Any], *, unnorm_key: str | None = None) -> "NormStats":
        if all(k in data for k in ("state_mean", "state_std", "action_mean", "action_std")):
            state_mean = data["state_mean"]
            state_std = data["state_std"]
            action_mean = data["action_mean"]
            action_std = data["action_std"]
        else:
            if unnorm_key is None:
                keys = [k for k, v in data.items() if isinstance(v, dict) and "action" in v]
                if len(keys) != 1:
                    raise ValueError(f"norm stats has {len(keys)} dataset blocks {keys}; pass unnorm_key")
                unnorm_key = keys[0]
            block = data[unnorm_key]
            state_block = block.get("state", {})
            action_block = block.get("action", {})
            state_mean = state_block.get("mean", 0.0)
            state_std = state_block.get("std", 1.0)
            action_mean = action_block["mean"]
            action_std = action_block["std"]

        return cls(
            state_mean=_as_1d_float32(state_mean),
            state_std=_as_1d_float32(state_std),
            action_mean=_as_1d_float32(action_mean),
            action_std=_as_1d_float32(action_std),
            active_state_slots=_read_active_slots(data, "state"),
            active_action_slots=_read_active_slots(data, "action"),
        )

    def expand_native72_mapping(
        self,
        mapping_path: str | Path,
        *,
        mapping_dataset_id: str | None = None,
    ) -> "NormStats":
        mapping = _select_mapping_dataset(_load_mapping(mapping_path), mapping_dataset_id)
        target_dim = int(mapping.get("target_dim", RHINO72_DIM))
        state_mapping = _native_slot_mapping(mapping, "state", target_dim)
        action_mapping = _native_slot_mapping(mapping, "action", target_dim)
        return NormStats(
            state_mean=_expand_norm_vector(
                self.state_mean,
                state_mapping,
                stat_kind="mean",
                default_value=0.0,
                mapping_path=mapping_path,
            ),
            state_std=_expand_norm_vector(
                self.state_std,
                state_mapping,
                stat_kind="std",
                default_value=1.0,
                mapping_path=mapping_path,
            ),
            action_mean=_expand_norm_vector(
                self.action_mean,
                action_mapping,
                stat_kind="mean",
                default_value=0.0,
                mapping_path=mapping_path,
            ),
            action_std=_expand_norm_vector(
                self.action_std,
                action_mapping,
                stat_kind="std",
                default_value=1.0,
                mapping_path=mapping_path,
            ),
            active_state_slots=list(state_mapping.active_slots),
            active_action_slots=list(action_mapping.active_slots),
        )


def _as_1d_float32(value: Any) -> np.ndarray:
    return np.asarray(value, dtype=np.float32).reshape(-1)


def _read_active_slots(data: dict[str, Any], kind: str) -> Optional[list[int]]:
    direct_keys = (
        f"active_{kind}_slots",
        f"active_{kind}_dims",
    )
    for key in direct_keys:
        value = data.get(key)
        if value is not None:
            return [int(x) for x in value]
    meta = data.get("_meta", {}) if isinstance(data.get("_meta", {}), dict) else {}
    for key in (f"{kind}_active_slots", f"{kind}_active_dims", "active_dims"):
        value = meta.get(key)
        if value is not None:
            return [int(x) for x in value]
    if kind == "action":
        value = data.get("active_action_dims")
        if value is not None:
            return [int(x) for x in value]
    return None


def _load_mapping(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"mapping file must contain a mapping object: {path}")
    return data


def _select_mapping_dataset(raw: dict[str, Any], dataset_id: str | None) -> dict[str, Any]:
    datasets = raw.get("datasets")
    if not datasets:
        return dict(raw)
    selected = None
    if dataset_id is not None:
        for entry in datasets:
            if str(entry.get("dataset_id", entry.get("id", ""))) == str(dataset_id):
                selected = dict(entry)
                break
        if selected is None:
            raise KeyError(f"dataset_id={dataset_id!r} not found in mapping datasets")
    else:
        selected = dict(datasets[0])
    merged = {k: v for k, v in raw.items() if k != "datasets"}
    merged.update(selected)
    return merged


def _native_slot_mapping(mapping: dict[str, Any], kind: str, target_dim: int) -> SlotMapping:
    groups = list(mapping.get("native_joint_groups", []) or [])
    if not groups:
        return _legacy_slot_mapping(mapping, kind, target_dim)

    source_indices: list[int] = []
    target_slots: list[int] = []
    transforms: list[str] = []
    indices_key = f"{kind}_source_indices"
    for group in groups:
        if kind == "state" and str(group.get("state_source", "")) == "none":
            continue
        indices = [int(x) for x in (group.get(indices_key) or [])]
        slots = [int(x) for x in (group.get("target_slots") or [])]
        if len(indices) != len(slots):
            raise ValueError(
                f"native_joint_groups.{group.get('slot_group', '<unknown>')} {indices_key} "
                f"length {len(indices)} != target_slots length {len(slots)}"
            )
        transform = str(group.get("value_transform", "identity") or "identity")
        source_indices.extend(indices)
        target_slots.extend(slots)
        transforms.extend([transform] * len(slots))

    active_key = f"active_{kind}_slots"
    active_slots = [int(x) for x in mapping.get(active_key, target_slots)]
    return SlotMapping(
        source_indices=source_indices,
        target_slots=target_slots,
        value_transforms=transforms,
        active_slots=active_slots,
        target_dim=int(target_dim),
    )


def _legacy_slot_mapping(mapping: dict[str, Any], kind: str, target_dim: int) -> SlotMapping:
    cfg = mapping.get(f"{kind}_mapping", {}) or {}
    source_indices = _flatten_dim_list(cfg.get("dims", []))
    slots = [int(x) for x in cfg.get("target_slots", range(len(source_indices)))]
    active_slots = [int(x) for x in mapping.get(f"active_{kind}_slots", slots)]
    return SlotMapping(
        source_indices=source_indices,
        target_slots=slots,
        value_transforms=["identity"] * len(slots),
        active_slots=active_slots,
        target_dim=int(mapping.get("target_dim", target_dim)),
    )


def _flatten_dim_list(spec: Any) -> list[int]:
    if not spec:
        return []
    if isinstance(spec, dict):
        out: list[int] = []
        for key in ("left_arm", "left_gripper", "right_arm", "right_gripper"):
            value = spec.get(key)
            if value is None:
                continue
            if isinstance(value, (list, tuple)):
                out.extend(int(x) for x in value)
            else:
                out.append(int(value))
        return out
    if isinstance(spec, (list, tuple)):
        return [int(x) for x in spec]
    return [int(spec)]


def _apply_stat_transform(value: np.float32, transform: str, stat_kind: str) -> np.float32:
    transform = transform or "identity"
    if transform == "identity":
        return value
    if transform == "one_minus_raw":
        if stat_kind == "mean":
            return np.float32(1.0) - value
        if stat_kind == "std":
            return value
    raise ValueError(f"unsupported value_transform {transform!r} for {stat_kind}")


def _expand_norm_vector(
    values: Any,
    slot_mapping: SlotMapping,
    *,
    stat_kind: str,
    default_value: float,
    mapping_path: str | Path,
) -> np.ndarray:
    arr = _as_1d_float32(values)
    target_dim = int(slot_mapping.target_dim)
    if arr.shape[0] == target_dim:
        return arr.astype(np.float32, copy=True)
    if not slot_mapping.source_indices:
        return np.full(target_dim, float(default_value), dtype=np.float32)

    source_indices = [int(x) for x in slot_mapping.source_indices]
    target_slots = [int(x) for x in slot_mapping.target_slots]
    transforms = [str(x or "identity") for x in slot_mapping.value_transforms]
    if len(source_indices) != len(target_slots) or len(source_indices) != len(transforms):
        raise ValueError("mapping source/target/transform lengths do not match")

    max_source = max(source_indices)
    if arr.shape[0] > max_source:
        selected = arr[source_indices]
    elif arr.shape[0] == len(target_slots):
        selected = arr
    else:
        raise ValueError(
            f"{stat_kind} length {arr.shape[0]} cannot be expanded with mapping {mapping_path}: "
            f"need {target_dim}D stats, native source dim >= {max_source + 1}, "
            f"or active dim {len(target_slots)}"
        )

    out = np.full(target_dim, float(default_value), dtype=np.float32)
    for value, target_slot, transform in zip(selected, target_slots, transforms):
        if target_slot < 0 or target_slot >= target_dim:
            raise ValueError(f"target slot {target_slot} outside target dim {target_dim}")
        out[target_slot] = _apply_stat_transform(value, transform, stat_kind)
    return out.astype(np.float32, copy=False)

