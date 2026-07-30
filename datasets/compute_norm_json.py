#!/usr/bin/env python3
"""Compute RhinoVLA-compatible LeRobot v3 normalization stats.

The output keeps legacy absolute `action_mean/std` fields and always writes
72D absolute and delta-from-current-state action spaces for native72 mappings.
`absolute_action_slots` only defines delta-space exceptions such as scalar
gripper opening; it never controls whether the two spaces are generated.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml


DEFAULT_STATE_KEY = "observation.state"
DEFAULT_ACTION_KEY = "action"
DEFAULT_TARGET_DIM = 16
DEFAULT_SAFE_HALF_RANGE_THRESHOLD = 0.01
DEFAULT_NORMALIZED_RANGE_WARNING = 2.0
DEFAULT_ZERO_CENTER_BALANCE_TOLERANCE = 0.15
DEFAULT_ZERO_CENTER_MEAN_OFFSET_FRACTION = 0.20
RHINO72_DIM = 72
COMMAND_NAME_HINTS = ("velocity", "vel", "cmd", "command", "control")
METHOD = (
    "quantile-minmax-as-affine (NO clip): "
    "x_norm=2*(x-p01)/(p99-p01)-1 == (x-mean)/std with "
    "mean=(p01+p99)/2, std=(p99-p01)/2; half-range<0.01 floored to 1.0"
)


def _load_mapping(mapping_path: Path | None) -> dict[str, Any]:
    if mapping_path is None:
        return {}
    with mapping_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"mapping file must contain a mapping object: {mapping_path}")
    return data


def _select_mapping_dataset(mapping: dict[str, Any], dataset_id: str | None) -> dict[str, Any]:
    datasets = mapping.get("datasets", None)
    if not datasets:
        return mapping
    selected = None
    if dataset_id:
        for entry in datasets:
            if str(entry.get("dataset_id", entry.get("id", ""))) == str(dataset_id):
                selected = dict(entry)
                break
        if selected is None:
            raise KeyError(f"dataset_id={dataset_id!r} not found in mapping datasets")
    else:
        selected = dict(datasets[0])
    merged = {k: v for k, v in mapping.items() if k != "datasets"}
    merged.update(selected)
    return merged


def _native_source_dim(mapping: dict[str, Any], *, kind: str) -> int | None:
    groups = mapping.get("native_joint_groups", [])
    if not groups:
        return None
    indices_key = "state_source_indices" if kind == "state" else "action_source_indices"
    indices: list[int] = []
    for group in groups:
        if kind == "state" and str(group.get("state_source", "")) == "none":
            continue
        group_indices = group.get(indices_key, [])
        if group_indices is None:
            group_indices = []
        indices.extend(int(x) for x in group_indices)
    return max(indices) + 1 if indices else None


def _resolve_contract(
    *,
    mapping_path: Path | None,
    mapping_dataset_id: str | None,
    state_key: str | None,
    action_key: str | None,
    target_dim: int | None,
) -> tuple[str, str, int, int]:
    mapping = _select_mapping_dataset(_load_mapping(mapping_path), mapping_dataset_id)
    state_mapping = mapping.get("state_mapping", {}) if isinstance(mapping.get("state_mapping", {}), dict) else {}
    action_mapping = mapping.get("action_mapping", {}) if isinstance(mapping.get("action_mapping", {}), dict) else {}
    resolved_state_key = state_key or state_mapping.get("source_key") or DEFAULT_STATE_KEY
    resolved_action_key = action_key or action_mapping.get("source_key") or DEFAULT_ACTION_KEY
    if mapping.get("native_joint_groups"):
        resolved_state_key = state_key or mapping.get("state_source") or resolved_state_key
        resolved_action_key = action_key or mapping.get("action_source") or resolved_action_key
        state_dim = target_dim or _native_source_dim(mapping, kind="state") or DEFAULT_TARGET_DIM
        action_dim = target_dim or _native_source_dim(mapping, kind="action") or DEFAULT_TARGET_DIM
    else:
        legacy_dim = target_dim or state_mapping.get("target_dim") or action_mapping.get("target_dim") or DEFAULT_TARGET_DIM
        state_dim = action_dim = legacy_dim
    state_dim = int(state_dim)
    action_dim = int(action_dim)
    if state_dim <= 0:
        raise ValueError(f"state_dim must be positive, got {state_dim}")
    if action_dim <= 0:
        raise ValueError(f"action_dim must be positive, got {action_dim}")
    return str(resolved_state_key), str(resolved_action_key), state_dim, action_dim


def _mapping_target_slots(group: dict[str, Any], count: int) -> list[int]:
    target_slots = group.get("target_slots", [])
    if not isinstance(target_slots, list):
        return []
    if len(target_slots) != count:
        return []
    return [int(x) for x in target_slots]


def _override_array_values(
    *,
    stats: dict[str, np.ndarray],
    key: str,
    values: Any,
    indices: list[int],
    group_name: str,
) -> list[float]:
    if key.endswith("_std"):
        value_name = "std"
    else:
        value_name = "mean"
    override = np.asarray(values, dtype=np.float32)
    if override.ndim != 1:
        raise ValueError(f"{group_name}: norm.{key} must be a 1D list")
    if len(override) != len(indices):
        raise ValueError(
            f"{group_name}: norm.{key} length {len(override)} does not match "
            f"{len(indices)} source indices"
        )
    if not np.all(np.isfinite(override)):
        raise ValueError(f"{group_name}: norm.{key} must contain only finite values")
    if value_name == "std" and np.any(override <= 0):
        raise ValueError(f"{group_name}: norm.{key} values must be positive")
    target = stats[key]
    for source_index, value in zip(indices, override):
        if source_index < 0 or source_index >= len(target):
            raise ValueError(
                f"{group_name}: source index {source_index} is outside {key} dim {len(target)}"
            )
        target[source_index] = value
    return override.tolist()


def _apply_mapping_norm_overrides(stats: dict[str, np.ndarray], mapping: dict[str, Any]) -> list[dict[str, Any]]:
    """Apply explicit norm overrides declared in mapping native_joint_groups.

    Supported schema per group:

    norm:
      state_mean: [...]
      state_std: [...]
      action_mean: [...]
      action_std: [...]
      method: optional free-form text
      reason: optional free-form text

    The list length must match the corresponding source index list.
    """
    groups = mapping.get("native_joint_groups", [])
    if not groups:
        return []

    applied: list[dict[str, Any]] = []
    for group in groups:
        if not isinstance(group, dict):
            continue
        norm = group.get("norm")
        if norm is None:
            continue
        if not isinstance(norm, dict):
            raise ValueError(f"{group.get('slot_group', '<unnamed>')}: norm must be a mapping")
        group_name = str(group.get("slot_group", "<unnamed>"))

        for kind in ("state", "action"):
            indices_key = f"{kind}_source_indices"
            indices = [int(x) for x in (group.get(indices_key) or [])]
            mean_key = f"{kind}_mean"
            std_key = f"{kind}_std"
            override_keys = [key for key in (mean_key, std_key) if key in norm]
            if not override_keys:
                continue
            if not indices:
                raise ValueError(f"{group_name}: norm override for {kind} requires {indices_key}")

            record: dict[str, Any] = {
                "slot_group": group_name,
                "kind": kind,
                "source_indices": indices,
                "target_slots": _mapping_target_slots(group, len(indices)),
            }
            reason = norm.get("reason")
            if reason is not None:
                record["reason"] = str(reason)
            method = norm.get("method")
            if method is not None:
                record["method"] = str(method)

            for key in override_keys:
                stat_key = key
                if stat_key not in stats:
                    raise ValueError(f"{group_name}: unsupported norm override {key}")
                values = _override_array_values(
                    stats=stats,
                    key=stat_key,
                    values=norm[key],
                    indices=indices,
                    group_name=group_name,
                )
                record[key.rsplit("_", 1)[1]] = values
            applied.append(record)
    return applied


def _mapping_dim_info(mapping: dict[str, Any]) -> dict[str, dict[int, dict[str, Any]]]:
    info: dict[str, dict[int, dict[str, Any]]] = {"state": {}, "action": {}}
    for group in mapping.get("native_joint_groups", []):
        if not isinstance(group, dict):
            continue
        group_name = str(group.get("slot_group", "<unnamed>"))
        target_slots = group.get("target_slots", [])
        if not isinstance(target_slots, list):
            target_slots = []
        for kind in ("state", "action"):
            indices = [int(x) for x in (group.get(f"{kind}_source_indices") or [])]
            for offset, source_index in enumerate(indices):
                dim_info: dict[str, Any] = {"slot_group": group_name}
                if offset < len(target_slots):
                    dim_info["target_slot"] = int(target_slots[offset])
                info[kind][source_index] = dim_info
    return info


def _json_float(value: float | np.floating | None) -> float | None:
    if value is None:
        return None
    value = float(value)
    if not np.isfinite(value):
        return None
    return value


def _is_command_like_action(dim_info: dict[str, Any]) -> bool:
    group_name = str(dim_info.get("slot_group", "")).lower()
    target_slot = dim_info.get("target_slot")
    if target_slot in (58, 59, 60):
        return True
    return any(hint in group_name for hint in COMMAND_NAME_HINTS)


def _warning_message(record: dict[str, Any], code: str) -> str:
    kind = record["kind"]
    index = record["index"]
    group = record.get("slot_group")
    target_slot = record.get("target_slot")
    label = f"{kind}[{index}]"
    if group:
        label += f" {group}"
    if target_slot is not None:
        label += f" -> D{target_slot}"
    if code == "nonfinite_values":
        return f"{label}: contains {record['nonfinite']} non-finite values"
    if code == "invalid_norm_std":
        return f"{label}: norm std is not finite and positive"
    if code == "normalized_range_outlier":
        return (
            f"{label}: normalized min/max [{record['normalized_min']:.4g}, "
            f"{record['normalized_max']:.4g}] exceed +/-{DEFAULT_NORMALIZED_RANGE_WARNING:g}"
        )
    if code == "zero_centered_bounded_command_candidate":
        return (
            f"{label}: raw min/max [{record['min']:.4g}, {record['max']:.4g}] look zero-centered, "
            f"but norm mean={record['mean']:.4g}, std={record['std']:.4g}; consider a symmetric "
            "physical-range norm override"
        )
    return f"{label}: {code}"


def _add_warning(warnings: list[dict[str, Any]], record: dict[str, Any], code: str) -> None:
    record["warning_codes"].append(code)
    warning = {
        "code": code,
        "kind": record["kind"],
        "index": record["index"],
        "message": _warning_message(record, code),
    }
    if "slot_group" in record:
        warning["slot_group"] = record["slot_group"]
    if "target_slot" in record:
        warning["target_slot"] = record["target_slot"]
    warnings.append(warning)


def _build_kind_norm_diagnostics(
    *,
    kind: str,
    values: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    dim_info: dict[int, dict[str, Any]],
    warnings: list[dict[str, Any]],
) -> dict[str, Any]:
    dims: list[dict[str, Any]] = []
    for index in range(values.shape[1]):
        column = values[:, index]
        finite_mask = np.isfinite(column)
        finite_values = column[finite_mask]
        nan_count = int(np.isnan(column).sum())
        posinf_count = int(np.isposinf(column).sum())
        neginf_count = int(np.isneginf(column).sum())
        nonfinite_count = int((~finite_mask).sum())
        raw_min = float(np.min(finite_values)) if len(finite_values) else None
        raw_max = float(np.max(finite_values)) if len(finite_values) else None
        mean_value = float(mean[index]) if index < len(mean) else float("nan")
        std_value = float(std[index]) if index < len(std) else float("nan")
        valid_std = np.isfinite(std_value) and std_value > 0
        normalized_min = (raw_min - mean_value) / std_value if raw_min is not None and valid_std else None
        normalized_max = (raw_max - mean_value) / std_value if raw_max is not None and valid_std else None

        record: dict[str, Any] = {
            "kind": kind,
            "index": index,
            "min": _json_float(raw_min),
            "max": _json_float(raw_max),
            "mean": _json_float(mean_value),
            "std": _json_float(std_value),
            "normalized_min": _json_float(normalized_min),
            "normalized_max": _json_float(normalized_max),
            "nonfinite": nonfinite_count,
            "nan": nan_count,
            "posinf": posinf_count,
            "neginf": neginf_count,
            "warning_codes": [],
        }
        record.update(dim_info.get(index, {}))

        if nonfinite_count:
            _add_warning(warnings, record, "nonfinite_values")
        if not valid_std:
            _add_warning(warnings, record, "invalid_norm_std")
        if normalized_min is not None and normalized_max is not None:
            if (
                normalized_min < -DEFAULT_NORMALIZED_RANGE_WARNING
                or normalized_max > DEFAULT_NORMALIZED_RANGE_WARNING
            ):
                _add_warning(warnings, record, "normalized_range_outlier")

        if kind == "action" and raw_min is not None and raw_max is not None and valid_std:
            max_abs = max(abs(raw_min), abs(raw_max))
            if max_abs > 0:
                balance_error = abs(abs(raw_min) - abs(raw_max)) / max_abs
                mean_offset = abs(mean_value) / max_abs
                if (
                    _is_command_like_action(record)
                    and raw_min < 0 < raw_max
                    and balance_error <= DEFAULT_ZERO_CENTER_BALANCE_TOLERANCE
                    and mean_offset >= DEFAULT_ZERO_CENTER_MEAN_OFFSET_FRACTION
                ):
                    _add_warning(warnings, record, "zero_centered_bounded_command_candidate")

        dims.append(record)
    return {"dims": dims}


def _build_norm_diagnostics(
    *,
    state_values: np.ndarray,
    action_values: np.ndarray,
    state_mean: np.ndarray,
    state_std: np.ndarray,
    action_mean: np.ndarray,
    action_std: np.ndarray,
    mapping: dict[str, Any],
) -> dict[str, Any]:
    dim_info = _mapping_dim_info(mapping)
    warnings: list[dict[str, Any]] = []
    diagnostics = {
        "state": _build_kind_norm_diagnostics(
            kind="state",
            values=state_values,
            mean=state_mean,
            std=state_std,
            dim_info=dim_info["state"],
            warnings=warnings,
        ),
        "action": _build_kind_norm_diagnostics(
            kind="action",
            values=action_values,
            mean=action_mean,
            std=action_std,
            dim_info=dim_info["action"],
            warnings=warnings,
        ),
        "warnings": warnings,
        "warning_count": len(warnings),
    }
    return diagnostics


def _format_norm_warning(warning: dict[str, Any]) -> str:
    return f"[WARN] norm {warning.get('message', warning.get('code', '<unknown>'))}"


def _range_label(record: dict[str, Any]) -> str:
    group = record.get("slot_group")
    target_slot = record.get("target_slot")
    if group and target_slot is not None:
        return f"{group}->D{target_slot}"
    if group:
        return str(group)
    if target_slot is not None:
        return f"D{target_slot}"
    return ""


def build_range_check(stats: dict[str, Any]) -> dict[str, Any]:
    meta = stats.get("_meta", {})
    diagnostics = meta.get("norm_diagnostics", {})
    output: dict[str, Any] = {
        "state": [],
        "action": [],
        "_meta": {
            "source": meta.get("source"),
            "dataset_root": meta.get("dataset_root"),
            "state_key": meta.get("state_key"),
            "action_key": meta.get("action_key"),
            "state_dim": meta.get("state_dim"),
            "action_dim": meta.get("action_dim"),
            "num_frames": meta.get("num_frames"),
        },
    }
    for kind in ("state", "action"):
        dims = diagnostics.get(kind, {}).get("dims", [])
        for record in dims:
            output[kind].append(
                {
                    "kind": kind,
                    "index": record.get("index"),
                    "label": _range_label(record),
                    "min": record.get("min"),
                    "max": record.get("max"),
                    "nonfinite_count": record.get("nonfinite", 0),
                    "nan_count": record.get("nan", 0),
                    "posinf_count": record.get("posinf", 0),
                    "neginf_count": record.get("neginf", 0),
                }
            )
    return output


def _format_range_value(value: float | None) -> str:
    return "" if value is None else f"{value:.9g}"


def print_range_check_table(range_check: dict[str, Any]) -> None:
    print("kind index label min max nonfinite nan +inf -inf")
    for record in [*range_check["state"], *range_check["action"]]:
        print(
            "{kind} {index} {label} {min} {max} {nonfinite_count} {nan_count} {posinf_count} {neginf_count}".format(
                **{
                    **record,
                    "label": record["label"] or "-",
                    "min": _format_range_value(record["min"]),
                    "max": _format_range_value(record["max"]),
                }
            )
        )


def print_range_check_csv(range_check: dict[str, Any]) -> None:
    writer = csv.DictWriter(
        sys.stdout,
        fieldnames=[
            "kind",
            "index",
            "label",
            "min",
            "max",
            "nonfinite_count",
            "nan_count",
            "posinf_count",
            "neginf_count",
        ],
    )
    writer.writeheader()
    for record in [*range_check["state"], *range_check["action"]]:
        writer.writerow(record)


def print_range_check(stats: dict[str, Any], output_format: str) -> None:
    if output_format == "none":
        return
    range_check = build_range_check(stats)
    if output_format == "json":
        print(json.dumps(range_check, ensure_ascii=False, indent=2))
    elif output_format == "csv":
        print_range_check_csv(range_check)
    elif output_format == "table":
        print_range_check_table(range_check)
    else:
        raise ValueError(f"unsupported range check format: {output_format}")


def _raise_on_norm_warnings(diagnostics: dict[str, Any]) -> None:
    warnings = diagnostics.get("warnings", [])
    if not warnings:
        return
    rendered = "; ".join(_format_norm_warning(warning) for warning in warnings[:5])
    if len(warnings) > 5:
        rendered += f"; ... {len(warnings) - 5} more"
    raise ValueError(f"norm diagnostics produced warnings: {rendered}")


def _iter_parquet_files(dataset_root: Path) -> list[Path]:
    data_root = dataset_root / "data"
    files = sorted(data_root.glob("**/*.parquet"))
    if not files:
        raise FileNotFoundError(f"no parquet files found under {data_root}")
    return files


def _read_vector_column(files: list[Path], column: str, dim: int) -> np.ndarray:
    try:
        import pyarrow.parquet as pq
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "pyarrow is required to read LeRobot parquet files; install pyarrow or run in the training env"
        ) from exc

    chunks: list[np.ndarray] = []
    for path in files:
        table = pq.read_table(path, columns=[column])
        values = table[column].to_pylist()
        arr = np.asarray(values, dtype=np.float32)
        if arr.ndim != 2:
            raise ValueError(f"{path}:{column} must be a 2D vector column, got shape {arr.shape}")
        if arr.shape[1] < dim:
            raise ValueError(f"{path}:{column} dim {arr.shape[1]} < required dim {dim}")
        chunks.append(arr[:, :dim])
    if not chunks:
        raise ValueError(f"no rows read for column {column}")
    return np.concatenate(chunks, axis=0)


def _read_scalar_column(files: list[Path], column: str, dtype: Any) -> np.ndarray:
    try:
        import pyarrow.parquet as pq
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "pyarrow is required to read LeRobot parquet files; install pyarrow or run in the training env"
        ) from exc

    chunks: list[np.ndarray] = []
    for path in files:
        table = pq.read_table(path, columns=[column])
        values = np.asarray(table[column].to_pylist(), dtype=dtype).reshape(-1)
        chunks.append(values)
    if not chunks:
        raise ValueError(f"no rows read for column {column}")
    return np.concatenate(chunks, axis=0)


def _apply_mapping_value_transforms(values: np.ndarray, transforms: list[str]) -> np.ndarray:
    out = np.asarray(values, dtype=np.float32).copy()
    for index, transform in enumerate(transforms):
        if transform in ("", "identity", None):
            continue
        if transform == "one_minus_raw":
            out[..., index] = 1.0 - out[..., index]
            continue
        raise ValueError(f"unsupported value_transform={transform!r} in combined action norm")
    return out


def _combined_action_mapping_spec(mapping: dict[str, Any]) -> dict[str, Any]:
    groups = list(mapping.get("native_joint_groups", []) or [])
    if not groups:
        raise ValueError("combined absolute/delta norm requires native_joint_groups")

    state_by_slot: dict[int, tuple[int, str]] = {}
    action_by_slot: dict[int, tuple[int, str]] = {}
    for group in groups:
        slots = [int(slot) for slot in (group.get("target_slots") or [])]
        state_indices = [int(index) for index in (group.get("state_source_indices") or [])]
        action_indices = [int(index) for index in (group.get("action_source_indices") or [])]
        transform = str(group.get("value_transform", "identity") or "identity")
        state_source = str(group.get("state_source", mapping.get("state_source", "observation.state")))
        if state_source != "none":
            if len(state_indices) != len(slots):
                raise ValueError(
                    f"{group.get('slot_group', '<unknown>')}: state_source_indices length "
                    f"{len(state_indices)} != target_slots length {len(slots)}"
                )
            for slot, source_index in zip(slots, state_indices):
                state_by_slot[slot] = (source_index, transform)
        if action_indices:
            if len(action_indices) != len(slots):
                raise ValueError(
                    f"{group.get('slot_group', '<unknown>')}: action_source_indices length "
                    f"{len(action_indices)} != target_slots length {len(slots)}"
                )
            for slot, source_index in zip(slots, action_indices):
                action_by_slot[slot] = (source_index, transform)

    active_state_slots = sorted(
        {int(slot) for slot in mapping.get("active_state_slots", state_by_slot.keys())}
    )
    active_action_slots = sorted(
        {int(slot) for slot in mapping.get("active_action_slots", action_by_slot.keys())}
    )
    missing_action_mapping = [slot for slot in active_action_slots if slot not in action_by_slot]
    if missing_action_mapping:
        raise ValueError(f"active action slots missing native mapping: {missing_action_mapping}")

    configured_absolute = {int(slot) for slot in mapping.get("absolute_action_slots", [])}
    absolute_slots = sorted(slot for slot in active_action_slots if slot in configured_absolute)
    relative_candidates = [slot for slot in active_action_slots if slot not in configured_absolute]
    passthrough_action_slots = [
        slot for slot in relative_candidates if slot not in state_by_slot or slot not in active_state_slots
    ]
    delta_slots = [slot for slot in relative_candidates if slot not in passthrough_action_slots]

    return {
        "state_by_slot": state_by_slot,
        "action_by_slot": action_by_slot,
        "active_action_slots": active_action_slots,
        "absolute_action_slots": absolute_slots,
        "delta_from_state_slots": delta_slots,
        "passthrough_action_slots": passthrough_action_slots,
    }


def _base_episode_chunk_mask(
    episode_indices: np.ndarray,
    frame_indices: np.ndarray,
    action_horizon: int,
) -> np.ndarray:
    n_rows = len(episode_indices)
    mask = np.zeros(n_rows, dtype=bool)
    last_start = n_rows - action_horizon + 1
    if last_start <= 0:
        return mask
    end_offset = action_horizon - 1
    same_episode = episode_indices[:last_start] == episode_indices[end_offset:]
    contiguous_frames = (frame_indices[end_offset:] - frame_indices[:last_start]) == end_offset
    mask[:last_start] = same_episode & contiguous_frames
    return mask


def _valid_chunk_start_indices(
    *,
    dataset_root: Path,
    files: list[Path],
    mapping: dict[str, Any],
    action_horizon: int,
) -> np.ndarray:
    episode_indices = _read_scalar_column(files, "episode_index", np.int64)
    frame_indices = _read_scalar_column(files, "frame_index", np.int64)
    valid_mask = _base_episode_chunk_mask(episode_indices, frame_indices, action_horizon)
    sampling = dict(mapping.get("sampling", {}) or {})
    mode = str(sampling.get("valid_chunk_filter", "episode_only") or "episode_only").lower()

    if mode == "valid_chunk_start":
        key = str(sampling.get("valid_chunk_start_key", "valid_chunk_start"))
        valid_mask &= _read_scalar_column(files, key, np.int64) == 1
    elif mode == "valid_intervals":
        import pandas as pd

        configured = sampling.get("valid_intervals_path")
        interval_path = Path(configured) if configured else dataset_root / "meta" / "episode_valid_intervals.parquet"
        if not interval_path.exists():
            raise FileNotFoundError(f"valid_intervals requires sidecar: {interval_path}")
        intervals = pd.read_parquet(interval_path)
        interval_mask = np.zeros(len(episode_indices), dtype=bool)
        by_episode: dict[int, list[tuple[int, int]]] = {}
        for row in intervals[["episode_index", "start_frame", "end_frame"]].itertuples(index=False):
            start, end = int(row.start_frame), int(row.end_frame)
            if end > start:
                by_episode.setdefault(int(row.episode_index), []).append((start, end))
        for episode, raw_intervals in by_episode.items():
            merged: list[list[int]] = []
            for start, end in sorted(raw_intervals):
                if not merged or start > merged[-1][1]:
                    merged.append([start, end])
                else:
                    merged[-1][1] = max(merged[-1][1], end)
            episode_rows = np.nonzero(episode_indices == episode)[0]
            frames = frame_indices[episode_rows]
            for start, end in merged:
                keep = (frames >= start) & ((frames + action_horizon) <= end)
                interval_mask[episode_rows[keep]] = True
        valid_mask &= interval_mask
    elif mode != "episode_only":
        raise ValueError(
            "sampling.valid_chunk_filter must be episode_only, valid_chunk_start, "
            f"or valid_intervals; got {mode!r}"
        )

    starts = np.nonzero(valid_mask)[0]
    if starts.size == 0:
        raise ValueError(f"no valid {action_horizon}-step chunk starts for combined delta norm")
    return starts


def _scatter_active_stats_to_rhino72(
    *,
    active_slots: list[int],
    mean: np.ndarray,
    std: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    mean72 = np.zeros(RHINO72_DIM, dtype=np.float32)
    std72 = np.ones(RHINO72_DIM, dtype=np.float32)
    mean72[active_slots] = np.asarray(mean, dtype=np.float32)
    std72[active_slots] = np.asarray(std, dtype=np.float32)
    return mean72, std72


def _compute_combined_action_stats(
    *,
    dataset_root: Path,
    files: list[Path],
    mapping: dict[str, Any],
    states: np.ndarray,
    actions: np.ndarray,
    absolute_source_mean: np.ndarray,
    absolute_source_std: np.ndarray,
    safe_half_range_threshold: float,
) -> dict[str, Any]:
    spec = _combined_action_mapping_spec(mapping)
    active_slots = spec["active_action_slots"]
    action_entries = [spec["action_by_slot"][slot] for slot in active_slots]
    action_source_indices = [entry[0] for entry in action_entries]
    action_transforms = [entry[1] for entry in action_entries]
    action_active = _apply_mapping_value_transforms(
        actions[:, action_source_indices],
        action_transforms,
    )

    abs_mean_active = np.asarray(
        [absolute_source_mean[source_index] for source_index in action_source_indices],
        dtype=np.float32,
    )
    abs_mean_active = _apply_mapping_value_transforms(abs_mean_active, action_transforms)
    abs_std_active = np.asarray(
        [absolute_source_std[source_index] for source_index in action_source_indices],
        dtype=np.float32,
    )
    abs_mean72, abs_std72 = _scatter_active_stats_to_rhino72(
        active_slots=active_slots,
        mean=abs_mean_active,
        std=abs_std_active,
    )

    action_horizon = int(mapping.get("action_horizon", 30))
    starts = _valid_chunk_start_indices(
        dataset_root=dataset_root,
        files=files,
        mapping=mapping,
        action_horizon=action_horizon,
    )
    chunk_indices = starts[:, None] + np.arange(action_horizon, dtype=np.int64)[None, :]
    action_delta_values = action_active[chunk_indices].copy()
    action_position_by_slot = {slot: position for position, slot in enumerate(active_slots)}
    for slot in spec["delta_from_state_slots"]:
        state_source_index, state_transform = spec["state_by_slot"][slot]
        state_values = _apply_mapping_value_transforms(
            states[starts, state_source_index][:, None],
            [state_transform],
        )[:, 0]
        action_position = action_position_by_slot[slot]
        action_delta_values[:, :, action_position] -= state_values[:, None]
    flat_delta_values = action_delta_values.reshape(-1, len(active_slots))
    delta_mean_active, delta_std_active = _quantile_affine_stats(
        flat_delta_values,
        safe_half_range_threshold,
    )
    delta_mean72, delta_std72 = _scatter_active_stats_to_rhino72(
        active_slots=active_slots,
        mean=delta_mean_active,
        std=delta_std_active,
    )
    return {
        "action_abs_mean": abs_mean72,
        "action_abs_std": abs_std72,
        "action_delta_mean": delta_mean72,
        "action_delta_std": delta_std72,
        "absolute_action_slots": spec["absolute_action_slots"],
        "delta_from_state_slots": spec["delta_from_state_slots"],
        "passthrough_action_slots": spec["passthrough_action_slots"],
        "valid_chunk_starts": int(starts.size),
        "delta_action_values": int(flat_delta_values.shape[0]),
        "action_horizon": action_horizon,
    }


def _quantile_affine_stats(values: np.ndarray, safe_half_range_threshold: float) -> tuple[np.ndarray, np.ndarray]:
    p01, p99 = np.quantile(values, [0.01, 0.99], axis=0)
    mean = ((p01 + p99) / 2.0).astype(np.float32)
    half_range = ((p99 - p01) / 2.0).astype(np.float32)
    std = np.where(half_range < safe_half_range_threshold, 1.0, half_range).astype(np.float32)
    return mean, std


def compute_norm_stats(
    *,
    dataset_root: str | Path,
    mapping_path: str | Path | None = None,
    mapping_dataset_id: str | None = None,
    state_key: str | None = None,
    action_key: str | None = None,
    target_dim: int | None = None,
    safe_half_range_threshold: float = DEFAULT_SAFE_HALF_RANGE_THRESHOLD,
    fail_on_norm_warning: bool = False,
) -> dict[str, Any]:
    dataset_root = Path(dataset_root)
    mapping = Path(mapping_path) if mapping_path is not None else None
    selected_mapping = _select_mapping_dataset(_load_mapping(mapping), mapping_dataset_id)
    state_key, action_key, state_dim, action_dim = _resolve_contract(
        mapping_path=mapping,
        mapping_dataset_id=mapping_dataset_id,
        state_key=state_key,
        action_key=action_key,
        target_dim=target_dim,
    )
    files = _iter_parquet_files(dataset_root)
    states = _read_vector_column(files, state_key, state_dim)
    actions = _read_vector_column(files, action_key, action_dim)
    state_mean, state_std = _quantile_affine_stats(states, safe_half_range_threshold)
    action_mean, action_std = _quantile_affine_stats(actions, safe_half_range_threshold)
    stats_arrays = {
        "state_mean": state_mean,
        "state_std": state_std,
        "action_mean": action_mean,
        "action_std": action_std,
    }
    norm_overrides = _apply_mapping_norm_overrides(stats_arrays, selected_mapping)
    combined_action_stats: dict[str, Any] | None = None
    if selected_mapping.get("native_joint_groups"):
        combined_action_stats = _compute_combined_action_stats(
            dataset_root=dataset_root,
            files=files,
            mapping=selected_mapping,
            states=states,
            actions=actions,
            absolute_source_mean=stats_arrays["action_mean"],
            absolute_source_std=stats_arrays["action_std"],
            safe_half_range_threshold=safe_half_range_threshold,
        )
    norm_diagnostics = _build_norm_diagnostics(
        state_values=states,
        action_values=actions,
        state_mean=stats_arrays["state_mean"],
        state_std=stats_arrays["state_std"],
        action_mean=stats_arrays["action_mean"],
        action_std=stats_arrays["action_std"],
        mapping=selected_mapping,
    )
    if fail_on_norm_warning:
        _raise_on_norm_warnings(norm_diagnostics)
    meta = {
        "method": METHOD,
        "active_dims": list(range(state_dim)) if state_dim == action_dim else None,
        "state_active_dims": list(range(state_dim)),
        "action_active_dims": list(range(action_dim)),
        "source": "data/**/*.parquet",
        "dataset_root": str(dataset_root),
        "state_key": state_key,
        "action_key": action_key,
        "target_dim": state_dim if state_dim == action_dim else None,
        "state_dim": state_dim,
        "action_dim": action_dim,
        "num_frames": int(states.shape[0]),
        "safe_half_range_threshold": float(safe_half_range_threshold),
        "norm_diagnostics": norm_diagnostics,
    }
    if norm_overrides:
        meta["mapping_norm_overrides"] = norm_overrides
    result = {
        "state_mean": stats_arrays["state_mean"].tolist(),
        "state_std": stats_arrays["state_std"].tolist(),
        "action_mean": stats_arrays["action_mean"].tolist(),
        "action_std": stats_arrays["action_std"].tolist(),
        "_meta": meta,
    }
    if combined_action_stats is not None:
        for key in (
            "action_abs_mean",
            "action_abs_std",
            "action_delta_mean",
            "action_delta_std",
        ):
            result[key] = combined_action_stats[key].tolist()
        meta.update(
            {
                "absolute_action_slots": combined_action_stats["absolute_action_slots"],
                "delta_from_state_slots": combined_action_stats["delta_from_state_slots"],
                "passthrough_action_slots": combined_action_stats["passthrough_action_slots"],
                "valid_chunk_starts": combined_action_stats["valid_chunk_starts"],
                "delta_action_values": combined_action_stats["delta_action_values"],
                "action_horizon": combined_action_stats["action_horizon"],
                "action_spaces": {
                    "absolute_joint_position": {
                        "mean_key": "action_abs_mean",
                        "std_key": "action_abs_std",
                    },
                    "delta_from_current_state": {
                        "mean_key": "action_delta_mean",
                        "std_key": "action_delta_std",
                    },
                },
            }
        )
    return result


def write_norm_json(stats: dict[str, Any], output_path: str | Path, *, overwrite: bool = False) -> Path:
    output_path = Path(output_path)
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"{output_path} already exists; pass --overwrite to replace it")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return output_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", type=Path, help="LeRobot v3 dataset root containing data/ and meta/")
    parser.add_argument("--mapping-path", type=Path, default=None, help="Optional data mapping YAML")
    parser.add_argument("--mapping-dataset-id", default=None, help="Dataset entry to select from mapping.datasets")
    parser.add_argument("--output", type=Path, default=None, help="Output path, defaults to DATASET_ROOT/meta/norm.json")
    parser.add_argument("--state-key", default=None, help=f"Override state column, default {DEFAULT_STATE_KEY}")
    parser.add_argument("--action-key", default=None, help=f"Override action column, default {DEFAULT_ACTION_KEY}")
    parser.add_argument("--target-dim", type=int, default=None, help=f"Override target dim, default {DEFAULT_TARGET_DIM}")
    parser.add_argument(
        "--safe-half-range-threshold",
        type=float,
        default=DEFAULT_SAFE_HALF_RANGE_THRESHOLD,
        help="Use std=1.0 when (p99-p01)/2 is below this threshold",
    )
    parser.add_argument(
        "--fail-on-norm-warning",
        action="store_true",
        help="Exit with an error if norm diagnostics detect risky dimensions",
    )
    parser.add_argument(
        "--range-check-format",
        choices=("none", "table", "json", "csv"),
        default="none",
        help="Print source state/action min/max and non-finite counts from norm diagnostics",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing output file")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    output = args.output or (args.dataset_root / "meta" / "norm.json")
    stats = compute_norm_stats(
        dataset_root=args.dataset_root,
        mapping_path=args.mapping_path,
        mapping_dataset_id=args.mapping_dataset_id,
        state_key=args.state_key,
        action_key=args.action_key,
        target_dim=args.target_dim,
        safe_half_range_threshold=args.safe_half_range_threshold,
        fail_on_norm_warning=args.fail_on_norm_warning,
    )
    path = write_norm_json(stats, output, overwrite=args.overwrite)
    for warning in stats["_meta"]["norm_diagnostics"]["warnings"]:
        print(_format_norm_warning(warning), file=sys.stderr)
    status_stream = sys.stderr if args.range_check_format != "none" else sys.stdout
    print(
        f"wrote {path} frames={stats['_meta']['num_frames']} "
        f"state_dim={stats['_meta']['state_dim']} action_dim={stats['_meta']['action_dim']}",
        file=status_stream,
    )
    print_range_check(stats, args.range_check_format)


if __name__ == "__main__":
    main()
