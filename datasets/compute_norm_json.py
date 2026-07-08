#!/usr/bin/env python3
"""Compute RhinoVLA-compatible LeRobot v3 normalization stats.

The output schema is the compact `norm.json` consumed by the native72 loader:
`state_mean`, `state_std`, `action_mean`, `action_std`, and `_meta`.
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
    return {
        "state_mean": stats_arrays["state_mean"].tolist(),
        "state_std": stats_arrays["state_std"].tolist(),
        "action_mean": stats_arrays["action_mean"].tolist(),
        "action_std": stats_arrays["action_std"].tolist(),
        "_meta": meta,
    }


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
