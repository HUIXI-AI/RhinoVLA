#!/usr/bin/env python3
"""Analyze RhinoVLA training metrics.jsonl files.

The script produces:
- summary.json: machine-readable speed/convergence/error-contribution summary
- report.md: human-readable report
- plots/*.png: optional plots when matplotlib is available
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Any


DIM_MSE_RE = re.compile(r"^fm/dim_mse/d(\d+)$")
DEFAULT_BASE_DIMS = [58, 59, 60]
DEFAULT_SMOOTH_WINDOW = 50
DEFAULT_EMA_ALPHA = 0.05


def _finite_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if not isinstance(value, (int, float)):
        return None
    value = float(value)
    if not math.isfinite(value):
        return None
    return value


def _step(row: dict[str, Any]) -> int:
    value = _finite_float(row.get("global_step"))
    return int(value) if value is not None else 0


def _series(rows: list[dict[str, Any]], key: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        value = _finite_float(row.get(key))
        if value is not None:
            values.append(value)
    return values


def _stat(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "median": None, "min": None, "max": None, "std": None}
    return {
        "mean": mean(values),
        "median": median(values),
        "min": min(values),
        "max": max(values),
        "std": pstdev(values) if len(values) > 1 else 0.0,
    }


def _mean(values: list[float]) -> float | None:
    return mean(values) if values else None


def _quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    pos = (len(values) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return values[lo]
    weight = pos - lo
    return values[lo] * (1.0 - weight) + values[hi] * weight


def _rolling_mean(values: list[float], window: int) -> list[float]:
    if window <= 0:
        raise ValueError("smooth window must be positive")
    out: list[float] = []
    running = 0.0
    for index, value in enumerate(values):
        running += value
        if index >= window:
            running -= values[index - window]
        count = min(index + 1, window)
        out.append(running / count)
    return out


def _ema(values: list[float], alpha: float) -> list[float]:
    if not 0 < alpha <= 1:
        raise ValueError("ema alpha must be in (0, 1]")
    if not values:
        return []
    out = [values[0]]
    for value in values[1:]:
        out.append(alpha * value + (1.0 - alpha) * out[-1])
    return out


def _linear_slope(xs: list[float], ys: list[float]) -> float | None:
    pairs = [(x, y) for x, y in zip(xs, ys) if y is not None]
    if len(pairs) < 2:
        return None
    xs = [x for x, _ in pairs]
    ys = [y for _, y in pairs]
    x_mean = mean(xs)
    y_mean = mean(ys)
    denom = sum((x - x_mean) ** 2 for x in xs)
    if denom == 0:
        return None
    return sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys)) / denom


def build_loss_trend(
    rows: list[dict[str, Any]],
    *,
    smooth_window: int = DEFAULT_SMOOTH_WINDOW,
    ema_alpha: float = DEFAULT_EMA_ALPHA,
) -> dict[str, list[float] | list[int]]:
    rows = sorted(rows, key=_step)
    pairs = [
        (_step(row), value)
        for row in rows
        for value in [_finite_float(row.get("loss/train"))]
        if value is not None
    ]
    steps = [step for step, _ in pairs]
    loss = [value for _, value in pairs]
    return {
        "steps": steps,
        "loss": loss,
        "rolling_mean": _rolling_mean(loss, smooth_window) if loss else [],
        "ema": _ema(loss, ema_alpha) if loss else [],
    }


def _last_window(rows: list[dict[str, Any]], late_window_steps: int) -> list[dict[str, Any]]:
    if not rows:
        return []
    last_step = max(_step(row) for row in rows)
    first_step = max(0, last_step - late_window_steps + 1)
    return [row for row in rows if _step(row) >= first_step]


def read_metrics_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}: invalid JSON") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{lineno}: expected a JSON object")
            rows.append(row)
    if not rows:
        raise ValueError(f"{path}: no metrics rows found")
    rows.sort(key=_step)
    return rows


def _infer_dim_ids(rows: list[dict[str, Any]]) -> list[int]:
    dims: set[int] = set()
    for row in rows:
        for key, value in row.items():
            match = DIM_MSE_RE.match(key)
            if match and _finite_float(value) is not None:
                dims.add(int(match.group(1)))
    active: list[int] = []
    for dim in sorted(dims):
        values = _series(rows, f"fm/dim_mse/d{dim:02d}")
        if any(abs(value) > 0 for value in values):
            active.append(dim)
    return active


def _build_windows(rows: list[dict[str, Any]], window_size: int) -> list[dict[str, Any]]:
    if window_size <= 0:
        raise ValueError("window_size must be positive")
    first_step = min(_step(row) for row in rows)
    last_step = max(_step(row) for row in rows)
    windows: list[dict[str, Any]] = []
    start = first_step
    while start <= last_step:
        end = start + window_size - 1
        win = [row for row in rows if start <= _step(row) <= end]
        if win:
            loss_values = _series(win, "loss/train")
            windows.append(
                {
                    "step_start": start,
                    "step_end": end,
                    "rows": len(win),
                    "loss_mean": _mean(loss_values),
                    "loss_median": median(loss_values) if loss_values else None,
                    "loss_p10": _quantile(loss_values, 0.10),
                    "loss_p90": _quantile(loss_values, 0.90),
                    "loss_min": min(loss_values) if loss_values else None,
                    "loss_max": max(loss_values) if loss_values else None,
                    "grad_norm_mean": _mean(_series(win, "grad_norm")),
                    "velocity_cosine_mean": _mean(_series(win, "fm/velocity_cosine")),
                    "learning_rate_mean": _mean(_series(win, "learning_rate")),
                }
            )
        start += window_size
    return windows


def _speed_summary(rows: list[dict[str, Any]], late_rows: list[dict[str, Any]]) -> dict[str, Any]:
    sec_values = _series(late_rows, "sec_per_step") or _series(late_rows, "optimizer_step_wall_sec")
    samples_values = _series(late_rows, "samples_per_sec")
    data_values = _series(late_rows, "data_time")
    model_values = _series(late_rows, "model_time")
    peak_mem = _series(late_rows, "cuda_mem_peak_allocated_gb")
    total_mem = _series(late_rows, "cuda_mem_total_gb")
    mfu = _series(late_rows, "hardware/mfu_percent_estimate")
    achieved_tflops = _series(late_rows, "hardware/achieved_tflops_estimate")

    last = rows[-1]
    last_step = _step(last)
    last_epoch = _finite_float(last.get("epoch"))
    steps_per_epoch = last_step / last_epoch if last_epoch and last_epoch > 0 else None
    sec_per_step = median(sec_values) if sec_values else None
    hours_per_epoch = steps_per_epoch * sec_per_step / 3600.0 if steps_per_epoch and sec_per_step else None
    hours_per_10k_steps = 10000.0 * sec_per_step / 3600.0 if sec_per_step else None
    mem_ratio = None
    if peak_mem and total_mem and max(total_mem) > 0:
        mem_ratio = max(peak_mem) / max(total_mem)

    return {
        "late_sec_per_step": _stat(sec_values),
        "late_samples_per_sec": _stat(samples_values),
        "late_samples_per_sec_mean": _mean(samples_values),
        "late_data_time": _stat(data_values),
        "late_model_time": _stat(model_values),
        "late_peak_allocated_gb": _stat(peak_mem),
        "cuda_total_gb": max(total_mem) if total_mem else None,
        "late_peak_allocated_ratio": mem_ratio,
        "late_mfu_percent": _stat(mfu),
        "late_achieved_tflops": _stat(achieved_tflops),
        "estimated_steps_per_epoch": steps_per_epoch,
        "estimated_hours_per_epoch": hours_per_epoch,
        "estimated_hours_per_10000_steps": hours_per_10k_steps,
    }


def _error_contribution(
    rows: list[dict[str, Any]],
    late_rows: list[dict[str, Any]],
    *,
    top_dims: int,
    base_dims: list[int],
) -> dict[str, Any]:
    active_dims = _infer_dim_ids(rows)
    dim_stats: list[dict[str, Any]] = []
    for dim in active_dims:
        key = f"fm/dim_mse/d{dim:02d}"
        values = _series(late_rows, key)
        if not values:
            continue
        dim_stats.append(
            {
                "dim": dim,
                "label": f"D{dim:02d}",
                "mean": mean(values),
                "median": median(values),
                "min": min(values),
                "max": max(values),
            }
        )
    dim_stats.sort(key=lambda item: item["mean"], reverse=True)

    base_set = set(base_dims)
    ratios: list[float] = []
    for row in late_rows:
        total = 0.0
        base = 0.0
        for dim in active_dims:
            value = _finite_float(row.get(f"fm/dim_mse/d{dim:02d}")) or 0.0
            total += value
            if dim in base_set:
                base += value
        if total > 0:
            ratios.append(base / total)

    return {
        "active_dims": active_dims,
        "base_dims": base_dims,
        "top_dims": dim_stats[:top_dims],
        "all_dim_stats": dim_stats,
        "base_loss_share": mean(ratios) if ratios else None,
        "base_dim_mse_mean": _mean(
            [
                _finite_float(row.get(f"fm/dim_mse/d{dim:02d}")) or 0.0
                for row in late_rows
                for dim in base_dims
                if dim in active_dims
            ]
        ),
        "nonbase_dim_mse_mean": _mean(
            [
                _finite_float(row.get(f"fm/dim_mse/d{dim:02d}")) or 0.0
                for row in late_rows
                for dim in active_dims
                if dim not in base_set
            ]
        ),
    }


def _interpret(summary: dict[str, Any]) -> list[str]:
    notes: list[str] = []
    top_dims = summary["error_contribution"]["top_dims"]
    if top_dims:
        labels = ", ".join(item["label"] for item in top_dims[:3])
        notes.append(f"Top error dimensions are {labels}; inspect their norm, labels, and data coverage first.")
    base_share = summary["error_contribution"].get("base_loss_share")
    if base_share is not None and base_share >= 0.30:
        notes.append(f"Base velocity dims contribute {base_share:.1%} of late active-dim error, which is high.")

    speed = summary["speed"]
    data_mean = speed["late_data_time"]["mean"]
    model_mean = speed["late_model_time"]["mean"]
    if data_mean is not None and model_mean is not None and model_mean > 0:
        if data_mean / model_mean < 0.05:
            notes.append("Dataloader time is small relative to model time; the observed speed is model-side, not data-loading-bound.")
        else:
            notes.append("Dataloader time is non-trivial; check video decode/cache and dataloader worker settings.")
    mem_ratio = speed.get("late_peak_allocated_ratio")
    if mem_ratio is not None and mem_ratio < 0.35:
        notes.append(f"Peak GPU memory ratio is only {mem_ratio:.1%}; there may be room to increase batch size.")

    conv = summary["convergence"]
    first_loss = conv.get("first_window_loss_mean")
    last_loss = conv.get("last_window_loss_mean")
    if first_loss and last_loss:
        improvement = 1.0 - (last_loss / first_loss)
        if improvement > 0.8:
            notes.append(f"Training loss improved by {improvement:.1%}; this looks like slowing convergence, not failure to learn.")
        elif improvement < 0.2:
            notes.append("Training loss improved little; check LR, optimizer stability, and data/norm issues.")
    return notes


def analyze_metrics(
    rows: list[dict[str, Any]],
    *,
    window_size: int = 100,
    late_window_steps: int = 200,
    top_dims: int = 12,
    base_dims: list[int] | None = None,
    smooth_window: int = DEFAULT_SMOOTH_WINDOW,
    ema_alpha: float = DEFAULT_EMA_ALPHA,
) -> dict[str, Any]:
    if not rows:
        raise ValueError("rows must not be empty")
    rows = sorted(rows, key=_step)
    base_dims = list(base_dims or DEFAULT_BASE_DIMS)
    late_rows = _last_window(rows, late_window_steps)
    windows = _build_windows(rows, window_size)
    loss_values = _series(rows, "loss/train")
    late_loss_values = _series(late_rows, "loss/train")
    last = rows[-1]
    first = rows[0]

    run = {
        "num_rows": len(rows),
        "first_step": _step(first),
        "last_step": _step(last),
        "first_epoch": _finite_float(first.get("epoch")),
        "last_epoch": _finite_float(last.get("epoch")),
    }
    speed = _speed_summary(rows, late_rows)
    run["estimated_steps_per_epoch"] = speed["estimated_steps_per_epoch"]

    convergence = {
        "windows": windows,
        "best_loss": min(loss_values) if loss_values else None,
        "best_loss_step": _step(min(rows, key=lambda row: row.get("loss/train", float("inf")))),
        "last_loss": loss_values[-1] if loss_values else None,
        "late_loss": _stat(late_loss_values),
        "first_window_loss_mean": windows[0]["loss_mean"] if windows else None,
        "last_window_loss_mean": windows[-1]["loss_mean"] if windows else None,
    }
    slope_windows = windows[-min(5, len(windows)) :]
    convergence["last_window_mean_slope_per_step"] = _linear_slope(
        [(window["step_start"] + window["step_end"]) / 2.0 for window in slope_windows],
        [window.get("loss_mean") for window in slope_windows],
    )

    summary = {
        "run": run,
        "speed": speed,
        "convergence": convergence,
        "loss_trend": build_loss_trend(rows, smooth_window=smooth_window, ema_alpha=ema_alpha),
        "error_contribution": _error_contribution(rows, late_rows, top_dims=top_dims, base_dims=base_dims),
        "tracked_norms": {
            "action_norm": _stat(_series(late_rows, "fm/action_norm")),
            "target_velocity_norm": _stat(_series(late_rows, "fm/target_velocity_norm")),
            "pred_velocity_norm": _stat(_series(late_rows, "fm/pred_velocity_norm")),
            "velocity_cosine": _stat(_series(late_rows, "fm/velocity_cosine")),
            "grad_norm": _stat(_series(late_rows, "grad_norm")),
            "learning_rate": _stat(_series(late_rows, "learning_rate")),
        },
    }
    summary["interpretation"] = _interpret(summary)
    return summary


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}g}"
    return str(value)


def render_report(summary: dict[str, Any]) -> str:
    run = summary["run"]
    speed = summary["speed"]
    conv = summary["convergence"]
    err = summary["error_contribution"]
    norms = summary["tracked_norms"]

    lines = [
        "# Training Metrics Report",
        "",
        "## Run",
        "",
        f"- rows: `{run['num_rows']}`",
        f"- steps: `{run['first_step']}` -> `{run['last_step']}`",
        f"- epoch: `{_fmt(run.get('first_epoch'))}` -> `{_fmt(run.get('last_epoch'))}`",
        f"- estimated steps/epoch: `{_fmt(run.get('estimated_steps_per_epoch'))}`",
        "",
        "## Speed",
        "",
        f"- late sec/step median: `{_fmt(speed['late_sec_per_step']['median'])}`",
        f"- late samples/sec mean: `{_fmt(speed['late_samples_per_sec']['mean'])}`",
        f"- estimated hours/epoch: `{_fmt(speed.get('estimated_hours_per_epoch'))}`",
        f"- estimated hours/10000 steps: `{_fmt(speed.get('estimated_hours_per_10000_steps'))}`",
        f"- data time mean: `{_fmt(speed['late_data_time']['mean'])}`",
        f"- model time mean: `{_fmt(speed['late_model_time']['mean'])}`",
        f"- peak allocated GPU memory ratio: `{_fmt(speed.get('late_peak_allocated_ratio'))}`",
        f"- MFU percent estimate mean: `{_fmt(speed['late_mfu_percent']['mean'])}`",
        "",
        "## Convergence",
        "",
        f"- best loss: `{_fmt(conv.get('best_loss'))}` at step `{_fmt(conv.get('best_loss_step'))}`",
        f"- last loss: `{_fmt(conv.get('last_loss'))}`",
        f"- first window loss mean: `{_fmt(conv.get('first_window_loss_mean'))}`",
        f"- last window loss mean: `{_fmt(conv.get('last_window_loss_mean'))}`",
        f"- recent window-mean slope/step: `{_fmt(conv.get('last_window_mean_slope_per_step'))}`",
        f"- late loss mean/std: `{_fmt(conv['late_loss']['mean'])}` / `{_fmt(conv['late_loss']['std'])}`",
        "- plots: `loss_curve.png` overlays raw loss, rolling mean and EMA; `loss_curve_log.png` uses log scale; `loss_window_band.png` shows window mean with p10/p90 band.",
        "",
        "## Late Norms",
        "",
        f"- grad norm mean: `{_fmt(norms['grad_norm']['mean'])}`",
        f"- action norm mean: `{_fmt(norms['action_norm']['mean'])}`",
        f"- target velocity norm mean: `{_fmt(norms['target_velocity_norm']['mean'])}`",
        f"- pred velocity norm mean: `{_fmt(norms['pred_velocity_norm']['mean'])}`",
        f"- velocity cosine mean: `{_fmt(norms['velocity_cosine']['mean'])}`",
        f"- learning rate mean: `{_fmt(norms['learning_rate']['mean'])}`",
        "",
        "## Top Error Dimensions",
        "",
        "| rank | dim | mean | median | min | max |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for idx, item in enumerate(err["top_dims"], start=1):
        lines.append(
            f"| {idx} | D{item['dim']:02d} | {_fmt(item['mean'])} | {_fmt(item['median'])} | "
            f"{_fmt(item['min'])} | {_fmt(item['max'])} |"
        )
    lines.extend(
        [
            "",
            f"- base dims: `{err['base_dims']}`",
            f"- base loss share: `{_fmt(err.get('base_loss_share'))}`",
            f"- base dim MSE mean: `{_fmt(err.get('base_dim_mse_mean'))}`",
            f"- non-base dim MSE mean: `{_fmt(err.get('nonbase_dim_mse_mean'))}`",
            "",
            "## Interpretation",
            "",
        ]
    )
    for note in summary.get("interpretation", []):
        lines.append(f"- {note}")
    return "\n".join(lines) + "\n"


def _plot_outputs(summary: dict[str, Any], rows: list[dict[str, Any]], plot_dir: Path) -> list[Path]:
    try:
        mpl_config_dir = plot_dir / ".matplotlib"
        mpl_config_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir))
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return []

    plot_dir.mkdir(parents=True, exist_ok=True)
    steps = [_step(row) for row in rows]
    outputs: list[Path] = []

    def save_current(name: str) -> None:
        path = plot_dir / name
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close()
        outputs.append(path)

    loss_trend = summary.get("loss_trend", {})
    trend_steps = loss_trend.get("steps", [])
    loss = loss_trend.get("loss", [])
    rolling = loss_trend.get("rolling_mean", [])
    ema = loss_trend.get("ema", [])
    if loss and trend_steps:
        plt.figure(figsize=(10, 5))
        plt.plot(trend_steps, loss, linewidth=0.8, alpha=0.25, label="raw loss")
        if rolling:
            plt.plot(trend_steps, rolling, linewidth=1.6, label="rolling mean")
        if ema:
            plt.plot(trend_steps, ema, linewidth=1.6, label="EMA")
        plt.xlabel("global_step")
        plt.ylabel("loss/train")
        plt.title("Training Loss")
        plt.legend()
        plt.grid(True, alpha=0.25)
        save_current("loss_curve.png")

        positive_pairs = [(step, value) for step, value in zip(trend_steps, loss) if value > 0]
        if positive_pairs:
            log_steps, log_loss = zip(*positive_pairs)
            plt.figure(figsize=(10, 5))
            plt.plot(log_steps, log_loss, linewidth=0.8, alpha=0.25, label="raw loss")
            positive_rolling = [(step, value) for step, value in zip(trend_steps, rolling) if value > 0]
            positive_ema = [(step, value) for step, value in zip(trend_steps, ema) if value > 0]
            if positive_rolling:
                rolling_steps, rolling_values = zip(*positive_rolling)
                plt.plot(rolling_steps, rolling_values, linewidth=1.6, label="rolling mean")
            if positive_ema:
                ema_steps, ema_values = zip(*positive_ema)
                plt.plot(ema_steps, ema_values, linewidth=1.6, label="EMA")
            plt.yscale("log")
            plt.xlabel("global_step")
            plt.ylabel("loss/train (log scale)")
            plt.title("Training Loss (Log Scale)")
            plt.legend()
            plt.grid(True, which="both", alpha=0.25)
            save_current("loss_curve_log.png")

    windows = summary.get("convergence", {}).get("windows", [])
    if windows:
        x = [(window["step_start"] + window["step_end"]) / 2.0 for window in windows]
        mean_values = [window.get("loss_mean") for window in windows]
        p10_values = [window.get("loss_p10") for window in windows]
        p90_values = [window.get("loss_p90") for window in windows]
        if all(value is not None for value in mean_values):
            plt.figure(figsize=(10, 5))
            plt.plot(x, mean_values, linewidth=1.8, label="window mean")
            if all(value is not None for value in p10_values + p90_values):
                plt.fill_between(x, p10_values, p90_values, alpha=0.20, label="p10-p90")
            plt.xlabel("global_step")
            plt.ylabel("loss/train")
            plt.title("Windowed Loss Trend")
            plt.legend()
            plt.grid(True, alpha=0.25)
            save_current("loss_window_band.png")

    sec_values = _series(rows, "sec_per_step")
    samples_values = _series(rows, "samples_per_sec")
    if sec_values or samples_values:
        plt.figure(figsize=(10, 5))
        if sec_values:
            plt.plot(steps[: len(sec_values)], sec_values, label="sec_per_step", linewidth=1.0)
        if samples_values:
            ax = plt.gca()
            ax2 = ax.twinx()
            ax2.plot(steps[: len(samples_values)], samples_values, color="tab:orange", label="samples_per_sec")
            ax2.set_ylabel("samples/sec")
        plt.xlabel("global_step")
        plt.ylabel("sec/step")
        plt.title("Training Speed")
        plt.grid(True, alpha=0.25)
        save_current("speed_curve.png")

    top_dims = summary["error_contribution"]["top_dims"]
    if top_dims:
        labels = [f"D{item['dim']:02d}" for item in top_dims]
        values = [item["mean"] for item in top_dims]
        plt.figure(figsize=(10, max(4, len(labels) * 0.35)))
        plt.barh(labels[::-1], values[::-1])
        plt.xlabel("late dim_mse mean")
        plt.title("Top Error Dimensions")
        plt.grid(True, axis="x", alpha=0.25)
        save_current("top_dim_mse.png")

    return outputs


def write_outputs(
    summary: dict[str, Any],
    rows: list[dict[str, Any]],
    output_dir: str | Path,
    *,
    make_plots: bool = True,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_json = output_dir / "summary.json"
    report_md = output_dir / "report.md"
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report_md.write_text(render_report(summary), encoding="utf-8")
    plots = _plot_outputs(summary, rows, output_dir / "plots") if make_plots else []
    return {"summary_json": summary_json, "report_md": report_md, "plots": plots}


def _parse_dim_list(text: str) -> list[int]:
    dims: list[int] = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        dims.append(int(item))
    return dims


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metrics_jsonl", type=Path, help="Path to metrics.jsonl")
    parser.add_argument("--output-dir", type=Path, default=Path("metrics_analysis"), help="Output directory")
    parser.add_argument("--window-size", type=int, default=100, help="Step window size for convergence summaries")
    parser.add_argument("--late-window-steps", type=int, default=200, help="Use this many final steps for late stats")
    parser.add_argument("--top-dims", type=int, default=12, help="Number of high-error dimensions to report")
    parser.add_argument("--base-dims", default="58,59,60", help="Comma-separated base velocity dims")
    parser.add_argument("--smooth-window", type=int, default=DEFAULT_SMOOTH_WINDOW, help="Rolling-mean window for loss plots")
    parser.add_argument("--ema-alpha", type=float, default=DEFAULT_EMA_ALPHA, help="EMA alpha for loss plots")
    parser.add_argument("--no-plots", action="store_true", help="Skip matplotlib plot generation")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    rows = read_metrics_jsonl(args.metrics_jsonl)
    summary = analyze_metrics(
        rows,
        window_size=args.window_size,
        late_window_steps=args.late_window_steps,
        top_dims=args.top_dims,
        base_dims=_parse_dim_list(args.base_dims),
        smooth_window=args.smooth_window,
        ema_alpha=args.ema_alpha,
    )
    outputs = write_outputs(summary, rows, args.output_dir, make_plots=not args.no_plots)
    print(f"wrote {outputs['summary_json']}")
    print(f"wrote {outputs['report_md']}")
    if outputs["plots"]:
        for path in outputs["plots"]:
            print(f"wrote {path}")
    else:
        print("plots skipped or matplotlib unavailable")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
