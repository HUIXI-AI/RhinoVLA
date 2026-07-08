import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

try:
    import swanlab
    HAS_SWANLAB = True
except ImportError:
    HAS_SWANLAB = False

from rhinovla.training._trainer_helpers import *  # noqa: F401,F403


def filter_swanlab_scalars(metrics: dict) -> dict[str, float]:
    """Return the scalar curves shown on the release SwanLab page."""

    numeric = {
        k: float(v)
        for k, v in metrics.items()
        if isinstance(v, (int, float, np.number))
    }
    out: dict[str, float] = {}

    def add(new_key: str, old_key: str) -> None:
        if old_key in numeric:
            out[new_key] = numeric[old_key]

    add("main_loss/train_total", "loss/train")
    global_step = int(numeric.get("global_step", 0))
    if global_step >= 20:
        add("eta_main/remaining_hours", "eta/remaining_hours")
        add("eta_main/avg_step_sec", "eta/avg_step_sec")
    add("throughput/data_time", "data_time")
    add("throughput/model_time", "model_time")
    add("eval_main/val_loss", "loss/val")
    add("lr/learning_rate", "learning_rate")
    for key, value in numeric.items():
        if str(key).startswith("lr/"):
            out[str(key)] = value
    return out


def _swanlab_login_hosts() -> tuple[str | None, str | None]:
    api_host = os.environ.get("SWANLAB_API_HOST")
    web_host = os.environ.get("SWANLAB_WEB_HOST")
    login_host = api_host or web_host
    if login_host:
        login_host = login_host.rstrip("/")
        if login_host.endswith("/api"):
            login_host = login_host[:-4]
    return login_host, web_host


def _swanlab_project_name(config) -> str:
    env_project = os.environ.get("SWANLAB_PROJECT")
    if env_project and env_project.strip():
        return env_project.strip()

    config_project = getattr(config, "swanlab_project", None)
    if config_project:
        return str(config_project)

    return "rhinovla-finetune"


class _MetricsMixin:
    """Metrics, logging, profiling and hardware-efficiency methods for VLATrainer."""

    def _swanlab_step(self, step: int | None = None) -> int:
        base_step = self.completed_steps if step is None else int(step)
        offset = int(getattr(self.config.trainer, "swanlab_step_offset", 0) or 0)
        return base_step + offset

    def _tracker_enabled(self, name: str) -> bool:
        trackers = getattr(self.config, "trackers", None)
        if trackers is None:
            return name == "swanlab"
        if isinstance(trackers, str):
            values = [x.strip().strip("'\"") for x in trackers.strip("[]").split(",") if x.strip()]
        else:
            values = [str(x) for x in trackers]
        return name.lower() in {x.lower() for x in values}

    def _init_trackers(self):
        """Initialize SwanLab."""
        if not self._swanlab_enabled:
            logger.info("SwanLab disabled by trackers config")
            return
        if self.accelerator.is_main_process:
            try:
                import swanlab
                run_note = getattr(self.config, "run_note", None) or (
                    f"{self.config.run_id}: "
                    f"freeze={getattr(self.config.trainer, 'freeze_modules', '')}, "
                    f"bs={self.config.datasets.vla_data.per_device_batch_size}, "
                    f"ga={self.config.trainer.gradient_accumulation_steps}"
                )
                # Build config dict for SwanLab
                from omegaconf import OmegaConf
                swanlab_config = OmegaConf.to_container(self.config, resolve=True)

                swanlab_api_key = os.environ.get("SWANLAB_API_KEY")
                swanlab_host, swanlab_web_host = _swanlab_login_hosts()
                if swanlab_api_key:
                    try:
                        swanlab.login(api_key=swanlab_api_key, host=swanlab_host, web_host=swanlab_web_host)
                    except TypeError:
                        swanlab.login(key=swanlab_api_key, host=swanlab_host)

                # SwanLab creates a new cloud run unless a stable id is passed.
                # For checkpoint resume we must keep logging into the same run;
                # otherwise a single training trajectory is split across UI runs.
                trainer_cfg = getattr(self.config, "trainer", None)
                swanlab_run_id = (
                    os.environ.get("SWANLAB_RUN_ID")
                    or getattr(trainer_cfg, "swanlab_run_id", None)
                    or getattr(self.config, "swanlab_run_id", None)
                )
                swanlab_resume = (
                    os.environ.get("SWANLAB_RESUME")
                    or getattr(trainer_cfg, "swanlab_resume", None)
                    or getattr(self.config, "swanlab_resume", None)
                )
                if isinstance(swanlab_resume, bool):
                    swanlab_resume = "allow" if swanlab_resume else "never"
                if swanlab_resume is not None:
                    swanlab_resume = str(swanlab_resume).lower()
                    if swanlab_resume in {"true", "yes", "1"}:
                        swanlab_resume = "allow"
                    elif swanlab_resume in {"false", "no", "0"}:
                        swanlab_resume = "never"

                if swanlab_run_id is None and bool(getattr(trainer_cfg, "is_resume", False)):
                    swanlab_run_id = str(self.config.run_id)
                if swanlab_resume is None and swanlab_run_id:
                    swanlab_resume = "allow"
                if swanlab_resume == "never" and swanlab_run_id:
                    logger.info(
                        "SwanLab clean run requested; dropping run id=%s because resume=never",
                        swanlab_run_id,
                    )
                    swanlab_run_id = None

                swanlab_project = _swanlab_project_name(self.config)
                swanlab_init_kwargs = {
                    "project": str(swanlab_project),
                    "experiment_name": self.config.run_id,
                    "description": run_note,
                    "config": swanlab_config,
                }
                swanlab_workspace = (
                    getattr(self.config, "swanlab_workspace", None)
                    or os.environ.get("SWANLAB_WORKSPACE")
                )
                if swanlab_workspace:
                    swanlab_init_kwargs["workspace"] = str(swanlab_workspace)
                if swanlab_run_id:
                    swanlab_init_kwargs["id"] = str(swanlab_run_id)
                if swanlab_resume:
                    swanlab_init_kwargs["resume"] = swanlab_resume
                    if swanlab_resume in {"allow", "must"} and not os.environ.get("SWANLAB_MODE"):
                        swanlab_init_kwargs["mode"] = "cloud"

                swanlab.init(**swanlab_init_kwargs)
                if swanlab_run_id:
                    logger.info(
                        "SwanLab resume configured: id=%s resume=%s",
                        swanlab_run_id,
                        swanlab_resume,
                    )

                logger.info("✅ SwanLab initialized")
            except ImportError:
                logger.warning("SwanLab not installed, skipping logging")
            except Exception as e:
                logger.warning(f"SwanLab init failed: {e}, continuing without logging")

            # Always save local visual diagnostics on the main process. SwanLab
            # image upload is best-effort inside the visualization helpers.
            try:
                self._log_sample_visualization()
                if bool(getattr(self.config.trainer, "tf_eval_on_start", False)):
                    self._log_tf_error_curves(use_ema=False)
                    self._log_attention_figure()
            except Exception as e:
                logger.warning(f"Startup visual diagnostics failed: {e}")

    def _collect_cuda_memory_metrics(self, *, reset_peak: bool) -> dict[str, float]:
        if not torch.cuda.is_available():
            return {}

        device_idx = torch.cuda.current_device()
        device = torch.device("cuda", device_idx)
        props = torch.cuda.get_device_properties(device_idx)
        values = torch.tensor(
            [
                float(torch.cuda.memory_allocated(device_idx)),
                float(torch.cuda.memory_reserved(device_idx)),
                float(torch.cuda.max_memory_allocated(device_idx)),
                float(torch.cuda.max_memory_reserved(device_idx)),
                float(props.total_memory),
            ],
            dtype=torch.float64,
            device=device,
        )
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(values, op=dist.ReduceOp.MAX)

        allocated, reserved, peak_allocated, peak_reserved, total = [float(x) for x in values.tolist()]
        if reset_peak:
            torch.cuda.reset_peak_memory_stats(device_idx)

        total_gb = total / 1e9 if total > 0 else 0.0
        peak_reserved_gb = peak_reserved / 1e9
        return {
            "cuda_mem_allocated_gb": allocated / 1e9,
            "cuda_mem_reserved_gb": reserved / 1e9,
            "cuda_mem_peak_allocated_gb": peak_allocated / 1e9,
            "cuda_mem_peak_reserved_gb": peak_reserved_gb,
            "cuda_mem_total_gb": total_gb,
            "cuda_mem_peak_allocated_ratio": (peak_allocated / total) if total > 0 else 0.0,
            "cuda_mem_peak_reserved_ratio": (peak_reserved / total) if total > 0 else 0.0,
            "cuda_mem_peak_headroom_gb": max(total_gb - peak_reserved_gb, 0.0),
        }

    def _mfu_param_counts(self) -> dict[str, float]:
        cached = getattr(self, "_cached_mfu_param_counts", None)
        if cached is not None:
            return cached

        raw_model = self.accelerator.unwrap_model(self.model)
        counts = {
            "total_params": 0.0,
            "trainable_params": 0.0,
            "qwen_total_params": 0.0,
            "qwen_trainable_params": 0.0,
            "non_qwen_total_params": 0.0,
            "non_qwen_trainable_params": 0.0,
        }
        for name, param in raw_model.named_parameters():
            n = float(param.numel())
            trainable = bool(param.requires_grad)
            is_qwen = str(name).startswith("qwen.")
            counts["total_params"] += n
            counts["trainable_params"] += n if trainable else 0.0
            if is_qwen:
                counts["qwen_total_params"] += n
                counts["qwen_trainable_params"] += n if trainable else 0.0
            else:
                counts["non_qwen_total_params"] += n
                counts["non_qwen_trainable_params"] += n if trainable else 0.0
        self._cached_mfu_param_counts = counts
        return counts

    def _mfu_peak_tflops_per_gpu(self) -> float | None:
        configured = _finite_float_or_none(getattr(self.config.trainer, "mfu_peak_tflops_per_gpu", None))
        if configured is not None and configured > 0:
            return configured
        if not torch.cuda.is_available():
            return None
        return _infer_peak_tflops_per_gpu(torch.cuda.get_device_name(torch.cuda.current_device()))

    def _estimate_hardware_efficiency_metrics(self, step_metrics: dict, step_wall_sec: float) -> dict[str, float | str]:
        peak_tflops_per_gpu = self._mfu_peak_tflops_per_gpu()
        if peak_tflops_per_gpu is None or peak_tflops_per_gpu <= 0 or step_wall_sec <= 0:
            return {"hardware/mfu_estimator_mode": "unavailable_no_peak_tflops"}

        trainer_cfg = self.config.trainer
        step_flops_override = _finite_float_or_none(getattr(trainer_cfg, "mfu_step_flops", None))
        flops_per_sample_override = _finite_float_or_none(getattr(trainer_cfg, "mfu_flops_per_sample", None))
        mode = "config_step_flops"
        if step_flops_override is not None and step_flops_override > 0:
            step_flops = step_flops_override
        elif flops_per_sample_override is not None and flops_per_sample_override > 0:
            step_flops = flops_per_sample_override * float(self.total_batch_size)
            mode = "config_flops_per_sample"
        else:
            counts = self._mfu_param_counts()
            micro_batches = _finite_float_or_none(step_metrics.get("grad_accum_micro_batches"))
            if micro_batches is None or micro_batches <= 0:
                micro_batches = float(getattr(self.accelerator, "gradient_accumulation_steps", 1) or 1)

            qwen_tokens_micro = (
                _finite_float_or_none(step_metrics.get("qwen/input_token_count"))
                or _finite_float_or_none(step_metrics.get("time/qwen/input_token_count"))
                or 0.0
            )
            if qwen_tokens_micro > 0 and counts["qwen_total_params"] > 0:
                qwen_global_tokens = qwen_tokens_micro * float(self.accelerator.num_processes) * float(micro_batches)
                qwen_coeff = 6.0 if counts["qwen_trainable_params"] > 0 else 2.0

                action_cfg = getattr(self.config.framework, "action_expert", None)
                action_horizon = int(getattr(action_cfg, "action_horizon", getattr(self.config.datasets.vla_data, "action_horizon", 30)))
                use_state_token = bool(getattr(action_cfg, "use_state_token", False))
                suffix_tokens = action_horizon + (1 if use_state_token else 0)
                ae_global_tokens = float(self.total_batch_size) * float(max(suffix_tokens, 1))
                ae_coeff = 6.0 if counts["non_qwen_trainable_params"] > 0 else 2.0

                step_flops = (
                    qwen_coeff * counts["qwen_total_params"] * qwen_global_tokens
                    + ae_coeff * counts["non_qwen_total_params"] * ae_global_tokens
                )
                mode = "token_param_proxy"
            else:
                frozen_params = max(counts["total_params"] - counts["trainable_params"], 0.0)
                flops_per_sample = 6.0 * counts["trainable_params"] + 2.0 * frozen_params
                step_flops = flops_per_sample * float(self.total_batch_size)
                mode = "param_count_proxy_no_token_count"

        peak_tflops_total = peak_tflops_per_gpu * float(self.accelerator.num_processes)
        achieved_tflops = step_flops / max(step_wall_sec, 1e-9) / 1e12
        mfu = achieved_tflops / peak_tflops_total
        out: dict[str, float | str] = {
            "hardware/mfu_estimate": float(mfu),
            "hardware/mfu_percent_estimate": float(mfu * 100.0),
            "hardware/achieved_tflops_estimate": float(achieved_tflops),
            "hardware/mfu_step_flops_estimated": float(step_flops),
            "hardware/mfu_peak_tflops_per_gpu": float(peak_tflops_per_gpu),
            "hardware/mfu_peak_tflops_total": float(peak_tflops_total),
            "hardware/mfu_estimator_mode": mode,
        }
        out.update({f"hardware/{k}": float(v) for k, v in self._mfu_param_counts().items()})
        return out

    def _log_metrics(self, metrics, force: bool = False):
        """Record training metrics.

        Logged metrics include train loss, PI flow diagnostics, eval arm/gripper
        metrics, timing, learning rate, and CUDA memory.

        force=True flushes the full metric dict regardless of logging_frequency. The eval
        call site passes force=do_light_eval so eval metrics are NEVER suppressed even when
        eval_interval is not a multiple of logging_frequency (otherwise eval-step metrics are
        silently dropped, e.g. a short run with eval_interval=3, logging_frequency=20).
        """
        log_interval = int(getattr(self.config.trainer, "logging_frequency", 1) or 1)
        weight_stats_interval = int(getattr(self.config.trainer, "weight_stats_interval", 0) or 0)
        should_log_metrics = bool(force) or self.completed_steps % log_interval == 0
        should_log_weight_stats = (
            weight_stats_interval > 0
            and self.completed_steps > 0
            and self.completed_steps % weight_stats_interval == 0
        )
        original_metrics = metrics
        should_log_profile = any(str(key).startswith("time/") for key in original_metrics)
        if not should_log_metrics and not should_log_weight_stats and not should_log_profile:
            return

        cuda_metrics = self._collect_cuda_memory_metrics(reset_peak=True) if torch.cuda.is_available() else {}
        if not self.accelerator.is_main_process:
            return

        metrics = dict(original_metrics) if should_log_metrics else {}
        if should_log_profile:
            metrics.update({key: value for key, value in original_metrics.items() if str(key).startswith("time/")})
        if should_log_weight_stats:
            metrics.update(self._collect_weight_range_stats())

        if metrics:
            metrics.update(self._collect_learning_rate_metrics())
            metrics["epoch"] = round(self.completed_steps / len(self.vla_train_dataloader), 2)
            metrics["global_step"] = self.completed_steps
            if self._data_phase_schedule:
                metrics["data_phase_index"] = float(self._data_phase_idx or 0)
                metrics["data_phase_name"] = self._data_phase_name
            if self._eta_start_time is not None:
                elapsed_sec = max(time.time() - self._eta_start_time, 0.0)
                steps_done = max(self.completed_steps - self._eta_start_step, 1)
                avg_step_sec = elapsed_sec / steps_done
                remaining_steps = max(int(self.config.trainer.max_train_steps) - self.completed_steps, 0)
                metrics["eta/remaining_min"] = remaining_steps * avg_step_sec / 60.0
                metrics["eta/remaining_hours"] = remaining_steps * avg_step_sec / 3600.0
                metrics["eta/elapsed_min"] = elapsed_sec / 60.0
                metrics["eta/avg_step_sec"] = avg_step_sec
            metrics.update(cuda_metrics)

            if HAS_SWANLAB and self._swanlab_enabled:
                try:
                    swanlab_metrics = self._build_swanlab_scalar_metrics(metrics)
                    swanlab.log(swanlab_metrics, step=self._swanlab_step())
                except Exception as exc:  # noqa: BLE001
                    # Don't silently swallow: a SwanLab schema/serialization error or
                    # transient host failure would otherwise freeze the dashboard with no
                    # trainer-side signal. Count it (persisted to metrics.jsonl) + rate-limit warn.
                    self._swanlab_log_failures = getattr(self, "_swanlab_log_failures", 0) + 1
                    metrics["swanlab/log_failed"] = float(self._swanlab_log_failures)
                    if self._swanlab_log_failures == 1 or self._swanlab_log_failures % 50 == 0:
                        logger.warning(
                            f"SwanLab metric upload failed (count={self._swanlab_log_failures}): {exc}"
                        )

            try:
                metrics_path = Path(self.config.output_dir) / "metrics.jsonl"
                metrics_path.parent.mkdir(parents=True, exist_ok=True)
                json_metrics = {}
                for k, v in metrics.items():
                    if isinstance(v, (int, float, np.number)):
                        json_metrics[k] = float(v)
                    elif isinstance(v, str):
                        json_metrics[k] = v
                json_metrics["timestamp"] = time.time()
                with metrics_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(json_metrics, ensure_ascii=False) + "\n")
            except Exception as e:
                logger.warning(f"Failed to write local metrics jsonl: {e}")

            logger.info(f"Step {self.completed_steps}, Loss: {metrics}")

    def _build_swanlab_scalar_metrics(self, metrics: dict) -> dict[str, float]:
        """Return only the scalar curves requested for the release run.

        Full diagnostics still go to local ``metrics.jsonl``. SwanLab is kept
        intentionally small so the run page exposes only the comparison curves
        needed for the public run page.
        """

        return filter_swanlab_scalars(metrics)

    def _collect_learning_rate_metrics(self) -> dict[str, float]:
        """Return LR curves for local jsonl and SwanLab scalar logging."""

        try:
            lrs = [float(lr) for lr in self.lr_scheduler.get_last_lr()]
        except Exception:  # noqa: BLE001
            lrs = []

        if not lrs:
            try:
                lrs = [float(group.get("lr", 0.0)) for group in self.optimizer.param_groups]
            except Exception:  # noqa: BLE001
                lrs = []

        out: dict[str, float] = {}
        for idx, lr in enumerate(lrs):
            if not np.isfinite(lr):
                continue
            group_name = None
            try:
                group = self.optimizer.param_groups[idx]
                group_name = group.get("name", None)
            except Exception:  # noqa: BLE001
                group_name = None
            safe_name = str(group_name or f"group_{idx}").replace("/", "_")
            out[f"lr/{safe_name}"] = lr

        if lrs and np.isfinite(float(lrs[0])):
            out["learning_rate"] = float(lrs[0])
            out.setdefault("lr/learning_rate", float(lrs[0]))
        return out

    def _collect_weight_range_stats(self) -> dict[str, float]:
        """Collect low-frequency parameter range diagnostics.

        The scan runs only on the main process and is intended for periodic
        health checks, not per-step logging.  By default we inspect trainable
        parameters only: this catches exploding Action Expert / LoRA weights
        without repeatedly scanning a frozen Qwen-VL backbone.
        """
        raw_model = self.accelerator.unwrap_model(self.model)
        trainable_only = bool(getattr(self.config.trainer, "weight_stats_trainable_only", True))
        groups = {
            "weights/ae": ("action_expert.",),
            "weights/qwen": ("qwen.",),
        }
        accum: dict[str, dict[str, float | int | None]] = {
            prefix: {
                "min": None,
                "max": None,
                "abs_max": None,
                "param_count": 0,
                "tensor_count": 0,
                "nonfinite_minmax_tensors": 0,
            }
            for prefix in groups
        }

        with torch.no_grad():
            for name, param in raw_model.named_parameters():
                if trainable_only and not param.requires_grad:
                    continue
                metric_prefix = None
                for prefix, name_prefixes in groups.items():
                    if name.startswith(name_prefixes):
                        metric_prefix = prefix
                        break
                if metric_prefix is None or param.numel() == 0:
                    continue
                tensor = param.detach()
                if tensor.is_sparse:
                    tensor = tensor.coalesce().values()
                tensor = tensor.float()
                try:
                    p_min = float(torch.amin(tensor).cpu())
                    p_max = float(torch.amax(tensor).cpu())
                    p_abs_max = float(torch.amax(torch.abs(tensor)).cpu())
                except RuntimeError:
                    accum[metric_prefix]["nonfinite_minmax_tensors"] = int(
                        accum[metric_prefix]["nonfinite_minmax_tensors"] or 0
                    ) + 1
                    continue

                bucket = accum[metric_prefix]
                bucket["param_count"] = int(bucket["param_count"] or 0) + int(param.numel())
                bucket["tensor_count"] = int(bucket["tensor_count"] or 0) + 1
                if not (np.isfinite(p_min) and np.isfinite(p_max) and np.isfinite(p_abs_max)):
                    bucket["nonfinite_minmax_tensors"] = int(bucket["nonfinite_minmax_tensors"] or 0) + 1
                    continue
                bucket["min"] = p_min if bucket["min"] is None else min(float(bucket["min"]), p_min)
                bucket["max"] = p_max if bucket["max"] is None else max(float(bucket["max"]), p_max)
                bucket["abs_max"] = (
                    p_abs_max if bucket["abs_max"] is None else max(float(bucket["abs_max"]), p_abs_max)
                )

        stats: dict[str, float] = {}
        for prefix, bucket in accum.items():
            stats[f"{prefix}/param_count"] = float(bucket["param_count"] or 0)
            stats[f"{prefix}/tensor_count"] = float(bucket["tensor_count"] or 0)
            stats[f"{prefix}/trainable_only"] = 1.0 if trainable_only else 0.0
            stats[f"{prefix}/nonfinite_minmax_tensors"] = float(bucket["nonfinite_minmax_tensors"] or 0)
            if bucket["min"] is not None:
                stats[f"{prefix}/min"] = float(bucket["min"])
                stats[f"{prefix}/max"] = float(bucket["max"])
                stats[f"{prefix}/abs_max"] = float(bucket["abs_max"])
        return stats

    def _should_profile_timing(self) -> bool:
        """Low-frequency timing profile for hidden CPU/GPU costs.

        The normal `data_time/model_time` counters stay cheap and run every
        step. This detailed profiler uses CUDA synchronization, so keep it to
        the startup window and very sparse later samples.
        """
        first_n = int(getattr(self.config.trainer, "profile_timing_first_n_steps", 0) or 0)
        if first_n > 0 and self.completed_steps < first_n:
            return True
        interval = int(getattr(self.config.trainer, "profile_timing_interval", 0) or 0)
        return interval > 0 and self.completed_steps > 0 and self.completed_steps % interval == 0

    def _sync_profile_clock(self, enabled: bool) -> float:
        if enabled and bool(getattr(self.config.trainer, "profile_timing_cuda_sync", True)) and torch.cuda.is_available():
            torch.cuda.synchronize()
        return time.perf_counter()

    def _will_log_step_metrics(self) -> bool:
        if not self.accelerator.is_main_process:
            return False
        log_interval = int(getattr(self.config.trainer, "logging_frequency", 1) or 1)
        return (self.completed_steps + 1) % log_interval == 0

    def _log_slow_data_sample(self, batch_vla: list[dict] | None) -> None:
        if not self.accelerator.is_main_process or not batch_vla:
            return
        threshold = float(getattr(self.config.trainer, "slow_sample_log_threshold_sec", 1.0) or 0.0)
        if threshold <= 0:
            return
        first_n = int(getattr(self.config.trainer, "slow_sample_log_first_n_steps", 200) or 0)
        interval = int(getattr(self.config.trainer, "slow_sample_log_interval", 1000) or 0)
        should_log = self.completed_steps <= first_n or (interval > 0 and self.completed_steps % interval == 0)
        if not should_log:
            return
        slowest = None
        slowest_sec = threshold
        for example in batch_vla:
            total_sec = float(example.get("_data_profile_total_sec", 0.0) or 0.0)
            if total_sec > slowest_sec:
                slowest = example
                slowest_sec = total_sec
        if slowest is None:
            return
        row = {
            "global_step": int(self.completed_steps),
            "dataset_id": str(slowest.get("_dataset_id", "")),
            "episode_id": str(slowest.get("_episode_id", "")),
            "frame_index": int(slowest.get("_frame_index", -1)),
            "total_sec": slowest_sec,
            "resolve_sec": float(slowest.get("_data_profile_resolve_sec", 0.0) or 0.0),
            "convert_sec": float(slowest.get("_data_profile_convert_sec", 0.0) or 0.0),
            "video_decode_sec": float(slowest.get("_data_profile_video_decode_sec", 0.0) or 0.0),
            "video_decode_max_view_sec": float(
                slowest.get("_data_profile_video_decode_max_view_sec", 0.0) or 0.0
            ),
            "video_decode_max_view_key": str(slowest.get("_data_profile_video_decode_max_view_key", "")),
            "image_preprocess_sec": float(slowest.get("_data_profile_image_preprocess_sec", 0.0) or 0.0),
            "instruction_sec": float(slowest.get("_data_profile_instruction_sec", 0.0) or 0.0),
            "view_decode": slowest.get("_data_profile_video_decode_by_view", []),
        }
        try:
            path = Path(self.config.output_dir) / "slow_samples.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.warning(f"Failed to write slow sample profile: {e}")

    def _log_training_config(self):
        """Record training config."""
        if self.accelerator.is_main_process:
            logger.info("***** Training Configuration *****")
            logger.info(f"  Total optimization steps = {self.config.trainer.max_train_steps}")
            logger.info(f"  Per device batch size = {self.config.datasets.vla_data.per_device_batch_size}")
            logger.info(f"  Eval batch size = {self.config.datasets.vla_data.get('eval_batch_size', self.config.datasets.vla_data.per_device_batch_size)}")
            logger.info(f"  Gradient accumulation steps = {self.config.trainer.gradient_accumulation_steps}")
            logger.info(f"  Total batch size = {self.total_batch_size}")
