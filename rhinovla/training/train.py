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
from accelerate.utils import set_seed
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_scheduler

# Local Modules
from rhinovla.dataloader import build_dataloader
from rhinovla.model.framework import build_framework
from rhinovla.training.trainer_utils.config_tracker import AccessTrackedConfig, wrap_config
from rhinovla.training.trainer_utils.trainer_tools import TrainerUtils, build_param_lr_groups, normalize_dotlist_args, resize_images

# Use pure DDP without DeepSpeed; optimized for communication efficiency
import datetime
from accelerate.utils import DistributedDataParallelKwargs, InitProcessGroupKwargs




from rhinovla.training._trainer_helpers import *  # noqa: F401,F403
from rhinovla.training.eval import _EvalMixin
from rhinovla.training.metrics import _MetricsMixin
from rhinovla.training.visualize import _VizMixin


class VLATrainer(TrainerUtils, _EvalMixin, _MetricsMixin, _VizMixin):
    def __init__(self, cfg, model, vla_train_dataloader, optimizer, lr_scheduler, accelerator):
        self.config = cfg
        self.model = model
        self.vla_train_dataloader = vla_train_dataloader
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.accelerator = accelerator

        self.completed_steps = 0
        self.total_batch_size = self._calculate_total_batch_size()
        self._eta_start_time = None
        self._eta_start_step = 0
        self._swanlab_enabled = self._tracker_enabled("swanlab")
        self._data_phase_schedule = _get_data_phase_schedule(cfg)
        self._data_phase_idx, self._data_phase = _select_data_phase(
            self._data_phase_schedule, self.completed_steps
        )
        self._data_phase_name = str(self._data_phase.get("name", "")) if self._data_phase else ""

    def prepare_training(self):
        rank = dist.get_rank() if dist.is_initialized() else 0
        seed = self.config.seed + rank if hasattr(self.config, "seed") else rank + 3047
        set_seed(seed)

        self._init_checkpointing()
        self.model = TrainerUtils.freeze_backbones(
            self.model,
            freeze_modules=getattr(self.config.trainer, "freeze_modules", None),
        )
        self.optimizer, self.lr_scheduler = setup_optimizer_and_scheduler(self.model, self.config)
        self._adjust_lr_scheduler_for_resume()

        self.print_trainable_parameters(self.model)

        self.model, self.optimizer, self.vla_train_dataloader = self.setup_distributed_training(
            self.accelerator,
            self.model,
            self.optimizer,
            self.vla_train_dataloader,
        )

        self._maybe_restore_full_state()
        self._maybe_seek_resume_data_start()

        self._init_val_dataloader()
        self._init_ema()
        self._init_trackers()

    def _init_ema(self):
        """Initialize EMA (Exponential Moving Average) for trainable parameters only."""
        ema_decay = getattr(self.config.trainer, "ema_decay", 0)
        if ema_decay > 0:
            raw_model = self.accelerator.unwrap_model(self.model)
            uses_framework_predict = hasattr(raw_model, "predict_action") and hasattr(raw_model, "forward")
            # Only track trainable params to save memory (~0.7 GB vs 4.8 GB for full model)
            self._ema_params = {}
            for name, param in raw_model.named_parameters():
                if param.requires_grad:
                    self._ema_params[name] = param.data.clone()
            self._ema_decay = ema_decay
            n_ema = sum(p.numel() for p in self._ema_params.values())
            logger.info(f"✅ EMA initialized: {n_ema/1e6:.1f}M params, decay={ema_decay}")

            # Load pretrained EMA if specified
            pretrained_ema = getattr(self.config.trainer, "pretrained_ema", None)
            if pretrained_ema and os.path.exists(pretrained_ema):
                ema_state = torch.load(pretrained_ema, map_location="cpu", weights_only=False)
                loaded = 0
                for name in self._ema_params:
                    if name in ema_state:
                        self._ema_params[name].copy_(ema_state[name].to(self._ema_params[name].device))
                        loaded += 1
                logger.info(f"✅ Loaded pretrained EMA: {loaded}/{len(self._ema_params)} params from {pretrained_ema}")
        else:
            self._ema_params = {}
            self._ema_decay = 0

    @torch.no_grad()
    def _update_ema(self):
        """Update EMA weights after each optimizer step."""
        if not self._ema_params:
            return
        raw_model = self.accelerator.unwrap_model(self.model)
        for name, param in raw_model.named_parameters():
            if name in self._ema_params:
                self._ema_params[name].lerp_(param.data, 1.0 - self._ema_decay)

    def _apply_ema(self):
        """Swap EMA weights into model (for eval). Returns backup of original weights."""
        if not self._ema_params:
            return {}
        raw_model = self.accelerator.unwrap_model(self.model)
        backup = {}
        for name, param in raw_model.named_parameters():
            if name in self._ema_params:
                backup[name] = param.data.clone()
                param.data.copy_(self._ema_params[name])
        return backup

    def _restore_from_ema(self, backup):
        """Restore original weights after EMA eval."""
        if not backup:
            return
        raw_model = self.accelerator.unwrap_model(self.model)
        for name, param in raw_model.named_parameters():
            if name in backup:
                param.data.copy_(backup[name])

    def _calculate_total_batch_size(self):
        """Calculate global batch size."""
        return (
            self.config.datasets.vla_data.per_device_batch_size
            * self.accelerator.num_processes
            * self.accelerator.gradient_accumulation_steps
        )

    def _shutdown_current_dataloader_iterator(self):
        """Best-effort shutdown for persistent DataLoader workers before a phase switch."""
        iterator = getattr(self, "vla_iter", None)
        shutdown = getattr(iterator, "_shutdown_workers", None)
        if callable(shutdown):
            try:
                shutdown()
            except Exception as e:
                logger.warning(f"Failed to shutdown old dataloader workers cleanly: {e}")
        self.vla_iter = None

    def _switch_data_phase(self, phase_idx: int, phase: dict[str, Any]) -> None:
        """Switch only the data source while keeping model/optimizer/scheduler/EMA continuous."""
        old_name = self._data_phase_name or "<none>"
        new_name = str(phase.get("name", f"phase_{phase_idx}"))
        _apply_data_phase(self.config, phase)
        if dist.is_initialized():
            dist.barrier()
        if self.accelerator.is_main_process:
            logger.info(
                f"DATA_PHASE_SWITCH step={self.completed_steps} {old_name} -> {new_name}; "
                f"vla_data overrides applied from phase_schedule[{phase_idx}]"
            )

        self._shutdown_current_dataloader_iterator()
        new_loader = prepare_data(
            cfg=self.config,
            accelerator=self.accelerator,
            output_dir=Path(self.config.output_dir),
            apply_initial_phase=False,
        )
        self.vla_train_dataloader = self.accelerator.prepare(new_loader)
        self.vla_epoch_count = 0
        self._create_data_iterators()
        self.total_batch_size = self._calculate_total_batch_size()

        # Eval / TF episode caches are data-dependent; rebuild them lazily after switching.
        self._val_dataloader = None
        self._val_iterator = None
        self._tf_eval_episodes = None
        self._init_val_dataloader()

        self._data_phase_idx = phase_idx
        self._data_phase = phase
        self._data_phase_name = new_name
        if dist.is_initialized():
            dist.barrier()

    def _maybe_switch_data_phase(self):
        if not self._data_phase_schedule:
            return
        phase_idx, phase = _select_data_phase(self._data_phase_schedule, self.completed_steps)
        if phase is None or phase_idx is None or phase_idx == self._data_phase_idx:
            return
        self._switch_data_phase(phase_idx, phase)

    def _init_checkpointing(self):
        """Initialize checkpoint directory and handle checkpoint loading."""
        self.checkpoint_dir = os.path.join(self.config.output_dir, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        pretrained_checkpoint = getattr(self.config.trainer, "pretrained_checkpoint", None)
        is_resume = getattr(self.config.trainer, "is_resume", False)
        self.resume_from_checkpoint = pretrained_checkpoint

        if is_resume:
            resume_from_checkpoint, self.completed_steps = self._get_latest_checkpoint(self.checkpoint_dir)
            if resume_from_checkpoint:
                self.resume_from_checkpoint = resume_from_checkpoint
                self.model = self.load_pretrained_backbones(self.model, self.resume_from_checkpoint, reload_modules=None)
                logger.info(
                    f"Resuming training from checkpoint: {self.resume_from_checkpoint}, steps: {self.completed_steps}"
                )
                return

            raise FileNotFoundError(
                f"trainer.is_resume=true but no valid checkpoint was found in {self.checkpoint_dir}. "
                "Refusing to restart from pretrained weights into a resume run."
            )

        if pretrained_checkpoint:
            reload_modules = getattr(self.config.trainer, "reload_modules", None)
            self.model = self.load_pretrained_backbones(
                self.model,
                pretrained_checkpoint,
                reload_modules=reload_modules,
            )
            self.completed_steps = 0
            self.resume_from_checkpoint = pretrained_checkpoint
            logger.info(f"Loaded pretrained checkpoint: {pretrained_checkpoint}, steps: {self.completed_steps}")
        else:
            # Fail-fast guard: refuse a silent from-scratch start into a run dir
            # that already has checkpoints -- it would overwrite same-step ckpts and mix
            # metrics.jsonl. Require explicit is_resume / pretrained_checkpoint / fresh run_id,
            # or trainer.allow_overwrite_run=true to override.
            existing_ckpts = (
                [f for f in os.listdir(self.checkpoint_dir)
                 if f.startswith("steps_") and (f.endswith("_pytorch_model.pt") or f.endswith("_model.safetensors"))]
                if os.path.isdir(self.checkpoint_dir) else []
            )
            if existing_ckpts and not bool(getattr(self.config.trainer, "allow_overwrite_run", False)):
                raise RuntimeError(
                    f"Refusing from-scratch start: {self.checkpoint_dir} already has "
                    f"{len(existing_ckpts)} checkpoint(s). This would overwrite checkpoints and mix "
                    f"metrics.jsonl. Set trainer.is_resume=true to continue, "
                    f"trainer.pretrained_checkpoint=<path> to init from weights, use a fresh run_id, "
                    f"or trainer.allow_overwrite_run=true to override."
                )
            logger.info("No pretrained checkpoint provided. Starting training from scratch.")
            self.completed_steps = 0

            merge_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    def _adjust_lr_scheduler_for_resume(self):
        """Adjust LR scheduler state without loading optimizer/scheduler state."""
        scheduler_offset = int(getattr(self.config.trainer, "lr_scheduler_step_offset", 0) or 0)
        scheduler_steps = int(self.completed_steps) + scheduler_offset
        if scheduler_steps > 0:
            logger.info(
                "Adjusting LR scheduler to global step %s "
                "(completed_steps=%s, configured_offset=%s)",
                scheduler_steps,
                self.completed_steps,
                scheduler_offset,
            )
            # Do not replay N scheduler steps on resume. Large runs resume at
            # step 30k+, and replaying step-by-step is both slow and noisy
            # (PyTorch warns because no optimizer step preceded the replay).
            # Setting last_epoch=N-1 then taking one scheduler step produces
            # the same LR as N replays for LambdaLR/cosine schedules.
            try:
                self.lr_scheduler.last_epoch = scheduler_steps - 1
                if hasattr(self.lr_scheduler, "_step_count"):
                    self.lr_scheduler._step_count = max(int(scheduler_steps), 1)
                self.lr_scheduler.step()
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Fast LR scheduler resume to step %s failed (%s); falling back to replay",
                    scheduler_steps,
                    exc,
                )
                for _ in range(scheduler_steps):
                    self.lr_scheduler.step()
            logger.info(
                f"LR scheduler adjusted to step {scheduler_steps}, current LR: {self.lr_scheduler.get_last_lr()}"
            )

    def _load_checkpoint(self, checkpoint_path):
        """Load checkpoint."""
        self.accelerator.load_state(checkpoint_path)
        self.accelerator.print(f"Resumed from checkpoint: {checkpoint_path}")

    def _maybe_restore_full_state(self):
        """Restore optimizer (Adam moments) from steps_{N}_optim.pt if present (post-prepare).

        Called after accelerator.prepare so self.optimizer is wrapped; each rank loads the
        rank-0-saved (DDP-replicated) state and PyTorch maps tensors to that rank's device.
        Legacy ckpts without _optim.pt keep existing behavior (fresh optimizer; LR recomputed).
        """
        if not bool(getattr(self.config.trainer, "is_resume", False)):
            return
        if int(getattr(self, "completed_steps", 0) or 0) <= 0:
            return
        optim_path = os.path.join(self.checkpoint_dir, f"steps_{self.completed_steps}_optim.pt")
        if not os.path.isfile(optim_path):
            if self.accelerator.is_main_process:
                logger.warning("[full-state] no %s; weight-only resume + fresh optimizer (Adam moments reinit)", optim_path)
            return
        try:
            sd = torch.load(optim_path, map_location="cpu", weights_only=False)
            self.optimizer.load_state_dict(sd["optimizer"])
            if self.accelerator.is_main_process:
                logger.info("[full-state] restored optimizer state from %s at step %s", optim_path, self.completed_steps)
        except Exception as e:
            if self.accelerator.is_main_process:
                logger.warning("[full-state] optimizer restore from %s failed (%s); fresh optimizer", optim_path, e)

    def _maybe_seek_resume_data_start(self):
        """On resume, begin data iteration at the next un-consumed sample (``start_idx``)
        instead of replaying the prefix.

        The train dataset's consumption order is the fixed, epoch-invariant permutation
        ``dataset._order`` (``__getitem__`` maps idx i -> ``_order[i]``). After ``completed_steps``
        optimizer steps we have consumed ``completed_steps * global_batch`` samples in that order,
        so the sample to resume at sits at position
        ``start_idx = (completed_steps * global_batch) % n``.

        We reorder ``_order`` so that position 0 *is* ``start_idx`` -- take ``order[start_idx:]``
        then wrap the already-consumed head ``order[:start_idx]`` onto the tail. This is the O(1)
        "begin at start_idx" seek: the prefix samples are simply never visited by ``__getitem__``
        (no read, no video decode), unlike a walk-and-skip which would still iterate the whole
        prefix and still burn optimizer/scheduler steps. (Mathematically identical to
        ``np.roll(order, -start_idx)``; written as an explicit concatenate so the intent --
        "iteration starts at start_idx" -- is visible in the code.)

        Correctness across ranks: ``start_idx`` is derived purely from (completed_steps, global_batch,
        n), all identical on every rank (completed_steps from the ckpt; global_batch from config;
        n from the shared index). accelerate shards the dataloader by *index* at iteration time and
        ``__getitem__`` reads ``_order`` lazily, so an identical reorder on every rank keeps the
        index->sample mapping consistent and sharding coherent. We reorder post-prepare but
        pre-iterator (workers fork at the first ``iter()`` in ``_create_data_iterators``), so the
        forked workers observe the new start.

        Guards: resume only (completed_steps > 0); deterministic, epoch-invariant orders only
        (episode/shard shuffle on the train split). For ``sample_shuffle`` the per-epoch order is
        drawn online and is not encoded in ``_order``, so a start_idx is meaningless -> skip (the
        prefix-replay fallback is correct there). Permanently beginning at start_idx is acceptable:
        every epoch is the same fixed order, so a cyclic shift only moves the epoch boundary, not
        the samples seen over training. Only applied at resume setup, never on later phase switches.
        """
        if not bool(getattr(self.config.trainer, "is_resume", False)):
            return
        if int(getattr(self, "completed_steps", 0) or 0) <= 0:
            return
        ds = getattr(self.vla_train_dataloader, "dataset", None)
        order = getattr(ds, "_order", None)
        if ds is None or order is None or len(order) == 0:
            return
        if str(getattr(ds, "split", "train")) != "train":
            return
        shuffle_mode = str(getattr(ds, "shuffle_mode", "episode_shuffle"))
        if shuffle_mode in {"", "sample", "sample_shuffle"}:
            if self.accelerator.is_main_process:
                logger.info(
                    "[resume] shuffle_mode=%s draws order online; skipping data start-seek "
                    "(prefix replay fallback)", shuffle_mode
                )
            return
        global_batch = self._calculate_total_batch_size()
        n = int(len(order))
        start_idx = int(int(self.completed_steps) * int(global_batch)) % n
        if start_idx <= 0:
            return
        # Begin iteration at start_idx: order[start_idx:] first, consumed head wrapped to tail.
        ds._order = np.concatenate([order[start_idx:], order[:start_idx]])
        if self.accelerator.is_main_process:
            logger.info(
                "[resume] data iteration begins at start_idx=%s (completed_steps=%s x "
                "global_batch=%s mod n=%s); prefix skipped without decode, not replayed.",
                start_idx, self.completed_steps, global_batch, n,
            )

    def _save_norm_stats(self, path_without_ext):
        """Save normalization stats alongside checkpoint for inference.

        Writes `<path_without_ext>_norm_stats.json` containing action/state
        mean/std used during training. Inference scripts can load this to
        avoid depending on the dataloader.
        """
        try:
            vla_cfg = self.config.datasets.vla_data
            action_mean = list(vla_cfg.get("action_mean", []))
            action_std = list(vla_cfg.get("action_std", []))
            state_mean = list(vla_cfg.get("state_mean", []))
            state_std = list(vla_cfg.get("state_std", []))
            active_action_dims = list(vla_cfg.get("active_action_dims", []))
            if not action_mean or not action_std:
                return
            payload = {
                "action_mean": action_mean,
                "action_std": action_std,
                "state_mean": state_mean,
                "state_std": state_std,
                "active_action_dims": active_action_dims,
                "action_horizon": int(vla_cfg.get("action_horizon", 0)),
                "state_dim": int(vla_cfg.get("state_dim", 0)),
                "image_size": list(vla_cfg.get("image_size", [])) if hasattr(vla_cfg.get("image_size", []), "__iter__") else [int(vla_cfg.get("image_size", 0))],
            }
            for key in ("action_abs_mean", "action_abs_std", "action_delta_from_state_dims"):
                val = vla_cfg.get(key, None)
                if val is not None:
                    payload[key] = list(val) if hasattr(val, "__iter__") and not isinstance(val, str) else val
            if vla_cfg.get("action_type", None) is not None:
                payload["action_type"] = str(vla_cfg.get("action_type"))
            _atomic_json_dump(payload, path_without_ext + "_norm_stats.json")
        except Exception as e:
            logger.warning(f"Failed to save norm_stats sidecar: {e}")

    def _save_checkpoint(self):
        """Save current training state."""
        if self.accelerator.is_main_process:
            save_format = getattr(self.config.trainer, "save_format", "pt")
            checkpoint_path = os.path.join(self.checkpoint_dir, f"steps_{self.completed_steps}")

            state_dict = self.accelerator.get_state_dict(self.model)
            if save_format == "safetensors":
                _atomic_safetensors_save(state_dict, checkpoint_path + "_model.safetensors")
            elif save_format == "pt":
                _atomic_torch_save(state_dict, checkpoint_path + "_pytorch_model.pt")
                if bool(getattr(self.config.trainer, "save_full_state", True)):
                    try:
                        _atomic_torch_save(
                            {"optimizer": self.optimizer.state_dict(), "completed_steps": self.completed_steps},
                            checkpoint_path + "_optim.pt",
                        )
                        logger.info("[full-state] saved optimizer state -> %s_optim.pt", checkpoint_path)
                    except Exception as e:
                        logger.warning("[full-state] optimizer save failed (model-only ckpt kept): %s", e)
            else:
                raise ValueError(f"Unsupported save_format `{save_format}`. Expected `pt` or `safetensors`.")

            summary_data = {"steps": self.completed_steps}
            with open(os.path.join(self.config.output_dir, "summary.jsonl"), "a") as f:
                f.write(json.dumps(summary_data) + "\n")
            # Save EMA weights alongside
            if self._ema_params:
                ema_path = checkpoint_path + "_ema_params.pt"
                _atomic_torch_save(self._ema_params, ema_path)

            # Save norm_stats sidecar for inference
            self._save_norm_stats(checkpoint_path)

            self.accelerator.print(f"✅ Checkpoint saved at {checkpoint_path}")

            # Archive old checkpoints, keep only the latest N (by numeric step)
            # 之前的 bug: glob 同时匹配 *_pytorch_model.pt + *_ema_params.pt, 且 sorted
            # 是字典序 (steps_1000 < steps_500), 导致 keep_n 计数错位 + 保留集错误.
            # 修复: 按 step 号数字排序，配套归档每个 step 的所有相关文件 (model + ema).
            keep_n = int(getattr(self.config.trainer, "keep_last_n_checkpoints", 3))
            if keep_n > 0:
                import glob
                import re
                # 只看 model 文件确定 step 集合
                model_files = glob.glob(os.path.join(self.checkpoint_dir, "steps_*_pytorch_model.pt"))
                model_files += glob.glob(os.path.join(self.checkpoint_dir, "steps_*_model.safetensors"))
                step_re = re.compile(r"steps_(\d+)_(?:pytorch_)?model")
                steps_seen: list[int] = []
                for fp in model_files:
                    m = step_re.search(os.path.basename(fp))
                    if m:
                        steps_seen.append(int(m.group(1)))
                steps_seen = sorted(set(steps_seen))
                if len(steps_seen) > keep_n:
                    to_archive = steps_seen[:-keep_n]
                    archive_dir = Path(self.config.output_dir) / "legacy" / "checkpoints"
                    archive_dir.mkdir(parents=True, exist_ok=True)
                    for s in to_archive:
                        # 归档该 step 的所有相关文件 (model + ema, 任何后缀)
                        victims = glob.glob(os.path.join(self.checkpoint_dir, f"steps_{s}_*"))
                        for fp in victims:
                            try:
                                target = archive_dir / os.path.basename(fp)
                                suffix = 0
                                while target.exists():
                                    suffix += 1
                                    target = archive_dir / f"{suffix}_{os.path.basename(fp)}"
                                shutil.move(fp, target)
                                self.accelerator.print(f"📦 Archived old checkpoint: {fp} -> {target}")
                            except OSError as e:
                                self.accelerator.print(f"⚠️ Failed to archive {fp}: {e}")

            if isinstance(self.config, AccessTrackedConfig):
                logger.info("📊 Saving accessed configuration...")
                output_dir = Path(self.config.output_dir)
                self.config.save_accessed_config(output_dir / "config.yaml", use_original_values=False)
                logger.info("✅ Configuration files saved")

        self.accelerator.wait_for_everyone()

    def _create_data_iterators(self):
        """Create data iterators."""
        self.vla_iter = iter(self.vla_train_dataloader)

    def _get_next_batch(self):
        """Get next batch (automatically handle data loop)."""
        try:
            batch_vla = next(self.vla_iter)
        except StopIteration:
            if not hasattr(self, "vla_epoch_count"):
                self.vla_epoch_count = 0
            self.vla_iter, self.vla_epoch_count = TrainerUtils._reset_dataloader(
                self.vla_train_dataloader, self.vla_epoch_count
            )
            batch_vla = next(self.vla_iter)

        return batch_vla

    def train(self):
        """Execute training loop."""
        self._log_training_config()
        self._create_data_iterators()
        self._eta_start_time = time.time()
        self._eta_start_step = self.completed_steps
        progress_bar = tqdm(
            range(self.config.trainer.max_train_steps), disable=not self.accelerator.is_local_main_process
        )

        while self.completed_steps < self.config.trainer.max_train_steps:
            self._maybe_switch_data_phase()
            t_start_data = time.perf_counter()
            batch_vla = self._get_next_batch()
            t_end_data = time.perf_counter()

            t_start_model = time.perf_counter()
            step_metrics = self._train_step(batch_vla)
            t_end_model = time.perf_counter()

            timing_accum = getattr(self, "_optimizer_step_timing", None)
            if timing_accum is None:
                timing_accum = {
                    "wall_start": t_start_data,
                    "data_sec": 0.0,
                    "model_sec": 0.0,
                    "micro_batches": 0,
                }
                self._optimizer_step_timing = timing_accum
            timing_accum["data_sec"] += t_end_data - t_start_data
            timing_accum["model_sec"] += t_end_model - t_start_model
            timing_accum["micro_batches"] += 1

            if self.accelerator.sync_gradients:
                self._update_ema()
                progress_bar.update(1)
                self.completed_steps += 1

            if self.accelerator.is_local_main_process:
                progress_bar.set_postfix(
                    {
                        "data_times": f"{t_end_data - t_start_data:.3f}",
                        "model_times": f"{t_end_model - t_start_model:.3f}",
                    }
                )

            if self.accelerator.sync_gradients:
                # 双频 eval cadence:
                #   eval_interval: 轻量 eval
                #   tf_eval_interval: 重 eval，渲染 TF 误差曲线
                tf_interval = int(getattr(
                    self.config.trainer, "tf_eval_interval",
                    self.config.trainer.eval_interval,
                ))
                do_light_eval = self.completed_steps % self.config.trainer.eval_interval == 0
                do_tf_eval = self.completed_steps % tf_interval == 0

                if do_light_eval:
                    torch.cuda.empty_cache()
                    step_metrics = self.eval_action_model(step_metrics)
                    # VLM input prompt cards on the val cadence (same as light eval).
                    self._log_vlm_prompt_cards(self.vla_train_dataloader.dataset, self._swanlab_step())
                    if do_tf_eval:
                        self._log_tf_error_curves(use_ema=False)
                        self._log_attention_figure()
                    if self._ema_params:
                        backup = self._apply_ema()
                        step_metrics = self.eval_action_model(step_metrics, use_ema=True)
                        if do_tf_eval:
                            self._log_tf_error_curves(use_ema=True)
                        self._restore_from_ema(backup)
                        self.accelerator.unwrap_model(self.model).train()
                    torch.cuda.empty_cache()

                step_wall_sec = t_end_model - float(timing_accum["wall_start"])
                step_data_sec = float(timing_accum["data_sec"])
                step_model_sec = float(timing_accum["model_sec"])
                micro_batches = int(timing_accum["micro_batches"])
                self._optimizer_step_timing = None
                step_metrics["data_time"] = step_data_sec
                step_metrics["model_time"] = step_model_sec
                step_metrics["optimizer_step_wall_sec"] = step_wall_sec
                step_metrics["grad_accum_micro_batches"] = float(micro_batches)
                step_metrics["last_micro_data_sec"] = t_end_data - t_start_data
                step_metrics["last_micro_model_sec"] = t_end_model - t_start_model
                if "time/data_loader/item_total_max_sec" in step_metrics:
                    data_minus_item = float(step_data_sec - step_metrics["time/data_loader/item_total_max_sec"])
                    step_metrics["time/data_loader/wait_minus_item_max_sec"] = data_minus_item
                    step_metrics["time/data_loader/unhidden_wait_sec"] = max(data_minus_item, 0.0)
                    step_metrics["time/data_loader/prefetch_hidden_item_sec"] = max(-data_minus_item, 0.0)
                self._log_slow_data_sample(batch_vla)
                total_step_time = max(step_wall_sec, 1e-9)
                step_metrics["samples_per_sec"] = float(self.total_batch_size) / total_step_time
                step_metrics.update(self._estimate_hardware_efficiency_metrics(step_metrics, total_step_time))
                # ETA to max_train_steps from amortized wall-clock per optimizer step
                # (rolling 100-step window; deliberately includes eval/ckpt pauses —
                # they delay completion too, so the honest rate is the amortized one).
                now_ts = time.time()
                prev_ts = getattr(self, "_eta_prev_ts", None)
                prev_step = getattr(self, "_eta_prev_step", -1)
                if prev_ts is not None and self.completed_steps > prev_step:
                    win = getattr(self, "_eta_sec_per_step_win", None)
                    if win is None:
                        win = deque(maxlen=100)
                        self._eta_sec_per_step_win = win
                    win.append((now_ts - prev_ts) / float(self.completed_steps - prev_step))
                    sec_per_step = float(sum(win) / len(win))
                    remaining_steps = max(int(self.config.trainer.max_train_steps) - int(self.completed_steps), 0)
                    step_metrics["sec_per_step"] = sec_per_step
                    step_metrics["eta_hours"] = remaining_steps * sec_per_step / 3600.0
                self._eta_prev_ts = now_ts
                self._eta_prev_step = int(self.completed_steps)
                # force=do_light_eval: when an eval ran this step, always flush its metrics even if
                # completed_steps % logging_frequency != 0 (otherwise eval metrics are silently dropped).
                self._log_metrics(step_metrics, force=do_light_eval)

                if self.completed_steps % self.config.trainer.save_interval == 0 and self.completed_steps > 0:
                    torch.cuda.empty_cache()
                    self._save_checkpoint()
                    torch.cuda.empty_cache()

            if self.completed_steps >= self.config.trainer.max_train_steps:
                break

        self._finalize_training()

    def _train_step(self, batch_vla, batch_vlm=None):
        """Execute single training step."""
        profile_timing = self._should_profile_timing()
        raw_model = self.accelerator.unwrap_model(self.model)
        previous_profile_timing = bool(getattr(raw_model, "_profile_timing", False))
        previous_collect_metrics = bool(getattr(raw_model, "_collect_detailed_metrics", True))
        raw_model._profile_timing = profile_timing
        # Compute detailed fm/mask metrics on EVERY rank on a log step (not just the
        # main rank), so the cross-rank averaging in the metric-logging block actually pools them. NOTE:
        # _will_log_step_metrics() short-circuits to False on non-main ranks (it gates the rank0-only
        # *logging*), so using it here left ranks 1..N producing no detailed metrics -> the _XRANK_AVG
        # gather saw NaN from those ranks and reduced to rank0's single-micro value (a no-op). Use a
        # rank-SYMMETRIC predicate (same log cadence, no is_main_process gate; self.completed_steps is
        # identical across ranks under DDP) so all ranks emit the detailed keys and the gather is a real
        # allreduce mean. Detailed metrics are cheap (norms/means on already-computed tensors; pred_base
        # is already computed for the decomposition loss), so this adds negligible per-rank cost, only on
        # log steps. The gather itself is unchanged + deadlock-safe (fixed key list), so even if this
        # predicate ever diverged it could not hang.
        _detail_log_interval = int(getattr(self.config.trainer, "logging_frequency", 1) or 1)
        _detail_metrics_step = (self.completed_steps + 1) % _detail_log_interval == 0
        raw_model._collect_detailed_metrics = _detail_metrics_step or profile_timing
        profile_metrics: dict[str, float] = {}
        with self.accelerator.accumulate(self.model):
            # Accelerate handles mixed precision (bf16) automatically
            t0 = self._sync_profile_clock(profile_timing)
            try:
                output_dict = self.model.forward(batch_vla)
                action_loss = output_dict["action_loss"]
                total_loss = action_loss
            finally:
                raw_model._profile_timing = previous_profile_timing
                raw_model._collect_detailed_metrics = previous_collect_metrics
            t1 = self._sync_profile_clock(profile_timing)
            if profile_timing:
                profile_metrics["time/train/forward_sec"] = t1 - t0

            t2 = self._sync_profile_clock(profile_timing)
            self.accelerator.backward(total_loss)
            t3 = self._sync_profile_clock(profile_timing)
            if profile_timing:
                profile_metrics["time/train/backward_sec"] = t3 - t2

            grad_norm = None
            if self.config.trainer.gradient_clipping is not None and self.accelerator.sync_gradients:
                t_clip0 = self._sync_profile_clock(profile_timing)
                grad_norm = self.accelerator.clip_grad_norm_(self.model.parameters(), self.config.trainer.gradient_clipping)
                t_clip1 = self._sync_profile_clock(profile_timing)
                if profile_timing:
                    profile_metrics["time/train/grad_clip_sec"] = t_clip1 - t_clip0

            t4 = self._sync_profile_clock(profile_timing)
            self.optimizer.step()
            # Step the (non-accelerate-prepared) cosine scheduler ONCE per optimizer
            # step, not per micro-batch; else grad_accum=8 consumes the 50k-step cosine
            # 8x too fast (LR -> 0 by ~max_train_steps/8).
            if self.accelerator.sync_gradients:
                self.lr_scheduler.step()
            # zero_grad AFTER step (not before forward): accelerate gates zero_grad on
            # sync_gradients, so a body-first zero_grad wipes the accumulated micro-batch
            # grads on the sync step -> only the last micro-batch counts (GA was broken,
            # effective batch = per_device*world, not *grad_accum).
            self.optimizer.zero_grad()
            t5 = self._sync_profile_clock(profile_timing)
            if profile_timing:
                profile_metrics["time/train/optimizer_sec"] = t5 - t4
                profile_metrics["time/train/profile_enabled"] = 1.0

        # Per-dim MSE in physical space: REMOVED - this was causing ~50% step time overhead
        # The train_step now only returns loss/train, metrics are computed in eval_action_model
        # loss/train + fm/velocity_mse: accumulate the per-micro losses over the WHOLE grad_accum
        # window, then gather across ranks on the sync (optimizer) step, so the logged value is the
                # mean over the full effective batch (grad_accum x world x per_device) instead of one
        # rank's single LAST micro-batch (~= per_device, ~8x noisier / jittery). Logging-only: the
        # loss used for backward/optimization (action_loss above) is unchanged.
        _fm_micro = output_dict.get("fm/velocity_mse")
        if getattr(self, "_eff_sum", None) is None:
            self._eff_sum = {"loss/train": action_loss.detach().clone()}
            if torch.is_tensor(_fm_micro):
                self._eff_sum["fm/velocity_mse"] = _fm_micro.detach().clone()
            self._eff_cnt = 1
        else:
            self._eff_sum["loss/train"] = self._eff_sum["loss/train"] + action_loss.detach()
            if torch.is_tensor(_fm_micro) and "fm/velocity_mse" in self._eff_sum:
                self._eff_sum["fm/velocity_mse"] = self._eff_sum["fm/velocity_mse"] + _fm_micro.detach()
            self._eff_cnt += 1
        _eff_log = {}
        if self.accelerator.sync_gradients:
            _cnt = max(int(self._eff_cnt), 1)
            for _k, _v in self._eff_sum.items():
                _pr = (_v / _cnt).reshape(1)
                if self.accelerator.num_processes > 1:
                    _eff_log[_k] = self.accelerator.gather(_pr).float().mean().item()
                else:
                    _eff_log[_k] = float(_pr.item())
            self._eff_sum = None
            self._eff_cnt = 0
            loss_train_val = _eff_log.get("loss/train", float(action_loss.item()))
        else:
            loss_train_val = action_loss.item()
        step_metrics = {"loss/train": loss_train_val}
        step_metrics.update(profile_metrics)
        if grad_norm is not None:
            step_metrics["grad_norm"] = float(grad_norm.detach().float().cpu()) if torch.is_tensor(grad_norm) else float(grad_norm)

        for key, value in output_dict.items():
            if key.startswith(
                (
                    "fm/",
                    "mask/",
                    "slot_loss/",
                    "slot_velocity_mae/",
                    "slot_mask/",
                    "slot_loss_weight/",
                    "slot_weighted_loss_weight/",
                    "dataset/",
                    "dataset_loss/",
                    "dataset_velocity_mae/",
                    "dataset_velocity_mae_unweighted/",
                    "dataset_sample_ratio/",
                    "dataset_active_dim_mean/",
                    "dataset_loss_dim_equiv_mean/",
                    "dataset_weighted_loss_dim_equiv_mean/",
                    "qwen/",
                    "time/",
                )
            ):
                step_metrics[key] = float(value.detach().float().cpu()) if torch.is_tensor(value) else float(value)

        # Cross-rank average the OTHER full-batch loss curves (fm/mask aggregates) on
        # the sync step too -> allreduce-averaged (less jitter) instead of one rank's single micro-batch.
        # CRITICAL: the all_gather below MUST use a rank-IDENTICAL key set/shape, or the collective sizes
        # differ and NCCL DEADLOCKS (watchdog 30-min timeout -> SIGABRT on every rank). Use a FIXED
        # key list (identical length on every rank), NaN-fill locally-absent keys, take a nan-aware
        # mean across ranks, and write back only keys the local rank actually logged. Per-GROUP keys
        # (dataset_*/<id>, slot_*/<slot>) are NOT averaged because a micro-batch can lack most groups.
        # loss/train + fm/velocity_mse are skipped here
        # (guarded by `not in _eff_log`) -> set to the effective-batch means by _eff_log below.
        if self.accelerator.sync_gradients and self.accelerator.num_processes > 1:
            _XRANK_AVG_KEYS = (
                "fm/velocity_mse", "fm/velocity_mae", "fm/velocity_mae_unweighted", "fm/velocity_cosine",
                "fm/action_norm", "fm/noise_norm", "fm/pred_velocity_norm", "fm/target_velocity_norm",
                "fm/x_t_norm", "fm/t_mean", "fm/t_std", "fm/loss_numerator", "fm/loss_denominator",
                "mask/active_dim_mean", "mask/active_ratio", "mask/loss_dim_equiv_mean",
                "mask/loss_weight_mean", "mask/state_active_dim_mean", "mask/state_active_ratio",
                "mask/weighted_loss_dim_equiv_mean", "mask/weighted_loss_weight_mean",
                "mask/dim_weight_enabled",
            )
            _av_vals = torch.tensor(
                [float(step_metrics.get(_k, float("nan"))) for _k in _XRANK_AVG_KEYS],
                dtype=torch.float32, device=action_loss.device,
            )  # FIXED length on every rank -> all_gather shapes always match (cannot deadlock)
            _av_g = self.accelerator.gather(_av_vals.reshape(1, -1)).float()  # [world, K]
            _av_valid = ~torch.isnan(_av_g)
            _av_mean = torch.nan_to_num(_av_g, nan=0.0).sum(dim=0) / _av_valid.sum(dim=0).clamp(min=1)
            _av_present = _av_valid.any(dim=0)
            for _i, _k in enumerate(_XRANK_AVG_KEYS):
                if bool(_av_present[_i]) and _k in step_metrics and _k not in _eff_log:
                    step_metrics[_k] = float(_av_mean[_i].item())
        # The fm/ loop wrote rank0-local fm/velocity_mse; on sync steps override loss/train +
        # fm/velocity_mse with the effective-batch means computed earlier -> jsonl + SwanLab.
        if _eff_log:
            step_metrics.update(_eff_log)

        if batch_vla and isinstance(batch_vla, list) and "_data_active_value_count" in batch_vla[0]:
            nonfinite = np.asarray([int(ex.get("_data_nonfinite_count", 0)) for ex in batch_vla], dtype=np.float64)
            range_violations = np.asarray(
                [int(ex.get("_data_range_violation_count", 0)) for ex in batch_vla],
                dtype=np.float64,
            )
            active_values = np.asarray(
                [max(int(ex.get("_data_active_value_count", 0)), 1) for ex in batch_vla],
                dtype=np.float64,
            )
            short_episodes = np.asarray(
                [int(ex.get("_data_short_episode_count", 0)) for ex in batch_vla],
                dtype=np.float64,
            )
            selected_episodes = np.asarray(
                [max(int(ex.get("_data_selected_episode_count", 0)), 1) for ex in batch_vla],
                dtype=np.float64,
            )
            step_metrics["data/nonfinite_count"] = float(nonfinite.sum())
            step_metrics["data/nonfinite_rate"] = float(nonfinite.sum() / active_values.sum())
            step_metrics["data/range_violation_count"] = float(range_violations.sum())
            step_metrics["data/range_violation_rate"] = float(range_violations.sum() / active_values.sum())
            step_metrics["data/short_horizon_filtered"] = float(short_episodes.mean())
            step_metrics["data/short_horizon_filtered_ratio"] = float((short_episodes / selected_episodes).mean())
            if "_data_static_chunk_resample_attempts" in batch_vla[0]:
                static_attempts = np.asarray(
                    [int(ex.get("_data_static_chunk_resample_attempts", 0)) for ex in batch_vla],
                    dtype=np.float64,
                )
                effective_motion = np.asarray(
                    [bool(ex.get("_data_effective_motion_chunk", True)) for ex in batch_vla],
                    dtype=bool,
                )
                step_metrics["data/static_chunk_resample_attempt_mean"] = float(static_attempts.mean())
                step_metrics["data/static_chunk_resample_rate"] = float((static_attempts > 0).mean())
                step_metrics["data/effective_motion_chunk_rate"] = float(effective_motion.mean())
            # Per-item dataloader timing is only logged during the profile window (profile_timing_*),
            # not every step — otherwise the time/data_item/* + time/data_loader/* keys flood metrics
            # and force a per-step log via should_log_profile. With profile_timing_first_n/interval=0
            # this is off (no overhead, no log spam).
            if self._should_profile_timing() and "_data_profile_total_sec" in batch_vla[0]:
                item_total_values = np.asarray(
                    [float(ex.get("_data_profile_total_sec", 0.0)) for ex in batch_vla],
                    dtype=np.float64,
                )
                for name in (
                    "total",
                    "resolve",
                    "convert",
                    "video_decode",
                    "image_preprocess",
                    "instruction",
                ):
                    values = np.asarray(
                        [float(ex.get(f"_data_profile_{name}_sec", 0.0)) for ex in batch_vla],
                        dtype=np.float64,
                    )
                    step_metrics[f"time/data_item/{name}_mean_sec"] = float(values.mean())
                    step_metrics[f"time/data_item/{name}_p95_sec"] = float(np.percentile(values, 95))
                    step_metrics[f"time/data_item/{name}_max_sec"] = float(values.max())
                if item_total_values.size:
                    slow_idx = int(item_total_values.argmax())
                    slow_ex = batch_vla[slow_idx]
                    step_metrics["time/data_loader/item_total_max_sec"] = float(item_total_values[slow_idx])
                    step_metrics["time/data_loader/slow_item_frame"] = float(slow_ex.get("_frame_index", -1))

        if batch_vla and isinstance(batch_vla, list) and "_dataset_id" in batch_vla[0]:
            dataset_ids = [str(ex.get("_dataset_id", "unknown")) for ex in batch_vla]
            total = max(len(dataset_ids), 1)
            for dataset_id in sorted(set(dataset_ids)):
                safe_id = dataset_id.replace("/", "_")
                step_metrics[f"batch_dataset_ratio/{safe_id}"] = float(dataset_ids.count(dataset_id) / total)
            step_metrics["batch_dataset_unique_count/value"] = float(len(set(dataset_ids)))

        if batch_vla and isinstance(batch_vla, list) and "_sample_weight" in batch_vla[0]:
            weights = np.asarray([float(ex.get("_sample_weight", 1.0)) for ex in batch_vla], dtype=np.float32)
            step_metrics["sampler/batch_weight_mean"] = float(weights.mean())
            step_metrics["sampler/batch_weight_max"] = float(weights.max())
            step_metrics["sampler/batch_weight_p95"] = float(np.percentile(weights, 95))
            step_metrics["sample_weight/batch_mean"] = step_metrics["sampler/batch_weight_mean"]
            step_metrics["sample_weight/batch_max"] = step_metrics["sampler/batch_weight_max"]
            step_metrics["sample_weight/batch_p95"] = step_metrics["sampler/batch_weight_p95"]

        # Preserve any auxiliary loss terms in local jsonl only; SwanLab scalar logging is filtered.
        for aux_key in ["loss_smooth", "loss_traj_recon", "loss_traj_acc", "loss_rollout", "loss_preclose_lift", "loss_state_grip_close"]:
            if aux_key in output_dict and output_dict[aux_key].item() > 0:
                step_metrics[f"{aux_key}/train"] = output_dict[aux_key].item()

        return step_metrics

    def _finalize_training(self):
        """Training end processing."""
        if self.accelerator.is_main_process:
            save_format = getattr(self.config.trainer, "save_format", "pt")
            final_checkpoint = os.path.join(self.config.output_dir, "final_model")
            os.makedirs(final_checkpoint, exist_ok=True)
            state_dict = self.accelerator.get_state_dict(self.model)
            if save_format == "safetensors":
                _atomic_safetensors_save(state_dict, os.path.join(final_checkpoint, "model.safetensors"))
            elif save_format == "pt":
                _atomic_torch_save(state_dict, os.path.join(final_checkpoint, "pytorch_model.pt"))
            else:
                raise ValueError(f"Unsupported save_format `{save_format}`. Expected `pt` or `safetensors`.")
            logger.info(f"Training complete. Final model saved at {final_checkpoint}")

            # Save EMA params if available
            if self._ema_params:
                ema_path = os.path.join(final_checkpoint, "ema_params.pt")
                _atomic_torch_save(self._ema_params, ema_path)
                logger.info(f"Final EMA params saved at {ema_path}")

            # Save norm_stats sidecar (final_model/pytorch_model_norm_stats.json)
            self._save_norm_stats(os.path.join(final_checkpoint, "pytorch_model"))

        if self.accelerator.is_main_process and HAS_SWANLAB and self._swanlab_enabled:
            # swanlab.finish() raises if init() was never called (e.g. when
            # `--trackers '[jsonl]'` runs without swanlab). Guard with a
            # try/except so the teardown doesn't mask a successful run.
            try:
                swanlab.finish()
            except RuntimeError as e:
                logger.info(f"swanlab.finish() skipped: {e}")

        self.accelerator.wait_for_everyone()


def main(cfg) -> None:
    print("VLA Training :: Warming Up")

    cfg = wrap_config(cfg)
    print("Configuration wrapped for access tracking")

    # attn_implementation: 优先使用 config 配置，仅当未配置时使用环境变量
    if "attn_implementation" not in cfg.framework.qwenvl:
        cfg.framework.qwenvl.attn_implementation = os.getenv("QWEN_ATTN_IMPLEMENTATION", "flash_attention_2")

    accelerator = build_accelerator(cfg)
    output_dir = setup_directories(cfg=cfg)
    save_startup_config(cfg, output_dir)
    vla = build_framework(cfg)
    freeze_modules = (
        cfg.trainer.freeze_modules
        if (cfg and hasattr(cfg.trainer, "freeze_modules"))
        else None
    )
    # Freeze/unfreeze before building optimizer groups so the optimizer only sees the final trainable set.
    vla = TrainerUtils.freeze_backbones(vla, freeze_modules=freeze_modules)
    vla_train_dataloader = prepare_data(cfg=cfg, accelerator=accelerator, output_dir=output_dir)
    optimizer, lr_scheduler = setup_optimizer_and_scheduler(model=vla, cfg=cfg)

    trainer = VLATrainer(
        cfg=cfg,
        model=vla,
        vla_train_dataloader=vla_train_dataloader,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        accelerator=accelerator,
    )

    trainer.prepare_training()
    trainer.train()

    print("... and that's all, folks!")
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_yaml",
        type=str,
        default="configs/training/demo_full_finetune.yaml",
        help="Path to YAML config",
    )
    args, clipargs = parser.parse_known_args()

    cfg = OmegaConf.load(args.config_yaml)
    dotlist = normalize_dotlist_args(clipargs)
    cli_cfg = OmegaConf.from_dotlist(dotlist)
    cfg = OmegaConf.merge(cfg, cli_cfg)
    cfg.config_yaml = args.config_yaml

    if OmegaConf.select(cfg, "is_debug", default=False) and dist.is_initialized() and dist.get_rank() == 0:
        import debugpy

        debugpy.listen(("0.0.0.0", 10092))
        print("🔍 Rank 0 waiting for debugger attach on port 10092...")
        debugpy.wait_for_client()

    main(cfg)
