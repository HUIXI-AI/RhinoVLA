"""Validation for the native continuous-action training paths."""

from __future__ import annotations

import numpy as np
import torch
import torch.distributed as dist
from accelerate.logging import get_logger

from rhinovla.dataloader import build_dataloader
from rhinovla.dataloader.lerobot_native96 import RHINO96_DIM


logger = get_logger(__name__)


def _warn(message: str) -> None:
    try:
        logger.warning(message)
    except RuntimeError:
        print(f"WARNING: {message}")


def _expected_dim(vla_cfg) -> int:
    dataset_py = str(vla_cfg.get("dataset_py", "lerobot_native96"))
    if dataset_py != "lerobot_native96":
        raise ValueError(f"V3 trainer requires dataset_py=lerobot_native96, got {dataset_py!r}")
    return RHINO96_DIM


def _validated_action_stats(profile, *, expected_dim: int, context: str) -> tuple[np.ndarray, np.ndarray]:
    """Return one validated action normalization profile."""
    action_mean = np.asarray(profile.get("action_mean", None), dtype=np.float32)
    action_std = np.asarray(profile.get("action_std", None), dtype=np.float32)
    if action_mean.shape != (expected_dim,) or action_std.shape != (expected_dim,):
        raise ValueError(
            f"{context} requires {expected_dim}D action stats, got "
            f"mean={action_mean.shape} std={action_std.shape}"
        )
    if not np.isfinite(action_mean).all() or not np.isfinite(action_std).all():
        raise ValueError(f"{context} action stats must be finite")
    if np.any(action_std <= 0):
        raise ValueError(f"{context} action_std must be strictly positive")
    return action_mean, action_std


def _resolve_batch_action_norm_stats(vla_cfg, examples) -> tuple[np.ndarray, np.ndarray]:
    """Broadcast the one global action normalization to a batch."""
    expected_dim = _expected_dim(vla_cfg)
    action_mean, action_std = _validated_action_stats(
        vla_cfg, expected_dim=expected_dim, context=f"{expected_dim}D eval"
    )
    batch_size = len(examples)
    return (
        np.broadcast_to(action_mean, (batch_size, expected_dim)).copy(),
        np.broadcast_to(action_std, (batch_size, expected_dim)).copy(),
    )


class _EvalMixin:
    """Validation methods for VLATrainer."""

    def _init_val_dataloader(self):
        vla_cfg = self.config.datasets.vla_data
        dataset_py = str(vla_cfg.get("dataset_py", "lerobot_native96"))
        self._val_dataloader = None
        self._val_iterator = None

        if dataset_py != "lerobot_native96":
            raise ValueError(f"V3 trainer requires dataset_py=lerobot_native96, got {dataset_py}")

        try:
            val_loader = build_dataloader(self.config, dataset_py=dataset_py, split="val")
        except Exception as exc:  # noqa: BLE001
            if vla_cfg.get("fixed_validation_loss", True):
                raise
            _warn(f"{dataset_py} val dataloader init failed: {exc}; eval will be skipped.")
            return
        if val_loader is None:
            _warn(f"{dataset_py} val split is not configured; eval will be skipped.")
            return

        self._val_dataloader = val_loader
        if vla_cfg.get("fixed_validation_loss", True):
            from torch.utils.data import DataLoader, Subset

            rank, world = self.accelerator.process_index, self.accelerator.num_processes
            subset = Subset(val_loader.dataset, list(range(rank, len(val_loader.dataset), world)))
            self._val_dataloader = DataLoader(
                subset,
                batch_size=int(vla_cfg.get("eval_batch_size", 8)),
                shuffle=False,
                num_workers=2,
                collate_fn=val_loader.collate_fn,
            )
        logger.info(f"✅ {dataset_py} val dataloader initialized: {len(val_loader.dataset)} samples")

    def eval_action_model(self, step_metrics: dict | None = None, use_ema: bool = False) -> dict:
        if step_metrics is None:
            step_metrics = {}
        if self._val_dataloader is None:
            _warn("Skipping eval because no val dataloader is available.")
            return step_metrics

        vla_cfg = self.config.datasets.vla_data
        if vla_cfg.get("fixed_validation_loss", True):
            return self._fixed_validation_loss(step_metrics, use_ema=use_ema)
        dataset_py = str(vla_cfg.get("dataset_py", "lerobot_native96"))
        n_eval_batches = int(vla_cfg.get("n_eval_batches", 4))

        metric_prefix = "ema_" if use_ema else ""
        raw_model = self.accelerator.unwrap_model(self.model)
        raw_model.eval()

        val_tf_losses: list[float] = []
        pred_batches: list[np.ndarray] = []
        gt_batches: list[np.ndarray] = []
        mask_batches: list[np.ndarray] = []

        if self._val_iterator is None:
            self._val_iterator = iter(self._val_dataloader)

        try:
            for _ in range(n_eval_batches):
                try:
                    examples = next(self._val_iterator)
                except StopIteration:
                    self._val_iterator = iter(self._val_dataloader)
                    examples = next(self._val_iterator)

                with torch.inference_mode():
                    tf_out = raw_model.forward(examples)
                    val_tf_losses.append(float(tf_out.get("loss_fm", tf_out["action_loss"]).detach().float().cpu()))
                    pred_actions = raw_model.predict_action(examples)

                if self.accelerator.is_main_process:
                    pred_np = pred_actions.detach().float().cpu().numpy()
                    batch_action_mean, batch_action_std = _resolve_batch_action_norm_stats(
                        vla_cfg,
                        examples,
                    )
                    if pred_np.shape[0] != batch_action_mean.shape[0]:
                        raise ValueError(
                            f"{dataset_py} eval prediction batch does not match examples: "
                            f"pred={pred_np.shape[0]} examples={batch_action_mean.shape[0]}"
                        )
                    pred_np = (
                        pred_np * batch_action_std[:, None, :]
                        + batch_action_mean[:, None, :]
                    )

                    gt = np.asarray(
                        [ex["action_raw"] if "action_raw" in ex else ex["action"] for ex in examples],
                        dtype=np.float32,
                    )
                    mask = np.asarray([ex["action_mask"] for ex in examples], dtype=np.float32)
                    expected_dim = _expected_dim(vla_cfg)
                    if gt.shape[-1] != expected_dim or mask.shape[-1] != expected_dim:
                        raise ValueError(
                            f"{dataset_py} eval expected {expected_dim}D gt/mask, "
                            f"got gt={gt.shape} mask={mask.shape}"
                        )

                    pred_batches.append(pred_np)
                    gt_batches.append(gt)
                    mask_batches.append(mask)
                del examples
        finally:
            if not use_ema:
                raw_model.train()

        if dist.is_initialized():
            dist.barrier()

        if not (self.accelerator.is_main_process and pred_batches):
            return step_metrics

        all_pred = np.concatenate(pred_batches, axis=0)
        all_gt = np.concatenate(gt_batches, axis=0)
        all_mask = np.concatenate(mask_batches, axis=0)
        horizon = min(all_pred.shape[1], all_gt.shape[1], all_mask.shape[1])
        all_pred = all_pred[:, :horizon, :]
        all_gt = all_gt[:, :horizon, :]
        all_mask = all_mask[:, :horizon, :]

        diff = all_pred - all_gt
        denom = max(float(all_mask.sum()), 1.0)
        masked_mse = float(((diff**2) * all_mask).sum() / denom)
        masked_mae = float((np.abs(diff) * all_mask).sum() / denom)

        step_metrics[f"{metric_prefix}loss/val"] = float(np.mean(val_tf_losses)) if val_tf_losses else 0.0
        step_metrics[f"{metric_prefix}rhino/val_masked_mse"] = masked_mse
        step_metrics[f"{metric_prefix}rhino/val_masked_mae"] = masked_mae
        # Keep the legacy key names for apples-to-apples JSONL comparisons.
        step_metrics[f"{metric_prefix}rhynix/val_masked_mse"] = masked_mse
        step_metrics[f"{metric_prefix}rhynix/val_masked_mae"] = masked_mae
        step_metrics[f"{metric_prefix}rhino/val_active_dim_mean"] = float(
            (all_mask[:, 0, :] > 0.5).sum(axis=1).mean()
        )
        step_metrics[f"{metric_prefix}rhino/val_active_ratio"] = float(all_mask.mean())
        return step_metrics

    def _fixed_validation_loss(self, step_metrics, use_ema=False):
        """Evaluate the full held-out split with fixed noise and RTC draws."""
        import random

        raw_model = self.accelerator.unwrap_model(self.model)
        was_training = raw_model.training
        device = self.accelerator.device
        totals = torch.zeros(3, dtype=torch.float64, device=device)
        py_rng, np_rng = random.getstate(), np.random.get_state()
        try:
            raw_model.eval()
            devices = [device.index] if device.type == "cuda" else []
            with torch.random.fork_rng(devices=devices), torch.inference_mode():
                seed = int(self.config.seed) + self.accelerator.process_index
                torch.manual_seed(seed)
                random.seed(seed)
                np.random.seed(seed)
                for examples in self._val_dataloader:
                    with self.accelerator.autocast():
                        out = raw_model(examples)
                    totals[0] += out["fm/loss_numerator"].double()
                    totals[1] += out["fm/loss_denominator"].double()
                    totals[2] += len(examples)
            if dist.is_initialized():
                dist.all_reduce(totals)
            prefix = "ema_" if use_ema else ""
            step_metrics[f"{prefix}loss/val"] = (totals[0] / totals[1].clamp_min(1)).item()
            step_metrics[f"{prefix}val/samples"] = totals[2].item()
            return step_metrics
        finally:
            raw_model.train(was_training)
            random.setstate(py_rng)
            np.random.set_state(np_rng)
