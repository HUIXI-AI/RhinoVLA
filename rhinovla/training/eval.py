"""Validation for the native72 training path."""

from __future__ import annotations

import numpy as np
import torch
import torch.distributed as dist
from accelerate.logging import get_logger

from rhinovla.dataloader import build_dataloader
from rhinovla.dataloader.lerobot_native72 import RHINO72_DIM


logger = get_logger(__name__)


def _warn(message: str) -> None:
    try:
        logger.warning(message)
    except RuntimeError:
        print(f"WARNING: {message}")


class _EvalMixin:
    """Validation methods for VLATrainer."""

    def _init_val_dataloader(self):
        vla_cfg = self.config.datasets.vla_data
        dataset_py = str(vla_cfg.get("dataset_py", "lerobot_native72"))
        self._val_dataloader = None
        self._val_iterator = None

        if dataset_py != "lerobot_native72":
            raise ValueError(f"Native72 trainer only supports lerobot_native72 eval, got {dataset_py}")

        try:
            val_loader = build_dataloader(self.config, dataset_py=dataset_py, split="val")
        except Exception as exc:  # noqa: BLE001
            _warn(f"Native72 val dataloader init failed: {exc}; eval will be skipped.")
            return
        if val_loader is None:
            _warn("Native72 val split is not configured; eval will be skipped.")
            return

        self._val_dataloader = val_loader
        logger.info(f"✅ Native72 val dataloader initialized: {len(val_loader.dataset)} samples")

    def eval_action_model(self, step_metrics: dict | None = None, use_ema: bool = False) -> dict:
        if step_metrics is None:
            step_metrics = {}
        if self._val_dataloader is None:
            _warn("Skipping eval because no val dataloader is available.")
            return step_metrics

        vla_cfg = self.config.datasets.vla_data
        n_eval_batches = int(vla_cfg.get("n_eval_batches", 4))
        action_mean = np.asarray(vla_cfg.get("action_mean", None), dtype=np.float32)
        action_std = np.asarray(vla_cfg.get("action_std", None), dtype=np.float32)
        if action_mean.shape[-1] != RHINO72_DIM or action_std.shape[-1] != RHINO72_DIM:
            raise ValueError(
                f"Native72 eval requires 72D action stats, got mean={action_mean.shape} std={action_std.shape}"
            )

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
                    pred_np = pred_np * action_std[None, None, :] + action_mean[None, None, :]

                    gt = np.asarray([ex.get("action_raw", ex["action"]) for ex in examples], dtype=np.float32)
                    mask = np.asarray([ex["action_mask"] for ex in examples], dtype=np.float32)
                    if gt.shape[-1] != RHINO72_DIM or mask.shape[-1] != RHINO72_DIM:
                        raise ValueError(f"native72 eval expected 72D gt/mask, got gt={gt.shape} mask={mask.shape}")

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
