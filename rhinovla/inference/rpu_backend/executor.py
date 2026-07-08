"""RhinoVLA RPU backend executor using the public rpu_backend API."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

from rhinovla.inference.rpu_backend.norm import NormStats


DEFAULT_RPU_VIEW_ROLES = ("top_head", "hand_left", "hand_right")
DEFAULT_RPU_VIEW_MODALITIES = ("rgb", "rgb", "rgb")


@dataclass
class InferenceResult:
    actions_raw: np.ndarray
    actions_norm: np.ndarray
    action_hz: float
    latency_ms: float
    extra: dict[str, Any] = field(default_factory=dict)

    def slots(self, indices: Sequence[int]) -> np.ndarray:
        return self.actions_raw[:, list(indices)]


class RhinoVLARPUBackendExecutor:
    """Single-frame RhinoVLA inference wrapper for `rpu_backend.api.RhinoVLAPolicy`.

    The current public RPU API consumes a prepared artifact. Training config and
    checkpoint paths are passed into the runtime environment so deployments can
    override artifact metadata without importing private rpu_backend modules.
    """

    def __init__(
        self,
        *,
        prepare_artifact: str | Path,
        norm_stats_path: str | Path,
        instruction: str,
        train_config: str | Path | None = None,
        checkpoint: str | Path | None = None,
        num_steps: int = 5,
        action_hz: float = 30.0,
        active_slots: Optional[Sequence[int]] = None,
        mapping_path: str | Path | None = None,
        mapping_dataset_id: str | None = None,
        view_roles: Optional[Sequence[str]] = None,
        view_modalities: Optional[Sequence[str]] = None,
        rhino_repo: str | Path | None = None,
        runtime_env: Optional[dict[str, str]] = None,
        artifact_strict: bool = False,
        noise_seed: int = 0,
    ) -> None:
        from rpu_backend.api import RhinoVLAPolicy

        self._prepare_artifact = str(prepare_artifact)
        self._train_config = str(train_config) if train_config is not None else None
        self._checkpoint = str(checkpoint) if checkpoint is not None else None
        self._instruction = str(instruction)
        self._action_hz = float(action_hz)
        self._noise_seed = int(noise_seed)
        self._norm = NormStats.load(
            norm_stats_path,
            mapping_path=mapping_path,
            mapping_dataset_id=mapping_dataset_id,
        )
        self._action_dim = self._norm.action_dim
        self._state_dim = self._norm.state_dim
        self._active_slots = self._default_active_slots(active_slots)
        self._active_state_slots = self._default_active_state_slots()
        self._view_roles = [str(x) for x in (view_roles or DEFAULT_RPU_VIEW_ROLES)]
        self._view_modalities = [str(x) for x in (view_modalities or DEFAULT_RPU_VIEW_MODALITIES)]

        self._state_mean = np.asarray(self._norm.state_mean, dtype=np.float32).reshape(-1)
        self._state_std = np.asarray(self._norm.state_std, dtype=np.float32).reshape(-1)
        self._state_std_safe = np.where(self._state_std == 0, 1.0, self._state_std)
        self._state_zero = self._state_std == 0
        self._action_mean = np.asarray(self._norm.action_mean, dtype=np.float32).reshape(-1)
        self._action_std = np.asarray(self._norm.action_std, dtype=np.float32).reshape(-1)

        self._runtime_env = _checkpoint_runtime_env(
            runtime_env,
            train_config=self._train_config,
            checkpoint=self._checkpoint,
        )
        self._policy = (
            RhinoVLAPolicy.from_prepare_artifact(
                self._prepare_artifact,
                strict=bool(artifact_strict),
                steps=int(num_steps),
                action_hz=self._action_hz,
                rhino_repo=str(rhino_repo) if rhino_repo is not None else None,
                instruction=self._instruction,
                active_slots=self._active_slots,
                view_roles=self._view_roles,
                view_modalities=self._view_modalities,
                runtime_env=dict(self._runtime_env),
            )
            .to("rpu")
        )

    @property
    def active_slots(self) -> list[int]:
        return list(self._active_slots)

    @property
    def active_state_slots(self) -> list[int]:
        return list(self._active_state_slots)

    @property
    def action_dim(self) -> int:
        return int(self._action_dim)

    @property
    def instruction(self) -> str:
        return self._instruction

    @property
    def prompt_locked(self) -> bool:
        return True

    @property
    def runtime_env(self) -> dict[str, str]:
        return dict(self._runtime_env)

    def infer(
        self,
        images: Sequence[Any],
        raw_state: Any,
        instruction: str | None = None,
    ) -> InferenceResult:
        if instruction is not None and instruction != self._instruction:
            raise ValueError("RPU backend prompt is locked; rebuild executor to change it.")

        out = self._policy.infer(
            images=self._prepare_images(images),
            state=self._normalize_state(raw_state),
            instruction=self._instruction,
            active_slots=self._active_slots,
            noise_seed=self._noise_seed,
        )
        actions_norm = self._to_numpy(out.actions_norm)
        actions_raw = self._denormalize_actions(actions_norm)
        return InferenceResult(
            actions_raw=actions_raw,
            actions_norm=actions_norm,
            action_hz=float(out.action_hz),
            latency_ms=float(out.latency_ms),
            extra={
                "active_slots": list(self._active_slots),
                "active_state_slots": list(self._active_state_slots),
                "prepare_artifact": self._prepare_artifact,
                "train_config": self._train_config,
                "checkpoint": self._checkpoint,
                "timing_ms": dict(getattr(out, "phase_ms", {}) or {}),
                "rpu_backend": dict(getattr(out, "extra", {}) or {}),
            },
        )

    def close(self) -> None:
        close = getattr(self._policy, "close", None)
        if callable(close):
            close()

    def _default_active_slots(self, active_slots: Optional[Sequence[int]]) -> list[int]:
        if active_slots is not None:
            return [int(x) for x in active_slots]
        if self._norm.active_action_slots is not None:
            return [int(x) for x in self._norm.active_action_slots]
        if self._action_dim >= 72:
            return list(range(16))
        return list(range(self._action_dim))

    def _default_active_state_slots(self) -> list[int]:
        if self._norm.active_state_slots is not None:
            return [int(x) for x in self._norm.active_state_slots]
        return list(self._active_slots)

    def _normalize_state(self, raw_state: Any) -> np.ndarray:
        raw = np.asarray(raw_state, dtype=np.float32).reshape(-1)
        if raw.shape[0] != self._state_dim:
            if raw.shape[0] == len(self._active_state_slots):
                expanded = self._state_mean.copy()
                for idx, slot in enumerate(self._active_state_slots):
                    if not 0 <= int(slot) < self._state_dim:
                        raise ValueError(f"active state slot {slot} outside state dim {self._state_dim}")
                    expanded[int(slot)] = raw[idx]
                raw = expanded
            else:
                raise ValueError(
                    f"state dim {raw.shape[0]} != expected {self._state_dim} "
                    f"or active state-slot dim {len(self._active_state_slots)}"
                )
        out = (raw - self._state_mean) / self._state_std_safe
        out[self._state_zero] = 0.0
        return out.astype(np.float32)

    def _denormalize_actions(self, actions_norm: Any) -> np.ndarray:
        norm = np.asarray(actions_norm, dtype=np.float32)
        if norm.ndim != 2:
            raise RuntimeError(f"RPU policy must return (H, D) actions, got {norm.shape}")
        if norm.shape[1] != self._action_dim:
            raise RuntimeError(f"RPU action dim {norm.shape[1]} != norm_stats action dim {self._action_dim}")
        return (norm * self._action_std[None, :] + self._action_mean[None, :]).astype(np.float32)

    def _prepare_images(self, images: Sequence[Any]) -> list[Any]:
        from PIL import Image

        out: list[Any] = []
        for img in images:
            if isinstance(img, Image.Image):
                out.append(img.convert("RGB") if img.mode != "RGB" else img)
            else:
                out.append(Image.fromarray(np.asarray(img).astype(np.uint8)).convert("RGB"))
        return out

    @staticmethod
    def _to_numpy(value: Any) -> np.ndarray:
        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "cpu"):
            value = value.cpu()
        if hasattr(value, "numpy"):
            value = value.numpy()
        return np.asarray(value, dtype=np.float32)


def _checkpoint_runtime_env(
    runtime_env: Optional[dict[str, str]],
    *,
    train_config: str | None,
    checkpoint: str | None,
) -> dict[str, str]:
    env = {str(k): str(v) for k, v in dict(runtime_env or {}).items()}
    if train_config:
        env.setdefault("RHINOVLA_CONFIG", str(train_config))
    if checkpoint:
        env.setdefault("RHINOVLA_CKPT", str(checkpoint))
    return env

