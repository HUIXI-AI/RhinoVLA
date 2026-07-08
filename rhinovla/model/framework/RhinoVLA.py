# Copyright 2026 The RhinoVLA authors.
#
# RhinoVLA has two top-level model modules:
# - qwen: frozen or partially trainable Qwen3-VL prefix encoder
# - action_expert: Rhino suffix denoising expert, including IO and mask conditioning

from __future__ import annotations

import time
from typing import List

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F

from rhinovla.model.modules import build_qwen
from rhinovla.model.modules.action_expert import (
    RhinoActionExpert,
    RhinoActionExpertConfig,
    estimate_rhino_action_params,
)
from rhinovla.training.trainer_utils.trainer_tools import resize_images


def _rhino_metric_label(value: object, max_len: int = 60) -> str:
    """Sanitize a dataset/root string into a SwanLab-safe metric label."""
    text = str(value).rstrip("/").split("/")[-1]
    safe = "".join(ch if (ch.isalnum() or ch in {"_", "-", "."}) else "_" for ch in text)
    return safe[:max_len] or "unknown"


class RhinoVLA(nn.Module):
    """Qwen3-VL prefix + Rhino 72D masked suffix denoising action expert.

    Training scaffold compatibility:
      - forward(examples) returns {"action_loss": loss, ...}
      - predict_action(examples) returns Tensor[B, H, action_dim]
    """

    def __init__(self, config=None, **kwargs):
        super().__init__()
        self.config = config
        self.qwen = build_qwen(config=self.config)

        qwen_text_config = self.qwen.model.config.text_config
        action_cfg = config.framework.action_expert
        self.action_horizon = int(getattr(action_cfg, "action_horizon", getattr(action_cfg, "future_action_window_size", 29) + 1))
        self.action_dim = int(getattr(action_cfg, "action_dim", 72))
        self.state_dim = int(getattr(action_cfg, "state_dim", 72))
        if self.action_dim != 72 or self.state_dim != 72:
            raise ValueError(
                f"RhinoVLA requires action_dim=72 and state_dim=72, "
                f"got action_dim={self.action_dim} state_dim={self.state_dim}"
            )

        use_all_qwen_text_layers = bool(getattr(action_cfg, "use_all_qwen_text_layers", False))
        expert_depth = (
            int(qwen_text_config.num_hidden_layers)
            if use_all_qwen_text_layers
            else int(getattr(action_cfg, "depth", 10))
        )

        self.expert_config = RhinoActionExpertConfig.from_qwen_text_config(
            qwen_text_config,
            action_dim=self.action_dim,
            state_dim=self.state_dim,
            action_horizon=self.action_horizon,
            width=int(getattr(action_cfg, "width", 1024)),
            depth=expert_depth,
            mlp_dim=int(getattr(action_cfg, "mlp_dim", 3072)),
            use_state_token=bool(getattr(action_cfg, "use_state_token", True)),
            use_adarms=bool(getattr(action_cfg, "use_adarms", True)),
            use_mask_condition=bool(getattr(action_cfg, "use_mask_condition", False)),
        )
        self.action_expert = RhinoActionExpert(
            self.expert_config,
            qwen_rotary_emb=self.qwen.model.model.language_model.rotary_emb,
        )
        default_prefix_layer_offset = 0 if use_all_qwen_text_layers else qwen_text_config.num_hidden_layers - self.expert_config.depth
        self.prefix_layer_offset = int(getattr(action_cfg, "prefix_layer_offset", default_prefix_layer_offset))
        if self.prefix_layer_offset < 0:
            raise ValueError(
                f"action expert depth {self.expert_config.depth} exceeds Qwen text layers {qwen_text_config.num_hidden_layers}"
            )
        if getattr(action_cfg, "init_from_qwen_last_n_layers", False):
            self.qwen_action_init_report = self._init_action_expert_from_qwen_last_layers(action_cfg)

        qwenvl_cfg = getattr(config.framework, "qwenvl", None)
        if bool(getattr(qwenvl_cfg, "freeze_qwen", True)):
            for param in self.qwen.parameters():
                param.requires_grad = False
        unfreeze_last_n = int(getattr(qwenvl_cfg, "unfreeze_last_n_layers", 0) or 0)
        if unfreeze_last_n > 0:
            language_model = self._get_qwen_language_model()
            if language_model is None or not hasattr(language_model, "layers"):
                raise RuntimeError("framework.qwenvl.unfreeze_last_n_layers is set, but Qwen language_model.layers was not found")
            n_layers = len(language_model.layers)
            start = max(0, n_layers - unfreeze_last_n)
            for layer in language_model.layers[start:]:
                for param in layer.parameters():
                    param.requires_grad = True
            if hasattr(language_model, "norm"):
                for param in language_model.norm.parameters():
                    param.requires_grad = True
            print(
                f"[Rhino] unfroze Qwen text layers {start}-{n_layers - 1} "
                f"and lm.norm via unfreeze_last_n_layers={unfreeze_last_n}",
                flush=True,
            )
        if bool(getattr(qwenvl_cfg, "freeze_vision", False)):
            visual = self._get_qwen_visual_encoder()
            if visual is None:
                print("[Rhino] freeze_vision=true but visual encoder was not found", flush=True)
            else:
                for param in visual.parameters():
                    param.requires_grad = False
                print("[Rhino] froze Qwen visual encoder", flush=True)

        self.beta_alpha = float(getattr(action_cfg, "noise_beta_alpha", 1.5))
        self.beta_beta = float(getattr(action_cfg, "noise_beta_beta", 1.0))
        self.num_inference_timesteps = int(getattr(action_cfg, "num_inference_timesteps", 10))
        # Flow time convention. Default "openpi" (x_t=t*noise+(1-t)*action, t:1->0,
        # target=noise-action) preserves the deployed D recipe. "rhinovla" mirrors it
        # (x_t=(1-t)*noise+t*action, t:0->1, target=action-noise) so a Rhino run shares
        # RhinoVLA's t-meaning; pair with Beta(1.0,1.5) to oversample high noise (t->0).
        # The flip is applied consistently in forward()/predict_action() (and thus TF-eval
        # + attention-GIF, which both call predict_action) so train/infer stay identical.
        self.flow_time_convention = str(getattr(action_cfg, "flow_time_convention", "openpi")).lower()
        if self.flow_time_convention not in ("openpi", "rhinovla"):
            raise ValueError(
                f"flow_time_convention must be 'openpi' or 'rhinovla', got {self.flow_time_convention!r}"
            )

        # Grasp weighting: per-sample + temporal + phase-dim weights
        gw = getattr(action_cfg, "grasp_weighting", None)
        self.grasp_weighting_enabled = bool(getattr(gw, "enabled", False)) if gw else False
        if self.grasp_weighting_enabled:
            self.use_sample_weight_in_loss = bool(getattr(gw, "use_sample_weight_in_loss", True))
            tw_cfg = getattr(gw, "temporal_weight", None)
            tw = torch.ones(self.action_horizon, dtype=torch.float32)
            if tw_cfg:
                tw[0:3] = float(getattr(tw_cfg, "steps_0_2", 2.0))
                tw[3:10] = float(getattr(tw_cfg, "steps_3_9", 1.5))
                tw[10:] = float(getattr(tw_cfg, "steps_10_plus", 1.0))
            self.register_buffer("temporal_weight", tw)
            pdw = getattr(gw, "phase_dim_weight", None)
            self._phase_dim_cfg = {}
            if pdw:
                for phase_name in ("close", "pregrasp", "lift", "approach", "place"):
                    pcfg = getattr(pdw, phase_name, None)
                    if pcfg:
                        self._phase_dim_cfg[phase_name] = {
                            "gripper_scale": float(getattr(pcfg, "gripper_scale", 1.0)),
                            "arm_scale": float(getattr(pcfg, "arm_scale", 1.0)),
                        }
            self._gripper_dims = list(getattr(gw, "gripper_dims", [7, 15]))
            self._arm_dims = list(getattr(gw, "arm_dims", list(range(0, 7)) + list(range(8, 15))))

    @property
    def action_param_estimate(self) -> dict[str, int]:
        return estimate_rhino_action_params(self.expert_config)

    def _get_qwen_language_model(self):
        model = self.qwen.model
        if hasattr(model, "model") and hasattr(model.model, "language_model"):
            return model.model.language_model
        return getattr(model, "language_model", None)

    def _get_qwen_visual_encoder(self):
        model = self.qwen.model
        if hasattr(model, "model") and hasattr(model.model, "visual"):
            return model.model.visual
        if hasattr(model, "model") and hasattr(model.model, "vision_model"):
            return model.model.vision_model
        for attr in ("visual", "vision_model", "vision_tower"):
            visual = getattr(model, attr, None)
            if visual is not None:
                return visual
        return None

    @staticmethod
    def _nested_attr(module, path: str):
        value = module
        for name in path.split("."):
            value = getattr(value, name, None)
            if value is None:
                return None
        return value

    @torch.no_grad()
    def _copy_qwen_param(
        self,
        dst_module,
        dst_path: str,
        src_module,
        src_path: str,
        label: str,
        copied: list[str],
        skipped: list[str],
        *,
        optional: bool = False,
    ):
        dst = self._nested_attr(dst_module, dst_path)
        src = self._nested_attr(src_module, src_path)
        if dst is None or src is None:
            if optional:
                return
            skipped.append(f"{label}: missing dst={dst is None} src={src is None}")
            return
        if tuple(dst.shape) != tuple(src.shape):
            skipped.append(f"{label}: shape dst={tuple(dst.shape)} src={tuple(src.shape)}")
            return
        dst.copy_(src.to(device=dst.device, dtype=dst.dtype))
        copied.append(label)

    @torch.no_grad()
    def _init_adarms_as_qwen_passthrough(self, module, *, gate_bias: float, labels: list[str], label: str):
        cond = getattr(module, "cond", None)
        if cond is None:
            return
        cond.weight.zero_()
        cond.bias.zero_()
        hidden = cond.bias.numel() // 3
        cond.bias[2 * hidden :] = gate_bias
        labels.append(label)

    @torch.no_grad()
    def _init_action_expert_from_qwen_last_layers(self, action_cfg) -> dict:
        """Warm-start action expert layers from the matching Qwen text tail.

        This only copies tensors whose shapes match exactly. For AdaRMSNorm,
        the Qwen RMSNorm weights are copied and the adaptive condition layers
        are initialized to preserve the copied Qwen residual path at step 0.
        """
        language_model = self._get_qwen_language_model()
        if language_model is None or not hasattr(language_model, "layers"):
            raise RuntimeError("init_from_qwen_last_n_layers=true but Qwen language_model.layers was not found")

        raw_n = getattr(action_cfg, "init_from_qwen_last_n_layers", False)
        init_n = self.expert_config.depth if isinstance(raw_n, bool) else int(raw_n)
        qwen_layers = list(language_model.layers)
        if init_n <= 0:
            raise ValueError("init_from_qwen_last_n_layers must be true or a positive integer")
        if init_n > len(qwen_layers):
            raise ValueError(f"cannot copy {init_n} Qwen layers from a {len(qwen_layers)}-layer text stack")
        if init_n > len(self.action_expert.layers):
            raise ValueError(f"cannot copy {init_n} Qwen layers into a {len(self.action_expert.layers)}-layer action expert")

        source_start = len(qwen_layers) - init_n
        copied: list[str] = []
        skipped: list[str] = []
        adarms_init: list[str] = []
        layer_pairs = zip(self.action_expert.layers[:init_n], qwen_layers[source_start:])

        for dst_idx, (dst_layer, src_layer) in enumerate(layer_pairs):
            src_idx = source_start + dst_idx
            prefix = f"action_layer_{dst_idx}<-qwen_layer_{src_idx}"
            self._copy_qwen_param(dst_layer, "input_layernorm.norm.weight", src_layer, "input_layernorm.weight", f"{prefix}.input_layernorm", copied, skipped)
            self._copy_qwen_param(dst_layer, "post_attention_layernorm.norm.weight", src_layer, "post_attention_layernorm.weight", f"{prefix}.post_attention_layernorm", copied, skipped)
            for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
                self._copy_qwen_param(dst_layer, f"self_attn.{proj}.weight", src_layer, f"self_attn.{proj}.weight", f"{prefix}.self_attn.{proj}.weight", copied, skipped)
                self._copy_qwen_param(dst_layer, f"self_attn.{proj}.bias", src_layer, f"self_attn.{proj}.bias", f"{prefix}.self_attn.{proj}.bias", copied, skipped, optional=True)
            self._copy_qwen_param(dst_layer, "self_attn.q_norm.weight", src_layer, "self_attn.q_norm.weight", f"{prefix}.self_attn.q_norm", copied, skipped)
            self._copy_qwen_param(dst_layer, "self_attn.k_norm.weight", src_layer, "self_attn.k_norm.weight", f"{prefix}.self_attn.k_norm", copied, skipped)
            for proj in ("gate_proj", "up_proj", "down_proj"):
                self._copy_qwen_param(dst_layer, f"mlp.{proj}.weight", src_layer, f"mlp.{proj}.weight", f"{prefix}.mlp.{proj}.weight", copied, skipped)
                self._copy_qwen_param(dst_layer, f"mlp.{proj}.bias", src_layer, f"mlp.{proj}.bias", f"{prefix}.mlp.{proj}.bias", copied, skipped, optional=True)

            if bool(getattr(action_cfg, "qwen_init_adarms_passthrough", True)):
                gate_bias = float(getattr(action_cfg, "qwen_init_adarms_gate_bias", 5.0))
                self._init_adarms_as_qwen_passthrough(dst_layer.input_layernorm, gate_bias=gate_bias, labels=adarms_init, label=f"{prefix}.input_adarms")
                self._init_adarms_as_qwen_passthrough(dst_layer.post_attention_layernorm, gate_bias=gate_bias, labels=adarms_init, label=f"{prefix}.post_adarms")

        self._copy_qwen_param(self.action_expert, "norm.norm.weight", language_model, "norm.weight", "action_expert.final_norm<-qwen_lm.norm", copied, skipped)
        if bool(getattr(action_cfg, "qwen_init_adarms_passthrough", True)):
            gate_bias = float(getattr(action_cfg, "qwen_init_adarms_gate_bias", 5.0))
            self._init_adarms_as_qwen_passthrough(self.action_expert.norm, gate_bias=gate_bias, labels=adarms_init, label="action_expert.final_adarms")

        report = {
            "source_start": source_start,
            "source_end": source_start + init_n - 1,
            "target_start": 0,
            "target_end": init_n - 1,
            "copied_tensors": len(copied),
            "skipped_tensors": len(skipped),
            "adarms_passthrough_modules": len(adarms_init),
            "skipped_examples": skipped[:8],
        }
        print(
            "[Rhino] initialized action expert from Qwen text "
            f"layers {report['source_start']}-{report['source_end']}: "
            f"copied={report['copied_tensors']} skipped={report['skipped_tensors']} "
            f"adarms_passthrough={report['adarms_passthrough_modules']}",
            flush=True,
        )
        if skipped:
            print(f"[Rhino] Qwen init skipped examples: {report['skipped_examples']}", flush=True)
        return report

    def _prepare_batch(self, examples: List[dict]):
        batch_images = [example["image"] for example in examples]
        instructions = [example["lang"] for example in examples]
        view_roles = [example.get("view_roles") for example in examples]
        view_modalities = [example.get("view_modalities") for example in examples]
        # pre_resize_images: resize images before passing to Qwen processor.
        # Default False — aligned with Qwen3-VL official: data pipeline provides
        # ≥256x256 images and Qwen processor handles smart_resize internally
        # via framework.qwenvl.image_min/max_pixels. Old 224 resize path is gone.
        # Set to True only for legacy 224 configs that really need explicit resize.
        pre_resize = bool(getattr(self.config.datasets.vla_data, "pre_resize_images", False))
        if pre_resize:
            train_obs_image_size = getattr(self.config.datasets.vla_data, "image_size", None)
            if train_obs_image_size:
                from omegaconf import OmegaConf
                image_size_container = OmegaConf.to_container(train_obs_image_size, resolve=True)
                if isinstance(image_size_container, (list, tuple)):
                    target_size = (int(image_size_container[0]), int(image_size_container[1]))
                else:
                    target_size = (int(image_size_container), int(image_size_container))
                batch_images = resize_images(batch_images, target_size=target_size)
        return batch_images, instructions, view_roles, view_modalities

    def _encode_prefix(self, examples: List[dict]):
        profile_timing = bool(getattr(self, "_profile_timing", False))
        self._last_prefix_timing = {}
        device = next(self.parameters()).device
        if profile_timing:
            if torch.cuda.is_available():
                torch.cuda.synchronize(device)
            t0 = time.perf_counter()
        # Fast path: when preprocess_qwen_in_dataloader=true, the dataloader worker already ran the
        # Qwen processor (apply_chat_template) and stashed the CPU inputs on examples[0]["_qwen_inputs_cpu"],
        # moving the dominant processor cost off the forward critical path (hidden by prefetch). Only the
        # H2D copy remains here. When absent (default/legacy), fall back to the in-forward processor call.
        precomputed = examples[0].get("_qwen_inputs_cpu") if examples else None
        if precomputed is not None:
            qwen_inputs = precomputed.to(device, non_blocking=True)
            if profile_timing:
                if torch.cuda.is_available():
                    torch.cuda.synchronize(device)
                t1 = t2 = time.perf_counter()
        else:
            batch_images, instructions, view_roles, view_modalities = self._prepare_batch(examples)
            if profile_timing:
                if torch.cuda.is_available():
                    torch.cuda.synchronize(device)
                t1 = time.perf_counter()
            qwen_inputs = self.qwen.build_qwenvl_inputs(
                images=batch_images,
                instructions=instructions,
                view_roles=view_roles,
                view_modalities=view_modalities,
                profile_timing=profile_timing,
            )
            if profile_timing:
                if torch.cuda.is_available():
                    torch.cuda.synchronize(device)
                t2 = time.perf_counter()
        model_inputs = {k: v for k, v in qwen_inputs.items() if k != "labels"}
        model_inputs["use_cache"] = True
        model_inputs["return_dict"] = True
        model_inputs["output_hidden_states"] = True
        input_ids = model_inputs.get("input_ids")
        pixel_values = model_inputs.get("pixel_values")
        self._last_prefix_timing = {
            "qwen/input_batch_size": float(len(examples)),
            "qwen/input_token_count": float(input_ids.numel()) if input_ids is not None else 0.0,
            "qwen/input_pixel_value_count": float(pixel_values.numel()) if pixel_values is not None else 0.0,
        }

        freeze_qwen = not any(p.requires_grad for p in self.qwen.parameters())
        qwen_forward = self.qwen.model.model
        if freeze_qwen:
            base_qwen = self.qwen.model
            if hasattr(base_qwen, "get_base_model"):
                base_qwen = base_qwen.get_base_model()
            qwen_forward = getattr(base_qwen, "model", qwen_forward)
        context = torch.no_grad() if freeze_qwen else torch.enable_grad()
        with context:
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
                outputs = qwen_forward(**model_inputs)
        if profile_timing:
            if torch.cuda.is_available():
                torch.cuda.synchronize(device)
            t3 = time.perf_counter()

        last_hidden = getattr(outputs, "last_hidden_state", None)
        if last_hidden is None:
            hidden_states = getattr(outputs, "hidden_states", None)
            if not hidden_states:
                raise RuntimeError("Qwen3-VL output has neither last_hidden_state nor hidden_states")
            last_hidden = hidden_states[-1]
        prefix_mask = model_inputs["attention_mask"].to(dtype=torch.bool, device=last_hidden.device)
        if profile_timing:
            if torch.cuda.is_available():
                torch.cuda.synchronize(device)
            t4 = time.perf_counter()
            self._last_prefix_timing.update(
                {
                    "time/qwen/prepare_batch_sec": t1 - t0,
                    "time/qwen/processor_h2d_sec": t2 - t1,
                    "time/qwen/prefix_forward_sec": t3 - t2,
                    "time/qwen/prefix_mask_sec": t4 - t3,
                    "time/qwen/input_batch_size": self._last_prefix_timing["qwen/input_batch_size"],
                    "time/qwen/input_token_count": self._last_prefix_timing["qwen/input_token_count"],
                    "time/qwen/input_pixel_value_count": self._last_prefix_timing["qwen/input_pixel_value_count"],
                }
            )
            self._last_prefix_timing.update(
                {
                    key: float(value)
                    for key, value in getattr(self.qwen, "_last_input_timing", {}).items()
                }
            )
        return outputs.past_key_values, prefix_mask

    def sample_noise(self, shape, device):
        return torch.randn(shape, dtype=torch.float32, device=device)

    def sample_time(self, batch_size, device):
        dist = torch.distributions.Beta(
            torch.tensor(self.beta_alpha, dtype=torch.float32, device=device),
            torch.tensor(self.beta_beta, dtype=torch.float32, device=device),
        )
        return (dist.sample((batch_size,)) * 0.999 + 0.001).to(torch.float32)

    def _prepare_masked_actions(self, examples: List[dict], device: torch.device) -> tuple[Tensor, Tensor]:
        actions = torch.as_tensor(np.asarray([ex["action"] for ex in examples]), dtype=torch.float32).to(
            device=device,
            non_blocking=True,
        )
        if actions.ndim != 3:
            raise ValueError(f"RhinoVLA expects action [B,H,72], got {tuple(actions.shape)}")
        actions = actions[:, -self.action_horizon :, : self.action_dim]
        if actions.shape[1:] != (self.action_horizon, self.action_dim):
            raise ValueError(
                f"RhinoVLA expects action horizon/dim {(self.action_horizon, self.action_dim)}, "
                f"got {tuple(actions.shape[1:])}"
            )
        if "action_mask" not in examples[0]:
            raise ValueError("RhinoVLA requires action_mask")
        masks = []
        for ex in examples:
            mask = np.asarray(ex["action_mask"], dtype=np.float32)
            if mask.ndim == 1:
                mask = np.broadcast_to(mask[None, :], (self.action_horizon, self.action_dim))
            masks.append(mask[-self.action_horizon :, : self.action_dim])
        action_mask = torch.as_tensor(np.asarray(masks), dtype=torch.float32).to(device=device, non_blocking=True)
        if action_mask.shape != actions.shape:
            raise ValueError(f"action_mask shape {tuple(action_mask.shape)} != action {tuple(actions.shape)}")
        return actions * action_mask, action_mask

    def _prepare_masked_state(self, examples: List[dict], device: torch.device) -> tuple[Tensor | None, Tensor | None]:
        if "state" not in examples[0] or self.state_dim <= 0:
            return None, None
        state = torch.as_tensor(np.asarray([ex["state"] for ex in examples]), dtype=torch.float32).to(
            device=device,
            non_blocking=True,
        )
        state = state[..., : self.state_dim]
        state_mask = None
        if "state_mask" in examples[0]:
            masks = []
            for ex in examples:
                mask = np.asarray(ex["state_mask"], dtype=np.float32)
                ex_state = np.asarray(ex["state"], dtype=np.float32)
                if mask.ndim == 1 and ex_state.ndim == 2:
                    mask = np.broadcast_to(mask[None, :], ex_state.shape)
                masks.append(mask[..., : self.state_dim])
            state_mask = torch.as_tensor(np.asarray(masks), dtype=torch.float32).to(device=device, non_blocking=True)
            if state_mask.shape != state.shape:
                raise ValueError(f"state_mask shape {tuple(state_mask.shape)} != state {tuple(state.shape)}")
            state = state * state_mask
        return state, state_mask

    def _action_forward_masked(
        self,
        prefix_key_values,
        prefix_mask: Tensor,
        state: Tensor | None,
        state_mask: Tensor | None,
        x_t: Tensor,
        action_mask: Tensor | None,
        time: Tensor,
    ) -> Tensor:
        return self.action_expert(
            prefix_key_values=prefix_key_values,
            prefix_mask=prefix_mask,
            state=state,
            state_mask=state_mask,
            x_t=x_t,
            action_mask=action_mask,
            time=time,
            prefix_layer_offset=self.prefix_layer_offset,
        )

    def forward(self, examples: List[dict] = None, **kwargs):
        if not examples:
            raise ValueError("examples must be a non-empty list")
        device = next(self.parameters()).device

        actions, action_mask = self._prepare_masked_actions(examples, device)
        state, state_mask = self._prepare_masked_state(examples, device)
        prefix_key_values, prefix_mask = self._encode_prefix(examples)

        noise = self.sample_noise(actions.shape, actions.device) * action_mask
        time = self.sample_time(actions.shape[0], actions.device)
        t = time[:, None, None]
        if self.flow_time_convention == "rhinovla":
            x_t = ((1 - t) * noise + t * actions) * action_mask
            target_velocity = (actions - noise) * action_mask
        else:
            x_t = (t * noise + (1 - t) * actions) * action_mask
            target_velocity = (noise - actions) * action_mask

        pred_velocity = self._action_forward_masked(
            prefix_key_values,
            prefix_mask,
            state,
            state_mask,
            x_t,
            action_mask,
            time,
        )
        per_elem_mse = (pred_velocity - target_velocity).pow(2)
        loss = (per_elem_mse * action_mask).sum() / action_mask.sum().clamp_min(1.0)

        pred_flat = (pred_velocity * action_mask).detach().float().flatten(1)
        target_flat = (target_velocity * action_mask).detach().float().flatten(1)
        per_elem_mse_det = per_elem_mse.detach().float()
        per_dim_mse = (per_elem_mse_det * action_mask).sum(dim=(0, 1)) / action_mask.sum(dim=(0, 1)).clamp_min(1.0)
        per_sample_mse = (per_elem_mse_det * action_mask).sum(dim=(1, 2)) / action_mask.sum(dim=(1, 2)).clamp_min(1.0)
        metrics = {
            "action_loss": loss,
            "loss_fm": loss.detach(),
            "fm/t_mean": time.detach().mean(),
            "fm/t_std": time.detach().std(unbiased=False),
            "fm/noise_norm": (noise * action_mask).detach().float().norm(dim=-1).mean(),
            "fm/action_norm": (actions * action_mask).detach().float().norm(dim=-1).mean(),
            "fm/x_t_norm": (x_t * action_mask).detach().float().norm(dim=-1).mean(),
            "fm/target_velocity_norm": (target_velocity * action_mask).detach().float().norm(dim=-1).mean(),
            "fm/pred_velocity_norm": (pred_velocity * action_mask).detach().float().norm(dim=-1).mean(),
            "fm/velocity_mse": loss.detach(),
            "fm/velocity_cosine": F.cosine_similarity(pred_flat, target_flat, dim=-1).mean(),
            "mask/active_dim_mean": action_mask[:, 0, :].sum(dim=-1).float().mean(),
            "mask/active_ratio": action_mask.float().mean(),
        }
        if state_mask is not None:
            metrics["mask/state_active_dim_mean"] = state_mask.reshape(state_mask.shape[0], -1, self.state_dim)[:, 0, :].sum(dim=-1).float().mean()
            metrics["mask/state_active_ratio"] = state_mask.float().mean()
        for i in range(per_dim_mse.shape[0]):
            metrics[f"fm/dim_mse/d{i:02d}"] = per_dim_mse[i]

        labels = [
            _rhino_metric_label(ex.get("_dataset_source_root") or ex.get("_dataset_id") or "unknown")
            for ex in examples
        ]
        for label in sorted(set(labels)):
            idx = torch.tensor([i for i, lab in enumerate(labels) if lab == label], device=device, dtype=torch.long)
            metrics[f"dataset_loss/{label}"] = per_sample_mse.index_select(0, idx).mean()
            metrics[f"dataset_sample_ratio/{label}"] = torch.tensor(float(idx.numel()) / float(len(examples)), device=device)
        return metrics

    @torch.no_grad()
    def predict_action(self, examples: List[dict] = None, num_steps: int | None = None, **kwargs) -> Tensor:
        if not examples:
            raise ValueError("examples must be a non-empty list")
        device = next(self.parameters()).device
        prefix_key_values, prefix_mask = self._encode_prefix(examples)
        bsz = prefix_mask.shape[0]

        state, state_mask = self._prepare_masked_state(examples, device)
        if "action_mask" in examples[0]:
            raw_masks = []
            for ex in examples:
                mask = np.asarray(ex["action_mask"], dtype=np.float32)
                if mask.ndim == 2:
                    mask = mask[0]
                raw_masks.append(mask[: self.action_dim])
            action_mask = torch.as_tensor(np.asarray(raw_masks), dtype=torch.float32, device=device)[:, None, :]
        else:
            action_mask = torch.ones((bsz, 1, self.action_dim), dtype=torch.float32, device=device)
        action_mask_full = action_mask.expand(-1, self.action_horizon, -1)

        steps = int(num_steps or self.num_inference_timesteps)
        x_t = self.sample_noise((bsz, self.action_horizon, self.action_dim), device=device) * action_mask
        if self.flow_time_convention == "rhinovla":
            dt = 1.0 / steps
            time = torch.zeros(bsz, dtype=torch.float32, device=device)
            for _ in range(steps):
                pred_velocity = self._action_forward_masked(
                    prefix_key_values,
                    prefix_mask,
                    state,
                    state_mask,
                    x_t,
                    action_mask_full,
                    time,
                )
                x_t = (x_t + dt * pred_velocity) * action_mask
                time = (time + dt).clamp(max=1.0)
        else:
            dt = -1.0 / steps
            time = torch.ones(bsz, dtype=torch.float32, device=device)
            for _ in range(steps):
                pred_velocity = self._action_forward_masked(
                    prefix_key_values,
                    prefix_mask,
                    state,
                    state_mask,
                    x_t,
                    action_mask_full,
                    time,
                )
                x_t = (x_t + dt * pred_velocity) * action_mask
                time = (time + dt).clamp(min=0.0)
        return x_t
