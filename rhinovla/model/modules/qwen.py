"""Qwen3-VL prefix encoder and preprocessing for RhinoVLA."""

import time
import torch
from typing import Any, Optional
from pathlib import Path
from transformers import Qwen3VLForConditionalGeneration, AutoConfig, AutoProcessor
from transformers.modeling_outputs import CausalLMOutputWithPast
from torch.nn.utils.rnn import pad_sequence
from transformers import BatchFeature

from qwen_vl_utils import process_vision_info


from accelerate.logging import get_logger

logger = get_logger(__name__)

IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = 151655
VIDEO_TOKEN_INDEX = 151656
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_VIDEO_TOKEN = "<video>"

# Qwen3-VL tokenizer vocabulary range reserved by the FAST-action variant.
# Plain RhinoVLA finetuning does not emit action tokens; this range is used
# only when optional text labels contain serialized action tokens.
_ACTION_TOKEN_MIN = 151669
_ACTION_TOKEN_MAX = 153716

_DEFAULT_VIEW_ROLE_VOCAB = (
    "image",
    "side",
    "hand",
    "top_head",
    "hand_left",
    "hand_right",
    "head",
    "head_center",
    "head_left",
    "head_right",
    "head_back",
    "back_left",
    "back_right",
    "cam_left_high",
    "cam_right_high",
    "cam_left_wrist",
    "cam_right_wrist",
    "cam_high",
)
_DEFAULT_VIEW_MODALITY_VOCAB = ("rgb", "fisheye")


def _resolve_pixel_budget(config):
    """Resolve (min_pixels, max_pixels) for the Qwen3-VL processor.

    IDENTICAL logic to Qwen.__init__ so the dataloader worker collate and the model
    load the AutoProcessor with the exact same pixel budget (else image_grid_thw / token count would
    differ -> input_ids/pixel_values mismatch -> loss jump). Single source of truth used by both.
    """
    qwenvl_config = config.framework.get("qwenvl", {})
    min_px = int(qwenvl_config.get("image_min_pixels", 0))
    max_px = int(qwenvl_config.get("image_max_pixels", 0))
    if max_px <= 0:
        image_size = config.datasets.vla_data.get("image_size", None)
        if image_size is None:
            raise ValueError(
                "Qwen3 processor pixel budget unset: provide framework.qwenvl.image_max_pixels "
                "or datasets.vla_data.image_size in your config (no hardcoded fallback)."
            )
        if hasattr(image_size, "__iter__"):
            from omegaconf import OmegaConf

            image_size = OmegaConf.to_container(image_size, resolve=True) if hasattr(image_size, "_metadata") else list(image_size)
            max_px = int(image_size[0]) * int(image_size[1])
        else:
            max_px = int(image_size) * int(image_size)
    if min_px <= 0:
        min_px = max_px
    return min_px, max_px


def _build_qwenvl_messages_cpu(
    *,
    images,
    instructions,
    solutions=None,
    view_roles=None,
    view_modalities=None,
    use_view_prompt: bool = True,
    allowed_roles=_DEFAULT_VIEW_ROLE_VOCAB,
    allowed_modalities=_DEFAULT_VIEW_MODALITY_VOCAB,
    cot_prompt=None,
    cot_prompt_present: bool = False,
):
    """Stateless mirror of Qwen._build_qwenvl_messages.

    All config-derived inputs are explicit args so this can run in a dataloader worker (no model
    instance). `cot_prompt_present` reproduces the `if "CoT_prompt" in vla_data:` KEY-PRESENCE test
    (NOT truthiness) so missing-key vs empty-string semantics match the in-forward path exactly.
    """
    messages = []
    assert len(images) == len(instructions), "Images and instructions must have the same length"
    allowed_roles = set(allowed_roles)
    allowed_modalities = set(allowed_modalities)
    for sample_idx, (imgs, instruction) in enumerate(zip(images, instructions)):
        content = []

        if cot_prompt_present:  # If using a grounding prompt to task
            prompt = (cot_prompt if cot_prompt is not None else "").replace("{instruction}", instruction)
        else:
            prompt = instruction

        if use_view_prompt:
            content.append({"type": "text", "text": f"Task: {prompt}\n\nViews:"})
            roles = (view_roles or [None] * len(images))[sample_idx] or []
            modalities = (view_modalities or [None] * len(images))[sample_idx] or []
            for image_idx, img in enumerate(imgs):
                role = str(roles[image_idx]) if image_idx < len(roles) else ("front" if image_idx == 0 else "external")
                modality = str(modalities[image_idx]) if image_idx < len(modalities) else "rgb"
                if role not in allowed_roles:
                    raise ValueError(f"Unknown view role '{role}'. Allowed roles: {sorted(allowed_roles)}")
                if modality not in allowed_modalities:
                    raise ValueError(
                        f"Unknown view modality '{modality}'. Allowed modalities: {sorted(allowed_modalities)}"
                    )
                content.append({"type": "text", "text": f"[{role} | {modality}]"})
                content.append({"type": "image", "image": img})
        else:
            content = [{"type": "image", "image": img} for img in imgs]
            content.append({"type": "text", "text": prompt})
        msg = [{"role": "user", "content": content}]

        if solutions is not None:
            solution = solutions[len(messages)]
            msg.append({"role": "assistant", "content": [{"type": "text", "text": solution}]})
        messages.append(msg)
    return messages


def build_qwenvl_cpu_inputs(
    processor,
    images,
    instructions,
    *,
    solutions=None,
    view_roles=None,
    view_modalities=None,
    use_view_prompt: bool = True,
    allowed_roles=_DEFAULT_VIEW_ROLE_VOCAB,
    allowed_modalities=_DEFAULT_VIEW_MODALITY_VOCAB,
    cot_prompt=None,
    cot_prompt_present: bool = False,
):
    """CPU-only Qwen3-VL input build, byte-identical to Qwen.build_qwenvl_inputs up to
    (but NOT including) the .to(device) transfer. Shared by the in-forward interface AND the dataloader
    worker collate so the two paths run the SAME code -> parity by construction.

    Returns a CPU BatchFeature (input_ids / attention_mask / pixel_values / image_grid_thw,
    + labels iff solutions is not None).
    """
    vision_process_kwargs = {"image_patch_size": 16}

    messages = _build_qwenvl_messages_cpu(
        images=images,
        instructions=instructions,
        solutions=solutions,
        view_roles=view_roles,
        view_modalities=view_modalities,
        use_view_prompt=use_view_prompt,
        allowed_roles=allowed_roles,
        allowed_modalities=allowed_modalities,
        cot_prompt=cot_prompt,
        cot_prompt_present=cot_prompt_present,
    )
    # Two-step path (verbatim from build_qwenvl_inputs): render text first, then let the processor
    # batch/pad text + vision tensors together. tokenize=True chat-template path is avoided.
    texts = [
        processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
        for m in messages
    ]
    image_inputs, video_inputs = process_vision_info(messages, **vision_process_kwargs)
    batch_inputs = processor(
        text=texts,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
        do_resize=False,
    )
    # if solutions, mask out the solution tokens in labels (training path is solutions=None -> skipped)
    if solutions is not None:
        action_token_min = _ACTION_TOKEN_MIN
        action_token_max = _ACTION_TOKEN_MAX
        labels = batch_inputs["input_ids"].clone()
        for i in range(labels.size(0)):
            seq = labels[i]
            mask_seq = (seq >= action_token_min) & (seq <= action_token_max)
            nonzero_indices = torch.nonzero(mask_seq, as_tuple=False)
            if nonzero_indices.numel() > 0:
                first_action_index = nonzero_indices[0].item()
                seq[:first_action_index] = IGNORE_INDEX
            else:
                seq[:] = IGNORE_INDEX
        labels[labels == processor.tokenizer.pad_token_id] = -100
        batch_inputs["labels"] = labels
    return batch_inputs


import torch.nn as nn


class Qwen(nn.Module):
    """Qwen3-VL prefix encoder used by RhinoVLA."""

    def __init__(self, config: Optional[dict] = None, **kwargs):
        """
        Initialize the Qwen3-VL wrapper.
        Following https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct

        """
        super().__init__()

        qwenvl_config = config.framework.get("qwenvl", {})
        model_id = qwenvl_config.get("base_vlm", "rhinovla/assets/qwen3_vl_processor")
        attn_implementation = qwenvl_config.get("attn_implementation", "flash_attention_2")
        model_path = Path(str(model_id))
        if not model_path.exists():
            raise FileNotFoundError(
                f"framework.qwenvl.base_vlm must point to a local Qwen3-VL config/processor asset directory, "
                f"got {model_id!r}. Finetuning initializes model structure from config and loads weights "
                "from trainer.pretrained_checkpoint; it does not download official Qwen3-VL weights."
            )

        qwen_config = AutoConfig.from_pretrained(model_path, local_files_only=True)
        qwen_config._attn_implementation = attn_implementation
        old_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.bfloat16)
        try:
            model = Qwen3VLForConditionalGeneration(qwen_config)
        finally:
            torch.set_default_dtype(old_dtype)
        if getattr(config.trainer, "enable_gradient_checkpointing", False):
            model.gradient_checkpointing_enable()
            model.config.use_cache = False
        # Pixel budget for Qwen3-VL vision encoder.
        # Prefer explicit framework.qwenvl.image_min/max_pixels; if absent, derive from
        # datasets.vla_data.image_size (NxN -> N*N pixels). 不再写死 256x256 fallback;
        # cfg 必须明确给出 image_size 或 image_min/max_pixels.
        min_px, max_px = _resolve_pixel_budget(config)
        print(f"[Qwen3] processor min_pixels={min_px} max_pixels={max_px} (~{int(max_px**0.5)}x{int(max_px**0.5)})")
        processor = AutoProcessor.from_pretrained(
            model_path,
            min_pixels=min_px,
            max_pixels=max_px,
            local_files_only=True,
        )
        processor.tokenizer.padding_side = "left"

        # Apply LoRA if configured
        lora_rank = getattr(qwenvl_config, "lora_rank", 0)
        if lora_rank and int(lora_rank) > 0:
            from peft import LoraConfig, get_peft_model
            lora_config = LoraConfig(
                r=int(lora_rank),
                lora_alpha=int(getattr(qwenvl_config, "lora_alpha", int(lora_rank) * 2)),
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                lora_dropout=float(getattr(qwenvl_config, "lora_dropout", 0.05)),
                bias="none",
                task_type="CAUSAL_LM",
            )
            model = get_peft_model(model, lora_config)
            model.print_trainable_parameters()

        self.model = model
        self.processor = processor
        self.config = config

        # alin qwen3 with qwen2.5
        self.model.config.hidden_size = self.model.config.text_config.hidden_size

        # only for fast base model
        if "-Action" in model_id:
            self._ACTION_TOKEN_MIN = _ACTION_TOKEN_MIN
            self._ACTION_TOKEN_MAX = _ACTION_TOKEN_MAX

    def forward(
        self,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        """
        Forward pass delegating to underlying Qwen2.5-VL backbone.
        """

        with torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = self.model(
                **kwargs,
            )

        return outputs

    def generate(
        self,
        **kwargs,
    ):
        """
        High-level generation interface (auto-regressive decoding), optionally vision-conditioned.

        Args:
            **kwargs: fully follow raw model.generate() signature.
        Returns:
            GenerateOutput | Model-dependent generation return.
        """
        with torch.autocast("cuda", dtype=torch.float16):
            generation_output = self.model.generate(
                **kwargs,
            )
        return generation_output

    def _build_qwenvl_messages(
        self,
        *,
        images: list[list[Any]],
        instructions: list[str],
        solutions: list[str] | None = None,
        view_roles: list[list[str]] | None = None,
        view_modalities: list[list[str]] | None = None,
    ) -> list[list[dict[str, Any]]]:
        """Build Qwen3-VL chat messages before processor tokenization.

        This is intentionally lightweight so View Registry prompt behavior can
        be smoke-tested without loading the full Qwen3-VL model.
        """

        vla_data = self.config.datasets.vla_data
        return _build_qwenvl_messages_cpu(
            images=images,
            instructions=instructions,
            solutions=solutions,
            view_roles=view_roles,
            view_modalities=view_modalities,
            use_view_prompt=bool(vla_data.get("use_view_registry_prompt", True)),
            allowed_roles=vla_data.get("view_role_vocab", _DEFAULT_VIEW_ROLE_VOCAB),
            allowed_modalities=vla_data.get("view_modality_vocab", _DEFAULT_VIEW_MODALITY_VOCAB),
            cot_prompt=vla_data.get("CoT_prompt", "") if "CoT_prompt" in vla_data else None,
            cot_prompt_present=("CoT_prompt" in vla_data),
        )

    def build_qwenvl_inputs(self, images, instructions, solutions=None, **kwargs):
        """
        Build model inputs from raw data (images + instructions + optional solutions).
        Follow official Qwen3-VL Instruct format: https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct
        """

        profile_timing = bool(kwargs.get("profile_timing", False))
        self._last_input_timing = {}
        t0 = time.perf_counter() if profile_timing else None

        # Delegate the CPU build to the shared module-level fn so the in-forward path and the
        # dataloader worker collate run THE SAME code (byte-identical inputs by construction).
        vla_data = self.config.datasets.vla_data
        batch_inputs = build_qwenvl_cpu_inputs(
            self.processor,
            images,
            instructions,
            solutions=solutions,
            view_roles=kwargs.get("view_roles"),
            view_modalities=kwargs.get("view_modalities"),
            use_view_prompt=bool(vla_data.get("use_view_registry_prompt", True)),
            allowed_roles=vla_data.get("view_role_vocab", _DEFAULT_VIEW_ROLE_VOCAB),
            allowed_modalities=vla_data.get("view_modality_vocab", _DEFAULT_VIEW_MODALITY_VOCAB),
            cot_prompt=vla_data.get("CoT_prompt", "") if "CoT_prompt" in vla_data else None,
            cot_prompt_present=("CoT_prompt" in vla_data),
        )
        t4 = time.perf_counter() if profile_timing else None

        device = self.model.device
        if profile_timing and torch.cuda.is_available():
            torch.cuda.synchronize(device)
        t6 = time.perf_counter() if profile_timing else None
        batch_inputs = batch_inputs.to(device, non_blocking=True)
        if profile_timing and torch.cuda.is_available():
            torch.cuda.synchronize(device)
        t7 = time.perf_counter() if profile_timing else None
        if profile_timing:
            input_ids = batch_inputs.get("input_ids")
            pixel_values = batch_inputs.get("pixel_values")
            self._last_input_timing = {
                "time/qwen/input_cpu_build_sec": t4 - t0,
                "time/qwen/input_h2d_sec": t7 - t6,
                "time/qwen/input_total_sec": t7 - t0,
                "time/qwen/input_batch_size": float(len(images)),
                "time/qwen/input_image_count": float(sum(len(imgs) for imgs in images)),
                "time/qwen/input_token_count": float(input_ids.numel()) if input_ids is not None else 0.0,
                "time/qwen/input_pixel_value_count": float(pixel_values.numel()) if pixel_values is not None else 0.0,
            }

        return batch_inputs
