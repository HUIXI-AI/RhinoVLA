# Copyright 2026 The RhinoVLA authors.
#
# Rhino suffix denoising action expert over a Qwen3-VL prefix KV cache.
#
# Modifications:
# - Replace PaliGemma/SigLIP prefix with Qwen3-VL prefix KV cache.
# - Use a Rhino action expert that can attend to Qwen prefix K/V.

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import torch
from torch import Tensor, nn
import torch.nn.functional as F

try:
    from transformers.cache_utils import Cache
    from transformers.models.qwen3_vl.modeling_qwen3_vl import (
        Qwen3VLTextConfig,
        Qwen3VLTextMLP,
        Qwen3VLTextRMSNorm,
        apply_rotary_pos_emb,
    )
except Exception:  # pragma: no cover - import-time compatibility guard
    Cache = object
    Qwen3VLTextConfig = object
    Qwen3VLTextMLP = None
    Qwen3VLTextRMSNorm = None
    apply_rotary_pos_emb = None


def create_sinusoidal_pos_embedding(
    time: Tensor,
    dimension: int,
    min_period: float = 4e-3,
    max_period: float = 4.0,
) -> Tensor:
    """Rhino scalar timestep embedding."""
    if dimension % 2 != 0:
        raise ValueError(f"dimension must be even, got {dimension}")
    if time.ndim != 1:
        raise ValueError(f"time must be shape (B,), got {tuple(time.shape)}")

    dtype = torch.float64 if time.device.type != "cpu" else torch.float32
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=time.device)
    period = min_period * (max_period / min_period) ** fraction
    phase = time[:, None].to(dtype) * (2 * math.pi / period)[None, :]
    return torch.cat([torch.sin(phase), torch.cos(phase)], dim=-1).to(torch.float32)


def make_suffix_attn_mask(prefix_mask: Tensor, suffix_mask: Tensor) -> Tensor:
    """Build Rhino suffix attention mask.

    Returned mask has shape (B, 1, S, P + S), where valid cells are True.
    Suffix tokens can attend to all valid prefix tokens and all valid suffix
    tokens. Prefix tokens are not queried in the suffix-only denoise path.
    """
    if prefix_mask.ndim != 2 or suffix_mask.ndim != 2:
        raise ValueError("prefix_mask and suffix_mask must be rank-2 bool tensors")
    bsz, prefix_len = prefix_mask.shape
    if suffix_mask.shape[0] != bsz:
        raise ValueError("prefix_mask and suffix_mask batch sizes differ")

    suffix_len = suffix_mask.shape[1]
    prefix_visible = prefix_mask[:, None, :].expand(bsz, suffix_len, prefix_len)
    suffix_visible = suffix_mask[:, None, :] & suffix_mask[:, :, None]
    full_visible = torch.cat([prefix_visible, suffix_visible], dim=-1)
    return full_visible[:, None, :, :]


def bool_mask_to_attention_bias(mask: Tensor, dtype: torch.dtype) -> Tensor:
    """Convert a bool visibility mask to additive attention bias."""
    min_value = torch.finfo(dtype).min
    return torch.where(mask, torch.zeros((), dtype=dtype, device=mask.device), torch.full((), min_value, dtype=dtype, device=mask.device))


@dataclass
class RhinoActionExpertConfig:
    """Rhino action expert config aligned to Qwen3-VL attention/cache shape."""

    action_dim: int = 72
    state_dim: int = 72
    action_horizon: int = 30
    width: int = 1024
    depth: int = 10
    mlp_dim: int = 3072
    num_attention_heads: int = 16
    num_key_value_heads: int = 8
    head_dim: int = 128
    rms_norm_eps: float = 1e-6
    rope_theta: float = 5_000_000.0
    hidden_act: str = "silu"
    attention_dropout: float = 0.0
    attention_bias: bool = False
    use_state_token: bool = True
    use_adarms: bool = True
    use_mask_condition: bool = False

    @classmethod
    def from_qwen_text_config(
        cls,
        qwen_text_config: Qwen3VLTextConfig,
        *,
        action_dim: int,
        state_dim: int = 72,
        action_horizon: int = 30,
        width: int = 1024,
        depth: int = 10,
        mlp_dim: int = 3072,
        use_state_token: bool = True,
        use_adarms: bool = True,
        use_mask_condition: bool = False,
    ) -> "RhinoActionExpertConfig":
        return cls(
            action_dim=action_dim,
            state_dim=state_dim,
            action_horizon=action_horizon,
            width=width,
            depth=depth,
            mlp_dim=mlp_dim,
            num_attention_heads=int(qwen_text_config.num_attention_heads),
            num_key_value_heads=int(qwen_text_config.num_key_value_heads),
            head_dim=int(qwen_text_config.head_dim),
            rms_norm_eps=float(qwen_text_config.rms_norm_eps),
            rope_theta=float(getattr(qwen_text_config, "rope_theta", None) or qwen_text_config.rope_parameters["rope_theta"]),
            hidden_act=str(qwen_text_config.hidden_act),
            attention_dropout=float(getattr(qwen_text_config, "attention_dropout", 0.0)),
            attention_bias=bool(getattr(qwen_text_config, "attention_bias", False)),
            use_state_token=use_state_token,
            use_adarms=use_adarms,
            use_mask_condition=use_mask_condition,
        )

    def to_qwen_text_config(self) -> Qwen3VLTextConfig:
        return Qwen3VLTextConfig(
            hidden_size=self.width,
            intermediate_size=self.mlp_dim,
            num_hidden_layers=self.depth,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_key_value_heads,
            head_dim=self.head_dim,
            rms_norm_eps=self.rms_norm_eps,
            rope_theta=self.rope_theta,
            hidden_act=self.hidden_act,
            attention_dropout=self.attention_dropout,
            attention_bias=self.attention_bias,
            vocab_size=1,
        )


class AdaRMSNorm(nn.Module):
    """Qwen RMSNorm with optional Rhino adaptive modulation."""

    def __init__(self, hidden_size: int, eps: float, cond_dim: int | None = None):
        super().__init__()
        self.norm = Qwen3VLTextRMSNorm(hidden_size, eps=eps)
        self.cond = nn.Linear(cond_dim, hidden_size * 3) if cond_dim is not None else None

    def forward(self, hidden_states: Tensor, cond: Tensor | None = None) -> tuple[Tensor, Tensor | None]:
        hidden_states = self.norm(hidden_states)
        if self.cond is None or cond is None:
            return hidden_states, None
        scale, shift, gate = self.cond(cond).chunk(3, dim=-1)
        hidden_states = hidden_states * (1 + scale[:, None, :]) + shift[:, None, :]
        return hidden_states, gate[:, None, :]


def gated_residual(residual: Tensor, update: Tensor, gate: Tensor | None) -> Tensor:
    if gate is None:
        return residual + update
    return residual + torch.tanh(gate) * update


def _repeat_kv(hidden_states: Tensor, n_rep: int) -> Tensor:
    if n_rep == 1:
        return hidden_states
    bsz, num_kv_heads, seq_len, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(bsz, num_kv_heads, n_rep, seq_len, head_dim)
    return hidden_states.reshape(bsz, num_kv_heads * n_rep, seq_len, head_dim)


class RhinoActionAttention(nn.Module):
    """Rhino suffix attention over Qwen3-VL prefix K/V tensors."""

    def __init__(self, config: RhinoActionExpertConfig):
        super().__init__()
        self.config = config
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_kv_heads
        self.head_dim = config.head_dim
        self.q_dim = self.num_heads * self.head_dim
        self.kv_dim = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5

        self.q_proj = nn.Linear(config.width, self.q_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(config.width, self.kv_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(config.width, self.kv_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.q_dim, config.width, bias=config.attention_bias)
        self.q_norm = Qwen3VLTextRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3VLTextRMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: Tensor,
        *,
        position_embeddings: tuple[Tensor, Tensor],
        prefix_kv: tuple[Tensor, Tensor] | None,
        attention_mask: Tensor,
    ) -> Tensor:
        bsz, suffix_len, _ = hidden_states.shape
        q = self.q_norm(self.q_proj(hidden_states).view(bsz, suffix_len, self.num_heads, self.head_dim)).transpose(1, 2)
        k = self.k_norm(self.k_proj(hidden_states).view(bsz, suffix_len, self.num_kv_heads, self.head_dim)).transpose(1, 2)
        v = self.v_proj(hidden_states).view(bsz, suffix_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        if prefix_kv is not None:
            prefix_k, prefix_v = prefix_kv
            k = torch.cat([prefix_k.to(k.dtype), k], dim=2)
            v = torch.cat([prefix_v.to(v.dtype), v], dim=2)

        k = _repeat_kv(k, self.num_key_value_groups)
        v = _repeat_kv(v, self.num_key_value_groups)

        attn_bias = bool_mask_to_attention_bias(attention_mask, q.dtype)
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_bias,
            dropout_p=self.config.attention_dropout if self.training else 0.0,
            scale=self.scaling,
        )
        out = out.transpose(1, 2).reshape(bsz, suffix_len, self.q_dim)
        return self.o_proj(out)


class RhinoActionLayer(nn.Module):
    def __init__(self, config: RhinoActionExpertConfig):
        super().__init__()
        cond_dim = config.width if config.use_adarms else None
        self.input_layernorm = AdaRMSNorm(config.width, eps=config.rms_norm_eps, cond_dim=cond_dim)
        self.self_attn = RhinoActionAttention(config)
        self.post_attention_layernorm = AdaRMSNorm(config.width, eps=config.rms_norm_eps, cond_dim=cond_dim)
        self.mlp = Qwen3VLTextMLP(config.to_qwen_text_config())

    def forward(
        self,
        hidden_states: Tensor,
        *,
        position_embeddings: tuple[Tensor, Tensor],
        prefix_kv: tuple[Tensor, Tensor] | None,
        attention_mask: Tensor,
        adarms_cond: Tensor | None,
    ) -> Tensor:
        residual = hidden_states
        hidden_states, gate = self.input_layernorm(hidden_states, adarms_cond)
        hidden_states = self.self_attn(
            hidden_states,
            position_embeddings=position_embeddings,
            prefix_kv=prefix_kv,
            attention_mask=attention_mask,
        )
        hidden_states = gated_residual(residual, hidden_states, gate)

        residual = hidden_states
        hidden_states, gate = self.post_attention_layernorm(hidden_states, adarms_cond)
        hidden_states = self.mlp(hidden_states)
        hidden_states = gated_residual(residual, hidden_states, gate)
        return hidden_states


class _RhinoActionTransformer(nn.Module):
    """Transformer stack for Rhino suffix denoising."""

    def __init__(self, config: RhinoActionExpertConfig, qwen_rotary_emb: nn.Module):
        super().__init__()
        self.config = config
        self.qwen_rotary_emb = qwen_rotary_emb
        self.layers = nn.ModuleList([RhinoActionLayer(config) for _ in range(config.depth)])
        self.norm = AdaRMSNorm(config.width, eps=config.rms_norm_eps, cond_dim=config.width if config.use_adarms else None)

    def forward(
        self,
        suffix_embeds: Tensor,
        *,
        prefix_key_values: Iterable[tuple[Tensor, Tensor]] | Cache,
        prefix_mask: Tensor,
        suffix_mask: Tensor,
        position_ids: Tensor,
        adarms_cond: Tensor | None,
        prefix_layer_offset: int = 0,
    ) -> Tensor:
        attention_mask = make_suffix_attn_mask(prefix_mask, suffix_mask)
        hidden_states = suffix_embeds
        position_embeddings = self.qwen_rotary_emb(hidden_states, position_ids)

        for layer_idx, layer in enumerate(self.layers):
            prefix_kv = get_cache_layer(prefix_key_values, prefix_layer_offset + layer_idx)
            hidden_states = layer(
                hidden_states,
                position_embeddings=position_embeddings,
                prefix_kv=prefix_kv,
                attention_mask=attention_mask,
                adarms_cond=adarms_cond,
            )
        hidden_states, _ = self.norm(hidden_states, adarms_cond)
        return hidden_states


def get_cache_layer(cache: Iterable[tuple[Tensor, Tensor]] | Cache, layer_idx: int) -> tuple[Tensor, Tensor]:
    """Extract a `(key, value)` pair from a Transformers Cache or tuple/list cache."""
    if hasattr(cache, "layers"):
        layer = cache.layers[layer_idx]
        return layer.keys, layer.values
    if hasattr(cache, "key_cache") and hasattr(cache, "value_cache"):
        return cache.key_cache[layer_idx], cache.value_cache[layer_idx]
    item = list(cache)[layer_idx]
    if len(item) < 2:
        raise ValueError("cache layer must contain at least key and value")
    return item[0], item[1]


class RhinoActionIO(nn.Module):
    """Rhino state/action/timestep projection and flow head."""

    def __init__(self, config: RhinoActionExpertConfig):
        super().__init__()
        self.config = config
        self.action_in_proj = nn.Linear(config.action_dim, config.width)
        self.action_out_proj = nn.Linear(config.width, config.action_dim)
        self.action_time_mlp_in = nn.Linear(config.width * 2, config.width)
        self.action_time_mlp_out = nn.Linear(config.width, config.width)
        self.state_proj = nn.Linear(config.state_dim, config.width) if config.use_state_token and config.state_dim > 0 else None

    def embed_suffix(self, state: Tensor | None, noisy_actions: Tensor, timestep: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        action_embeds = self.action_in_proj(noisy_actions)
        time_emb = create_sinusoidal_pos_embedding(timestep, self.config.width).to(action_embeds.device, action_embeds.dtype)
        per_action_time = time_emb[:, None, :].expand_as(action_embeds)
        action_time_embeds = torch.cat([action_embeds, per_action_time], dim=-1)
        action_time_embeds = self.action_time_mlp_out(F.silu(self.action_time_mlp_in(action_time_embeds)))

        tokens = []
        masks = []
        if self.state_proj is not None:
            if state is None:
                raise ValueError("state is required when use_state_token=True")
            state_token = self.state_proj(state[:, 0] if state.ndim == 3 else state)[:, None, :]
            tokens.append(state_token)
            masks.append(torch.ones(state_token.shape[:2], dtype=torch.bool, device=state_token.device))
        tokens.append(action_time_embeds)
        masks.append(torch.ones(action_time_embeds.shape[:2], dtype=torch.bool, device=action_time_embeds.device))
        return torch.cat(tokens, dim=1), torch.cat(masks, dim=1), time_emb

    def decode_actions(self, suffix_hidden: Tensor) -> Tensor:
        action_hidden = suffix_hidden[:, -self.config.action_horizon :]
        return self.action_out_proj(action_hidden.to(torch.float32))


def build_suffix_position_ids(
    prefix_mask: Tensor,
    suffix_mask: Tensor,
    prefix_rope_deltas: Tensor,
) -> Tensor:
    """Build AE suffix mRoPE ids from Qwen3-VL's continuation delta."""
    if prefix_mask.ndim != 2 or suffix_mask.ndim != 2:
        raise ValueError("prefix_mask and suffix_mask must be rank-2 tensors")
    if prefix_mask.shape[0] != suffix_mask.shape[0]:
        raise ValueError("prefix_mask and suffix_mask batch sizes differ")

    if prefix_rope_deltas.dtype != torch.long:
        raise TypeError(
            f"prefix_rope_deltas must have dtype torch.long, got {prefix_rope_deltas.dtype}"
        )
    if prefix_rope_deltas.ndim == 1:
        prefix_rope_deltas = prefix_rope_deltas[:, None]
    if tuple(prefix_rope_deltas.shape) != (prefix_mask.shape[0], 1):
        raise ValueError(
            "prefix_rope_deltas must have shape [batch, 1], got "
            f"{tuple(prefix_rope_deltas.shape)}"
        )
    prefix_offsets = prefix_mask.long().sum(dim=-1, keepdim=True)
    prefix_offsets = prefix_offsets + prefix_rope_deltas.to(prefix_offsets.device)

    suffix_pos = prefix_offsets + torch.cumsum(suffix_mask.long(), dim=1) - 1
    return suffix_pos[None, ...].expand(3, -1, -1)


class RhinoActionExpert(nn.Module):
    """Complete Rhino action expert: IO, mask conditioning, transformer, and head."""

    def __init__(self, config: RhinoActionExpertConfig, qwen_rotary_emb: nn.Module):
        super().__init__()
        self.config = config
        self.transformer = _RhinoActionTransformer(config, qwen_rotary_emb)
        self.io = RhinoActionIO(config)
        if config.use_mask_condition:
            self.state_mask_proj = nn.Linear(config.state_dim, config.width)
            self.action_mask_proj = nn.Linear(config.action_dim, config.width)
            nn.init.zeros_(self.state_mask_proj.weight)
            nn.init.zeros_(self.state_mask_proj.bias)
            nn.init.zeros_(self.action_mask_proj.weight)
            nn.init.zeros_(self.action_mask_proj.bias)
        else:
            self.state_mask_proj = None
            self.action_mask_proj = None

    @property
    def layers(self):
        return self.transformer.layers

    @property
    def norm(self):
        return self.transformer.norm

    def _apply_mask_condition(
        self,
        suffix_embeds: Tensor,
        state_mask: Tensor | None,
        action_mask: Tensor | None,
    ) -> Tensor:
        if not self.config.use_mask_condition:
            return suffix_embeds
        if self.action_mask_proj is None or self.state_mask_proj is None:
            raise RuntimeError("use_mask_condition=True but mask projectors are missing")
        if action_mask is None:
            raise ValueError("action_mask is required when use_mask_condition=True")

        action_token_start = 1 if self.io.state_proj is not None else 0
        if self.io.state_proj is not None:
            if state_mask is None:
                raise ValueError("state_mask is required when use_mask_condition=True and use_state_token=True")
            state_mask_token = state_mask[:, 0] if state_mask.ndim == 3 else state_mask
            suffix_embeds[:, :1] = suffix_embeds[:, :1] + self.state_mask_proj(
                state_mask_token.to(device=suffix_embeds.device, dtype=suffix_embeds.dtype)
            )[:, None, :]

        expected_action_tokens = suffix_embeds.shape[1] - action_token_start
        if action_mask.ndim == 2:
            action_mask = action_mask[:, None, :]
        if action_mask.shape[1] == 1 and expected_action_tokens > 1:
            action_mask = action_mask.expand(-1, expected_action_tokens, -1)
        if action_mask.shape[:2] != (suffix_embeds.shape[0], expected_action_tokens):
            raise ValueError(
                f"action_mask tokens {tuple(action_mask.shape[:2])} do not match suffix action tokens "
                f"{(suffix_embeds.shape[0], expected_action_tokens)}"
            )
        suffix_embeds[:, action_token_start:] = suffix_embeds[:, action_token_start:] + self.action_mask_proj(
            action_mask.to(device=suffix_embeds.device, dtype=suffix_embeds.dtype)
        )
        return suffix_embeds

    def forward(
        self,
        *,
        prefix_key_values: Iterable[tuple[Tensor, Tensor]] | Cache,
        prefix_mask: Tensor,
        state: Tensor | None,
        state_mask: Tensor | None,
        x_t: Tensor,
        action_mask: Tensor | None,
        time: Tensor,
        prefix_rope_deltas: Tensor,
        prefix_layer_offset: int = 0,
    ) -> Tensor:
        suffix_embeds, suffix_mask, adarms_cond = self.io.embed_suffix(state, x_t, time)
        suffix_embeds = self._apply_mask_condition(suffix_embeds, state_mask, action_mask)
        suffix_embeds = suffix_embeds.to(dtype=next(self.transformer.parameters()).dtype)
        position_ids = build_suffix_position_ids(
            prefix_mask,
            suffix_mask,
            prefix_rope_deltas,
        )
        suffix_hidden = self.transformer(
            suffix_embeds,
            prefix_key_values=prefix_key_values,
            prefix_mask=prefix_mask,
            suffix_mask=suffix_mask,
            position_ids=position_ids,
            adarms_cond=adarms_cond,
            prefix_layer_offset=prefix_layer_offset,
        )
        return self.io.decode_actions(suffix_hidden)


def estimate_rhino_action_params(config: RhinoActionExpertConfig) -> dict[str, int]:
    q_dim = config.num_attention_heads * config.head_dim
    kv_dim = config.num_key_value_heads * config.head_dim
    attn = config.width * q_dim + config.width * kv_dim * 2 + q_dim * config.width
    qk_norm = config.head_dim * 2
    mlp = 3 * config.width * config.mlp_dim
    rms = 2 * config.width
    adarms = 2 * (config.width * config.width * 3 + config.width * 3) if config.use_adarms else 0
    layer = attn + qk_norm + mlp + rms + adarms
    stack = layer * config.depth
    final_norm = config.width + (config.width * config.width * 3 + config.width * 3 if config.use_adarms else 0)
    io = (
        config.action_dim * config.width
        + config.width * config.action_dim
        + (config.width * 2) * config.width + config.width
        + config.width * config.width + config.width
    )
    if config.use_state_token and config.state_dim > 0:
        io += config.state_dim * config.width + config.width
    mask_condition = 0
    if config.use_mask_condition:
        mask_condition += config.state_dim * config.width + config.width
        mask_condition += config.action_dim * config.width + config.width
    return {
        "attention_per_layer": attn + qk_norm,
        "mlp_per_layer": mlp,
        "adarms_per_layer": adarms,
        "layer_total": layer,
        "expert_stack": stack,
        "final_norm": final_norm,
        "io": io,
        "mask_condition": mask_condition,
        "total": stack + final_norm + io + mask_condition,
    }
